#!/usr/bin/env python3
"""What does this Databento key actually deliver? Connect and find out.

    python scripts/dbn_probe.py --symbols ES.c.0 --seconds 20

Subscribes to each candidate schema in turn, waits, and reports what
arrived: the record type, whether it carries a book, whether it carries
a trade, and the first few records with their real field values.

WHY THIS EXISTS

A subscription that is not entitled, or a schema whose records have no
book in them, both look identical from the dashboard: a panel that
never fills. They need opposite fixes. This answers which one it is,
with the records themselves rather than an inference.

WHAT EACH SCHEMA CARRIES (verified against the databento package, not
assumed -- the probe re-checks on your own data)

    trades    TradeMsg   no levels, action 'T'   -> candles, NO OFI
    bbo-1s    BBOMsg     1 level, NO action      -> OFI, no trades
    bbo-1m    BBOMsg     as above, once a minute
    mbp-1     MBP1Msg    1 level + action        -> both
    mbp-10    MBP10Msg   10 levels + action      -> both, plus depth
    tbbo      MBP1Msg    trade + the book at it  -> both

So `trades` alone cannot produce an OFI reading: there is no bid or ask
in the record to difference. That is not a bug to fix, it is the wrong
schema for the question.

IT READS AND PRINTS. It places nothing, stores nothing, and stops on
its own.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liqmap import dbn                                      # noqa: E402

DATASET = "GLBX.MDP3"
# How long after the package's own timeout the watchdog gives up on it.
WATCHDOG_GRACE = 5.0
CANDIDATES = ("mbp-10", "mbp-1", "tbbo", "bbo-1s", "bbo-1m", "trades")


def describe(rec) -> dict:
    """What this one record actually carries."""
    levels = dbn._get(rec, "levels")
    if levels is not None and not isinstance(levels, (list, tuple)):
        levels = [levels]
    out = {
        "type": type(rec).__name__,
        "action": dbn._action(rec),
        "side": dbn._side(rec),
        "levels": len(levels) if levels else 0,
        "ts": dbn.seconds(dbn._get(rec, "ts_event")),
        "sequence": dbn._get(rec, "sequence"),
        "instrument": dbn._get(rec, "instrument_id"),
    }
    px = dbn._get(rec, "price")
    if px is not None:
        out["price"] = dbn.price(px)
        out["size"] = dbn._get(rec, "size")
    if levels:
        lv = levels[0]
        out["bid"] = dbn.price(dbn._get(lv, "bid_px", 0))
        out["bid_sz"] = dbn._get(lv, "bid_sz", 0)
        out["ask"] = dbn.price(dbn._get(lv, "ask_px", 0))
        out["ask_sz"] = dbn._get(lv, "ask_sz", 0)
    out["trade"] = dbn.trade_from(rec) is not None
    out["has_book"] = bool(out["levels"]) and out.get("bid", 0) > 0
    return out


def verdict(rows: list[dict]) -> str:
    """What this schema can drive, from what actually arrived."""
    if not rows:
        return "NOTHING ARRIVED"
    book = sum(1 for r in rows if r["has_book"])
    trade = sum(1 for r in rows if r["trade"])
    bits = []
    bits.append("OFI yes" if book else "OFI NO (no book in the record)")
    bits.append("candles yes" if trade
                else "candles NO (no trade in the record)")
    return f"{book}/{len(rows)} with a book, {trade}/{len(rows)} with a " \
           f"trade -> " + ", ".join(bits)


def probe(schema: str, symbols: list[str], dataset: str, stype: str,
          seconds: float, show: int) -> dict:
    """Listen to one schema for `seconds`, whatever arrives -- or doesn't.

    CALLBACKS, NOT ITERATION. Iterating the client blocks in the socket
    read until a record arrives, so a deadline checked inside that loop
    is only consulted when there is already something to look at. A
    schema that accepts the subscription and then sends nothing -- a
    quiet market, precisely the case this script exists to identify --
    never reaches the check and the probe hangs for good.

    `block_for_close(timeout=)` is a real wall-clock deadline: the
    package calls `terminate()` when it expires, whether or not a byte
    ever came down the wire. The watchdog below is a second line for
    the case where the session never opens at all.
    """
    import databento as db
    out = {"schema": schema, "records": 0, "rows": [], "error": "",
           "elapsed": 0.0}
    client, watchdog = None, None
    started = time.time()

    def on_record(rec) -> None:
        name = type(rec).__name__
        # Session chatter, not market data.
        if name in ("SystemMsg", "SymbolMappingMsg", "ErrorMsg"):
            if name == "ErrorMsg" and not out["error"]:
                out["error"] = str(getattr(rec, "err", rec))
            return
        out["records"] += 1
        if len(out["rows"]) < show:
            out["rows"].append(describe(rec))

    def on_error(exc: Exception) -> None:
        if not out["error"]:
            out["error"] = f"{type(exc).__name__}: {exc}"

    try:
        client = db.Live()
        client.subscribe(dataset=dataset, schema=schema,
                         stype_in=stype, symbols=symbols)
        client.add_callback(on_record, on_error)
        client.start()
        # Belt and braces: if the package's own timeout somehow does
        # not fire, this does. A probe that hangs is worse than a
        # probe that reports nothing, because it tells you nothing
        # AND costs you the terminal.
        watchdog = threading.Timer(
            seconds + WATCHDOG_GRACE,
            lambda: _hush(client.terminate))
        watchdog.daemon = True
        watchdog.start()
        client.block_for_close(timeout=seconds)
    except Exception as exc:                            # noqa: BLE001
        if not out["error"]:
            out["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if watchdog is not None:
            watchdog.cancel()
        if client is not None:
            _hush(client.terminate)
        out["elapsed"] = round(time.time() - started, 1)
    return out


def _hush(fn) -> None:
    """Call it; a client that never opened raises on being closed."""
    try:
        fn()
    except Exception:                                   # noqa: BLE001
        pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--symbols", default="ES.c.0")
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--stype-in", default="continuous")
    ap.add_argument("--seconds", type=float, default=15.0,
                    help="how long to listen per schema")
    ap.add_argument("--show", type=int, default=3,
                    help="records to print per schema")
    ap.add_argument("--schemas", default=",".join(CANDIDATES))
    args = ap.parse_args(argv)

    if not os.environ.get("DATABENTO_API_KEY"):
        print("DATABENTO_API_KEY is not set.\n"
              "    export DATABENTO_API_KEY=<your key>", file=sys.stderr)
        return 2
    try:
        import databento  # noqa: F401
    except ImportError:
        print("databento is not installed.\n    pip install databento",
              file=sys.stderr)
        return 2

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    # De-duplicated: the gateway rejects the same schema twice, and a
    # repeat in a comma list is easy to type and invisible in the error
    # that comes back from the far end.
    schemas, _seen = [], set()
    for x in args.schemas.split(","):
        n = x.strip().lower()
        if n and n not in _seen:
            _seen.add(n)
            schemas.append(n)

    print(f"dataset   {args.dataset}")
    print(f"symbols   {', '.join(symbols)}  (stype_in={args.stype_in})")
    print(f"listening {args.seconds:.0f}s per schema\n")
    print("NOTE: outside CME trading hours every schema is legitimately "
          "quiet. A clean")
    print("subscription with 0 records is a closed market, not a broken "
          "key -- the error")
    print("line is what separates them. Each schema returns after the "
          "listen window")
    print(f"whether or not anything arrives, so the whole run takes about "
          f"{len(schemas) * args.seconds:.0f}s.\n")

    results = []
    for schema in schemas:
        print(f"--- {schema}", flush=True)
        print(f"    listening {args.seconds:.0f}s ...", end="", flush=True)
        r = probe(schema, symbols, args.dataset, args.stype_in,
                  args.seconds, args.show)
        results.append(r)
        print(f"\r    {' ' * 28}\r", end="")
        if r["error"]:
            print(f"    ERROR  {r['error']}")
        print(f"    {r['records']:,} records in {r['elapsed']:.1f}s")
        for row in r["rows"]:
            bits = [f"{row['type']}", f"action={row['action']!r}",
                    f"levels={row['levels']}"]
            if "bid" in row:
                bits.append(f"bid {row['bid']}x{row['bid_sz']}")
                bits.append(f"ask {row['ask']}x{row['ask_sz']}")
            if "price" in row:
                bits.append(f"last {row['price']}x{row['size']}")
            bits.append("TRADE" if row["trade"] else "-")
            print("      " + "  ".join(str(b) for b in bits))
        if r["records"]:
            print(f"    {verdict(r['rows'])}")
        print()

    print("=" * 62)
    print("SUMMARY")
    ok_ofi, ok_candles = [], []
    for r in results:
        if not r["records"]:
            state = "nothing" if not r["error"] else "error"
        else:
            book = any(x["has_book"] for x in r["rows"])
            trade = any(x["trade"] for x in r["rows"])
            if book:
                ok_ofi.append(r["schema"])
            if trade:
                ok_candles.append(r["schema"])
            state = ("book+trade" if book and trade else
                     "book only" if book else
                     "trade only" if trade else "neither")
        print(f"  {r['schema']:<9} {r['records']:>7,} records   {state}"
              + (f"   {r['error'][:60]}" if r["error"] else ""))
    print()
    print(f"  can drive OFI     : {', '.join(ok_ofi) or 'NONE'}")
    print(f"  can drive candles : {', '.join(ok_candles) or 'NONE'}")
    print()
    if ok_ofi and ok_candles:
        both = [s for s in ok_ofi if s in ok_candles]
        if both:
            print(f"  -> one subscription does everything: {both[0]}")
        else:
            print(f"  -> subscribe to BOTH {ok_ofi[0]} (for OFI) and "
                  f"{ok_candles[0]} (for candles)")
    elif ok_ofi:
        print(f"  -> {ok_ofi[0]} gives OFI; nothing here gives trades, "
              f"so no candles")
    elif ok_candles:
        print(f"  -> {ok_candles[0]} gives candles; nothing here carries "
              f"a book, so no OFI")
    else:
        print("  -> nothing usable arrived. If the errors above mention "
              "entitlement or")
        print("     schema, that is the subscription; if the market is "
              "closed, retry in hours.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
