#!/usr/bin/env python3
"""Check the DBN reader against a real file, and print what it found.

    python scripts/verify_dbn.py data/ES_c_0_mbp-10_2026-09-25.dbn.zst

Nothing is uploaded anywhere. This prints a short report you can paste back
into the chat -- a few hundred bytes instead of a few hundred megabytes.

WHAT IT IS ACTUALLY CHECKING

The reader was written from Databento's documentation, which settles field
names, price scaling and enum values. It cannot settle one thing: whether a
trade marked 'B' really is a buy aggressor by the time it reaches these
records. If that is backwards, nothing crashes -- every delta in the engine
simply reads with the sign inverted, absorption reads as initiative, and the
whole system is confidently wrong. So the last section re-derives the
convention from the tape: crossing buyers pay the ask, so correctly-labelled
buys print above the mid. The file answers the question about itself.

It also dumps the raw field names off the first record, because the surest
way to find out that an assumption about the format was wrong is to look at
one.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liqmap import dbn                                      # noqa: E402


def show(label: str, value) -> None:
    print(f"  {label:<24} {value}")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2

    path = Path(argv[1])
    if not path.exists():
        print(f"No such file: {path}", file=sys.stderr)
        return 2

    # How many records to walk. The whole point is to be quick.
    limit = int(argv[2]) if len(argv) > 2 else 150_000

    try:
        store = dbn.open_file(str(path))
    except ImportError:
        print("pip install databento", file=sys.stderr)
        return 2
    except Exception as exc:                              # noqa: BLE001
        print(f"Could not open the file: {exc}", file=sys.stderr)
        return 1

    print(f"\n=== {path.name} ===\n")

    # --- what the records actually look like ---------------------------
    first = None
    for r in store:
        first = r
        break
    if first is None:
        print("The file has no records.", file=sys.stderr)
        return 1

    fields = [f for f in dir(first) if not f.startswith("_")
              and not callable(getattr(first, f, None))]
    print("RAW RECORD")
    show("type", type(first).__name__)
    show("fields", ", ".join(sorted(fields)[:18]))
    for name in ("ts_event", "action", "side", "price", "size", "depth"):
        if hasattr(first, name):
            show(f"  .{name}", repr(getattr(first, name)))
    lv = getattr(first, "levels", None)
    if lv:
        show("  .levels[0]", ", ".join(
            f"{k}={getattr(lv[0], k, '?')}"
            for k in ("bid_px", "bid_sz", "bid_ct", "ask_px", "ask_sz")
            if hasattr(lv[0], k)))
    print()

    # --- what the reader makes of them ---------------------------------
    actions: Counter = Counter()
    sides: Counter = Counter()
    trades = books = 0
    first_ts = last_ts = 0.0
    sample_book = None
    bad_books = 0
    px_lo, px_hi = float("inf"), 0.0

    for i, r in enumerate(store):
        if i >= limit:
            break
        actions[dbn._action(r)] += 1
        sides[dbn._side(r)] += 1

        ts = dbn.seconds(getattr(r, "ts_event", 0))
        if ts > 0:
            first_ts = first_ts or ts
            last_ts = ts

        t = dbn.trade_from(r)
        if t is not None:
            trades += 1
            px_lo, px_hi = min(px_lo, t.px), max(px_hi, t.px)

        if getattr(r, "levels", None):
            b = dbn.book_from(r)
            if not b.empty:
                books += 1
                sample_book = sample_book or b
                # A crossed or inverted book means the sides arrived
                # swapped -- silent, and it poisons every depth read.
                if b.bids and b.asks and b.best_bid >= b.best_ask:
                    bad_books += 1

    print("PARSED")
    show("records walked", f"{min(limit, i + 1):,}")
    show("actions", dict(actions.most_common()))
    show("sides", dict(sides.most_common()))
    show("trades parsed", f"{trades:,}")
    show("books parsed", f"{books:,}")
    if first_ts and last_ts:
        import datetime as _dt
        fmt = "%Y-%m-%d %H:%M:%S"
        a = _dt.datetime.fromtimestamp(first_ts, _dt.timezone.utc)
        z = _dt.datetime.fromtimestamp(last_ts, _dt.timezone.utc)
        show("span (UTC)", f"{a.strftime(fmt)} -> {z.strftime(fmt)}")
    if px_hi > 0:
        show("trade prices", f"{px_lo:,.2f} .. {px_hi:,.2f}")
    print()

    if sample_book is not None:
        b = sample_book
        print("SAMPLE BOOK")
        show("levels", f"{len(b.bids)} bids x {len(b.asks)} asks")
        show("touch", f"{b.best_bid:,.2f} / {b.best_ask:,.2f}")
        show("spread", f"{(b.best_ask - b.best_bid):.2f} "
                       f"({b.spread_bps:.3f} bps)")
        show("top sizes", f"{b.bids[0].sz:g} x {b.asks[0].sz:g}")
        show("crossed books", f"{bad_books:,}"
                              + ("  <-- SIDES MAY BE SWAPPED" if bad_books
                                 else ""))
        print()

    # --- the question documentation cannot answer ----------------------
    store2 = dbn.open_file(str(path))
    report = dbn.audit(store2, limit=limit)
    print("AGGRESSOR CONVENTION")
    show("samples", f"{report.samples:,}")
    show("agree with docs", f"{report.agree:.1%}")
    show("verdict", report.verdict.upper())
    if report.flip:
        print("\n  The feed is INVERTED against the documentation.")
        print("  Pass flip=True everywhere trade_from is called.")
    elif report.ok:
        print("\n  Matches the documentation. No flip needed.")
    else:
        print("\n  Inconclusive. Do not trust delta until this is settled.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
