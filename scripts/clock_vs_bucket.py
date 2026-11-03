#!/usr/bin/env python3
"""The clock version against the contract-bucket version, on one tape.

THE QUESTION. The OFI candles end on a quantity of flow -- every
2,000 contracts -- and they flip often enough to whipsaw the trade.
This drives the same raw events through a 15-minute clock and reports
both, so the trade-off can be seen rather than argued: fewer
unconfirmed flips is only an improvement if it does not collapse into
too few signals to trade.

EVENT BY EVENT, IN ts_event ORDER. There is no resample here and no
pass over a pre-built frame. The replay drives the same state machine
the live service drives, so the backtest and the panel cannot
disagree about what a window was.

    python scripts/clock_vs_bucket.py data/ES_*.dbn.zst
    python scripts/clock_vs_bucket.py data/ES_*.dbn.zst --period 300

WHAT YOU HAVE TO CONFIRM, and what was assumed:

  * MBP-10 is the configured schema, so OFI is LEVEL-LEVEL at the
    touch. Per-order reconstruction needs MBO and is not possible on
    this subscription. If the account is actually on MBO, this
    understates nothing but ignores queue position entirely.
  * `side` is read raw from the record. 'N' is COUNTED, never
    dropped -- the service's own decoder returns None for those
    trades, so the CVD everywhere else omits them. The count is
    printed; if it is large, every CVD figure in this app is
    understated and that matters more than anything else here.
  * The candle is MID-PRICE from the book, which moves on every book
    update. Its high and low include levels no trade happened at.
    The traded OHLC is carried beside it.

Nothing here places, sizes or times an order.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liqmap.clockwin import PERIOD_S, ClockWindows    # noqa: E402
from liqmap.ofibar import OfiBarBuilder              # noqa: E402


def replay(paths, period_s: float, target: int, level: int,
           limit: int = 0):
    """Drive both state machines from one pass of raw events."""
    import databento as db

    from liqmap import dbn

    cw = ClockWindows(period_s=period_s, keep=10_000)
    ob = OfiBarBuilder(tick_size=0.25, threshold=target, max_s=900.0,
                       min_updates=2, call_level=level)
    # The bucket version's own flips, counted the same way: a change
    # of side between CONSECUTIVE closed bars.
    b_flips = b_conf = 0
    b_last = ""
    seen = 0

    for path in paths:
        for rec in db.DBNStore.from_file(path):
            ts = float(dbn.seconds(dbn._get(rec, "ts_event")) or 0.0)

            if dbn._action(rec) in (dbn.ACTION_TRADE, dbn.ACTION_FILL):
                sz = float(dbn._get(rec, "size", 0) or 0)
                px = dbn.price(dbn._get(rec, "price"))
                cw.on_trade_px(ts, px, sz, dbn._side(rec))
                t = dbn.trade_from(rec)
                if t is not None:
                    ob.on_trade(float(t.ts), float(t.sz), t.aggressor,
                                int(dbn._get(rec, "price") or 0))

            lv = dbn._get(rec, "levels")
            if not lv:
                continue
            t0 = lv[0]
            bpx = int(dbn._get(t0, "bid_px", 0) or 0)
            apx = int(dbn._get(t0, "ask_px", 0) or 0)
            bsz = int(dbn._get(t0, "bid_sz", 0) or 0)
            asz = int(dbn._get(t0, "ask_sz", 0) or 0)
            if bpx <= 0 or apx <= 0:
                continue
            cw.on_book(ts, bpx, bsz, apx, asz)
            before = len(ob.bars)
            ob.on_book(ts, bpx, bsz, apx, asz)
            if len(ob.bars) > before:
                bar = ob.bars[-1]
                d = ("up" if bar.ofi > 0 else "down" if bar.ofi < 0
                     else "flat")
                if d in ("up", "down"):
                    if b_last and d != b_last:
                        b_flips += 1
                        dl = bar.delta
                        if (d == "up" and dl > 0) or (d == "down"
                                                      and dl < 0):
                            b_conf += 1
                    b_last = d
            seen += 1
            if limit and seen >= limit:
                return cw, ob, b_flips, b_conf
    return cw, ob, b_flips, b_conf


def frame(cw: ClockWindows):
    """The aligned comparison, as a DataFrame when pandas is here."""
    rows = [{
        "start": w.start_ts, "candle": w.mid_dir,
        "traded_candle": w.last_dir,
        "net_ofi": w.net_ofi, "ofi": w.ofi_dir,
        "net_cvd": w.net_cvd, "cvd": w.cvd_dir,
        "agree": w.agree, "ticks": w.ticks,
        "trades": w.trades, "unknown_side": w.unknown,
    } for w in cw.done]
    try:
        import pandas as pd

        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["start"] = pd.to_datetime(df["start"], unit="s", utc=True)
        return df.set_index("start")
    except ImportError:
        return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help=".dbn / .dbn.zst MBP files")
    ap.add_argument("--period", type=float, default=PERIOD_S,
                    help="window length in seconds (900 = 15 minutes)")
    ap.add_argument("--target", type=int, default=2000,
                    help="the contract-bucket size to compare against")
    ap.add_argument("--level", type=int, default=500,
                    help="the OFI deadband the bucket version uses")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--csv", default="",
                    help="write the aligned windows here")
    a = ap.parse_args()

    if not a.files:
        print("\n  No files. This one does not run a control: it is a"
              " comparison of two")
        print("  reconstructions of the same tape, and a synthetic"
              " tape would be")
        print("  comparing them on a market that does not exist."
              " Point it at your .dbn files.")
        return 2

    cw, ob, b_flips, b_conf = replay(a.files, a.period, a.target,
                                     a.level, a.limit)
    s = cw.summary()
    mins = a.period / 60.0

    print(f"\n{'=' * 78}")
    print(f"CLOCK vs CONTRACT BUCKET   {len(a.files)} file"
          f"{'' if len(a.files) == 1 else 's'}")
    print(f"{mins:g}-minute windows, rebuilt from raw events"
          f"   |   {a.target:,}-contract buckets")
    print("=" * 78)

    print(f"\n  THE CLOCK VERSION   {s['windows']:,} windows")
    print(f"    all three agree      {s['agreed']:,} of"
          f" {s['decided']:,} decided"
          f"   ({100 * s['agree_pct']:.1f}%)")
    print(f"    OFI flips            {s['ofi_flips']:,}"
          f"   ({s['flips_per_window']:.2f} per window)")
    print(f"    CVD confirmed        {s['flips_cvd_confirmed']:,}"
          f"   unconfirmed {s['flips_unconfirmed']:,}")

    n_bars = len(ob.bars)
    print(f"\n  THE BUCKET VERSION  {n_bars:,} bars")
    print(f"    OFI flips            {b_flips:,}"
          f"   ({b_flips / n_bars if n_bars else 0:.2f} per bar)")
    print(f"    CVD confirmed        {b_conf:,}"
          f"   unconfirmed {b_flips - b_conf:,}")

    print(f"\n  {'-' * 74}")
    if s["windows"] and n_bars:
        print(f"  Signals: {s['windows']:,} windows against"
              f" {n_bars:,} bars.")
        print("  FEWER FLIPS IS ONLY BETTER IF THERE ARE STILL ENOUGH"
              " SIGNALS TO TRADE.")
        print("  One window an hour that never flips is not an"
              " improvement, and the")
        print("  two numbers above have to be read together or either"
              " can be made")
        print("  to look good on its own.")

    if s["unknown_side_trades"]:
        print(f"\n  {s['unknown_side_trades']:,} of {s['trades']:,}"
              f" prints had NO side stated ('N').")
        print("  They are counted here and DROPPED everywhere else in"
              " this service --")
        print("  `dbn.trade_from` returns None for them. If that share"
              " is large, every")
        print("  CVD figure on the panel is understated by roughly"
              " this much.")
    else:
        print(f"\n  Every one of {s['trades']:,} prints stated a side.")

    print("\n  ASSUMPTIONS. OFI is level-level at the touch because"
          " the subscription is")
    print("  MBP-10; per-order reconstruction needs MBO. The candle is"
          " mid-price from")
    print("  the book, so its high and low include levels nothing"
          " traded at -- the")
    print("  traded OHLC is in the frame beside it.")

    if a.csv:
        df = frame(cw)
        try:
            df.to_csv(a.csv)
            print(f"\n  Wrote {a.csv}")
        except AttributeError:
            import csv as _csv

            with open(a.csv, "w", newline="") as fh:
                if df:
                    wr = _csv.DictWriter(fh, fieldnames=list(df[0]))
                    wr.writeheader()
                    wr.writerows(df)
            print(f"\n  Wrote {a.csv} (no pandas; plain csv)")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
