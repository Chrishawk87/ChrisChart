#!/usr/bin/env python3
"""Fills in the regime table from your own tape.

THE TABLE IS THE OUTPUT, NOT THE INPUT.

    OFI signal + low VPIN + tight spread   -> high confidence?
    OFI signal + high VPIN + tight spread  -> dangerous?
    OFI signal + high VPIN + wide spread   -> low confidence?
    OFI signal + high cancellation         -> suspect?

Those are four hypotheses. Each says a regime should change how well
the imbalance calls the next five-minute candle, and every one of
them is checkable: take the reading, note the regime it arrived in,
and see what the candle did. This runs that.

WHAT IT CANNOT DO IS CONFIRM THEM BY ITSELF. Eight regimes split a
session's seventy-eight candles into cells of nine, and nine candles
cannot distinguish a 70% regime from a 50% one. The script says how
many it has and refuses to read a cell that is too thin, which on a
single day will be most of them.

THE BASELINE IS THE READ WITHOUT ANY REGIME. A cell at 58% means
nothing next to an overall 57%; it means something next to an
overall 48%. So the overall rate is printed first and every cell is
shown as a difference from it.

Nothing here predicts, and nothing here places, sizes or times an
order.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liqmap.flowread import FlowRead, HOLD_S, PERIOD_S, WARMUP_S  # noqa: E402
from liqmap.ofi import Touch, level_ofi                           # noqa: E402
from liqmap.regime import BUCKET_VOL, Regime, RegimeTally         # noqa: E402

RAW_TICK = 250_000_000.0


def from_files(paths, limit: int = 0):
    """(ts, kind, payload) in feed order. One pass, nothing buffered."""
    import databento as db

    from liqmap import dbn

    out, n = [], 0
    for path in paths:
        for rec in db.DBNStore.from_file(path):
            ts = float(dbn.seconds(dbn._get(rec, "ts_event")) or 0.0)
            tr = dbn.trade_from(rec)
            if tr is not None:
                out.append((ts, "trade", (float(tr.sz), tr.aggressor)))
            lv = dbn._get(rec, "levels")
            if not lv:
                continue
            t0 = lv[0]
            bpx = int(dbn._get(t0, "bid_px", 0) or 0)
            apx = int(dbn._get(t0, "ask_px", 0) or 0)
            bsz = int(dbn._get(t0, "bid_sz", 0) or 0)
            asz = int(dbn._get(t0, "ask_sz", 0) or 0)
            out.append((ts, "book", (bpx, bsz, apx, asz,
                                     dbn._action(rec),
                                     float(dbn._get(rec, "size", 0) or 0))))
            n += 1
            if limit and n >= limit:
                return out
    return out


def walk(n: int, seed: int, p_move: float = 0.12, dt: float = 0.01):
    """A control, not data. A book with nothing in it to find."""
    rng = random.Random(seed)
    bid, ts = 7_700 * 10 ** 9, 0.0
    out = []
    for _ in range(n):
        if rng.random() < p_move:
            bid += RAW_TICK if rng.random() < 0.5 else -RAW_TICK
        ts += dt
        if rng.random() < 0.3:
            out.append((ts, "trade",
                        (float(rng.randint(1, 20)),
                         "buy" if rng.random() < 0.5 else "sell")))
        wide = rng.random() < 0.1
        out.append((ts, "book",
                    (int(bid), rng.randint(50, 900),
                     int(bid + (2 if wide else 1) * RAW_TICK),
                     rng.randint(50, 900),
                     "C" if rng.random() < 0.5 else "A",
                     float(rng.randint(1, 40)))))
    return out


def run(events, period_s, warmup_s, hold_s, bucket_vol):
    """Every candle: the read it settled on, the regime, the outcome."""
    fr = FlowRead(period_s=period_s, warmup_s=warmup_s, hold_s=hold_s)
    rg = Regime(bucket_vol=bucket_vol)
    prev = None
    rows = []
    cur = None          # candle open ts
    settled = None      # (read, tag, px) at the moment it settled
    last_px = 0.0
    for ts, kind, payload in events:
        if kind == "trade":
            sz, side = payload
            rg.on_trade(ts, sz, side)
            continue
        bpx, bsz, apx, asz, action, size = payload
        t = Touch(bid_px=bpx, bid_sz=bsz, ask_px=apx, ask_sz=asz)
        if not t.valid or t.crossed:
            continue
        mid = (bpx + apx) / 2.0
        rg.on_book(ts, bpx, bsz, apx, asz, RAW_TICK)
        rg.on_action(ts, action, size)
        if prev is not None:
            fr.on_event(ts, float(level_ofi(prev, t)))
        prev = t
        s = fr.snapshot
        if s.candle_ts != cur:
            if cur is not None and settled is not None:
                read, tag, px0 = settled
                move = (last_px - px0) / RAW_TICK
                rows.append({"read": read, "tag": tag, "ticks": move,
                             "outcome": ("up" if move > 0
                                         else "down" if move < 0 else "")})
            cur, settled = s.candle_ts, None
        # THE REGIME IS TAKEN WHEN THE READ SETTLES, not at the
        # candle's close. Reading it at the close would tag the trade
        # with a book state that arrived after the decision was made,
        # which is the most flattering mistake available here.
        if settled is None and s.read:
            rg.set_gross(s.gross)
            rs = rg.snapshot
            # The agreement is part of the regime for scoring: "the
            # book said up and the tape was also lifting" is a
            # different state from "the book said up and the tape was
            # hitting bids", and folding them together would measure
            # neither.
            ag = rs.agrees_with(s.read)
            tag = f"{rs.tag}-{ag}" if (rs.tag and ag) else rs.tag
            settled = (s.read, tag, mid)
        last_px = mid
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*",
                    help=".dbn / .dbn.zst MBP files; omit for a control")
    ap.add_argument("--period", type=float, default=PERIOD_S)
    ap.add_argument("--warmup", type=float, default=WARMUP_S)
    ap.add_argument("--hold", type=float, default=HOLD_S)
    ap.add_argument("--bucket-vol", type=float, default=BUCKET_VOL,
                    help="VPIN bucket size in contracts")
    ap.add_argument("--min-n", type=int, default=30,
                    help="decided candles before a cell is readable")
    ap.add_argument("--updates", type=int, default=600_000)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    if a.files:
        ev = from_files(a.files, a.limit)
        src = f"{len(a.files)} file" + ("" if len(a.files) == 1 else "s")
        real = True
    else:
        ev = walk(a.updates, a.seed)
        src = "A CONTROL, NOT DATA -- a random book with nothing in it"
        real = False
    if len(ev) < 1000:
        print("  Not enough records to measure anything.")
        return 2

    rows = run(ev, a.period, a.warmup, a.hold, a.bucket_vol)
    if not rows:
        print("  The read never settled on a direction. Nothing to"
              " score.")
        return 2

    dec = [r for r in rows if r["outcome"]]
    right = sum(1 for r in dec if r["outcome"] == r["read"])
    base = (right / len(dec)) if dec else 0.0

    print(f"\n{'=' * 86}")
    print(f"REGIME OUTCOMES   {src}")
    print(f"{len(ev):,} records   {len(rows):,} candles with a read   "
          f"VPIN bucket {a.bucket_vol:g}")
    print("=" * 86)
    print(f"  BASELINE, every regime together: {100 * base:.1f}% of"
          f" {len(dec):,} decided candles.")
    print("  Every cell below is a DIFFERENCE from that. A regime at"
          " 58% is worth")
    print("  nothing beside a baseline of 57%, and worth a lot beside"
          " one of 48%.")

    t = RegimeTally()
    for r in rows:
        t.record(r["tag"], r["read"], r["outcome"])
    if not t.rows:
        print("\n  NO CANDLE CARRIED A COMPLETE REGIME TAG. The bands"
              " need enough of a")
        print("  session behind them before they mean high or low;"
              " on a short file")
        print("  that never happens. Use more data.")
        return 0

    print(f"\n  {'regime':>34}{'candles':>9}{'decided':>9}"
          f"{'right':>7}{'rate':>8}{'vs base':>9}")
    thin = 0
    for tag in sorted(t.rows):
        got = t.of(tag, min_n=a.min_n)
        if not got["enough"]:
            thin += 1
            print(f"  {tag:>34}{got['n']:>9,}{got['decided']:>9,}"
                  f"{'--':>7}{'--':>8}{'too few':>9}")
            continue
        print(f"  {tag:>34}{got['n']:>9,}{got['decided']:>9,}"
              f"{got['right']:>7,}{100 * got['pct']:>7.1f}%"
              f"{100 * (got['pct'] - base):>+8.1f}%")

    if thin:
        print(f"\n  {thin} of {len(t.rows)} regimes have fewer than"
              f" {a.min_n} decided candles.")
        print("  Eight regimes split a session's seventy-eight"
              " candles into cells of")
        print("  nine, and nine cannot tell a 70% regime from a 50%"
              " one. This needs")
        print("  weeks of files, not a day.")

    readable = [t.of(x, a.min_n) for x in t.rows
                if t.of(x, a.min_n)["enough"]]
    if readable:
        best = max(readable, key=lambda r: r["pct"])
        worst = min(readable, key=lambda r: r["pct"])
        spread = best["pct"] - worst["pct"]
        print(f"\n  Best readable regime {best['tag']} at"
              f" {100 * best['pct']:.1f}%; worst {worst['tag']} at"
              f" {100 * worst['pct']:.1f}%.")
        if spread < 0.05:
            print("  THAT GAP IS NOTHING. The regimes are not"
                  " separating outcomes on this")
            print("  data, which means the tags are decoration and"
                  " the reading is as good")
            print("  or as bad whatever state it arrives in.")
        else:
            print("  Before reading anything into that gap: check the"
                  " counts above, then")
            print("  run the same thing on files this did not see. A"
                  " gap that does not")
            print("  survive that was a property of these days.")

    if not real:
        print("\n  THIS WAS A CONTROL. Every figure above is what"
              " zero looks like, and")
        print("  that is the only reason to run it without files."
              " Point it at your")
        print("  .dbn files for an answer.")
    else:
        print("\n  Costs are in none of these numbers, and a candle"
              " called correctly is")
        print("  not a candle traded profitably -- this scores"
              " direction only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
