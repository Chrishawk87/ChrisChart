#!/usr/bin/env python3
"""Replay the TIMED rule over real ES tape and count what it did.

    python scripts/timed_replay.py data/ES_*.dbn.zst
    python scripts/timed_replay.py data/ES_*.dbn.zst --tp 8 --sl 4
    python scripts/timed_replay.py data/ES_*.dbn.zst --sweep
    python scripts/timed_replay.py data/ES_*.dbn.zst --csv trades.csv

THE RULE, EXACTLY AS THE SERVICE RUNS IT

    Entry   the 1-minute and the 5-minute candles point the same way
            (close against open, both of them), at most one trade per
            5-minute candle.
    Exit    take profit, stop loss, or the 1-minute closing back
            through its own open against the position.

The entry and exit tests are IMPORTED from `liqmap.timed`. They are
not reimplemented here, because a backtest that rewrites the rule
measures the backtest.

WHAT THIS IS AND IS NOT

    IS: every trade the rule would have taken on this tape, with the
    fill price, the exit reason, and the ticks.

    IS NOT: a promise. It fills at the price the signal fired, which
    nobody is ever filled at, and it charges a fixed round-trip cost
    instead of a real spread. Both are stated in the output, both
    flatter the result, and the second one is adjustable.

Reads files. Nothing here places, sizes, routes or times an order.
"""
from __future__ import annotations

import argparse
import csv
import glob
import sys
from dataclasses import dataclass, field

sys.path.insert(0, ".")

from liqmap import timed as tm                         # noqa: E402

TICK = 0.25
TICK_VALUE = 12.50
# ES round trip, in ticks. One tick is the usual all-in cost; it is a
# parameter because it is the single assumption most able to turn a
# winning column into a losing one.
ROUND_TRIP = 1.0

M1, M5 = 60.0, 300.0


@dataclass
class Bar:
    start: float
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0

    def feed(self, px: float) -> None:
        if not self.open:
            self.open = self.high = self.low = px
        else:
            self.high = max(self.high, px)
            self.low = min(self.low, px)
        self.close = px

    def as_dict(self) -> dict:
        return {"open": self.open, "close": self.close,
                "start_ts": self.start}


@dataclass
class Trade:
    side: str
    entry_ts: float
    entry: float
    bar: float
    exit_ts: float = 0.0
    exit: float = 0.0
    why: str = ""

    @property
    def ticks(self) -> float:
        if not self.exit:
            return 0.0
        raw = (self.exit - self.entry) / TICK
        return raw if self.side == "up" else -raw


@dataclass
class Result:
    trades: list = field(default_factory=list)
    bars5: int = 0
    signals_blocked_by_cap: int = 0


def replay(prints, tp_ticks: float, sl_ticks: float,
           cost: float = ROUND_TRIP) -> Result:
    """One pass over (ts, px). Bars are built here, the rule is not."""
    res = Result()
    b1 = b5 = None
    open_t: Trade | None = None
    taken_bar = -1.0

    for ts, px in prints:
        k1 = float(int(ts // M1) * M1)
        k5 = float(int(ts // M5) * M5)
        if b1 is None or b1.start != k1:
            b1 = Bar(start=k1)
        if b5 is None or b5.start != k5:
            if b5 is not None:
                res.bars5 += 1
            b5 = Bar(start=k5)
        b1.feed(px)
        b5.feed(px)

        d1 = tm.direction(b1.as_dict())
        d5 = tm.direction(b5.as_dict())

        if open_t is not None:
            t = open_t
            raw = (px - t.entry) / TICK
            ticks = raw if t.side == "up" else -raw
            why = ""
            # Stop before target: a bar through both is recorded as
            # stopped out, the pessimistic reading and the one that
            # does not assume a fill.
            if sl_ticks > 0 and ticks <= -sl_ticks:
                why = "sl"
            elif tp_ticks > 0 and ticks >= tp_ticks:
                why = "tp"
            else:
                rev, _w = tm.reversed_against(t.side, d1)
                if rev:
                    why = "reversal"
            if why:
                t.exit_ts, t.exit, t.why = ts, px, why
                res.trades.append(t)
                open_t = None
            continue

        if taken_bar == k5:
            continue
        side, _why = tm.entry_side(d1, d5)
        if side:
            open_t = Trade(side=side, entry_ts=ts, entry=px, bar=k5)
            taken_bar = k5

    if open_t is not None:
        open_t.why = "still open at the end"
        res.trades.append(open_t)
    return res


def report(res: Result, tp: float, sl: float, cost: float) -> dict:
    done = [t for t in res.trades if t.exit]
    if not done:
        return {"trades": 0}
    net = [t.ticks - cost for t in done]
    wins = [x for x in net if x > 0]
    by_why: dict[str, list] = {}
    for t in done:
        by_why.setdefault(t.why, []).append(t.ticks - cost)
    return {
        "trades": len(done),
        "bars5": res.bars5,
        "wins": len(wins),
        "win_pct": round(100.0 * len(wins) / len(done), 1),
        "net_ticks": round(sum(net), 2),
        "net_dollars": round(sum(net) * TICK_VALUE, 2),
        "avg_ticks": round(sum(net) / len(done), 3),
        "best": round(max(net), 2),
        "worst": round(min(net), 2),
        "by_reason": {k: {"n": len(v), "ticks": round(sum(v), 2)}
                      for k, v in sorted(by_why.items())},
        "tp": tp, "sl": sl, "cost": cost,
    }


def load(paths: list[str]) -> list:
    """(ts, price) for every trade print, in order."""
    try:
        import databento as db                          # noqa: F401
        from liqmap import dbn
    except ImportError:
        raise SystemExit("pip install databento")
    out = []
    for path in paths:
        store = db.DBNStore.from_file(path)
        for rec in store:
            if dbn._action(rec) not in (dbn.ACTION_TRADE, dbn.ACTION_FILL):
                continue
            ts = float(dbn.seconds(dbn._get(rec, "ts_event")) or 0.0)
            px = dbn.price(dbn._get(rec, "price"))
            if ts > 0 and px > 0:
                out.append((ts, px))
    out.sort(key=lambda r: r[0])
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--tp", type=float, default=8.0, help="ticks, 0 = off")
    ap.add_argument("--sl", type=float, default=4.0, help="ticks, 0 = off")
    ap.add_argument("--cost", type=float, default=ROUND_TRIP,
                    help="round-trip cost in ticks")
    ap.add_argument("--sweep", action="store_true",
                    help="a grid of take profit and stop loss")
    ap.add_argument("--csv", default="", help="write every trade here")
    a = ap.parse_args(argv)

    paths: list[str] = []
    for p in a.files:
        paths.extend(sorted(glob.glob(p)) or [p])
    print(f"reading {len(paths)} file(s) ...")
    prints = load(paths)
    if not prints:
        print("no trade prints found in those files")
        return 1
    span = (prints[-1][0] - prints[0][0]) / 3600.0
    print(f"{len(prints):,} prints over {span:.1f} hours\n")

    if a.sweep:
        print(f"{'tp':>4} {'sl':>4} {'trades':>7} {'win%':>6} "
              f"{'net ticks':>10} {'$':>10} {'avg':>7}")
        print("-" * 52)
        for tp in (0.0, 4.0, 6.0, 8.0, 12.0, 16.0):
            for sl in (0.0, 4.0, 6.0, 8.0):
                r = report(replay(prints, tp, sl, a.cost), tp, sl, a.cost)
                if not r.get("trades"):
                    continue
                print(f"{tp:>4.0f} {sl:>4.0f} {r['trades']:>7} "
                      f"{r['win_pct']:>6.1f} {r['net_ticks']:>10.2f} "
                      f"{r['net_dollars']:>10.2f} {r['avg_ticks']:>7.3f}")
        print("\nA take profit moves the win RATE. Whether it moves the "
              "TOTAL is the\nonly question worth asking of this table.")
        return 0

    res = replay(prints, a.tp, a.sl, a.cost)
    r = report(res, a.tp, a.sl, a.cost)
    if not r.get("trades"):
        print("the rule took no trades on this tape")
        return 0

    print(f"  {'5-minute candles':<26}{r['bars5']}")
    print(f"  {'trades':<26}{r['trades']}"
          f"  ({r['trades'] / max(1, r['bars5']) * 100:.0f}% of candles)")
    print(f"  {'won':<26}{r['wins']}  ({r['win_pct']}%)")
    print(f"  {'net':<26}{r['net_ticks']:+.2f} ticks"
          f"  (${r['net_dollars']:+,.2f})")
    print(f"  {'average':<26}{r['avg_ticks']:+.3f} ticks a trade")
    print(f"  {'best / worst':<26}{r['best']:+.2f} / {r['worst']:+.2f}")
    print(f"\n  how they ended")
    for why, v in r["by_reason"].items():
        print(f"    {why:<22}{v['n']:>5}   {v['ticks']:+10.2f} ticks")

    print(f"\n  tp {a.tp} / sl {a.sl} ticks, {a.cost} tick round trip")
    print("  FILLED AT THE SIGNAL PRICE, which nobody is ever filled at.")
    print("  Raise --cost to see how much of this survives a real one.")

    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["entry_ts", "side", "entry", "exit_ts", "exit",
                        "why", "ticks_gross"])
            for t in res.trades:
                w.writerow([f"{t.entry_ts:.3f}", t.side, t.entry,
                            f"{t.exit_ts:.3f}", t.exit, t.why,
                            f"{t.ticks:.2f}"])
        print(f"\n  wrote {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
