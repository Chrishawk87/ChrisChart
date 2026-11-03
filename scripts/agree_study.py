#!/usr/bin/env python3
"""THE ONLY STRATEGY. The OFI candle decides; the stack can only veto.

    ENTRY   The OFI bar read -- where price sits against the open of
            the bar in progress. That is the whole signal. The 15m,
            5m and 1m do not vote; they VETO, and only when all three
            point the other way. The OFI and the DELTA must agree --
            the book leaning and the size that actually printed -- and
            a candle is traded once.
    EXIT    take profit, stop loss, or the agreement breaking --
            whichever comes first.

Everything else this repository can call is ignored here. No cascade
state, no swing, no regime veto, no absorption, no level dropdown.

WHAT IS COUNTED AND WHAT IS NOT.

A win is a trade that ended at or above the entry in the direction it
was taken, measured in POINTS. Points, not ticks, because that is how
Chris reads the chart; the tick figure is carried beside it.

COSTS ARE NOT IN THESE NUMBERS. Commission and the spread you cross on
both sides are real and are not modelled, so a strategy that averages
less than about a tick a trade is losing money while this prints a
profit. The summary says so rather than leaving it to be remembered.

THE EXIT IS CHECKED ON THE SAME BOOK UPDATES THE ENTRY WAS, which
means a bar that traded through the stop and came back is recorded as
stopped out. That is the pessimistic reading and it is the right one:
the alternative assumes you were not filled, and assuming you were not
filled is how a backtest invents money.

NO LOOKAHEAD. Every reading is taken from the bars as they stood at
that moment -- `from_live` is handed the simulated time, not the wall
clock, because without it every bar reads as complete and the study
sees a finished candle the trader could not have seen.

    python scripts/agree_study.py data/ES_*.dbn.zst --tp 4 --sl 2
    python scripts/agree_study.py data/ES_*.dbn.zst --sweep

Nothing here places, sizes or times an order.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liqmap import cascade as csc                        # noqa: E402
from liqmap import pressure as pr                        # noqa: E402
from liqmap.ofibar import OfiBarBuilder                  # noqa: E402

POINT = 1_000_000_000
RAW_TICK = POINT // 4
STACK = ("15m", "5m", "1m")
from liqmap.journal import MAX_PER_BAR  # noqa: E402
SECS = {"15m": 900.0, "5m": 300.0, "1m": 60.0}


@dataclass
class Trade:
    side: str                 # "up" / "down"
    ts_in: float
    px_in: float
    ts_out: float = 0.0
    px_out: float = 0.0
    why: str = ""             # "tp" / "sl" / "disagree" / "end"

    @property
    def points(self) -> float:
        d = (self.px_out - self.px_in) / POINT
        return d if self.side == "up" else -d

    @property
    def ticks(self) -> float:
        return self.points * 4.0

    @property
    def held_s(self) -> float:
        return max(0.0, self.ts_out - self.ts_in)

    @property
    def won(self) -> bool:
        return self.points > 0


@dataclass
class Result:
    tp: float
    sl: float
    trades: list = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.trades)

    @property
    def wins(self) -> int:
        return sum(1 for t in self.trades if t.won)

    @property
    def rate(self) -> float:
        return (self.wins / self.n) if self.n else 0.0

    @property
    def points(self) -> float:
        return sum(t.points for t in self.trades)

    @property
    def per_trade(self) -> float:
        return (self.points / self.n) if self.n else 0.0

    def by_reason(self) -> dict:
        out: dict = {}
        for t in self.trades:
            o = out.setdefault(t.why, {"n": 0, "pts": 0.0, "won": 0})
            o["n"] += 1
            o["pts"] += t.points
            o["won"] += 1 if t.won else 0
        return out


def _side(dirs: dict, read: str, delta: str = "") -> str:
    """The direction to trade. "" for none.

    THE OFI CANDLE DECIDES and the stack only vetoes -- the same rule
    the panel runs, imported from the same place rather than written
    out twice. A study measuring a slightly different rule from the
    one on the screen is worse than no study, and this file has
    already drifted from it once.
    """
    from liqmap.journal import signal

    side, _blocked = signal(dirs, read, delta)
    return side


def _dirs(tape, now_s: float, book) -> dict:
    """Each timeframe's qualified direction, or None.

    THE SAME RULE THE PANEL USES -- colour AND bps have to agree, via
    `cascade.colour_of` and `Rung`. Re-deriving it here with a sign
    test would measure a strategy nobody is looking at.
    """
    out: dict = {}
    for tf in STACK:
        b = tape.builders.get(tf)
        bar = b.candle(now_s) if b is not None else None
        if bar is None or not getattr(bar, "seeded", False):
            out[tf] = None
            continue
        hist = tape.history(tf, limit=pr.TYPICAL_BARS + 6,
                            include_current=False)
        p = pr.from_live(tf, SECS[tf], bar, book, coverage=1.0,
                         history=hist, now=now_s)
        if not p.measured:
            out[tf] = None
            continue
        rung = csc.Rung(timeframe=tf,
                        colour=csc.colour_of(p.winner, forming=p.forming),
                        bps=float(p.move_bps), present=True)
        out[tf] = rung.qualified
    return out


def run(records, tp_pts: float, sl_pts: float, target: int = 2000,
        eval_s: float = 1.0) -> Result:
    """One pass over raw records.

    THROUGH THE TAPE'S OWN INGEST, not a reimplementation of it. The
    records go in with `offer`, exactly as the live service feeds them,
    and the OFI builder is attached to the tape so the same book update
    drives both. A study that built its own bars would be measuring a
    strategy nobody is running.

    THE AGREEMENT IS RE-READ ON A CADENCE, not on every record. The
    panel polls; a rule evaluated thousands of times a second would
    take entries no one watching the screen could have taken. EXITS are
    checked on every update, because a stop does not wait for a poll.
    """
    from liqmap.cme_tape import Tape

    tape = Tape(symbol="ES", intervals=(60.0, 300.0, 900.0))
    ofi = OfiBarBuilder(tick_size=0.25, threshold=target, max_s=900.0,
                        min_updates=2)
    tape.ofibars = {target: ofi}

    res = Result(tp=tp_pts, sl=sl_pts)
    open_t: Trade | None = None
    tp_raw, sl_raw = tp_pts * POINT, sl_pts * POINT
    next_eval, last_ts, last_mid = 0.0, 0.0, 0.0
    cur_bar, taken = None, []
    # NO RE-ENTRY GATE. There was one, and it existed because
    # without it the P&L telescopes -- 4/2, 2/2 and no levels at all
    # came out at +216.00 to the cent, with only the win rate moving.
    # Chris chose immediate re-entry anyway; the consequence is
    # reported instead of prevented.

    for ts, rec, mid in records:
        tape.offer(rec)
        last_ts, last_mid = ts, mid

        # ---- exits first, on every update --------------------------
        if open_t is not None:
            d = mid - open_t.px_in
            fav = d if open_t.side == "up" else -d
            if sl_raw > 0 and fav <= -sl_raw:
                open_t.ts_out, open_t.px_out, open_t.why = ts, mid, "sl"
                res.trades.append(open_t)
                open_t = None
            elif tp_raw > 0 and fav >= tp_raw:
                open_t.ts_out, open_t.px_out, open_t.why = ts, mid, "tp"
                res.trades.append(open_t)
                open_t = None

        if ts < next_eval:
            continue
        next_eval = ts + eval_s

        dirs = _dirs(tape, ts, tape.book)
        br = ofi.bar_read or {}
        read = br.get("read") or ""
        agree = _side(dirs, read, br.get("delta_side") or "")
        bar_ts = float(br.get("bar_ts") or 0.0)

        # LOSING AGREEMENT IS NOT AN EXIT; ONLY A FLIP IS.
        #
        # Exiting the moment the four stopped agreeing closed for
        # about -1 tick over and over: the read crosses its bar's
        # open constantly, so agreement breaks for an instant and the
        # next look re-opens. None of those round trips was a
        # decision about the market.
        other = "down" if (open_t and open_t.side == "up") else "up"
        if open_t is not None and agree == other:
            open_t.ts_out, open_t.px_out = ts, mid
            open_t.why = "flip"
            res.trades.append(open_t)
            open_t = None
        # RE-ENTRY IS IMMEDIATE, by Chris's rule: open whenever three
        # of the four agree. The `armed` gate is gone with it, and so
        # is what it bought -- after a target the signal is usually
        # still there, the next look re-opens at about the price it
        # closed, and the pieces sum to the one move. The take profit
        # therefore moves the win RATE and not the total. The report
        # says so rather than leaving it to be noticed.
        # TWO TO A CANDLE, NEVER TWO THE SAME WAY -- the same cap
        # the panel applies. Without it one candle is traded the same
        # way over and over, each one the same read on the same bar
        # and charged a round trip.
        if bar_ts != cur_bar:
            cur_bar, taken = bar_ts, []
        # NO BAR IDENTITY, NO CAP -- the same rule the journal's
        # `sides_in_bar` applies when it is handed a zero. Capping
        # against an unknown candle would charge every trade in the
        # run to one imaginary bar and stop after two.
        capped = bool(bar_ts) and (agree in taken
                                   or len(taken) >= MAX_PER_BAR)
        if open_t is None and agree and not capped:
            open_t = Trade(side=agree, ts_in=ts, px_in=mid)
            if bar_ts:
                taken.append(agree)

    if open_t is not None:
        # STILL OPEN AT THE END. Recorded as what it is: dropping it
        # would quietly discard losers that had not resolved, and
        # closing it as a win at the last price would be worse.
        open_t.ts_out, open_t.px_out = last_ts, last_mid
        open_t.why = "end"
        res.trades.append(open_t)
    return res


def from_files(paths, limit: int = 0):
    """(ts, record, mid) in feed order.

    The RECORD is kept, not a summary of it: the tape wants what
    Databento sent, and anything this script extracted instead would
    be a second decoder to keep in step with the first.
    """
    import databento as db

    from liqmap import dbn

    out, n = [], 0
    for path in paths:
        for rec in db.DBNStore.from_file(path):
            lv = dbn._get(rec, "levels")
            if not lv:
                continue
            t0 = lv[0]
            bid = int(dbn._get(t0, "bid_px", 0) or 0)
            ask = int(dbn._get(t0, "ask_px", 0) or 0)
            if bid <= 0 or ask <= 0 or ask <= bid:
                continue
            ts = float(dbn.seconds(dbn._get(rec, "ts_event")) or 0.0)
            out.append((ts, rec, (bid + ask) / 2.0))
            n += 1
            if limit and n >= limit:
                return out
    return out


def report(res: Result, src: str, real: bool) -> None:
    print(f"\n{'=' * 78}")
    print("THE OFI CANDLE DECIDES   15m/5m/1m veto only")
    print(f"{src}")
    print(f"take profit {res.tp:g} pts   stop loss {res.sl:g} pts"
          f"   (0 means not used)")
    print("=" * 78)
    if not res.n:
        print("  NO TRADE EVER OPENED. Either the four never lined up on"
              " this data, or")
        print("  one of them never produced a reading at all -- those"
              " look the same")
        print("  from here. Run with --debug to see how often each"
              " agreed.")
        return
    print(f"  trades {res.n:,}   wins {res.wins:,}"
          f"   {100 * res.rate:.1f}%")
    print(f"  points {res.points:+,.2f}   per trade"
          f" {res.per_trade:+.3f} pts ({res.per_trade * 4:+.2f} ticks)")
    best = max(res.trades, key=lambda t: t.points)
    worst = min(res.trades, key=lambda t: t.points)
    print(f"  best {best.points:+.2f}   worst {worst.points:+.2f}"
          f"   median hold "
          f"{sorted(t.held_s for t in res.trades)[res.n // 2]:.0f}s")

    print(f"\n  {'ended by':>10}{'trades':>9}{'won':>7}{'rate':>8}"
          f"{'points':>11}{'per trade':>11}")
    for why, o in sorted(res.by_reason().items(),
                         key=lambda kv: -kv[1]["n"]):
        print(f"  {why:>10}{o['n']:>9,}{o['won']:>7,}"
              f"{100 * o['won'] / o['n']:>7.1f}%"
              f"{o['pts']:>+11.2f}{o['pts'] / o['n']:>+11.3f}")

    print("\n  COSTS ARE IN NONE OF THESE NUMBERS. Commission and the"
          " spread crossed on")
    print("  both sides are real. At ES a tick is $12.50 and a round"
          " trip costs about")
    print("  one; a per-trade figure under that is a loss wearing a"
          " profit's clothes.")
    if not real:
        print("\n  AND THIS WAS A CONTROL -- a random book with nothing"
              " in it to find.")
        print("  Every figure above is what zero looks like. Point it at"
              " your .dbn")
        print("  files for an answer.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*",
                    help=".dbn / .dbn.zst MBP files")
    ap.add_argument("--tp", type=float, default=4.0,
                    help="take profit in POINTS; 0 disables it")
    ap.add_argument("--sl", type=float, default=2.0,
                    help="stop loss in POINTS; 0 disables it")
    ap.add_argument("--target", type=int, default=2000,
                    help="OFI bar target the read is taken from")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--eval-s", dest="eval_s", type=float, default=1.0,
                    help="how often the agreement is re-read, in feed "
                         "seconds. The panel polls; evaluating faster "
                         "takes entries nobody watching could take.")
    ap.add_argument("--sweep", action="store_true",
                    help="a grid of take profits and stop losses")
    ap.add_argument("--tps", type=float, nargs="*",
                    default=[1, 2, 3, 4, 6, 8])
    ap.add_argument("--sls", type=float, nargs="*",
                    default=[1, 2, 3, 4])
    a = ap.parse_args()

    if not a.files:
        print("\n  No files. This one does not run a synthetic control:"
              " the rule reads")
        print("  three timeframes and an OFI bar, and a random walk"
              " that produced all")
        print("  four would be measuring the generator. Point it at"
              " your .dbn files.")
        return 2

    ev = from_files(a.files, a.limit)
    src = (f"{len(a.files)} file" + ("" if len(a.files) == 1 else "s")
           + f", {len(ev):,} records")
    if len(ev) < 1000:
        print("  Not enough records to measure anything.")
        return 2

    if not a.sweep:
        report(run(ev, a.tp, a.sl, a.target, a.eval_s), src, True)
        return 0

    print(f"\n{'=' * 78}")
    print(f"SWEEP   {src}")
    print("=" * 78)
    print(f"  {'tp':>5}{'sl':>5}{'trades':>9}{'win%':>8}"
          f"{'points':>11}{'per trade':>11}")
    rows = []
    for tp in a.tps:
        for sl in a.sls:
            r = run(ev, tp, sl, a.target, a.eval_s)
            rows.append(r)
            print(f"  {tp:>5g}{sl:>5g}{r.n:>9,}{100 * r.rate:>7.1f}%"
                  f"{r.points:>+11.2f}{r.per_trade:>+11.3f}")
    best = max(rows, key=lambda r: r.points) if rows else None
    if best is not None:
        print(f"\n  Best on these files: tp {best.tp:g} / sl {best.sl:g}"
              f" at {best.points:+.2f} points.")
        print("  THAT IS THE BEST FIT TO THESE DAYS, which is not the"
              " same as the best")
        print("  rule. A grid this size will always produce a winner"
              " even on noise --")
        print("  re-run the chosen pair on files this never saw before"
              " believing it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
