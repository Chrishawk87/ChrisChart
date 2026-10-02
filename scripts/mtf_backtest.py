#!/usr/bin/env python3
"""Walk the 4H/15m/1m signals through the ES tape and count the outcomes.

    python scripts/mtf_backtest.py data/ES_c_0_ohlcv-1s_2024-01-01.dbn.zst

THE RULE, AS STATED

Take the signal. Target first is a win, stop first is a loss, and either
way you wait for the NEXT 15-minute candle before taking anything again.
One position at a time.

WHAT YOU WILL SEE, AND WHY THE SECOND NUMBER IS THE ONE THAT MATTERS

A 10-tick target against a ~20-tick structural stop wins about two thirds
of the time on a pure coin flip -- that is the shape of the barriers, not
a read on the market. So this prints the real win rate NEXT TO the same
entries on the same bars with the directions shuffled. If the real one
does not clear the shuffled one, the engine is reporting geometry.

It also prints net dollars after a taker's true round trip: $5 commission
plus the $12.50 spread given up. A gross figure on a 10-tick target is
most of a fiction.

WHEN YOU CAN HAVE TRADED IT

A signal is stamped with its 1-minute bar's START. It is not known until
that bar CLOSES, sixty seconds later, so the fill is the first price at or
after that close -- never inside the minute whose own close produced the
signal. Getting this wrong scored 98.4% on a pure random walk against a
79.8% shuffled control, which is exactly what the control is for.

TWO GRIDS

Signals are decided on 1-minute bars. Outcomes are resolved on the finest
grid in the file, because a bar that touches both barriers does not record
which tick came first -- and this scores those as the stop, every time.
Resolving on 1-minute bars would make most outcomes that coin flip, so
the script refuses to run on a grid that coarse.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from liqmap import bars as bl                                   # noqa: E402
from liqmap import mtf, mtfsim as sim                           # noqa: E402
from liqmap.levels import trade_date                            # noqa: E402

TICK = 0.25


def hour_of(ts: float) -> int:
    return datetime.fromtimestamp(ts, tz=timezone.utc).hour


def say(label: str, s: dict, extra: str = "") -> None:
    if not s["resolved"]:
        print(f"  {label:<22s} {s['trades']:>5d} signals, none resolved")
        return
    print(f"  {label:<22s} {s['resolved']:>5d}  "
          f"{s['wins']:>5d}W {s['losses']:>5d}L   "
          f"{s['win_rate']*100:5.1f}%  "
          f"[{s['ci_low']*100:4.1f}-{s['ci_high']*100:4.1f}]   "
          f"need {s['breakeven']*100:5.1f}%   "
          f"${s['per_trade_usd']:>8.2f}/trade{extra}")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", help=".dbn/.dbn.zst OHLCV files")
    p.add_argument("--fine", type=float, default=5.0,
                   help="resolution grid in seconds (default 5)")
    p.add_argument("--hold-min", type=float, default=240.0,
                   help="minutes before a trade is called unresolved")
    p.add_argument("--target-ticks", type=float, default=10.0)
    p.add_argument("--control-runs", type=int, default=200)
    p.add_argument("--breakeven-at", type=float, default=4.0,
                   help="ticks onside at which the stop moves to the entry, "
                        "in the breakeven variant (default 4)")
    p.add_argument("--sweep-entry", default=None,
                   help="comma-separated caps on how many minutes into the "
                        "15m candle a trigger may fire, e.g. 2,3,5,8,15. "
                        "Searched WITH --sweep as a grid, on the first half "
                        "only.")
    p.add_argument("--block", type=float, default=14400.0,
                   help="direction timeframe in seconds (default 14400 = 4H)")
    p.add_argument("--trade", type=float, default=900.0,
                   help="traded candle in seconds -- supplies the stop (its "
                        "open) and the clock (its close). Default 900 = 15m")
    p.add_argument("--trigger", type=float, default=60.0,
                   help="continuation bar in seconds (default 60 = 1m). A "
                        "coarser one enters further from the traded "
                        "candle's open, which WIDENS the stop")
    p.add_argument("--trail", type=float, default=2.0,
                   help="ticks the trailing stop follows behind the best "
                        "price once armed (default 2)")
    p.add_argument("--trail-arm", type=float, default=3.0,
                   help="ticks onside before the trail arms (default 3)")
    p.add_argument("--require-1h", action="store_true",
                   help="also require the 1H to agree (an earlier form of "
                        "the rule; the main run reports both either way)")
    p.add_argument("--max-minutes-in", type=float, default=None,
                   help="single entry-window cap for the main run")
    p.add_argument("--no-clock", action="store_true",
                   help="let trades outlive their 15m candle (for "
                        "comparison only -- the rule says they do not)")
    p.add_argument("--sweep", default=None,
                   help="comma-separated target sizes in ticks to try, e.g. "
                        "5,10,15,20,30,40. Swept on the FIRST half of the "
                        "data only; the winner is then run once on the "
                        "sealed second half, which is the only number that "
                        "counts.")
    p.add_argument("--keep-rolls", action="store_true",
                   help="do not drop sessions near a quarterly expiry")
    p.add_argument("--json", default=None, help="write the full result here")
    a = p.parse_args()

    if a.fine >= 60.0:
        print("refusing: the resolution grid must be finer than the 1-minute "
              "signal grid, or almost every outcome is the tie rule rather "
              "than the tape.", file=sys.stderr)
        return 2

    # ---------------------------------------------------------- load
    fine: list = []
    for f in a.files:
        path = Path(f)
        if not path.exists():
            print(f"missing: {path}", file=sys.stderr)
            return 2
        mb = path.stat().st_size / 1e6
        print(f"reading {path.name} ({mb:,.0f} MB compressed) ...",
              flush=True)
        # Streams and aggregates as it reads. Two years of one-second ES
        # is about twenty million records, and materialising those as
        # Python objects before aggregating costs a couple of gigabytes
        # for no reason.
        fine.extend(bl.load(str(path), seconds=a.fine))
    fine.sort(key=lambda b: b.ts)
    if len(fine) < 5000:
        print("not enough bars to run on", file=sys.stderr)
        return 2

    first, last = fine[0].ts, fine[-1].ts
    span_days = (last - first) / 86400.0
    print(f"  timeframes: block {a.block/60:g}m / traded {a.trade/60:g}m "
          f"/ trigger {a.trigger/60:g}m")
    print(f"  {len(fine):,} bars at {a.fine:g}s  "
          f"{trade_date(first)} to {trade_date(last)}  "
          f"({span_days:.0f} days)")

    # Quarterly rolls. A contract change is a price discontinuity, and a
    # 4H open carried across one is an open from a different instrument.
    if not a.keep_rolls:
        rolls = bl.roll_dates(trade_date(first), trade_date(last))
        before = len(fine)
        fine = [b for b in fine if not bl.near_roll(trade_date(b.ts), rolls)]
        print(f"  dropped {before - len(fine):,} bars within "
              f"{bl.ROLL_WINDOW_DAYS} days of {len(rolls)} quarterly rolls")

    # ------------------------------------------------- signals, per day
    # Per trade date, so a 4H block never spans the overnight break and
    # the state never carries across a gap.
    days: dict = {}
    for b in fine:
        days.setdefault(trade_date(b.ts), []).append(b)

    per_day: dict = {}
    for day in sorted(days):
        minute = bl.resample(days[day], 60.0)
        if len(minute) >= 120:
            per_day[day] = minute

    def scan_all(max_in=None, target=None, want_1h=None):
        out: list = []
        for m in per_day.values():
            out.extend(mtf.scan(m, tick=TICK,
                                target_ticks=target or a.target_ticks,
                                max_minutes_in=max_in,
                                require_1h=(a.require_1h if want_1h is None
                                            else want_1h),
                                block_s=a.block, trade_s=a.trade,
                                trigger_s=a.trigger))
        out.sort(key=lambda g: g.ts)
        return out

    signals = scan_all(a.max_minutes_in)

    if not signals:
        print("\nno signals in this data.")
        return 0

    status = Counter(s.status for s in signals)
    print(f"\n{len(signals):,} signals  "
          f"({status.get('confirmed', 0)} confirmed, "
          f"{status.get('withdrawn', 0)} withdrawn by their 15m close)")
    print(f"  {Counter(s.side for s in signals)['long']} long / "
          f"{Counter(s.side for s in signals)['short']} short")

    # ------------------------------------------------------------ walk
    tape = sim.Tape(fine)
    expire = not a.no_clock
    trades = sim.walk(signals, tape, tick=TICK, hold_minutes=a.hold_min,
                      expire=expire, signal_seconds=a.trigger)
    overall = sim.summarise(trades)

    # The second reading of "we go back to break even": the stop is pulled
    # to the entry once the trade is onside, so a loser costs only fees.
    be_trades = sim.walk(signals, tape, tick=TICK, hold_minutes=a.hold_min,
                         expire=expire, breakeven_at=a.breakeven_at,
                         signal_seconds=a.trigger)
    be = sim.summarise(be_trades)

    # The hour, both ways, on the same bars. Chris stated the rule with
    # the 1H once and without it once; this answers which is better
    # rather than picking.
    tr_trades = sim.walk(signals, tape, tick=TICK, hold_minutes=a.hold_min,
                         expire=expire, signal_seconds=a.trigger,
                         trail_ticks=a.trail, trail_arm=a.trail_arm)
    trail = sim.summarise(tr_trades)

    other = scan_all(a.max_minutes_in, want_1h=not a.require_1h)
    other_sum = sim.summarise(sim.walk(other, tape, tick=TICK,
                                       hold_minutes=a.hold_min,
                                       expire=expire,
                                       signal_seconds=a.trigger))

    print(f"\n{'=' * 78}\nTAKEN, one at a time, re-arming on the next 15m "
          f"candle\n{'=' * 78}")
    print(f"  {'':<22s} {'n':>5s}  {'':>5s}  {'':>5s}   "
          f" rate   {'95% CI':^11s}   breakeven      net")
    say("hard stop at 15m open", overall)
    say(f"stop to BE at +{a.breakeven_at:g}t", be)
    say(f"trail {a.trail:g}t from +{a.trail_arm:g}t", trail)
    # WITH A TRAIL THE WIN RATE IS THE WRONG NUMBER. Nearly every exit is
    # technically a stop, including the profitable ones, so the rate
    # collapses while the money does not. Average ticks and the share of
    # trades that finished green are what describe it.
    tr_ticks = (trail["gross_usd"] / sim.ES_TICK_USD / trail["booked"]
                if trail["booked"] else 0.0)
    print(f"\n  the trail exits through the stop even when it is winning, "
          f"so its\n  win rate is not comparable: it averaged "
          f"{tr_ticks:+.2f} ticks a trade and "
          f"{trail['profitable_rate']*100:.1f}% finished green")
    say(("4H+15m only" if a.require_1h else "4H+1H+15m"), other_sum,
        "   <- the other reading of the agreement")
    print(f"\n  outcome mix, hard stop: "
          f"{overall['wins']} target / {overall['losses']} stop / "
          f"{overall['timed_out']} closed by the 15m clock")
    print(f"  outcome mix, BE stop:   "
          f"{be['wins']} target / {be['losses']} stop / "
          f"{be['scratched']} scratched at the entry / "
          f"{be['timed_out']} closed by the clock")
    print(f"  counting the clock exits too, "
          f"{overall['profitable_rate']*100:.1f}% of trades finished green")
    print(f"\n  held {overall['avg_minutes']:.0f} min on average; "
          f"barriers averaged {overall['avg_target_ticks']:.1f}t target "
          f"against {overall['avg_stop_ticks']:.1f}t stop")
    if overall["unresolved"]:
        print(f"  {overall['unresolved']} trades touched neither barrier "
              f"within {a.hold_min:g} minutes and are excluded from the "
              f"rate rather than counted as either")

    # --------------------------------------------------------- control
    print(f"\n{'=' * 78}\nTHE SAME ENTRIES WITH THE DIRECTIONS SHUFFLED"
          f"\n{'=' * 78}")
    ctrl = sim.control(signals, tape, runs=a.control_runs, tick=TICK,
                       hold_minutes=a.hold_min, expire=expire)
    if ctrl.get("runs"):
        print(f"  {ctrl['runs']} runs: "
              f"{ctrl['win_rate_mean']*100:.1f}% "
              f"+/- {ctrl['win_rate_sd']*100:.1f}  "
              f"(95th pct {ctrl['win_rate_p95']*100:.1f}%), "
              f"${ctrl['per_trade_mean']:.2f}/trade")
        edge = (overall["win_rate"] - ctrl["win_rate_mean"]) * 100
        print(f"\n  observed {overall['win_rate']*100:.1f}%  "
              f"vs shuffled {ctrl['win_rate_mean']*100:.1f}%  "
              f"=  {edge:+.1f} points")
        print(f"  clears the control by 2 sd: "
              f"{'YES' if sim.beats(overall, ctrl) else 'NO'}")
        print("\n  The shuffled run holds the bars, the entry times, the "
              "barrier\n  widths and the time-of-day mix fixed, and varies "
              "only the call.\n  Whatever it scores is what the shape of "
              "the trade is worth.")

    # ---------------------------------------------------------- splits
    cells = (len({t.side for t in trades})
             + len({trade_date(t.ts).year for t in trades})
             + len({t.later or "unknown" for t in trades})
             + len({hour_of(t.ts) for t in trades}))
    print(f"\n{'=' * 78}\nBREAKDOWN  ({cells} cells -- at a nominal 5% "
          f"each, the chance that at\nleast one reads well on noise alone "
          f"is about {(1 - 0.95 ** cells) * 100:.0f}%)\n{'=' * 78}")
    for name, key in (("by side", lambda t: t.side),
                      ("by year",
                       lambda t: trade_date(t.ts).year),
                      ("by 15m close",
                       lambda t: t.later or "unknown"),
                      ("by UTC hour", lambda t: f"{hour_of(t.ts):02d}:00")):
        print(f"\n  -- {name}")
        for k, s in sim.split(trades, key).items():
            say("     " + k, s)

    # ----------------------------------------------------------- sweep
    if a.sweep or a.sweep_entry:
        try:
            targets = ([float(x) for x in a.sweep.split(",") if x.strip()]
                       if a.sweep else [a.target_ticks])
            windows = ([float(x) for x in a.sweep_entry.split(",")
                        if x.strip()] if a.sweep_entry else [None])
        except ValueError:
            print("--sweep and --sweep-entry want numbers",
                  file=sys.stderr)
            return 2

        # SPLIT BY TIME AND SEARCH ON ONE HALF ONLY.
        #
        # This is a grid, so the search is wider than it looks: every
        # extra entry window multiplies by every extra target. The best
        # cell of a wide grid is partly the luckiest cell, and reporting
        # that as a result is how a strategy arrives already dead. The
        # search happens on the first half; the winner runs ONCE on the
        # second, which it has never touched.
        half = fine[len(fine) // 2].ts
        combos = len(targets) * len(windows)
        print(f"\n{'=' * 78}\nGRID SEARCH  ({len(windows)} entry windows "
              f"x {len(targets)} targets = {combos} combinations, on the "
              f"FIRST half only)\n{'=' * 78}")
        print(f"  search  up to  {trade_date(half)}")
        print(f"  sealed  after  {trade_date(half)}\n")
        print(f"  {'entry':>7s} {'target':>7s} {'n':>6s} {'win':>7s} "
              f"{'need':>7s} {'net/trade':>11s}  {'shuffled':>9s} "
              f"{'edge':>7s}")

        best = None
        for win in windows:
            got_all = scan_all(win)
            early = [g for g in got_all if g.ts < half]
            if len(early) < 50:
                print(f"  {str(win or 'any'):>7s}  -- only {len(early)} "
                      f"signals in the search half, skipped")
                continue
            for target in targets:
                for g in early:
                    g.target_ticks = target
                tr = sim.walk(early, tape, tick=TICK,
                              hold_minutes=a.hold_min, expire=expire,
                              signal_seconds=a.trigger)
                got = sim.summarise(tr)
                ctl = sim.control(early, tape,
                                  runs=max(20, a.control_runs // 5),
                                  tick=TICK, hold_minutes=a.hold_min,
                                  expire=expire)
                edge = (got["win_rate"] - ctl.get("win_rate_mean", 0)) * 100
                print(f"  {str(win or 'any'):>6s}m {target:>6.0f}t "
                      f"{got['booked']:>6d} {got['win_rate']*100:>6.1f}% "
                      f"{got['breakeven']*100:>6.1f}% "
                      f"{got['per_trade_usd']:>10.2f}  "
                      f"{ctl.get('win_rate_mean', 0)*100:>8.1f}% "
                      f"{edge:>+6.1f}")
                if best is None or got["per_trade_usd"] > best[2]:
                    best = (win, target, got["per_trade_usd"])

        if best is None:
            print("\n  nothing in the grid produced enough signals")
        else:
            win, target, net = best
            print(f"\n  best on the search half: entry within "
                  f"{win or 'any'} min, {target:.0f}t target, "
                  f"${net:.2f}/trade")

            got_all = scan_all(win)
            late = [g for g in got_all if g.ts >= half]
            for g in late:
                g.target_ticks = target
            sealed = sim.summarise(sim.walk(late, tape, tick=TICK,
                                            hold_minutes=a.hold_min,
                                            expire=expire,
                                            signal_seconds=a.trigger))
            sealed_be = sim.summarise(sim.walk(late, tape, tick=TICK,
                                               hold_minutes=a.hold_min,
                                               expire=expire,
                                               breakeven_at=a.breakeven_at,
                                               signal_seconds=a.trigger))
            sctl = sim.control(late, tape,
                               runs=max(20, a.control_runs // 2),
                               tick=TICK, hold_minutes=a.hold_min,
                               expire=expire)
            print(f"\n{'-' * 78}\n  THE SEALED HALF, run once with that "
                  f"setting:\n{'-' * 78}")
            say("   hard stop", sealed)
            say("   stop to BE", sealed_be)
            print(f"     shuffled control "
                  f"{sctl.get('win_rate_mean', 0)*100:.1f}% +/- "
                  f"{sctl.get('win_rate_sd', 0)*100:.1f}   clears by 2sd: "
                  f"{'YES' if sim.beats(sealed, sctl) else 'NO'}")
            print(f"\n  The grid row and the sealed row will differ. The "
                  f"sealed one is the\n  estimate; the grid is where it "
                  f"came from, and the gap between them\n  is how much of "
                  f"the search was luck. With {combos} combinations, a gap "
                  f"is\n  expected -- a LARGE one means the grid found "
                  f"noise.")

    print(f"\n{'=' * 78}")
    print("Gross and net both shown per trade; net charges "
          f"${sim.COST_RT:.2f} a round trip\n($5 commission + $12.50 "
          "spread).")
    if a.sweep or a.sweep_entry:
        print("A grid was searched. The only number here that is an "
              "estimate of\nfuture performance is the sealed-half row; "
              "every grid row is a\ndescription of the search, not a "
              "result.")
    else:
        print("Nothing here was fitted or searched.")

    if a.json:
        Path(a.json).write_text(json.dumps({
            "overall": overall, "breakeven_variant": be,
            "trailing_variant": trail, "control": ctrl,
            "by_side": sim.split(trades, lambda t: t.side),
            "by_year": sim.split(trades,
                                 lambda t: trade_date(t.ts).year),
            "by_close": sim.split(trades, lambda t: t.later or "unknown"),
            "trades": [t.to_dict() for t in trades],
        }, indent=2, default=str))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
