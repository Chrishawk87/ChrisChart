#!/usr/bin/env python3
"""The candle walkthrough as a dataset: one row per minute, two years.

    python scripts/walk_export.py data/*.dbn.zst --out ~/Downloads/ES_walk

Writes:

    ES_walk_minutes.csv     one row per 1-minute bar  (~700,000 rows)
    ES_walk_candles.csv     one row per 15m candle    (~47,000 rows)
    ES_walk_minutes.parquet when pyarrow is installed -- same rows, ~10x
                            smaller and far faster to load

This is the same computation `candle_walk.py` draws on its pages. The
pages are for reading with your eyes; this is for asking questions of.
The whole file as pages would be ~47,000 of them and about half a
gigabyte, which no reader will open -- and you cannot filter a PDF.

WHAT IS ON EACH ROW

Identity          session, candle, minute, idx (0-14 inside the candle),
                  contract
The 1m bar        open high low close volume
The context       o5, o15 -- the opens of the 5m and 15m bars this
                  minute sits in
What was showing  d1, d5, d15 (+1 up, -1 down, 0 flat) and `agree`

All of the above is read CAUSALLY: at this minute the 5m and 15m bars
have not finished, so their direction is their open compared with THIS
minute's close. Nothing reads a bar that had not formed.

THE FORWARD COLUMNS, AND WHY `to_close` IS NOT ONE OF THEM

`to_close` is on the pages and is here too, because it is what the
pages show. It is also the column NOT to build a strategy on: it runs
to the end of the same 15-minute candle, which is a deadline the market
does not observe. A minute agreeing at idx 1 has fourteen minutes to
work with; one at idx 14 has none. Sorting by it ranks position in the
candle, not signal.

The columns to use instead do not depend on where in the candle you
are:

    fwd_1 fwd_5 fwd_15   ticks after 1, 5 and 15 minutes, signed by the
                         direction showing, measured across candle and
                         session boundaries
    mfe mae              best and worst excursion over those 15 minutes
    first_4t             which of +4 / -4 ticks came first within 15
                         minutes: target, stop, neither, or ambiguous
    secs_to_first        how long that took

ENTRY IS NOT THE CLOSE. You cannot trade a bar's close -- you learn it
when the bar is over. Every forward column is measured from `entry`,
the first print STRICTLY AFTER the minute ended, with `entry_lag_s`
saying how long that took. Measuring from the close instead is the fill
-timing defect that produced a 98.4% win rate on a random walk earlier
in this project.

FIRST TOUCH IS READ FROM THE 1-SECOND PRINTS, not from 1-minute bars.
A 1-minute bar whose range covers both +4 and -4 cannot say which came
first; the seconds underneath it usually can. When a single second
covers both, that is `ambiguous` -- reported, never guessed.

NOTHING IS RESOLVED ACROSS A CONTRACT ROLL. The forward walk stops at
a change of instrument and the row is marked `neither` with whatever
time elapsed, because the next print is a different contract at a
different price.

COSTS ARE NOT APPLIED HERE. These are raw measurements. A 4-tick target
against a 4-tick stop needs the spread and fees taken off at analysis
time: `breakeven = (stop + cost) / (target + stop)`.

NOTHING HERE PLACES, SIZES OR TIMES AN ORDER. It writes rows.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import candle_walk as W                                     # noqa: E402
import chart_pdf as C                                       # noqa: E402

TICK = C.TICK
SPAN15, SPAN5, SPAN1 = W.SPAN15, W.SPAN5, W.SPAN1

HORIZONS = (1, 5, 15)                   # minutes
BRACKET_TICKS = 4                       # the 4-tick target and stop
BRACKET_CAP_S = 15 * 60                 # how long to wait for a touch

MINUTE_COLS = (
    "session", "candle", "minute", "idx", "contract",
    "open", "high", "low", "close", "volume",
    "o5", "o15", "d1", "d5", "d15", "agree",
    "to_close",
    "entry", "entry_lag_s",
    "fwd_1", "fwd_5", "fwd_15", "mfe", "mae",
    "first_4t", "secs_to_first",
    "utc",
)

CANDLE_COLS = (
    "session", "candle", "contract",
    "open", "high", "low", "close", "chg_ticks", "range_ticks",
    "volume", "minutes", "agreed", "first_agree_idx",
    "rollup_ok", "partial", "utc",
)


# ------------------------------------------------- the forward tape

class Tape:
    """One session's 1-second bars, in time order, searchable forward.

    The bracket is resolved against THESE, not against 1-minute bars.
    A minute whose range covers both the target and the stop cannot say
    which came first; the seconds underneath it almost always can.

    The forward walks run on numpy slices rather than Python loops. A
    row has to look 15 minutes ahead and there are ~700,000 rows: at
    ~900 seconds each that is 600 million steps per pass, which does
    not finish in a Python loop. Frozen once per session, every lookup
    after that is a slice.
    """

    def __init__(self) -> None:
        self._ts: list = []
        self._hi: list = []
        self._lo: list = []
        self._px: list = []
        self._kind: list = []
        self.frozen = False

    def add(self, ts, h, lo, c, contract) -> None:
        self._ts.append(ts)
        self._hi.append(h)
        self._lo.append(lo)
        self._px.append(c)
        self._kind.append(contract)

    def freeze(self) -> "Tape":
        """Arrays, plus where each contract's run ends.

        `seg_end[i]` is the first index at or after `i` that belongs to
        a DIFFERENT instrument. Every forward walk stops there: across
        a roll the next print is another contract at another price, and
        a bracket resolved through it would be resolved by carry.
        """
        import numpy as np
        self.ts = np.asarray(self._ts, dtype="float64")
        self.hi = np.asarray(self._hi, dtype="float64")
        self.lo = np.asarray(self._lo, dtype="float64")
        self.px = np.asarray(self._px, dtype="float64")
        n = len(self._ts)
        seg = np.empty(n, dtype="int64")
        end = n
        for i in range(n - 1, -1, -1):
            if i + 1 < n and self._kind[i + 1] != self._kind[i]:
                end = i + 1
            seg[i] = end
        self.seg_end = seg
        self.kind = self._kind
        self.frozen = True
        return self

    def __len__(self) -> int:
        return len(self._ts)

    def after(self, ts: float) -> int:
        """Index of the first print STRICTLY after `ts`.

        Strictly: a print stamped at the same second the minute ended
        is part of that minute. Treating it as the fill is the
        look-ahead that lets a backtest enter inside the bar that
        produced its own signal -- the defect that produced a 98.4%
        win rate on a random walk earlier in this project.
        """
        return int(bisect.bisect_right(self._ts, ts))

    def price_at(self, i: int):
        return self._px[i] if 0 <= i < len(self._ts) else None

    def _limit(self, i0: int, cap_s: float) -> int:
        """One past the last index this walk may read."""
        import numpy as np
        j = int(np.searchsorted(self.ts, self.ts[i0] + cap_s,
                                side="right"))
        return min(j, int(self.seg_end[i0]))

    def bracket(self, i0: int, entry: float, side: int,
                ticks: int = BRACKET_TICKS,
                cap_s: float = BRACKET_CAP_S):
        """Which of target / stop was touched first, from the seconds.

        Returns (outcome, seconds). `side` is +1 long, -1 short:

            target     the favourable level came first
            stop       the adverse level came first
            ambiguous  ONE second's range covered both -- which came
                       first is not in this data, so it is not claimed
            neither    the cap ran out, the data ran out, or the
                       contract rolled
        """
        import numpy as np
        if side == 0 or not self.frozen or i0 >= len(self.ts):
            return ("neither", None)
        j = self._limit(i0, cap_s)
        if j <= i0:
            return ("neither", None)
        offset = ticks * TICK
        good = entry + offset * side        # favourable for this side
        bad = entry - offset * side         # adverse
        hi_lvl, lo_lvl = (good, bad) if side > 0 else (bad, good)
        up = self.hi[i0:j] >= hi_lvl - W.EPS
        dn = self.lo[i0:j] <= lo_lvl + W.EPS
        any_hit = up | dn
        if not any_hit.any():
            return ("neither", None)
        k = int(np.argmax(any_hit))
        secs = float(self.ts[i0 + k] - self.ts[i0])
        if up[k] and dn[k]:
            return ("ambiguous", secs)
        hit_good = up[k] if side > 0 else dn[k]
        return (("target" if hit_good else "stop"), secs)

    def excursion(self, i0: int, entry: float, side: int,
                  cap_s: float = BRACKET_CAP_S):
        """(best, worst) excursion in ticks, signed by `side`."""
        if side == 0 or not self.frozen or i0 >= len(self.ts):
            return (None, None)
        j = self._limit(i0, cap_s)
        if j <= i0:
            return (None, None)
        hi = float(self.hi[i0:j].max())
        lo = float(self.lo[i0:j].min())
        if side > 0:
            return (max(0.0, (hi - entry) / TICK),
                    min(0.0, (lo - entry) / TICK))
        return (max(0.0, (entry - lo) / TICK),
                min(0.0, (entry - hi) / TICK))

    def forward(self, i0: int, entry: float, side: int, minutes: int):
        """Ticks `minutes` later, signed by side.

        The price `minutes` later is the LAST print at or before that
        moment -- what a screen would have been showing. Taking the
        next print after a quiet stretch instead would report a price
        from minutes further on and call it this horizon.
        """
        import numpy as np
        if side == 0 or not self.frozen or i0 >= len(self.ts):
            return None
        want = self.ts[i0] + minutes * 60
        end = int(self.seg_end[i0])
        # The tape has to REACH the horizon. "The last print at or
        # before that moment" is the right price only while the data
        # continues past it; where the session simply ends, the last
        # print is not the price fifteen minutes later, and reporting
        # it fills the tail of every session with rows that say price
        # did not move.
        if end <= i0 or self.ts[end - 1] < want:
            return None
        j = int(np.searchsorted(self.ts, want, side="right")) - 1
        if j < i0 or j >= end:
            return None
        return float((self.px[j] - entry) * side / TICK)


# ---------------------------------------------------------- the rows

def minute_rows(c: W.Candle, tape: Tape) -> list[dict]:
    """Every column on the page, plus the forward ones."""
    out = []
    walk = c.walk()
    _s, o15, _h, _lo, _c15, _v, _k = c.bar15
    fives = {b[0]: b for b in c.bars5}
    for idx, r in enumerate(walk):
        slot = r["slot"]
        five = fives.get(slot - (slot % SPAN5))
        side = r["d1"]
        end = slot + SPAN1                 # the minute's last instant
        i0 = tape.after(end - 1)           # first print after it closed
        entry = tape.price_at(i0)
        lag = (tape.ts[i0] - (end - 1)) if entry is not None else None

        fwd = {}
        mfe = mae = None
        first, secs = "neither", None
        # `and side` is belt-and-braces: each of forward/excursion/
        # bracket refuses a side of 0 on its own, and those refusals
        # are what the tests pin. Removing this line changes no output
        # -- it only saves the work.
        if entry is not None and side:
            for m in HORIZONS:
                fwd[m] = tape.forward(i0, entry, side, m)
            mfe, mae = tape.excursion(i0, entry, side)
            first, secs = tape.bracket(i0, entry, side)

        out.append({
            "session": c.day,
            "candle": C.clock(c.slot),
            "minute": C.clock(slot),
            "idx": idx,
            "contract": c.contract,
            "open": round(r["open"], 4),
            "high": round(r["high"], 4),
            "low": round(r["low"], 4),
            "close": round(r["close"], 4),
            "volume": int(r["volume"]),
            "o5": round(five[1], 4) if five else "",
            "o15": round(o15, 4),
            "d1": r["d1"], "d5": r["d5"], "d15": r["d15"],
            "agree": 1 if r["agree"] else 0,
            "to_close": ("" if r["to_close"] is None
                         else f"{r['to_close']:.0f}"),
            "entry": "" if entry is None else round(entry, 4),
            "entry_lag_s": "" if lag is None else f"{lag:.0f}",
            "fwd_1": _num(fwd.get(1)),
            "fwd_5": _num(fwd.get(5)),
            "fwd_15": _num(fwd.get(15)),
            "mfe": _num(mfe), "mae": _num(mae),
            "first_4t": first,
            "secs_to_first": "" if secs is None else f"{secs:.0f}",
            "utc": dt.datetime.fromtimestamp(
                slot, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
    return out


def _num(x):
    return "" if x is None else f"{x:.2f}"


def candle_row(c: W.Candle) -> dict:
    s = c.summary()
    _sl, o, h, lo, cl, v, _k = c.bar15
    r = c.rollup()
    walk = c.walk()
    first_idx = ""
    for i, w in enumerate(walk):
        if w["agree"]:
            first_idx = i
            break
    return {
        "session": c.day,
        "candle": C.clock(c.slot),
        "contract": c.contract,
        "open": round(o, 4), "high": round(h, 4),
        "low": round(lo, 4), "close": round(cl, 4),
        "chg_ticks": f"{s['chg']:.0f}",
        "range_ticks": f"{(h - lo) / TICK:.0f}",
        "volume": int(v),
        "minutes": s["minutes"], "agreed": s["agreed"],
        "first_agree_idx": first_idx,
        "rollup_ok": 1 if (r and all(r.values())) else 0,
        "partial": 1 if c.partial else 0,
        "utc": dt.datetime.fromtimestamp(
            c.slot, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# -------------------------------------------------------- the writing

class Sink:
    """Streams rows to CSV. Two years will not fit in a list."""

    def __init__(self, path: Path, cols) -> None:
        self.path, self.cols, self.n = path, cols, 0
        self.fh = open(path, "w", newline="")
        self.w = csv.DictWriter(self.fh, fieldnames=cols,
                                extrasaction="ignore")
        self.w.writeheader()

    def add(self, rows) -> None:
        for r in rows:
            self.w.writerow(r)
            self.n += 1

    def close(self) -> None:
        self.fh.close()


def to_parquet(csv_path: Path, out: Path) -> str:
    """Same rows, ~10x smaller. Optional -- CSV is always written."""
    try:
        import pyarrow.csv as pv
        import pyarrow.parquet as pq
    except ImportError:
        return "pyarrow not installed -- CSV only"
    table = pv.read_csv(csv_path)
    pq.write_table(table, out, compression="snappy")
    return f"{out.stat().st_size / 1e6:,.1f} MB"


# ------------------------------------------------------------- run

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+", help=".dbn / .dbn.zst files")
    ap.add_argument("--out", default="ES_walk",
                    help="directory to write into")
    ap.add_argument("--month", default="")
    ap.add_argument("--from", dest="start", default="")
    ap.add_argument("--to", dest="stop", default="")
    ap.add_argument("--rth", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N records per file -- a trial run")
    ap.add_argument("--no-parquet", action="store_true")
    args = ap.parse_args(argv)

    start, stop = args.start, args.stop
    if args.month:
        start = start or f"{args.month}-01"
        stop = stop or f"{args.month}-31"

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    ohlcv, depth = [], []
    for p in args.paths:
        (ohlcv if "ohlcv" in Path(p).name.lower() else depth).append(p)
    ordered = [(p, "ohlcv-1s") for p in ohlcv] + \
              [(p, "mbp-10") for p in depth]

    print(f"out        {out}")
    print(f"dates      {start or 'all'} to {stop or 'all'}")
    print(f"window     {'08:30-15:15 RTH' if args.rth else 'full session'}")
    print(f"rows       one per 1-minute bar, plus a per-candle summary")
    print(f"forward    {HORIZONS} minutes, and a "
          f"{BRACKET_TICKS}-tick bracket from the 1-second prints")
    print("progress on stderr")
    print()

    mins = Sink(out / "ES_walk_minutes.csv", MINUTE_COLS)
    cands = Sink(out / "ES_walk_candles.csv", CANDLE_COLS)

    sessions = 0
    candles = 0
    agreed = 0
    bracket = {"target": 0, "stop": 0, "neither": 0, "ambiguous": 0}
    sources = []
    cur_day = None
    builders = None
    tape = None

    def close_day():
        nonlocal sessions, candles, agreed, builders, tape
        if builders is None:
            return
        tape.freeze()
        cs = W.group_candles(builders["15m"].finish(),
                             builders["5m"].finish(),
                             builders["1m"].finish())
        builders = None
        if cs:
            sessions += 1
        for c in cs:
            candles += 1
            rows = minute_rows(c, tape)
            for r in rows:
                if r["agree"]:
                    agreed += 1
                bracket[r["first_4t"]] = bracket.get(
                    r["first_4t"], 0) + 1
            mins.add(rows)
            cands.add([candle_row(c)])
        tape = None

    try:
        for path, schema in ordered:
            src = C.Source(path=path, schema=schema)
            pr = C.Progress(f"{schema} {Path(path).name[:22]}")
            for ts, o, h, lo, c, v, iid in C.records(
                    path, src, progress=pr, limit=args.limit,
                    rth=args.rth):
                day = C.session_day(ts)
                if start and day < start:
                    continue
                if stop and day > stop:
                    continue
                if day != cur_day:
                    close_day()
                    cur_day = day
                    builders = {"15m": C.Bars(SPAN15),
                                "5m": C.Bars(SPAN5),
                                "1m": C.Bars(SPAN1)}
                    tape = Tape()
                    pr.extra = f"{day}  {candles:,} candles"
                tape.add(ts, h, lo, c, iid)
                for b in builders.values():
                    b.feed(ts, o, h, lo, c, v, iid)
            pr.done()
            sources.append(src)
        close_day()
    finally:
        mins.close()
        cands.close()

    if not candles:
        print("No candles -- nothing matched the dates.",
              file=sys.stderr)
        return 1

    print("SOURCES")
    for s in sources:
        print(f"  {Path(s.path).name}  kept {s.kept:,}  "
              f"skipped {s.skipped:,}")
    print()
    print("FILES")
    for sink in (mins, cands):
        size = sink.path.stat().st_size / 1e6
        print(f"  {sink.path}   {sink.n:,} rows   {size:,.1f} MB")
    if not args.no_parquet:
        note = to_parquet(mins.path,
                          out / "ES_walk_minutes.parquet")
        if note.endswith("CSV only"):
            print(f"      parquet skipped: {note}")
        else:
            print(f"  {out / 'ES_walk_minutes.parquet'}   {note}")
    print()
    print(f"{candles:,} candles over {sessions:,} sessions; "
          f"{mins.n:,} minutes, {agreed:,} of them agreeing "
          f"({100.0 * agreed / mins.n:.1f}%).")
    tot = sum(bracket.values()) or 1
    print(f"4-tick bracket from entry, 15-minute cap:")
    for k in ("target", "stop", "neither", "ambiguous"):
        print(f"  {k:<10} {bracket.get(k, 0):>9,}  "
              f"({100.0 * bracket.get(k, 0) / tot:5.1f}%)")
    print()
    print("`to_close` is in the file because it is on the pages. Do "
          "not rank on it -- it runs to the")
    print("end of the same candle, so it measures position in the "
          "candle. Use fwd_1 / fwd_5 / fwd_15,")
    print("mfe / mae, or first_4t, all measured from `entry` (the "
          "first print AFTER the minute closed).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
