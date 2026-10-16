#!/usr/bin/env python3
"""Every ES session as one page of candles: 15m, 5m and 1m, stacked.

    python scripts/chart_pdf.py data/*.dbn.zst --out ~/Downloads/ES.pdf

One page per CME trade date. The three panels share an x-axis in real
time, so 10:15 on the 15m panel sits directly above 10:15 on the 5m and
the 1m. That vertical alignment is the point of the page: it is how you
read what a 15-minute candle was doing while it was forming.

    --rth              08:30-15:15 Chicago only -- the readable view
    --from / --to      YYYY-MM-DD, inclusive, to chart part of the file
    --max-sessions N   stop after N pages -- a trial run
    --limit N          stop after N records per file -- a faster trial run

SESSIONS

A CME equity session runs 17:00 to 16:00 Chicago and is named for the day
it CLOSES on, so Sunday evening belongs to Monday. This is the same rule
`data_audit.py` counts with, so the page count here should equal the
session count it reported.

A BAR NEVER SPANS A CONTRACT ROLL -- the same rule `export_bars.py`
builds under. Across a roll the price steps by carry, and a bar built
through one would draw a range nobody traded. Where a session contains a
roll it is marked on the page and ruled vertically.

READING THE CANDLES

Up bars are hollow with a blue edge, down bars are solid red. The fill
carries the direction as well as the colour, so the page survives
greyscale printing and red-green colour blindness; neither reading
depends on hue alone.

A bar whose open and close are equal has no body and draws as a single
horizontal line. That is a real state, not a rendering fault.

NOTHING HERE PLACES, SIZES OR TIMES AN ORDER. It draws rectangles.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liqmap import dbn                                      # noqa: E402

FRAMES = (("15m", 900), ("5m", 300), ("1m", 60))
EXCHANGE_TZ = "America/Chicago"

# The regular session, in seconds from midnight Chicago.
RTH_OPEN = 8 * 3600 + 30 * 60                   # 08:30
RTH_CLOSE = 15 * 3600 + 15 * 60                 # 15:15

TICK = 0.25                                     # ES, one tick, in points

# Up and down are a POLARITY, so the pair is diverging: one cool pole,
# one warm, no third hue. Validated against a white page --  CVD dE 21.6
# (protan), normal-vision dE 32.3, both clear of the floors.
UP = "#2a78d6"
DOWN = "#e34948"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d8d7d2"
ROLL = "#eda100"

BODY_WIDTH = 0.7            # fraction of the bar's span
HEARTBEAT = 2_000_000


class Progress:
    """Heartbeat on stderr. Two years of seconds take minutes."""

    def __init__(self, label: str, every: int = HEARTBEAT, out=None) -> None:
        self.label, self.every, self.n = label, max(1, every), 0
        self.extra = ""
        self.out = out if out is not None else sys.stderr
        self.start = time.time()

    def tick(self) -> None:
        self.n += 1
        if self.n % self.every:
            return
        el = max(1e-9, time.time() - self.start)
        print(f"  [{self.label}] {self.n:,}  {el:,.0f}s  "
              f"{self.n / el:,.0f}/s  {self.extra}", file=self.out,
              flush=True)

    def done(self) -> None:
        el = max(1e-9, time.time() - self.start)
        print(f"  [{self.label}] done: {self.n:,} in {el:,.0f}s",
              file=self.out, flush=True)


def _tz():
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(EXCHANGE_TZ)
    except Exception:                                   # noqa: BLE001
        return dt.timezone.utc


TZ = _tz()


def session_day(ts: float) -> str:
    """The trade date an epoch second belongs to.

    CME equity sessions run 17:00 to 16:00 Chicago and are named for the
    day they CLOSE on, so 17:00 Sunday onwards is Monday's session. The
    same rule `data_audit._session_day` counts sessions with.
    """
    d = dt.datetime.fromtimestamp(ts, TZ)
    if d.hour >= 17:
        d += dt.timedelta(days=1)
    return d.strftime("%Y-%m-%d")


def clock(ts: float) -> str:
    """HH:MM in exchange time."""
    return dt.datetime.fromtimestamp(ts, TZ).strftime("%H:%M")


def seconds_into_day(ts: float) -> int:
    """Seconds from midnight Chicago -- for the RTH window test."""
    d = dt.datetime.fromtimestamp(ts, TZ)
    return d.hour * 3600 + d.minute * 60 + d.second


def in_rth(ts: float) -> bool:
    """Inside 08:30-15:15 Chicago.

    The window is a half-open interval on the close: 15:15:00 itself
    belongs to the session that is ending, and a bar stamped 15:15:00
    opens the minute AFTER the cash close.
    """
    s = seconds_into_day(ts)
    return RTH_OPEN <= s < RTH_CLOSE


# --------------------------------------------------------------- bars

@dataclass
class Live:
    """One timeframe's bar in progress."""

    slot: float
    contract: object
    open: float
    high: float
    low: float
    close: float
    volume: float

    def tuple(self) -> tuple:
        return (self.slot, self.open, self.high, self.low, self.close,
                self.volume, self.contract)


class Bars:
    """Aggregates 1-second bars (or prints) to one span.

    A new slot OR a change of contract closes the bar in progress. The
    second is the one that matters: see the module docstring on rolls.
    This mirrors `export_bars.Builder`, which writes the same bars as
    spreadsheet rows; the rule is stated in both places because the two
    scripts must not be able to drift apart silently.
    """

    def __init__(self, span: int) -> None:
        self.span = span
        self.cur: Live | None = None
        self.out: list[tuple] = []

    def feed(self, ts: float, o: float, h: float, lo: float, c: float,
             v: float, contract) -> None:
        s = math.floor(ts / self.span) * self.span
        cur = self.cur
        if cur is None or s != cur.slot or contract != cur.contract:
            if cur is not None:
                self.out.append(cur.tuple())
            self.cur = Live(slot=s, contract=contract, open=o, high=h,
                            low=lo, close=c, volume=v)
            return
        cur.high = max(cur.high, h)
        cur.low = min(cur.low, lo)
        cur.close = c
        cur.volume += v

    def finish(self) -> list[tuple]:
        if self.cur is not None:
            self.out.append(self.cur.tuple())
            self.cur = None
        return self.out


@dataclass
class Session:
    """One trade date's bars, at all three timeframes."""

    day: str
    builders: dict = field(default_factory=dict)
    frames: dict = field(default_factory=dict)
    contracts: list = field(default_factory=list)
    rolls: list = field(default_factory=list)
    seconds: int = 0

    @classmethod
    def start(cls, day: str) -> "Session":
        s = cls(day=day)
        s.builders = {name: Bars(span) for name, span in FRAMES}
        return s

    def feed(self, ts, o, h, lo, c, v, contract) -> None:
        if contract not in self.contracts:
            if self.contracts:
                self.rolls.append(ts)
            self.contracts.append(contract)
        self.seconds += 1
        for b in self.builders.values():
            b.feed(ts, o, h, lo, c, v, contract)

    def close(self) -> None:
        self.frames = {name: b.finish()
                       for name, b in self.builders.items()}
        self.builders = {}

    @property
    def empty(self) -> bool:
        return not any(self.frames.values())

    def stats(self) -> dict:
        """Open, high, low, close and range for the header.

        Any of the three frames would give the same four numbers: they
        are fed the same prints, so the extremes and the first and last
        price agree by construction. The 1m frame is used because it is
        the finest, and a test asserts the agreement rather than taking
        it on trust.
        """
        bars = self.frames.get("1m") or []
        if not bars:
            return {}
        bars = sorted(bars, key=lambda b: b[0])
        hi = max(b[2] for b in bars)
        lo = min(b[3] for b in bars)
        return {
            "open": bars[0][1], "high": hi, "low": lo,
            "close": bars[-1][4],
            "range": hi - lo,
            "ticks": int(round((hi - lo) / TICK)),
            "volume": sum(b[5] for b in bars),
            "first": bars[0][0], "last": bars[-1][0],
        }


# ------------------------------------------------------------- reading

@dataclass
class Source:
    path: str
    schema: str = ""
    kept: int = 0
    skipped: int = 0
    first: float | None = None
    last: float = 0.0


def records(path: str, src: Source, progress=None, limit: int = 0,
            rth: bool = False):
    """Yield (ts, o, h, lo, c, v, instrument) from one file, in order.

    `ohlcv` files give the venue's own 1-second bars. Depth files give
    trade prints, each of which becomes a one-tick bar and lets the
    aggregation do the rest.
    """
    import itertools
    store = dbn.open_file(path)
    if limit:
        store = itertools.islice(store, limit)
    depth = src.schema != "ohlcv-1s"
    for rec in store:
        if progress is not None:
            progress.tick()
        if depth:
            t = dbn.trade_from(rec)
            if t is None:
                continue
            ts, o, h, lo, c, v = t.ts, t.px, t.px, t.px, t.px, t.sz
        else:
            ts = dbn.seconds(dbn._get(rec, "ts_event"))
            o = dbn.price(dbn._get(rec, "open"))
            h = dbn.price(dbn._get(rec, "high"))
            lo = dbn.price(dbn._get(rec, "low"))
            c = dbn.price(dbn._get(rec, "close"))
            v = float(dbn._get(rec, "volume", 0) or 0)
        if ts <= 0 or min(o, h, lo, c) <= 0:
            src.skipped += 1
            continue
        if not (lo <= o <= h and lo <= c <= h):
            src.skipped += 1
            continue
        if rth and not in_rth(ts):
            continue
        src.kept += 1
        if src.first is None:
            src.first = ts
        src.last = max(src.last, ts)
        yield (ts, o, h, lo, c, v, dbn._get(rec, "instrument_id"))


# ------------------------------------------------------------- drawing

def candle_geometry(bars, span: int):
    """Split bars into the four collections a panel needs.

    Returns (up_wicks, down_wicks, up_bodies, down_bodies) where a wick
    is a two-point segment and a body is a four-point polygon.

    A bar is UP when its close is strictly above its open. A bar that
    closed where it opened is neither, and is drawn with the down
    collection's geometry but no fill -- it has no height, so what shows
    is its edge. Classing it as "up" instead would overstate how often
    price finished higher, which is the number these pages exist to let
    someone eyeball.
    """
    half = span * BODY_WIDTH / 2.0
    uw, dw, ub, db = [], [], [], []
    for slot, o, h, lo, c, _v, _k in bars:
        mid = slot + span / 2.0
        up = c > o
        (uw if up else dw).append(((mid, lo), (mid, h)))
        x0, x1 = mid - half, mid + half
        top, bot = max(o, c), min(o, c)
        poly = ((x0, bot), (x0, top), (x1, top), (x1, bot))
        (ub if up else db).append(poly)
    return uw, dw, ub, db


def draw_panel(ax, bars, span: int, label: str, xlim, rolls=()) -> None:
    """One timeframe, drawn as four collections rather than N artists.

    A session is up to 1,380 one-minute bars. Drawn one Rectangle and
    one Line2D at a time that is ~2,800 artists a panel and the run does
    not finish; as collections it is four.
    """
    from matplotlib.collections import LineCollection, PolyCollection

    ax.set_xlim(*xlim)
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, which="major", color=GRID, linewidth=0.4, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=6, length=2, width=0.4)

    if not bars:
        ax.text(0.5, 0.5, "no bars in this window", transform=ax.transAxes,
                ha="center", va="center", color=MUTED, fontsize=7)
        ax.set_yticks([])
        ax.set_ylabel(label, color=INK, fontsize=8, fontweight="bold")
        return

    uw, dw, ub, db = candle_geometry(bars, span)
    lw = 0.5 if span >= 300 else 0.35

    # Wicks under bodies, so a body's fill covers the line through it.
    ax.add_collection(LineCollection(uw, colors=UP, linewidths=lw,
                                     zorder=2))
    ax.add_collection(LineCollection(dw, colors=DOWN, linewidths=lw,
                                     zorder=2))
    # Up is hollow, down is solid: the direction is carried by the fill
    # as well as the hue, so greyscale and CVD both still read.
    ax.add_collection(PolyCollection(ub, facecolors="white",
                                     edgecolors=UP, linewidths=lw,
                                     zorder=3))
    ax.add_collection(PolyCollection(db, facecolors=DOWN,
                                     edgecolors=DOWN, linewidths=lw,
                                     zorder=3))

    lows = [b[3] for b in bars]
    highs = [b[2] for b in bars]
    lo, hi = min(lows), max(highs)
    pad = max((hi - lo) * 0.06, TICK)
    ax.set_ylim(lo - pad, hi + pad)

    for r in rolls:
        ax.axvline(r, color=ROLL, linewidth=0.8, linestyle=(0, (4, 2)),
                   zorder=4)

    ax.set_ylabel(label, color=INK, fontsize=8, fontweight="bold")


def time_ticks(lo: float, hi: float, want: int = 9):
    """Tick positions on the hour, about `want` of them."""
    hours = max(1.0, (hi - lo) / 3600.0)
    step = max(1, int(round(hours / want)))
    for mult in (1, 2, 3, 4, 6, 8, 12, 24):
        if mult >= step:
            step = mult
            break
    start = math.ceil(lo / (step * 3600)) * (step * 3600)
    out = []
    t = start
    while t <= hi:
        out.append(t)
        t += step * 3600
    return out


def session_page(fig, sess: Session, page: int, rth: bool) -> None:
    """Draw one trade date onto a cleared figure."""
    st = sess.stats()
    spans = [b[0] for f in sess.frames.values() for b in f]
    if not spans:
        return
    lo_x = min(spans)
    hi_x = max(b[0] + span
               for (name, span) in FRAMES
               for b in sess.frames.get(name, []))

    axes = fig.subplots(3, 1, sharex=True,
                        gridspec_kw={"height_ratios": [1, 1, 1],
                                     "hspace": 0.12,
                                     "left": 0.065, "right": 0.985,
                                     "top": 0.86, "bottom": 0.075})
    for ax, (name, span) in zip(axes, FRAMES):
        draw_panel(ax, sess.frames.get(name, []), span, name,
                   (lo_x, hi_x), sess.rolls)

    ticks = time_ticks(lo_x, hi_x)
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels([clock(t) for t in ticks])
    axes[-1].set_xlabel(f"{EXCHANGE_TZ}", color=MUTED, fontsize=7)

    day = dt.datetime.strptime(sess.day, "%Y-%m-%d")
    window = "08:30-15:15" if rth else "17:00 prior day - 16:00"
    fig.text(0.065, 0.955, f"{sess.day}  {day.strftime('%A')}",
             fontsize=14, fontweight="bold", color=INK, va="top")
    fig.text(0.065, 0.915, f"ES  ·  {window} {EXCHANGE_TZ}  ·  page {page}",
             fontsize=7.5, color=MUTED, va="top")

    if st:
        bits = [
            f"open {st['open']:,.2f}",
            f"high {st['high']:,.2f}",
            f"low {st['low']:,.2f}",
            f"close {st['close']:,.2f}",
            f"range {st['range']:,.2f} pts ({st['ticks']} ticks)",
            f"volume {int(st['volume']):,}",
        ]
        fig.text(0.985, 0.955, "   ".join(bits), fontsize=7.5,
                 color=INK, va="top", ha="right")
    counts = "   ".join(f"{n} {len(sess.frames.get(n, [])):,}"
                        for n, _ in FRAMES)
    tail = f"bars: {counts}   ·   contract {sess.contracts[0]}"
    if len(sess.contracts) > 1:
        tail += (f" -> {sess.contracts[-1]}  ROLL, ruled in amber; "
                 f"no bar is built across it")
    fig.text(0.985, 0.915, tail, fontsize=7.5, color=MUTED, va="top",
             ha="right")

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor="white", edgecolor=UP, label="up (close > open)"),
        Patch(facecolor=DOWN, edgecolor=DOWN,
              label="down (close <= open)"),
    ]
    if sess.rolls:
        legend.append(Line2D([0], [0], color=ROLL, linewidth=0.8,
                             linestyle=(0, (4, 2)), label="contract roll"))
    fig.legend(handles=legend, loc="upper left",
               bbox_to_anchor=(0.065, 0.895), ncol=3, frameon=False,
               fontsize=7, labelcolor=INK, handlelength=1.6,
               columnspacing=1.4)


def cover_page(fig, meta: dict) -> None:
    """What the document is, where it came from, and how to read it."""
    y = 0.90
    fig.text(0.08, y, "ES futures", fontsize=26, fontweight="bold",
             color=INK, va="top")
    y -= 0.075
    fig.text(0.08, y, "every session, three timeframes, one page each",
             fontsize=13, color=MUTED, va="top")
    y -= 0.085

    rows = [
        ("Sessions", f"{meta['sessions']:,} pages"),
        ("Span", f"{meta['first']} to {meta['last']}"),
        ("Window", meta["window"]),
        ("Panels", "15m on top, then 5m, then 1m -- one shared x-axis, "
                   "so the same minute lines up vertically"),
        ("Source", meta["sources"]),
        ("Records", f"{meta['kept']:,} kept, {meta['skipped']:,} rejected "
                    f"as malformed"),
        ("Built", meta["built"]),
    ]
    for k, v in rows:
        fig.text(0.08, y, k, fontsize=8.5, fontweight="bold", color=INK,
                 va="top")
        fig.text(0.22, y, v, fontsize=8.5, color=MUTED, va="top",
                 wrap=True)
        y -= 0.047

    y -= 0.02
    fig.text(0.08, y, "Reading a page", fontsize=11, fontweight="bold",
             color=INK, va="top")
    y -= 0.045
    notes = [
        "Up bars are hollow with a blue edge; down bars are solid red. "
        "The fill carries the direction as well as the colour, so the",
        "page reads in greyscale and under red-green colour blindness. "
        "A bar that closed where it opened has no body and draws as a",
        "single line -- that is a real state, not a fault.",
        "",
        "A bar is never built across a contract roll. Where a session "
        "contains one it is ruled in amber and named in the header.",
        "",
        "Sessions are CME trade dates: 17:00 to 16:00 Chicago, named for "
        "the day they close on, so Sunday evening belongs to Monday.",
        "",
        "These pages show what price did. They do not contain a signal, "
        "a recommendation, or an order.",
    ]
    for line in notes:
        fig.text(0.08, y, line, fontsize=8, color=MUTED, va="top")
        y -= 0.030


def index_pages(pdf, fig, rows, per_page: int = 44) -> int:
    """A contents table -- 517 pages need one. Returns pages written."""
    cols = 3
    chunk = per_page * cols
    written = 0
    for start in range(0, len(rows), chunk):
        part = rows[start:start + chunk]
        fig.clf()
        fig.text(0.065, 0.955, "Contents", fontsize=14,
                 fontweight="bold", color=INK, va="top")
        fig.text(0.065, 0.925,
                 "trade date, session range in ticks, page",
                 fontsize=7.5, color=MUTED, va="top")
        for i, (day, ticks, page) in enumerate(part):
            col, row = divmod(i, per_page)
            x = 0.065 + col * 0.315
            y = 0.885 - row * 0.0188
            fig.text(x, y, day, fontsize=6.6, color=INK, va="top",
                     family="DejaVu Sans Mono")
            fig.text(x + 0.112, y, f"{ticks:>5}", fontsize=6.6,
                     color=MUTED, va="top", family="DejaVu Sans Mono")
            fig.text(x + 0.168, y, f"{page:>4}", fontsize=6.6,
                     color=MUTED, va="top", family="DejaVu Sans Mono")
        pdf.savefig(fig)
        written += 1
    return written


# ------------------------------------------------------------- the run

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+", help=".dbn / .dbn.zst files")
    ap.add_argument("--out", default="ES_sessions.pdf")
    ap.add_argument("--rth", action="store_true",
                    help="08:30-15:15 Chicago only -- far more readable")
    ap.add_argument("--from", dest="start", default="",
                    help="YYYY-MM-DD, inclusive")
    ap.add_argument("--to", dest="stop", default="",
                    help="YYYY-MM-DD, inclusive")
    ap.add_argument("--max-sessions", type=int, default=0,
                    help="stop after N pages -- a trial run")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N records per file -- a trial run")
    ap.add_argument("--no-index", action="store_true")
    args = ap.parse_args(argv)

    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.backends.backend_pdf import PdfPages
        from matplotlib.figure import Figure
    except ImportError:
        print("matplotlib is not installed in this interpreter.\n"
              "    source .venv/bin/activate\n"
              "    pip install matplotlib", file=sys.stderr)
        return 2

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    ohlcv, depth = [], []
    for p in args.paths:
        (ohlcv if "ohlcv" in Path(p).name.lower() else depth).append(p)
    ordered = [(p, "ohlcv-1s") for p in ohlcv] + \
              [(p, "mbp-10") for p in depth]

    print(f"out        {out}")
    print(f"window     {'08:30-15:15 RTH' if args.rth else 'full session'}"
          f"  ({EXCHANGE_TZ})")
    print(f"panels     15m, 5m, 1m -- shared x-axis")
    print(f"files      {len(ordered)}")
    if args.start or args.stop:
        print(f"dates      {args.start or '...'} to {args.stop or '...'}")
    if args.max_sessions:
        print(f"MAX        {args.max_sessions} sessions -- a trial run")
    if args.limit:
        print(f"LIMIT      {args.limit:,} records per file -- a trial run")
    print("progress on stderr")
    print()

    fig = Figure(figsize=(11, 8.5), dpi=150)
    fig.patch.set_facecolor("white")

    sessions_done = 0
    index: list[tuple] = []
    seen: set[str] = set()
    out_of_order: list[str] = []
    sources: list[Source] = []
    pending: list = []                 # pages are written after the cover
    cur: Session | None = None
    stop_now = False

    scratch = out.with_suffix(".pages.pdf")
    body = PdfPages(scratch)

    def flush(s: Session) -> bool:
        """Close a session and draw its page. True when a page was made."""
        nonlocal sessions_done
        s.close()
        if s.empty:
            return False
        sessions_done += 1
        fig.clf()
        session_page(fig, s, sessions_done, args.rth)
        body.savefig(fig)
        st = s.stats()
        index.append((s.day, st.get("ticks", 0), sessions_done))
        return True

    try:
        for path, schema in ordered:
            if stop_now:
                break
            src = Source(path=path, schema=schema)
            pr = Progress(f"{schema} {Path(path).name[:22]}")
            for ts, o, h, lo, c, v, iid in records(
                    path, src, progress=pr, limit=args.limit,
                    rth=args.rth):
                day = session_day(ts)
                if args.start and day < args.start:
                    continue
                if args.stop and day > args.stop:
                    continue
                if cur is None or cur.day != day:
                    if cur is not None:
                        flush(cur)
                        if (args.max_sessions
                                and sessions_done >= args.max_sessions):
                            cur = None
                            stop_now = True
                            break
                    if day in seen:
                        out_of_order.append(day)
                    seen.add(day)
                    cur = Session.start(day)
                    pr.extra = f"{day}  {sessions_done} pages"
                cur.feed(ts, o, h, lo, c, v, iid)
            pr.done()
            sources.append(src)
        if cur is not None:
            flush(cur)
    finally:
        body.close()

    if not sessions_done:
        print("No sessions drawn -- nothing matched the window or the "
              "dates.", file=sys.stderr)
        scratch.unlink(missing_ok=True)
        return 1

    # Front matter is written last because it has to count the pages it
    # introduces, then the body is appended behind it.
    first = min((s.first for s in sources if s.first is not None),
                default=0.0)
    last = max((s.last for s in sources), default=0.0)
    meta = {
        "sessions": sessions_done,
        "first": session_day(first) if first else "?",
        "last": session_day(last) if last else "?",
        "window": ("08:30-15:15 Chicago (RTH)" if args.rth
                   else "17:00-16:00 Chicago (the whole session)"),
        "sources": ", ".join(Path(s.path).name for s in sources),
        "kept": sum(s.kept for s in sources),
        "skipped": sum(s.skipped for s in sources),
        "built": dt.datetime.now(TZ).strftime("%Y-%m-%d %H:%M %Z"),
    }

    front = out.with_suffix(".front.pdf")
    with PdfPages(front) as fp:
        fig.clf()
        cover_page(fig, meta)
        fp.savefig(fig)
        if not args.no_index:
            index_pages(fp, fig, index)

    merged = merge(front, scratch, out, meta)
    front.unlink(missing_ok=True)
    scratch.unlink(missing_ok=True)

    print("SOURCES")
    for s in sources:
        span = (f"  {session_day(s.first)} -> {session_day(s.last)}"
                if s.first else "")
        print(f"  {Path(s.path).name}")
        print(f"      {s.schema}  kept {s.kept:,}  "
              f"skipped {s.skipped:,}{span}")
    print()
    if out_of_order:
        print(f"WARNING  {len(out_of_order)} session(s) reappeared after "
              f"being closed, so they have more than one page:")
        print(f"         {', '.join(sorted(set(out_of_order))[:8])}")
        print()
    size = out.stat().st_size / 1e6 if out.exists() else 0.0
    print(f"{out}")
    print(f"  {merged} pages   {size:,.1f} MB   "
          f"{sessions_done:,} sessions")
    print()
    print("Up is hollow with a blue edge, down is solid red. The three "
          "panels share one x-axis, so the")
    print("same minute sits in the same place on all three.")
    if not args.rth:
        print("For a far more readable 1m panel, re-run with --rth.")
    return 0


def merge(front: Path, body: Path, out: Path, meta: dict) -> int:
    """Front matter then the body, into one file. Returns page count."""
    try:
        from pypdf import PdfWriter, PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfWriter, PdfReader       # noqa: F401
        except ImportError:
            # No merger available: ship the body and say where the cover
            # went, rather than losing 517 pages to a missing dependency.
            body.replace(out)
            front.replace(out.with_name(out.stem + "_cover.pdf"))
            print("pypdf not installed -- cover and contents were written "
                  "beside the charts as a separate file.\n"
                  "    pip install pypdf      (then re-run to get one "
                  "file)", file=sys.stderr)
            return 0
    w = PdfWriter()
    n = 0
    for p in (front, body):
        r = PdfReader(str(p))
        for page in r.pages:
            w.add_page(page)
            n += 1
    w.add_metadata({
        "/Title": f"ES sessions {meta['first']} to {meta['last']}",
        "/Subject": f"{meta['sessions']} sessions, 15m/5m/1m candles",
        "/Creator": "liqmap scripts/chart_pdf.py",
    })
    with open(out, "wb") as fh:
        w.write(fh)
    return n


if __name__ == "__main__":
    raise SystemExit(main())
