#!/usr/bin/env python3
"""Every 15-minute ES candle, taken apart: the 3 fives and the 15 ones.

    python scripts/candle_walk.py data/*.dbn.zst --month 2026-08 \\
        --out ~/Downloads/ES_candle_walk.pdf

ONE PAGE PER 15-MINUTE CANDLE. On it:

    the 15m candle drawn whole, beside the three 5m candles that built
    it, beside the fifteen 1m candles that built those -- one shared
    price axis, so the 15m bar is visibly the envelope of the ones to
    its right;

    the OPEN and CLOSE of the 15m candle ruled straight across all
    three panels, which is the thing worth seeing: the first 1m bar
    opens exactly where the 15m opens and the last 1m bar closes
    exactly where the 15m closes, because they are the same prints
    aggregated differently;

    a table of every bar at every timeframe -- date, time, open, high,
    low, close, and the change in ticks;

    and, for each of the fifteen minutes, what the three timeframes
    were showing AT THAT MOMENT, and whether they agreed.

WHAT "AGREE" MEANS HERE, AND WHY IT IS CAUSAL

At minute N inside the candle, the 1m bar has closed. The 5m bar it
sits in has NOT -- but its open is known and its price so far is this
1m close. The 15m bar is the same. So:

    1m  direction   = this bar's close vs this bar's open
    5m  direction   = this bar's close vs the open of its 5m bar
    15m direction   = this bar's close vs the open of the 15m candle

Nothing here reads a bar that had not finished forming, and nothing
reads a price from later in the candle. That is what you would have had
on the screen at that minute.

They AGREE when all three are the same direction and none is flat.

`to close` is how far price then travelled to the end of the 15m candle,
signed by the direction that was showing: positive means the move
continued, negative means it gave back.

MISSING MINUTES ARE SHOWN AS MISSING. Overnight, ES has minutes with no
trades. A candle with 11 of 15 minutes says so on its page rather than
drawing four bars that never happened.

A CANDLE IS NEVER BUILT ACROSS A CONTRACT ROLL, so a 15m slot holding
one produces two partial candles, each labelled.

THIS IS A DESCRIPTION, NOT A TEST. It shows what the candles did. It
does not claim that agreement predicts anything -- the controlled
version of that question is `agree_travel.py`, and its answer was below
a coin-flip null on every rule and horizon tried.

NOTHING HERE PLACES, SIZES OR TIMES AN ORDER.
"""

from __future__ import annotations

import argparse
import datetime as dt
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chart_pdf as C                                       # noqa: E402

TICK = C.TICK
SPAN15, SPAN5, SPAN1 = 900, 300, 60
PER_CANDLE_5 = SPAN15 // SPAN5                              # 3
PER_CANDLE_1 = SPAN15 // SPAN1                              # 15

UP, DOWN, INK, MUTED, GRID = C.UP, C.DOWN, C.INK, C.MUTED, C.GRID
FLAT = "#8a8984"
AGREE_BAND = "#eef4fd"

MONO = "DejaVu Sans Mono"
EPS = 1e-6              # dbn.price decodes are not exact to the tick


def direction(a: float, b: float, eps: float = EPS) -> int:
    """+1, -1 or 0 for b relative to a.

    A tolerance is required, not optional: `dbn.price` decodes a tick
    as 7827.250000000001, so two prices that ARE the same tick differ
    by ~1e-12. Compared with `>` that noise becomes a direction, and
    every flat bar would be classed up or down at random.
    """
    if b - a > eps:
        return 1
    if a - b > eps:
        return -1
    return 0


def arrow(d: int) -> str:
    return {1: "up", -1: "dn", 0: "--"}[d]


def ticks(points: float) -> float:
    return points / TICK


# ------------------------------------------------------- the candles

@dataclass
class Candle:
    """One 15m bar with the finer bars that built it."""

    slot: float
    contract: object
    bar15: tuple
    bars5: list = field(default_factory=list)
    bars1: list = field(default_factory=list)

    @property
    def day(self) -> str:
        return C.session_day(self.slot)

    @property
    def partial(self) -> bool:
        """Fewer minutes than a whole candle -- a gap or a roll."""
        return len(self.bars1) < PER_CANDLE_1

    def rollup(self) -> dict:
        """The 15m bar against the 1m bars underneath it.

        These must agree: both are the same 1-second prints aggregated
        to different spans. The check is printed on every page because
        it is exactly what was asked to be shown, and because a failure
        would mean the aggregation is broken rather than the market
        doing something interesting.
        """
        if not self.bars1:
            return {}
        ones = sorted(self.bars1, key=lambda b: b[0])
        _s, o, h, lo, c, _v, _k = self.bar15
        return {
            "open": abs(ones[0][1] - o) < EPS,
            "high": abs(max(b[2] for b in ones) - h) < EPS,
            "low": abs(min(b[3] for b in ones) - lo) < EPS,
            "close": abs(ones[-1][4] - c) < EPS,
        }

    def walk(self) -> list[dict]:
        """Minute by minute: what each timeframe showed, and agreement.

        See the module docstring. Nothing here reads a price later than
        the minute being described.
        """
        _s, o15, _h, _lo, c15, _v, _k = self.bar15
        fives = {b[0]: b for b in self.bars5}
        out = []
        for b in sorted(self.bars1, key=lambda b: b[0]):
            slot1, o1, h1, lo1, c1, v1, _k1 = b
            five_slot = slot1 - (slot1 % SPAN5)
            five = fives.get(five_slot)
            o5 = five[1] if five else o1
            d1 = direction(o1, c1)
            d5 = direction(o5, c1)
            d15 = direction(o15, c1)
            agreed = d1 == d5 == d15 != 0
            # Signed by the direction showing at that minute, so a
            # positive number always means "it kept going".
            #
            # A FLAT minute has no direction to sign by, so its travel
            # is None rather than zero. Multiplying by d1 == 0 would
            # print "+0 ticks to the close" for a minute after which
            # price moved a hundred ticks -- a fabricated number that
            # looks like a measurement.
            #
            # The `+ 0.0` is not decoration: a DOWN minute that closed
            # exactly at the candle's close gives 0.0 * -1 == -0.0,
            # which prints as "-0" and reads as a loss of nothing.
            travel = (ticks(c15 - c1) * d1) + 0.0 if d1 else None
            out.append({
                "slot": slot1, "open": o1, "high": h1, "low": lo1,
                "close": c1, "volume": v1,
                "d1": d1, "d5": d5, "d15": d15, "agree": agreed,
                "to_close": travel,
            })
        return out

    def summary(self) -> dict:
        w = self.walk()
        agreed = [r for r in w if r["agree"]]
        first = agreed[0] if agreed else None
        return {
            "minutes": len(w),
            "agreed": len(agreed),
            "first_agree": first["slot"] if first else None,
            "first_travel": first["to_close"] if first else None,
            "chg": ticks(self.bar15[4] - self.bar15[1]),
        }


def group_candles(b15, b5, b1) -> list[Candle]:
    """Hang the 5m and 1m bars under the 15m bar they belong to.

    Matching is on the slot range AND the contract. The contract half
    matters at a roll: the slot holds bars from two instruments, and
    without it the new contract's minutes would be filed under the old
    contract's 15m candle and show as a vertical jump that nobody
    traded.
    """
    out = []
    for bar in sorted(b15, key=lambda b: (b[0], str(b[6]))):
        slot, k = bar[0], bar[6]
        c = Candle(slot=slot, contract=k, bar15=bar)
        c.bars5 = [b for b in b5
                   if slot <= b[0] < slot + SPAN15 and b[6] == k]
        c.bars1 = [b for b in b1
                   if slot <= b[0] < slot + SPAN15 and b[6] == k]
        out.append(c)
    return out


# -------------------------------------------------------- aggregates

@dataclass
class Census:
    """Counts across every candle drawn. Counts, not probabilities."""

    candles: int = 0
    partial: int = 0
    minutes: int = 0
    agreed: int = 0
    agree_travel: list = field(default_factory=list)
    other_travel: list = field(default_factory=list)

    def add(self, c: Candle) -> None:
        self.candles += 1
        if c.partial:
            self.partial += 1
        for r in c.walk():
            self.minutes += 1
            if r["agree"]:
                self.agreed += 1
            # A flat minute has no signed travel (see `walk`), so it is
            # counted as a minute and left out of the travel samples
            # rather than entered as a zero.
            if r["to_close"] is None:
                continue
            (self.agree_travel if r["agree"]
             else self.other_travel).append(r["to_close"])

    @staticmethod
    def quartiles(xs) -> tuple:
        if not xs:
            return (0.0, 0.0, 0.0)
        s = sorted(xs)
        return (s[len(s) // 4], statistics.median(s),
                s[(3 * len(s)) // 4])


# ----------------------------------------------------------- drawing

def mini_panel(ax, bars, span, label, ylim, rules=()):
    """One timeframe's candles, on a shared price axis."""
    from matplotlib.collections import LineCollection, PolyCollection

    ax.set_facecolor("white")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_ylim(*ylim)
    ax.tick_params(colors=MUTED, labelsize=6, length=2, width=0.4)
    ax.set_yticks([])
    ax.set_title(label, color=INK, fontsize=8, fontweight="bold", pad=4)

    for y in rules:
        ax.axhline(y, color=MUTED, linewidth=0.6, linestyle=(0, (3, 2)),
                   zorder=1)

    if not bars:
        ax.text(0.5, 0.5, "no bars", transform=ax.transAxes,
                ha="center", va="center", color=MUTED, fontsize=7)
        ax.set_xticks([])
        return

    bars = sorted(bars, key=lambda b: b[0])
    lo_x = bars[0][0]
    hi_x = bars[-1][0] + span
    ax.set_xlim(lo_x - span * 0.3, hi_x + span * 0.3)

    uw, dw, ub, db = C.candle_geometry(bars, span)
    lw = 1.2 if span >= SPAN15 else (0.9 if span >= SPAN5 else 0.7)
    ax.add_collection(LineCollection(uw, colors=UP, linewidths=lw,
                                     zorder=2))
    ax.add_collection(LineCollection(dw, colors=DOWN, linewidths=lw,
                                     zorder=2))
    ax.add_collection(PolyCollection(ub, facecolors="white",
                                     edgecolors=UP, linewidths=lw,
                                     zorder=3))
    ax.add_collection(PolyCollection(db, facecolors=DOWN,
                                     edgecolors=DOWN, linewidths=lw,
                                     zorder=3))
    # Ticks sit under the candle's middle but are LABELLED with the
    # time the bar opened, which is what the `time` column of the table
    # holds. Labelling the midpoint instead puts "09:37" under a candle
    # the table calls 09:30.
    step = max(1, len(bars) // 5)
    shown = bars[::step]
    ax.set_xticks([b[0] + span / 2 for b in shown])
    ax.set_xticklabels([C.clock(b[0]) for b in shown], fontsize=5.5)


def agreement_strip(ax, walk, span=SPAN1):
    """One square a minute: agreed up, agreed down, or neither."""
    from matplotlib.collections import PolyCollection

    ax.set_facecolor("white")
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([])
    if not walk:
        return
    lo_x = walk[0]["slot"]
    hi_x = walk[-1]["slot"] + span
    ax.set_xlim(lo_x - span * 0.3, hi_x + span * 0.3)
    ax.set_ylim(0, 1)
    polys, colors = [], []
    for r in walk:
        x0 = r["slot"] + span * 0.15
        x1 = r["slot"] + span * 0.85
        polys.append(((x0, 0.15), (x0, 0.85), (x1, 0.85), (x1, 0.15)))
        if r["agree"]:
            colors.append(UP if r["d1"] > 0 else DOWN)
        else:
            colors.append("#e6e5e1")
    ax.add_collection(PolyCollection(polys, facecolors=colors,
                                     edgecolors="white", linewidths=0.4))
    ax.text(-0.012, 0.5, "agree", transform=ax.transAxes, ha="right",
            va="center", fontsize=6.5, color=INK, fontweight="bold")


TABLE_COLS = (
    ("tf",    0.000, "l"),
    ("date",  0.045, "l"),
    ("time",  0.150, "l"),
    ("open",  0.225, "r"),
    ("high",  0.320, "r"),
    ("low",   0.415, "r"),
    ("close", 0.510, "r"),
    ("chg",   0.590, "r"),
    ("1m",    0.650, "c"),
    ("5m",    0.700, "c"),
    ("15m",   0.750, "c"),
    ("agree", 0.815, "c"),
    ("to close", 0.900, "r"),
)


def table(fig, c: Candle, top: float, bottom: float) -> None:
    """Every bar at every timeframe, with the agreement columns."""
    x0, x1 = 0.055, 0.965
    width = x1 - x0

    def put(col, y, text, bold=False, color=INK, size=6.4):
        label, frac, align = col
        ha = {"l": "left", "r": "right", "c": "center"}[align]
        fig.text(x0 + frac * width, y, text, fontsize=size,
                 color=color, va="top", ha=ha, family=MONO,
                 fontweight="bold" if bold else "normal")

    y = top
    for col in TABLE_COLS:
        put(col, y, col[0], bold=True, size=6.4)
    y -= 0.016
    fig.add_artist(_rule(x0, x1, y + 0.008))
    y -= 0.004

    rows = []
    _s, o, h, lo, cl, v, _k = c.bar15
    rows.append(("15m", c.day, C.clock(c.slot), o, h, lo, cl, None))
    for b in sorted(c.bars5, key=lambda b: b[0]):
        rows.append(("5m", "", C.clock(b[0]), b[1], b[2], b[3], b[4],
                     None))
    walk = {r["slot"]: r for r in c.walk()}
    for b in sorted(c.bars1, key=lambda b: b[0]):
        rows.append(("1m", "", C.clock(b[0]), b[1], b[2], b[3], b[4],
                     walk.get(b[0])))

    step = (top - bottom) / max(len(rows) + 1, 1)
    step = min(step, 0.0175)
    for tf, date, t, o_, h_, l_, c_, w in rows:
        shade = w is not None and w["agree"]
        if shade:
            fig.add_artist(_band(x0 - 0.006, x1 + 0.006, y, step))
        chg = ticks(c_ - o_)
        base = INK if tf == "15m" else MUTED
        put(TABLE_COLS[0], y, tf, bold=(tf == "15m"), color=base)
        put(TABLE_COLS[1], y, date, color=MUTED)
        put(TABLE_COLS[2], y, t, color=base)
        for col, val in zip(TABLE_COLS[3:7], (o_, h_, l_, c_)):
            put(col, y, f"{val:,.2f}", color=base)
        put(TABLE_COLS[7], y, f"{chg:+.0f}",
            color=UP if chg > 0 else (DOWN if chg < 0 else FLAT))
        if w is not None:
            for col, d in zip(TABLE_COLS[8:11],
                              (w["d1"], w["d5"], w["d15"])):
                put(col, y, arrow(d),
                    color=UP if d > 0 else (DOWN if d < 0 else FLAT))
            put(TABLE_COLS[11], y, "YES" if w["agree"] else "-",
                bold=w["agree"],
                color=INK if w["agree"] else FLAT)
            tc = w["to_close"]
            put(TABLE_COLS[12], y,
                "--" if tc is None else f"{tc:+.0f}",
                color=FLAT if tc is None or tc == 0
                else (UP if tc > 0 else DOWN))
        y -= step


def _rule(x0, x1, y):
    """A hairline in FIGURE coordinates.

    `Figure.add_artist` leaves an artist's transform at the figure's
    own when none was set, so these are placed in the same 0-1 space as
    the `fig.text` calls around them and stay put when the axes move.
    """
    from matplotlib.lines import Line2D
    return Line2D([x0, x1], [y, y], color=GRID, linewidth=0.5)


def _band(x0, x1, y, h):
    from matplotlib.patches import Rectangle
    return Rectangle((x0, y - h + 0.004), x1 - x0, h,
                     facecolor=AGREE_BAND, edgecolor="none", zorder=0)


def candle_page(fig, c: Candle, page: int) -> None:
    """One 15-minute candle, taken apart, on one page."""
    walk = c.walk()
    s = c.summary()
    _sl, o15, h15, lo15, c15, v15, _k = c.bar15

    lows = [b[3] for b in c.bars1] or [lo15]
    highs = [b[2] for b in c.bars1] or [h15]
    lo, hi = min(min(lows), lo15), max(max(highs), h15)
    pad = max((hi - lo) * 0.10, TICK)
    ylim = (lo - pad, hi + pad)

    gs = fig.add_gridspec(1, 3, width_ratios=[2, 4, 13],
                          left=0.055, right=0.965, top=0.845,
                          bottom=0.595, wspace=0.07)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    mini_panel(axes[0], [c.bar15], SPAN15, "15m", ylim, (o15, c15))
    mini_panel(axes[1], c.bars5, SPAN5,
               f"5m  ({len(c.bars5)} of {PER_CANDLE_5})", ylim,
               (o15, c15))
    mini_panel(axes[2], c.bars1, SPAN1,
               f"1m  ({len(c.bars1)} of {PER_CANDLE_1})", ylim,
               (o15, c15))

    strip = fig.add_axes([axes[2].get_position().x0, 0.545,
                          axes[2].get_position().width, 0.033])
    agreement_strip(strip, walk)

    # ------------------------------------------------------ header
    day = dt.datetime.strptime(c.day, "%Y-%m-%d")
    fig.text(0.055, 0.962,
             f"{c.day}  {C.clock(c.slot)}-{C.clock(c.slot + SPAN15)}",
             fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(0.055, 0.928,
             f"{day.strftime('%A')}  ·  ES  ·  15-minute candle  ·  "
             f"contract {c.contract}  ·  page {page}",
             fontsize=7, color=MUTED, va="top")

    chg = s["chg"]
    head = (f"open {o15:,.2f}   high {h15:,.2f}   low {lo15:,.2f}   "
            f"close {c15:,.2f}   {chg:+.0f} ticks   "
            f"range {ticks(h15 - lo15):.0f} ticks   "
            f"volume {int(v15):,}")
    fig.text(0.965, 0.962, head, fontsize=7.5, color=INK, va="top",
             ha="right")

    r = c.rollup()
    if r:
        bits = "  ".join(f"{k} {'ok' if v else 'MISMATCH'}"
                         for k, v in r.items())
        colour = INK if all(r.values()) else DOWN
        fig.text(0.965, 0.928,
                 f"1m rolls up to 15m:  {bits}", fontsize=7,
                 color=colour, va="top", ha="right")

    note = (f"agreed on {s['agreed']} of {s['minutes']} minutes")
    if s["first_agree"] is not None:
        note += (f"   ·   first at {C.clock(s['first_agree'])}, "
                 f"{s['first_travel']:+.0f} ticks to the candle's close")
    if c.partial:
        note += "   ·   PARTIAL: minutes with no trades are absent"
    fig.text(0.055, 0.893, note, fontsize=7.5, color=INK, va="top")

    from matplotlib.patches import Patch
    fig.legend(handles=[
        Patch(facecolor="white", edgecolor=UP, label="up"),
        Patch(facecolor=DOWN, edgecolor=DOWN, label="down"),
        Patch(facecolor=AGREE_BAND, edgecolor=GRID,
              label="all three agree"),
    ], loc="upper right", bbox_to_anchor=(0.965, 0.905), ncol=3,
        frameon=False, fontsize=6.5, labelcolor=INK, handlelength=1.4,
        columnspacing=1.2)

    table(fig, c, top=0.495, bottom=0.035)


def summary_page(fig, cen: Census, meta: dict) -> None:
    """The counts across everything drawn, and what they are not."""
    y = 0.90
    fig.text(0.08, y, "What the agreement column adds up to",
             fontsize=17, fontweight="bold", color=INK, va="top")
    y -= 0.06
    fig.text(0.08, y, f"Across {cen.candles:,} fifteen-minute candles "
                      f"and {cen.minutes:,} minutes inside them.",
             fontsize=9, color=MUTED, va="top")
    y -= 0.07

    pct = 100.0 * cen.agreed / cen.minutes if cen.minutes else 0.0
    a = Census.quartiles(cen.agree_travel)
    b = Census.quartiles(cen.other_travel)
    rows = [
        ("Minutes where 1m, 5m and 15m all agreed",
         f"{cen.agreed:,} of {cen.minutes:,}  ({pct:.1f}%)"),
        ("Ticks to the candle's close, when they agreed",
         f"p25 {a[0]:+.1f}   median {a[1]:+.1f}   p75 {a[2]:+.1f}"),
        ("Ticks to the candle's close, when they did not",
         f"p25 {b[0]:+.1f}   median {b[1]:+.1f}   p75 {b[2]:+.1f}"),
        ("Candles with a minute missing (no trades, or a roll)",
         f"{cen.partial:,} of {cen.candles:,}"),
    ]
    for k, v in rows:
        fig.text(0.08, y, k, fontsize=8.5, color=INK, va="top")
        fig.text(0.62, y, v, fontsize=8.5, color=MUTED, va="top",
                 family=MONO)
        y -= 0.05

    y -= 0.03
    fig.text(0.08, y, "What this is not", fontsize=11,
             fontweight="bold", color=INK, va="top")
    y -= 0.045
    for line in [
        "These are counts of what the candles did, not a test of "
        "whether agreement predicts anything.",
        "",
        "`to close` is measured to the END OF THE SAME CANDLE, which is "
        "a deadline the market does not observe. A minute that",
        "agrees at 09:46 has fourteen minutes left; one that agrees at "
        "09:59 has one. The median above mixes the two, so it is a",
        "description of this shape and not a number to size a trade "
        "from.",
        "",
        "The controlled version of this question -- a horizon that does "
        "not depend on where in the candle you are, and a null of",
        "random duration-matched starts -- is `agree_travel.py`. It was "
        "run on 361,053 prints of this same data and the signal",
        "came in BELOW the null in all 36 cells, under both the loose "
        "and the cascade agreement rules, on both the 5m and the",
        "15m trading frame. Nothing on these pages revises that.",
    ]:
        fig.text(0.08, y, line, fontsize=8, color=MUTED, va="top")
        y -= 0.029


def markdown_candle(c: Candle) -> str:
    """The same page, as text.

    A PDF of candle drawings is the wrong thing to hand another model
    to check: it has to read the picture. This is the identical
    breakdown -- the same `walk()`, `rollup()` and `summary()` the page
    draws from, so the two cannot disagree -- written as a table it can
    read directly.
    """
    _s, o, h, lo, cl, v, _k = c.bar15
    st = c.summary()
    r = c.rollup()
    out = [f"### {c.day}  {C.clock(c.slot)}-{C.clock(c.slot + SPAN15)}"
           f"  ·  contract {c.contract}", ""]
    out.append(f"15m candle: open {o:,.2f}  high {h:,.2f}  "
               f"low {lo:,.2f}  close {cl:,.2f}  "
               f"({st['chg']:+.0f} ticks, range "
               f"{ticks(h - lo):.0f} ticks, volume {int(v):,})")
    if r:
        out.append("1m rolls up to 15m: "
                   + "  ".join(f"{k} {'ok' if ok else 'MISMATCH'}"
                               for k, ok in r.items()))
    note = (f"agreed on {st['agreed']} of {st['minutes']} minutes"
            f" ({len(c.bars5)} of {PER_CANDLE_5} fives, "
            f"{len(c.bars1)} of {PER_CANDLE_1} ones)")
    if st["first_agree"] is not None:
        note += (f"; first at {C.clock(st['first_agree'])}, "
                 f"{st['first_travel']:+.0f} ticks to the close")
    if c.partial:
        note += "; PARTIAL -- minutes with no trades are absent"
    out += [note, ""]
    out.append("| tf | time | open | high | low | close | chg | "
               "1m | 5m | 15m | agree | to close |")
    out.append("|----|------|------|------|-----|-------|-----|"
               "----|----|-----|-------|----------|")

    def row(tf, t, o_, h_, l_, c_, w=None):
        chg = ticks(c_ - o_)
        cells = [tf, t, f"{o_:,.2f}", f"{h_:,.2f}", f"{l_:,.2f}",
                 f"{c_:,.2f}", f"{chg:+.0f}"]
        if w is None:
            cells += ["", "", "", "", ""]
        else:
            tc = w["to_close"]
            cells += [arrow(w["d1"]), arrow(w["d5"]), arrow(w["d15"]),
                      "YES" if w["agree"] else "-",
                      "--" if tc is None else f"{tc:+.0f}"]
        return "| " + " | ".join(cells) + " |"

    out.append(row("15m", C.clock(c.slot), o, h, lo, cl))
    for b in sorted(c.bars5, key=lambda b: b[0]):
        out.append(row("5m", C.clock(b[0]), b[1], b[2], b[3], b[4]))
    walk = {w["slot"]: w for w in c.walk()}
    for b in sorted(c.bars1, key=lambda b: b[0]):
        out.append(row("1m", C.clock(b[0]), b[1], b[2], b[3], b[4],
                       walk.get(b[0])))
    out.append("")
    return "\n".join(out)


TEXT_HEADER = """# ES 15-minute candles, broken down tick by tick

Every 15-minute candle, with the three 5-minute candles and the fifteen
1-minute candles that built it. Prices are index points; `chg` and
`to close` are TICKS (1 tick = 0.25 points = $12.50; 4 ticks = 1 point).

WHAT THE DIRECTION COLUMNS MEAN. At each minute the 5m and 15m bars
have NOT finished forming, so their direction is their OPEN compared
with THIS minute's close -- what a screen was showing at that moment.
Nothing in this file reads a bar that had not formed, or a price from
later than the minute it describes.

    1m   this minute's close vs this minute's open
    5m   this minute's close vs the open of the 5m bar it sits in
    15m  this minute's close vs the open of the 15m candle
    agree = all three the same direction, none flat

`to close` is the ticks from that minute's close to the END OF THE SAME
15-minute candle, signed by the direction showing, so positive means
the move continued. It is NOT a measure of edge: it runs to a deadline
the market does not observe, so a minute agreeing at the first bar has
fourteen minutes to work with and one agreeing at the last has none.
A flat minute has no direction to sign by and shows `--`.

`1m rolls up to 15m` checks that the fifteen 1-minute bars reproduce
the 15-minute bar's open, high, low and close. They are the same
1-second prints aggregated differently, so it should always say ok;
it is printed because a MISMATCH would mean the aggregation is broken.

A bar is never built across a contract roll.

Source: Databento `ohlcv-1s` for CME ES. Sessions run 17:00-16:00
America/Chicago and are named for the day they close on.
"""


def cover_page(fig, meta: dict) -> None:
    y = 0.90
    fig.text(0.08, y, "ES, candle by candle", fontsize=26,
             fontweight="bold", color=INK, va="top")
    y -= 0.07
    fig.text(0.08, y, "every 15-minute candle taken apart into its "
                      "5-minute and 1-minute bars",
             fontsize=13, color=MUTED, va="top")
    y -= 0.085
    for k, v in [
        ("Month", meta["month"]),
        ("Candles", f"{meta['candles']:,}, one page each"),
        ("Sessions", f"{meta['sessions']:,}"),
        ("Window", meta["window"]),
        ("Source", meta["sources"]),
        ("Built", meta["built"]),
    ]:
        fig.text(0.08, y, k, fontsize=8.5, fontweight="bold", color=INK,
                 va="top")
        fig.text(0.22, y, v, fontsize=8.5, color=MUTED, va="top")
        y -= 0.047

    y -= 0.025
    fig.text(0.08, y, "On every page", fontsize=11, fontweight="bold",
             color=INK, va="top")
    y -= 0.045
    for line in [
        "Three panels on one price axis: the 15m candle, the three 5m "
        "candles inside it, the fifteen 1m candles inside those.",
        "The 15m open and close are ruled across all three -- the first "
        "1m bar opens on the first line and the last 1m bar closes",
        "on the second, because they are the same prints aggregated "
        "differently. The page states that check in words too.",
        "",
        "Under the 1m panel, one square a minute: coloured when all "
        "three timeframes were showing the same direction at that",
        "minute, grey when they were not. Agreeing minutes are shaded "
        "in the table below.",
        "",
        "`to close` is the ticks from that minute's close to the end of "
        "the 15m candle, signed by the direction that was showing,",
        "so positive means the move kept going. Nothing on the page "
        "reads a price from later than the minute it describes.",
    ]:
        fig.text(0.08, y, line, fontsize=8, color=MUTED, va="top")
        y -= 0.029


def contents_page(pdf, fig, by_day) -> int:
    fig.clf()
    fig.text(0.065, 0.955, "Contents", fontsize=14, fontweight="bold",
             color=INK, va="top")
    fig.text(0.065, 0.925, "trade date, candles, first page",
             fontsize=7.5, color=MUTED, va="top")
    per_col = 30
    for i, (day, n, first) in enumerate(by_day):
        col, row = divmod(i, per_col)
        x = 0.065 + col * 0.235
        y = 0.885 - row * 0.0255
        fig.text(x, y, day, fontsize=7.4, color=INK, va="top",
                 family=MONO)
        fig.text(x + 0.105, y, f"{n:>4}", fontsize=7.4, color=MUTED,
                 va="top", family=MONO)
        fig.text(x + 0.150, y, f"{first:>5}", fontsize=7.4, color=MUTED,
                 va="top", family=MONO)
    pdf.savefig(fig)
    return 1


# ------------------------------------------------------------- run

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+", help=".dbn / .dbn.zst files")
    ap.add_argument("--out", default="ES_candle_walk.pdf")
    ap.add_argument("--month", default="",
                    help="YYYY-MM; default is the last month present")
    ap.add_argument("--from", dest="start", default="")
    ap.add_argument("--to", dest="stop", default="")
    ap.add_argument("--rth", action="store_true",
                    help="08:30-15:15 Chicago only")
    ap.add_argument("--max-candles", type=int, default=0,
                    help="stop after N pages -- a trial run")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N records per file -- a trial run")
    ap.add_argument("--text", default="",
                    help="also write the SAME breakdown as markdown, "
                         "for handing to a reader that parses numbers "
                         "rather than pictures")
    ap.add_argument("--no-pdf", action="store_true",
                    help="skip the pages; requires --text")
    args = ap.parse_args(argv)

    if args.no_pdf and not args.text:
        print("--no-pdf without --text would produce nothing.",
              file=sys.stderr)
        return 2

    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.backends.backend_pdf import PdfPages
        from matplotlib.figure import Figure
    except ImportError:
        print("matplotlib is not installed in this interpreter.\n"
              "    source .venv/bin/activate\n"
              "    pip install matplotlib pypdf", file=sys.stderr)
        return 2

    start, stop = args.start, args.stop
    if args.month:
        start = start or f"{args.month}-01"
        stop = stop or f"{args.month}-31"

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    ohlcv, depth = [], []
    for p in args.paths:
        (ohlcv if "ohlcv" in Path(p).name.lower() else depth).append(p)
    ordered = [(p, "ohlcv-1s") for p in ohlcv] + \
              [(p, "mbp-10") for p in depth]

    span_label = args.month or f"{start or '...'} to {stop or '...'}"
    window_label = "08:30-15:15 RTH" if args.rth else "full session"
    print(f"out          {out}")
    print(f"month        {span_label}")
    print(f"window       {window_label}  ({C.EXCHANGE_TZ})")
    print("one page     per 15-minute candle: 15m, its 3 fives, "
          "its 15 ones")
    if args.max_candles:
        print(f"MAX          {args.max_candles} candles -- a trial run")
    if args.limit:
        print(f"LIMIT        {args.limit:,} records per file")
    print("progress on stderr")
    print()

    fig = Figure(figsize=(11, 8.5), dpi=150)
    fig.patch.set_facecolor("white")

    text_fh = None
    if args.text:
        text_path = Path(args.text).expanduser()
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_fh = open(text_path, "w")
        text_fh.write(TEXT_HEADER + "\n")

    scratch = out.with_suffix(".pages.pdf")
    body = None if args.no_pdf else PdfPages(scratch)
    cen = Census()
    by_day: list[tuple] = []
    pages = 0
    sources: list = []
    cur_day = None
    builders = None
    stop_now = False

    def close_day(day):
        """Group the day's bars into candles and draw a page each."""
        nonlocal pages, builders
        if builders is None:
            return
        b15 = builders["15m"].finish()
        b5 = builders["5m"].finish()
        b1 = builders["1m"].finish()
        builders = None
        cands = group_candles(b15, b5, b1)
        if not cands:
            return
        by_day.append((day, len(cands), pages + 1))
        for c in cands:
            if args.max_candles and pages >= args.max_candles:
                return
            cen.add(c)
            pages += 1
            if text_fh is not None:
                text_fh.write(markdown_candle(c) + "\n")
            if body is not None:
                fig.clf()
                candle_page(fig, c, pages)
                body.savefig(fig)

    try:
        for path, schema in ordered:
            if stop_now:
                break
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
                    close_day(cur_day)
                    if args.max_candles and pages >= args.max_candles:
                        stop_now = True
                        cur_day = None
                        break
                    cur_day = day
                    builders = {"15m": C.Bars(SPAN15),
                                "5m": C.Bars(SPAN5),
                                "1m": C.Bars(SPAN1)}
                    pr.extra = f"{day}  {pages} pages"
                for b in builders.values():
                    b.feed(ts, o, h, lo, c, v, iid)
            pr.done()
            sources.append(src)
        if cur_day is not None:
            close_day(cur_day)
    finally:
        if body is not None:
            body.close()
        if text_fh is not None:
            text_fh.close()

    if not pages:
        print("No candles drawn -- nothing matched the month or the "
              "dates.", file=sys.stderr)
        scratch.unlink(missing_ok=True)
        return 1

    if args.no_pdf:
        size = text_path.stat().st_size / 1e6
        print(f"{text_path}")
        print(f"  {pages:,} candles over {len(by_day)} sessions   "
              f"{size:,.2f} MB")
        return 0

    meta = {
        "month": args.month or f"{start or '...'} to {stop or '...'}",
        "candles": pages,
        "sessions": len(by_day),
        "window": ("08:30-15:15 Chicago (RTH)" if args.rth
                   else "17:00-16:00 Chicago (the whole session)"),
        "sources": ", ".join(Path(s.path).name for s in sources),
        "built": dt.datetime.now(C.TZ).strftime("%Y-%m-%d %H:%M %Z"),
    }
    front = out.with_suffix(".front.pdf")
    with PdfPages(front) as fp:
        fig.clf()
        cover_page(fig, meta)
        fp.savefig(fig)
        fig.clf()
        summary_page(fig, cen, meta)
        fp.savefig(fig)
        contents_page(fp, fig, by_day)

    total = C.merge(front, scratch, out, {
        "first": meta["month"], "last": meta["month"],
        "sessions": pages})
    front.unlink(missing_ok=True)
    scratch.unlink(missing_ok=True)

    size = out.stat().st_size / 1e6 if out.exists() else 0.0
    if text_fh is not None:
        t_size = text_path.stat().st_size / 1e6
        print(f"  {text_path}   {t_size:,.2f} MB   "
              f"the same breakdown as markdown")
        print()
    print("SOURCES")
    for s in sources:
        print(f"  {Path(s.path).name}  kept {s.kept:,}  "
              f"skipped {s.skipped:,}")
    print()
    print(f"{out}")
    print(f"  {total} pages   {size:,.1f} MB   {pages:,} candles over "
          f"{len(by_day)} sessions")
    print()
    pct = 100.0 * cen.agreed / cen.minutes if cen.minutes else 0.0
    print(f"All three timeframes agreed on {cen.agreed:,} of "
          f"{cen.minutes:,} minutes ({pct:.1f}%).")
    a = Census.quartiles(cen.agree_travel)
    b = Census.quartiles(cen.other_travel)
    print(f"  ticks to the candle's close, agreed:     "
          f"p25 {a[0]:+.1f}  median {a[1]:+.1f}  p75 {a[2]:+.1f}")
    print(f"  ticks to the candle's close, not agreed: "
          f"p25 {b[0]:+.1f}  median {b[1]:+.1f}  p75 {b[2]:+.1f}")
    print()
    print("These are counts, not a test: `to close` runs to the end of "
          "the same candle, a deadline")
    print("the market does not observe. The controlled version is "
          "agree_travel.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
