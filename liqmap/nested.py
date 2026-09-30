"""Two volume profiles, one inside the other.

NESTED BALANCE AREAS

A structural profile (SVP) over a long trailing window, and a micro profile
(MVP) over a short one. Each is a complete, independent auction with its own
POC, VAH, VAL, HVNs and LVNs. The SVP says where value has been accepted over
the last hour or so; the MVP says where it is being accepted right now.

The interesting object is not either profile. It is WHERE THE SMALL ONE SITS
INSIDE THE BIG ONE, which is what this module reports.

WHY BOTH ARE BUILT FROM THE SAME BARS ON THE SAME GRID

The two profiles differ by WINDOW, not by bar granularity. Volume-by-price
barely changes when you rebuild the same window out of coarser bars -- a
60-minute profile from 1m bars and one from 5m bars come out nearly the same
shape, because the volume went to the same prices either way. Finer bars only
sharpen the intra-bar distribution estimate. So "1 hour vs 1 minute" is a
statement about how far back each profile looks, and the MVP is literally the
tail of the SVP's own bars.

They also share one row size, computed once over the union. Two profiles
quantised differently cannot be compared: "MVP VAH at 7780.50 vs SVP VAH at
7780.50" is only a fact if both were rounded onto the same grid. A per-profile
row size would make every comparison in this module an artefact of binning.

NO PREDICTION

The state is a classification of two shapes that already exist, made from
bars that have already closed. The `playbook` line attached to each state is
the user's own stated rule for that state, carried through so the panel can
show it; it is a suggestion for a human to accept or ignore, and nothing here
places, sizes or times an order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from . import profile as pf

# ---------------------------------------------------------------------------
# FIXED CONVENTIONS. Stated, not tuned.
# ---------------------------------------------------------------------------

# POC displacement between one MVP window and the next that counts as the
# micro auction MIGRATING rather than sitting still. Four ticks, which on ES
# is one point -- below that the POC moves by binning noise alone.
MIGRATE_TICKS = 4.0

# How far past the middle of the SVP value area the MVP POC must sit before
# it is described as pulled toward an edge. Expressed as a share of the SVP
# value area half-width, so it means the same thing on any instrument.
EDGE_SIDE = 0.40

# MVP window volume, as a share of the previous equal window, below which the
# micro auction is described as FADING. A rejection forming on falling volume
# is a different fact from one forming on rising volume, and the distinction
# is the whole content of the user's third rule.
FADING_RATIO = 0.70
BUILDING_RATIO = 1.30

# Default windows, in minutes.
SVP_MINUTES = 60.0
MVP_MINUTES = 10.0


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------

NESTED = "NESTED"
MIGRATING_UP = "MIGRATING_UP"
MIGRATING_DOWN = "MIGRATING_DOWN"
TESTING_SVP_VAH = "TESTING_SVP_VAH"
TESTING_SVP_VAL = "TESTING_SVP_VAL"
BREAKOUT_UP = "BREAKOUT_UP"
BREAKOUT_DOWN = "BREAKOUT_DOWN"
ENGULFING = "ENGULFING"
UNKNOWN = "UNKNOWN"

# The user's stated rule for each state. These are suggestions shown to a
# human, not instructions to a machine -- nothing in this package places an
# order, and the test suite enforces that.
PLAYBOOK = {
    NESTED: ("Consolidating. Both auctions agree on value. Fade the MVP "
             "edges back toward the MVP POC; small only. Do not expect "
             "expansion."),
    MIGRATING_UP: ("Value is being pulled toward the SVP VAH. This is the "
                   "setup, not the trade -- wait for the reaction at the "
                   "edge."),
    MIGRATING_DOWN: ("Value is being pulled toward the SVP VAL. This is the "
                     "setup, not the trade -- wait for the reaction at the "
                     "edge."),
    TESTING_SVP_VAH: ("The micro auction is forming ON the structural high. "
                      "If the MVP POC sets at the SVP VAH and MVP volume is "
                      "fading, that is the rejection back into value. "
                      "Reference target: the SVP POC."),
    TESTING_SVP_VAL: ("The micro auction is forming ON the structural low. "
                      "If the MVP POC sets at the SVP VAL and MVP volume is "
                      "fading, that is the rejection back into value. "
                      "Reference target: the SVP POC."),
    BREAKOUT_UP: ("The micro auction has left the structural value area "
                  "upward. Not a fade. Stand aside, or follow with the MVP "
                  "VAL as trailing reference."),
    BREAKOUT_DOWN: ("The micro auction has left the structural value area "
                    "downward. Not a fade. Stand aside, or follow with the "
                    "MVP VAH as trailing reference."),
    ENGULFING: ("The MVP value area is wider than the SVP's -- the micro "
                "window is carrying as much range as the structural one. "
                "The hierarchy has collapsed; lengthen the SVP window or "
                "shorten the MVP window before reading anything from it."),
    UNKNOWN: "Not enough bars in one of the two windows to build a profile.",
}


@dataclass
class Nested:
    """Where the micro auction sits inside the structural one."""

    state: str = UNKNOWN

    # Geometry
    overlap: float = 0.0        # share of the MVP value area inside the SVP's
    poc_position: float = 0.0   # MVP POC in SVP VA units: 0 = VAL, 1 = VAH
    gap_ticks: float = 0.0      # MVP POC to the nearer SVP edge, signed:
                                # negative below the VAL, positive above VAH,
                                # zero while inside

    # Movement
    drift_ticks: float = 0.0    # MVP POC now vs one MVP window ago
    volume_ratio: float = 1.0   # MVP window volume vs the previous equal one
    volume_trend: str = "flat"  # fading | flat | building
    has_history: bool = False   # was there a previous window to compare to

    note: str = ""
    playbook: str = ""

    def to_dict(self) -> dict:
        return {"state": self.state,
                "overlap": round(self.overlap, 3),
                "poc_position": round(self.poc_position, 3),
                "gap_ticks": round(self.gap_ticks, 1),
                "drift_ticks": round(self.drift_ticks, 1),
                "volume_ratio": round(self.volume_ratio, 2),
                "volume_trend": self.volume_trend,
                "has_history": self.has_history,
                "note": self.note, "playbook": self.playbook}


def window(bars: Sequence, minutes: float, end: float | None = None
           ) -> list:
    """The bars in a trailing window of `minutes` ending at `end`.

    Trailing, never centred -- a centred window would be half made of bars
    the read has not seen. `end` defaults to the last bar's timestamp, and
    an explicit earlier value gives the same window shifted back, which is
    how the previous-window comparison is taken.
    """
    if not bars:
        return []
    stop = float(end) if end is not None else float(bars[-1].ts)
    start = stop - minutes * 60.0
    return [b for b in bars if start <= float(b.ts) <= stop]


def _overlap(lo_a: float, hi_a: float, lo_b: float, hi_b: float) -> float:
    """Share of span A that lies inside span B."""
    width = hi_a - lo_a
    if width <= 0:
        return 1.0 if lo_b <= lo_a <= hi_b else 0.0
    cut = min(hi_a, hi_b) - max(lo_a, lo_b)
    return max(0.0, min(1.0, cut / width))


def classify(svp: pf.Profile | None, mvp: pf.Profile | None,
             drift_ticks: float = 0.0, volume_ratio: float = 1.0,
             has_history: bool = False) -> Nested:
    """Which of the states the two profiles are in, right now.

    Order matters. "Entirely outside" is checked before "straddling the
    edge", and both before "inside", because a value area that has cleared
    the SVP VAH also technically sits above the SVP midpoint -- resolving
    that the other way round would report every breakout as a migration.
    """
    out = Nested()
    if svp is None or mvp is None or svp.empty or mvp.empty:
        out.playbook = PLAYBOOK[UNKNOWN]
        out.note = "no profile in one of the two windows"
        return out

    tick = svp.tick if svp.tick > 0 else pf.TICK
    va = svp.vah - svp.val
    mid = (svp.vah + svp.val) / 2.0

    out.drift_ticks = drift_ticks
    out.volume_ratio = volume_ratio
    out.has_history = has_history
    if has_history:
        if volume_ratio <= FADING_RATIO:
            out.volume_trend = "fading"
        elif volume_ratio >= BUILDING_RATIO:
            out.volume_trend = "building"

    out.overlap = _overlap(mvp.val, mvp.vah, svp.val, svp.vah)
    out.poc_position = ((mvp.poc - svp.val) / va) if va > 0 else 0.5
    if mvp.poc > svp.vah:
        out.gap_ticks = (mvp.poc - svp.vah) / tick
    elif mvp.poc < svp.val:
        out.gap_ticks = (mvp.poc - svp.val) / tick

    swings = f"MVP POC {mvp.poc:,.2f}, SVP value {svp.val:,.2f}-{svp.vah:,.2f}"

    # 1. Entirely outside: the micro auction has rejected structural value.
    if mvp.val > svp.vah:
        out.state = BREAKOUT_UP
        out.note = (f"the whole MVP value area is above the SVP VAH "
                    f"({svp.vah:,.2f}) -- {swings}")
    elif mvp.vah < svp.val:
        out.state = BREAKOUT_DOWN
        out.note = (f"the whole MVP value area is below the SVP VAL "
                    f"({svp.val:,.2f}) -- {swings}")

    # 2. Wider than the structure it is supposed to sit inside. Not one of
    #    the four tradeable states -- it means the windows are wrong.
    elif mvp.val <= svp.val and mvp.vah >= svp.vah and va > 0:
        out.state = ENGULFING
        out.note = (f"the MVP value area ({mvp.val:,.2f}-{mvp.vah:,.2f}) "
                    f"contains the SVP's ({svp.val:,.2f}-{svp.vah:,.2f})")

    # 3. Straddling a structural edge: the golden state.
    elif mvp.val <= svp.vah <= mvp.vah:
        out.state = TESTING_SVP_VAH
        out.note = (f"the MVP value area straddles the SVP VAH "
                    f"({svp.vah:,.2f}); MVP volume {out.volume_trend} -- "
                    f"{swings}")
    elif mvp.val <= svp.val <= mvp.vah:
        out.state = TESTING_SVP_VAL
        out.note = (f"the MVP value area straddles the SVP VAL "
                    f"({svp.val:,.2f}); MVP volume {out.volume_trend} -- "
                    f"{swings}")

    # 4. Inside. Either drifting toward an edge, or genuinely balanced.
    else:
        toward_high = (mvp.poc - mid) >= EDGE_SIDE * (va / 2.0)
        toward_low = (mid - mvp.poc) >= EDGE_SIDE * (va / 2.0)
        if toward_high and drift_ticks >= MIGRATE_TICKS:
            out.state = MIGRATING_UP
            out.note = (f"MVP POC has moved up {drift_ticks:,.0f} ticks in "
                        f"one window and sits in the upper part of SVP "
                        f"value -- {swings}")
        elif toward_low and drift_ticks <= -MIGRATE_TICKS:
            out.state = MIGRATING_DOWN
            out.note = (f"MVP POC has moved down {abs(drift_ticks):,.0f} "
                        f"ticks in one window and sits in the lower part of "
                        f"SVP value -- {swings}")
        else:
            out.state = NESTED
            out.note = (f"the MVP value area ({mvp.val:,.2f}-{mvp.vah:,.2f}) "
                        f"is inside SVP value ({svp.val:,.2f}-"
                        f"{svp.vah:,.2f}) -- both auctions agree")

    out.playbook = PLAYBOOK[out.state]
    return out


def study(bars: Sequence, svp_minutes: float = SVP_MINUTES,
          mvp_minutes: float = MVP_MINUTES, tick: float | None = None,
          share: float = pf.VALUE_AREA
          ) -> tuple[pf.Profile, pf.Profile, Nested]:
    """Build both profiles off one bar array and classify the relation.

    `tick` None means "work it out once, from the SVP window, and use the
    SAME value for both". That is the whole point: comparing levels that
    were binned differently compares the binning.
    """
    svp_bars = window(bars, svp_minutes)
    mvp_bars = window(bars, mvp_minutes)
    if not svp_bars or not mvp_bars:
        return pf.final([]), pf.final([]), classify(None, None)

    if tick is None:
        tick = pf.row_size(svp_bars)

    svp = pf.final(svp_bars, tick=tick, share=share)
    mvp = pf.final(mvp_bars, tick=tick, share=share)

    # The previous equal MVP window, for drift and for the volume trend.
    # Both are facts about closed bars; neither looks forward.
    prev_end = float(mvp_bars[0].ts) - 1e-9
    prev_bars = window(bars, mvp_minutes, end=prev_end)
    drift = 0.0
    ratio = 1.0
    has_history = False
    if len(prev_bars) >= 3:
        prev = pf.final(prev_bars, tick=tick, share=share)
        if not prev.empty and tick > 0:
            drift = (mvp.poc - prev.poc) / tick
            if prev.total > 0:
                ratio = mvp.total / prev.total
            has_history = True

    return svp, mvp, classify(svp, mvp, drift, ratio, has_history)
