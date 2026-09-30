"""Where price is against the profile, and what it is doing there. Now.

NO PREDICTION, NO TRADE, NO STOP

This answers one question: given the volume profile and the bars so far,
what is price doing at this moment in auction terms. Is it rotating inside
value, has it been accepted somewhere new, is it sitting on a node, is it
moving through thin ground. Every word of the answer is a fact about bars
that have already closed.

It does not say where price is going and it does not suggest a trade.

WHAT "ACCEPTED" MEANS, AND WHY IT IS NOT THE SAME AS "BEYOND"

Price pokes outside the value area constantly. Acceptance is the auction
agreeing to trade there: two consecutive closes beyond the edge. One close
is an excursion, and calling it acceptance is the single most common way a
profile read goes wrong -- it turns every probe into a trend.

So LOCATION says where price is, and ACCEPTANCE says whether the market has
agreed to be there. They are reported separately because they disagree
often, and the disagreement is usually the interesting part.

THE PROFILE IS THE PRIOR SESSION'S

That is what is on the chart at the open and it needs no assumption about
today. A developing profile is available separately for the session in
progress, but it describes a shape that is still being made -- reading
today's own POC as a level price is "reacting to" is circular, because the
reaction is part of what set it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

TICK = 0.25

# ---------------------------------------------------------------------------
# FIXED CONVENTIONS. Stated, not tuned.
# ---------------------------------------------------------------------------

# Closes beyond an edge that constitute acceptance.
ACCEPT_CLOSES = 2

# How close counts as "at" a node.
AT_TICKS = 3.0

# Bars of history used to describe what price has been doing at a node.
LOOKBACK = 6

# A node is "held" if price has been near it this many of the last LOOKBACK
# bars without closing through.
HOLD_BARS = 3


@dataclass
class Read:
    """The current structural picture. Every field is about closed bars."""

    price: float = 0.0

    # WHERE
    zone: str = "unknown"          # above value | inside value | below value
    at_node: str | None = None     # "POC" / "VAH" / "HVN" / "LVN" / None
    node_price: float | None = None
    ticks_from_node: float | None = None

    # WHAT
    accepted: str = "no"           # no | above | below
    doing: str = "unclear"         # rotating | holding | through | leaving
                                   # | traversing | probing

    # CONTEXT
    nearest_above: float | None = None
    nearest_below: float | None = None
    thin_above: bool = False
    thin_below: bool = False

    bars_seen: int = 0
    note: str = ""

    def to_dict(self) -> dict:
        return {"price": round(self.price, 2), "zone": self.zone,
                "at_node": self.at_node,
                "node_price": (round(self.node_price, 2)
                               if self.node_price else None),
                "ticks_from_node": (round(self.ticks_from_node, 1)
                                    if self.ticks_from_node is not None
                                    else None),
                "accepted": self.accepted, "doing": self.doing,
                "nearest_above": (round(self.nearest_above, 2)
                                  if self.nearest_above else None),
                "nearest_below": (round(self.nearest_below, 2)
                                  if self.nearest_below else None),
                "thin_above": self.thin_above, "thin_below": self.thin_below,
                "bars_seen": self.bars_seen, "note": self.note}


def _accepted(closes: Sequence[float], edge: float, above: bool,
              n: int = ACCEPT_CLOSES) -> bool:
    tail = closes[-n:]
    if len(tail) < n:
        return False
    return all(c > edge for c in tail) if above else all(c < edge for c in tail)


def read(bars: Sequence, prof, tick: float = TICK,
         at_ticks: float = AT_TICKS, lookback: int = LOOKBACK) -> Read:
    """One structural sentence about right now.

    `prof` is a `profile.Profile` -- the prior session's, normally.
    `bars` are today's closed bars, oldest first.
    """
    out = Read()
    if not bars or prof is None or prof.empty:
        out.note = "no profile yet"
        return out

    last = bars[-1]
    closes = [b.close for b in bars]
    px = last.close
    out.price = px
    out.bars_seen = len(bars)

    # -- WHERE: zone, and the node price is nearest ---------------------
    if px > prof.vah:
        out.zone = "above value"
    elif px < prof.val:
        out.zone = "below value"
    else:
        out.zone = "inside value"

    nodes: list[tuple[float, str]] = [(prof.poc, "POC"), (prof.vah, "VAH"),
                                      (prof.val, "VAL")]
    nodes += [(x, "HVN") for x in prof.hvn]
    nodes += [(x, "LVN") for x in prof.lvn]

    near = min(nodes, key=lambda n: abs(n[0] - px))
    gap = (px - near[0]) / tick
    if abs(gap) <= at_ticks:
        out.at_node, out.node_price = near[1], near[0]
        out.ticks_from_node = gap

    above = [n for n, _ in nodes if n > px + tick]
    below = [n for n, _ in nodes if n < px - tick]
    out.nearest_above = min(above) if above else None
    out.nearest_below = max(below) if below else None

    # Thin ground is an LVN between here and the next node that way.
    out.thin_above = any(px < x <= (out.nearest_above or px)
                         for x in prof.lvn)
    out.thin_below = any((out.nearest_below or px) <= x < px
                         for x in prof.lvn)

    # -- WHAT: acceptance, then behaviour --------------------------------
    if _accepted(closes, prof.vah, above=True):
        out.accepted = "above"
    elif _accepted(closes, prof.val, above=False):
        out.accepted = "below"

    recent = bars[-lookback:]
    if out.at_node and out.node_price is not None:
        node = out.node_price
        touching = sum(1 for b in recent
                       if abs(b.close - node) / tick <= at_ticks)
        crossed = any((b.close - node) * (recent[0].close - node) < 0
                      for b in recent)
        if touching >= HOLD_BARS and not crossed:
            out.doing = "holding"
            out.note = (f"sitting on the {out.at_node} at {node:,.2f} -- "
                        f"{touching} of the last {len(recent)} bars closed "
                        f"within {at_ticks:g} ticks of it")
        elif crossed:
            out.doing = "through"
            out.note = (f"crossed the {out.at_node} at {node:,.2f} inside "
                        f"the last {len(recent)} bars")
        else:
            out.doing = "probing"
            out.note = f"just reached the {out.at_node} at {node:,.2f}"
    else:
        moved = (px - recent[0].close) / tick if recent else 0.0
        if out.accepted == "above":
            out.doing = "leaving"
            out.note = (f"accepted above the VAH ({prof.vah:,.2f}) -- "
                        f"seeking value higher")
        elif out.accepted == "below":
            out.doing = "leaving"
            out.note = (f"accepted below the VAL ({prof.val:,.2f}) -- "
                        f"seeking value lower")
        elif (out.thin_above and moved > 0) or (out.thin_below and moved < 0):
            out.doing = "traversing"
            direction = "up" if moved > 0 else "down"
            out.note = (f"moving {direction} through thin ground toward "
                        f"{(out.nearest_above if moved > 0 else out.nearest_below) or 0:,.2f}")
        elif out.zone == "inside value":
            out.doing = "rotating"
            out.note = (f"rotating inside value "
                        f"({prof.val:,.2f} - {prof.vah:,.2f})")
        else:
            out.doing = "probing"
            side = "above" if px > prof.vah else "below"
            out.note = (f"{side} value but NOT accepted -- fewer than "
                        f"{ACCEPT_CLOSES} closes beyond the edge")

    return out


def histogram_rows(prof, max_rows: int = 120) -> list[dict]:
    """The profile as rows a chart can draw, merged only for display.

    One tick per row is the right resolution to compute on and far too fine
    to draw -- a 400 tick session would be 400 bars two pixels apart. Rows
    are merged in equal groups for the picture; the POC, VAH, VAL and node
    prices are reported separately and unmerged, so nothing a reader acts
    on is rounded by the drawing.
    """
    if prof is None or prof.empty:
        return []
    v = prof.volumes
    group = max(1, (v.size + max_rows - 1) // max_rows)
    peak = 0.0
    rows: list[dict] = []
    for i in range(0, v.size, group):
        chunk = v[i:i + group]
        total = float(chunk.sum())
        peak = max(peak, total)
        rows.append({"lo": (prof.lo_row + i) * prof.tick,
                     "hi": (prof.lo_row + i + group) * prof.tick,
                     "v": total})
    for r in rows:
        r["share"] = round(r["v"] / peak, 4) if peak > 0 else 0.0
        r["v"] = round(r["v"], 1)
    return rows
