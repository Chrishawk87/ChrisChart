#!/usr/bin/env python3
"""Step 1 sanity check: five sessions of profile, plus an event census.

    python scripts/profile_check.py data/ES_c_0_mbp-10_2026-09-21.5s.slices.jsonl

Prints, per RTH session: POC, VAH, VAL, VA width, node counts, and an ASCII
histogram so the shape can be eyeballed against a chart. Then a census of
how many auction events a week can physically contain, which is the number
that decides whether Steps 2-5 can conclude anything at all.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liqmap import escondense as ec, profile as pf           # noqa: E402
from liqmap.levels import Phase, phase, trade_date           # noqa: E402


def histogram(p: pf.Profile, width: int = 46, rows: int = 22) -> list[str]:
    """Coarse ASCII profile. Rows are merged only for DISPLAY."""
    v = p.volumes
    if v.size == 0 or v.sum() <= 0:
        return ["  (empty)"]
    group = max(1, v.size // rows)
    out = []
    peak = max(v[i:i + group].sum() for i in range(0, v.size, group))
    # Mark the group that CONTAINS the POC row, not the one whose centre
    # price is nearest it. Grouping boundaries do not align with the POC, so
    # the price test put the marker on a neighbouring row and made the
    # longest bar look like it was not the POC.
    poc_row = int(p.poc / p.tick) - p.lo_row
    for i in range(v.size - group, -1, -group):
        chunk = v[i:i + group]
        px = (p.lo_row + i + group / 2.0) * p.tick
        bar = "#" * int(round(width * chunk.sum() / peak)) if peak else ""
        mark = " "
        if i <= poc_row < i + group:
            mark = "<- POC"
        elif p.val <= px <= p.vah:
            mark = "|"
        out.append(f"   {px:>9.2f} {bar:<{width}} {mark}")
    return out


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    path = Path(argv[1])
    if not path.exists():
        print(f"No such file: {path}", file=sys.stderr)
        return 2

    slices = ec.load(path)
    slices.sort(key=lambda s: s.ts)
    print(f"\n{len(slices):,} bars from {path.name}")

    # Bars are grouped by CME trade date and split RTH / overnight. The
    # session boundary is exchange-local, so it survives the DST change.
    sessions: dict = defaultdict(list)
    for s in slices:
        if phase(s.ts) is Phase.RTH:
            sessions[trade_date(s.ts)].append(s)

    days = sorted(sessions)
    print(f"{len(days)} RTH sessions: "
          f"{', '.join(d.isoformat() for d in days)}")

    print("\n=== CONVENTIONS (fixed, not tuned) ===")
    print(f"  row height          {pf.ROW_TICKS} tick = {pf.TICK}")
    print(f"  value area          {pf.VALUE_AREA:.0%} (Steidlmayer)")
    print(f"  HVN / LVN           >= {pf.HVN_FRACTION:.0%} / "
          f"<= {pf.LVN_FRACTION:.0%} of the POC row")
    print(f"  node window         +/- {pf.NODE_WINDOW} rows "
          f"({pf.NODE_WINDOW * pf.TICK:g} points)")
    print(f"  rolling window      {pf.ROLLING_SECONDS:g}s")
    print("  volume split        proportional to row overlap of [low, high]")

    print("\n=== FIVE SESSIONS (final profiles -- display only) ===")
    for d in days[:5]:
        bars = sessions[d]
        p = pf.final(bars)
        rng = (max(b.high for b in bars) - min(b.low for b in bars)) / pf.TICK
        print(f"\n  {d}   {len(bars):,} bars   "
              f"session range {rng:.0f} ticks")
        print(f"    POC {p.poc:,.2f}   VAH {p.vah:,.2f}   VAL {p.val:,.2f}"
              f"   VA width {p.va_width_ticks:.0f} ticks")
        print(f"    HVN {len(p.hvn)}  {[round(x, 2) for x in p.hvn[:6]]}")
        print(f"    LVN {len(p.lvn)}  {[round(x, 2) for x in p.lvn[:6]]}")
        for line in histogram(p):
            print(line)

    # --- the number that decides whether Steps 2-5 can conclude anything
    print("\n=== EVENT CENSUS ===")
    print("  How many times could an auction event physically fire in this")
    print("  file? This is an UPPER BOUND -- Step 3's rules will cut it.\n")

    rth_hours = sum(len(v) for v in sessions.values()) * 5 / 3600.0
    print(f"  RTH hours                {rth_hours:,.1f}")
    print(f"  1h bars in RTH           {int(rth_hours)}")

    # VA crossings on the PRIOR session's value area, which is the level
    # genuinely known at today's open.
    crossings = 0
    for i, d in enumerate(days[1:], start=1):
        prior = pf.final(sessions[days[i - 1]])
        was_in = None
        for s in sessions[d]:
            now_in = prior.inside(s.close)
            if was_in is not None and now_in != was_in:
                crossings += 1
            was_in = now_in
    print(f"  prior-VA crossings       {crossings}  "
          f"(raw, before any acceptance rule)")

    lvn_total = sum(len(pf.final(v).lvn) for v in sessions.values())
    print(f"  LVN rows, all sessions   {lvn_total}")

    print("\n  At n = 20, a hit-rate interval spans roughly +/- 22 points.")
    print("  At n = 100 it is +/- 10. At n = 400 it is +/- 5.")
    print("  Nothing here can pass or fail the pre-registered test. One")
    print("  week sizes the machinery; months decide the hypothesis.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
