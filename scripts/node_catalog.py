#!/usr/bin/env python3
"""Conditional behaviour at volume nodes. A catalog, not a verdict.

    python scripts/node_catalog.py data/ES_c_0_mbp-10_2026-09-21.5s.slices.jsonl
    python scripts/node_catalog.py <file> --by node_kind,approach,regime

No expectancy, no costs, no pass/fail. What it reports is: given that price
arrived at a node of a given kind from a given side, how often did each
thing happen next, how far did it go, how long did it take, and how many
observations is that based on.

WHERE THE NODES COME FROM

The PRIOR session's completed profile -- what is on the chart at the open.
Nothing is read from the session being measured, so no level is defined by
the move it is being used to explain.

READ THE n COLUMN FIRST

Splitting by node, approach, regime, session and volatility produces
hundreds of cells, and several will show a striking split from nothing at
all. Cells under 30 observations are marked `thin` and should be read as
"not measured", not as "measured and small".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liqmap import (bars as bl, escondense as ec, nodestudy as ns,  # noqa: E402
                    profile as pf)
from liqmap.levels import Phase, phase, trade_date                   # noqa: E402

THIN = 30


def load_sessions(path: Path, resample_s: float = 5.0):
    """RTH sessions from either input, keyed by trade date.

    A .dbn.zst OHLCV file is read straight through `bars`, never through
    the slices format. Two years of five-second bars as JSON Lines runs to
    well over a gigabyte and takes longer to re-read than the DBN takes to
    parse -- the intermediate file earns its place at one week and stops
    earning it at one year.

    Sessions flagged suspect by the roll detector are dropped here rather
    than downstream, because the prior session's profile is the input to
    every touch and across a contract roll it describes a different
    instrument.
    """
    if path.suffix == ".zst" or path.suffix == ".dbn":
        raw = bl.load(str(path), seconds=resample_s)
        found = bl.sessions(raw, rth_only=True)
        dropped = [s for s in found if s.suspect]
        good = {s.day: s.bars for s in found if not s.suspect and s.bars}
        return good, dropped

    slices = ec.load(path)
    slices.sort(key=lambda s: s.ts)
    out: dict = {}
    for s in slices:
        if phase(s.ts) is Phase.RTH:
            out.setdefault(trade_date(s.ts), []).append(s)
    return out, []


def parse(argv):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("slices")
    p.add_argument("--by", default="node_kind,approach",
                   help="Touch fields to condition on, comma separated. "
                        "Any of: node_kind, approach, regime, session, vol, "
                        "trend_1h, strength")
    p.add_argument("--horizon-min", type=float, default=60.0)
    p.add_argument("--hist", action="store_true",
                   help="Print distance distributions, not just medians.")
    p.add_argument("--control", action="store_true",
                   help="Also run the identical classifier at RANDOM prices "
                        "instead of nodes. If the splits match, the node "
                        "carried no information and the table is measuring "
                        "the thresholds.")
    p.add_argument("--json", default=None,
                   help="Write the catalog to a JSON file for the site to "
                        "serve. Includes every conditioning split, not just "
                        "the one printed.")
    return p.parse_args(argv)


# Splits written to JSON. Coarse first, so the site can fall back to a
# wider cell when the narrow one is thin -- which, at any useful depth of
# conditioning, it often will be.
JSON_SPLITS = (
    ("node_kind",),
    ("node_kind", "approach"),
    ("node_kind", "approach", "regime"),
    ("node_kind", "approach", "vol"),
    ("node_kind", "approach", "strength"),
    ("node_kind", "approach", "trend_1h"),
    ("node_kind", "approach", "regime", "vol"),
)


def write_json(path: Path, pairs, sessions: int, span: tuple) -> None:
    import json

    doc = {
        "generated_from": {"sessions": sessions,
                           "first": span[0].isoformat() if span[0] else None,
                           "last": span[1].isoformat() if span[1] else None,
                           "interactions": len(pairs)},
        "conventions": {
            "touch_ticks": ns.TOUCH_TICKS, "rearm_ticks": ns.REARM_TICKS,
            "horizon_s": ns.HORIZON_S, "reject_ticks": ns.REJECT_TICKS,
            "break_ticks": ns.BREAK_TICKS,
            "through_ticks": ns.THROUGH_TICKS,
            "thin_below": THIN,
            "nodes_from": "prior session's completed profile",
        },
        "splits": {},
    }
    for by in JSON_SPLITS:
        rows = [c.to_dict() for c in ns.catalog(pairs, by=by)]
        for r in rows:
            r["key"] = list(r["key"])
        doc["splits"]["/".join(by)] = rows

    # Distance distributions per outcome, so the site can show shape rather
    # than a median that hides a bimodal move.
    doc["distributions"] = {
        label: [{"lo": lo, "hi": hi, "n": n} for lo, hi, n in
                ns.histogram([o.dist_ticks for _, o in pairs
                              if o.label == label], 0.0, 60.0, 24)]
        for label in ns.LABELS}

    path.write_text(json.dumps(doc, indent=1))


def session_context(bars, prior: pf.Profile) -> dict:
    """Regime, volatility and trend -- from the PRIOR session only.

    Every field here is knowable at the open. A regime label computed from
    the session being measured would encode its own outcome, and "trend
    days trend" would come back as a discovery.
    """
    if prior.empty:
        return {}
    rng = prior.va_width_ticks
    return {
        "regime": ("balance" if rng < 40 else
                   "range" if rng < 90 else "trend"),
        "vol": "high" if rng >= 60 else "low",
        "session": "rth",
    }


def nodes_from(p: pf.Profile) -> list[tuple[float, str, str]]:
    """(price, kind, strength) for everything on the prior profile."""
    if p.empty:
        return []
    peak = float(p.volumes.max()) if p.volumes.size else 0.0

    def strength(px: float) -> str:
        row = int(px / p.tick) - p.lo_row
        if not (0 <= row < p.volumes.size) or peak <= 0:
            return "unknown"
        return ("strong" if p.volumes[row] >= ns.STRONG_FRACTION * peak
                else "weak")

    out = [(p.poc, "POC", strength(p.poc)),
           (p.vah, "VAH", strength(p.vah)),
           (p.val, "VAL", strength(p.val))]
    out += [(x, "HVN", strength(x)) for x in p.hvn]
    out += [(x, "LVN", strength(x)) for x in p.lvn]

    # One node per price. POC often coincides with an HVN, and counting it
    # twice would double every cell it lands in.
    seen: dict[float, tuple[float, str, str]] = {}
    for px, kind, s in out:
        if px > 0 and round(px, 4) not in seen:
            seen[round(px, 4)] = (px, kind, s)
    return list(seen.values())


def random_nodes(prior_bars, n: int, rng) -> list[tuple[float, str, str]]:
    """Pseudo-nodes at random prices inside the prior session's range.

    THE CONTROL THIS STUDY NEEDS

    "45% reject, 30% break" means nothing on its own. A price level chosen
    with a dart produces some split too, purely from the geometry of the
    thresholds -- move 8 ticks against without first closing 2 ticks
    through, and so on. If darts give the same answer as volume nodes, the
    catalog is a description of the classifier rather than of the market.

    Same count per session, same price range, same everything else. Only
    the reason for the level is destroyed.
    """
    if not prior_bars or n <= 0:
        return []
    lo = min(b.low for b in prior_bars)
    hi = max(b.high for b in prior_bars)
    if hi <= lo:
        return []
    out = []
    for _ in range(n):
        px = round(rng.uniform(lo, hi) * 4) / 4
        out.append((px, "RANDOM", "unknown"))
    return out


def main(argv=None) -> int:
    a = parse(argv or sys.argv[1:])
    path = Path(a.slices)
    if not path.exists():
        print(f"No such file: {path}", file=sys.stderr)
        return 2

    days, dropped = load_sessions(path)
    ordered = sorted(days)
    n_bars = sum(len(v) for v in days.values())

    print(f"\n{n_bars:,} RTH bars, {len(ordered)} sessions")
    if dropped:
        print(f"  {len(dropped)} sessions dropped at contract rolls or "
              f"gaps: "
              f"{', '.join(s.day.isoformat() for s in dropped[:6])}"
              f"{' ...' if len(dropped) > 6 else ''}")
    print(f"nodes: prior session's completed profile "
          f"(POC, VAH, VAL, HVN, LVN)")
    print(f"touch {ns.TOUCH_TICKS:g} ticks | re-arm {ns.REARM_TICKS:g} | "
          f"horizon {a.horizon_min:g} min | "
          f"reject/break {ns.REJECT_TICKS:g} ticks")

    import random
    rng = random.Random(0)

    pairs: list[tuple[ns.Touch, ns.Outcome]] = []
    ctrl: list[tuple[ns.Touch, ns.Outcome]] = []
    for i in range(1, len(ordered)):
        prior_bars = days[ordered[i - 1]]
        prior = pf.final(prior_bars)
        today = days[ordered[i]]
        nodes = nodes_from(prior)
        if not nodes:
            continue
        ctx = session_context(today, prior)
        for t in ns.find_touches(today, nodes, context=ctx):
            pairs.append((t, ns.resolve(today, t,
                                        horizon_s=a.horizon_min * 60.0)))
        if a.control:
            for t in ns.find_touches(
                    today, random_nodes(prior_bars, len(nodes), rng),
                    context=ctx):
                ctrl.append((t, ns.resolve(today, t,
                                           horizon_s=a.horizon_min * 60.0)))

    print(f"\n{len(pairs):,} node interactions from "
          f"{len(ordered) - 1} tradeable sessions")
    if not pairs:
        print("No interactions. Too few sessions, or price never reached "
              "a prior node.", file=sys.stderr)
        return 1

    by = tuple(x.strip() for x in a.by.split(",") if x.strip())
    cells = ns.catalog(pairs, by=by)

    head = " / ".join(by)
    print(f"\n=== conditional outcomes by {head} ===\n")
    # MFE and MAE are the market's numbers, measured over the whole
    # horizon. `dist` is the distance at which MY OWN rule fired, so it
    # sits at the threshold by construction and says nothing -- it is kept
    # only because its spread shows when a move blew straight past.
    print(f"  {'condition':<34}{'outcome':<12}{'freq':>7}{'n':>7}"
          f"{'MFE':>7}{'MAE':>7}{'trig':>7}{'p25-p75':>11}"
          f"{'time':>7}{'back':>7}")

    last_key = None
    for c in cells:
        key = " / ".join(str(x) for x in c.key)
        shown = key if c.key != last_key else ""
        last_key = c.key
        flag = "  THIN" if c.thin else ""
        print(f"  {shown:<34}{c.label:<12}{c.freq:>6.0%}{c.n:>7}"
              f"{c.median_mfe:>6.0f}t{c.median_mae:>6.0f}t"
              f"{c.median_dist:>6.0f}t"
              f"{c.p25_dist:>6.0f}-{c.p75_dist:<4.0f}"
              f"{c.median_time_s / 60:>6.0f}m{c.returned_share:>6.0%}{flag}")

    if a.control and ctrl:
        print(f"\n=== CONTROL: the same classifier at RANDOM prices "
              f"(n={len(ctrl):,}) ===")
        print("  Same sessions, same count of levels, same range, same")
        print("  thresholds. Only the reason for the level is destroyed.\n")
        # Frequency AND distance. Comparing only how often each outcome
        # happens leaves the question a reader actually asks -- "how far
        # does it go from a node" -- completely unchecked. If darts travel
        # the same distance, those columns describe ES, not the node.
        print(f"  {'outcome':<12}{'freq':>7}{'rand':>7}{'gap':>7}"
              f"{'MFE':>7}{'rand':>7}{'MAE':>7}{'rand':>7}")
        node_cells = {c.label: c for c in ns.catalog(pairs, by=())}
        ctrl_cells = {c.label: c for c in ns.catalog(ctrl, by=())}
        biggest = 0.0
        # NOT `a` and `b`: this function's argparse namespace is `a`, and
        # binding a Cell to it here left `a.hist` further down looking for
        # an attribute on the wrong object. The run produced every table
        # correctly and then died before writing the JSON.
        for label in ns.LABELS:
            real = node_cells.get(label)
            rand = ctrl_cells.get(label)
            if real is None or rand is None:
                continue
            biggest = max(biggest, abs(real.freq - rand.freq))
            print(f"  {label:<12}{real.freq:>6.0%}{rand.freq:>7.0%}"
                  f"{real.freq - rand.freq:>+7.1%}"
                  f"{real.median_mfe:>6.0f}t{rand.median_mfe:>6.0f}t"
                  f"{real.median_mae:>6.0f}t{rand.median_mae:>6.0f}t")
        print(f"\n  Largest difference: {biggest:.1%}.")
        if biggest < 0.03:
            print("  The darts and the nodes behave the same. On this "
                  "evidence the node\n  is not what is producing the "
                  "split -- the thresholds are.")
        else:
            print("  The nodes differ from the darts. That difference is "
                  "the part of the\n  table that is about the market "
                  "rather than about the method.")

    # What is left AFTER the outcome is knowable.
    #
    # "A reject runs 18 ticks" is not something anyone can act on: the
    # reject is only a reject once price has already moved 8 ticks against
    # the approach. The tradeable quantity is the remainder -- total
    # excursion minus the distance at which the label became true -- and it
    # is the only number in this whole catalog that a live reader could use
    # in the moment.
    print("\n=== what remains after the outcome is confirmed ===")
    print("  The label is not knowable until the move that defines it has")
    print("  happened. This is total excursion minus the trigger distance.\n")
    print(f"  {'outcome':<12}{'n':>7}{'median total':>15}"
          f"{'at confirm':>13}{'remaining':>12}")
    for label in ("reject", "break_go", "break_fail"):
        rows = [(t, o) for t, o in pairs if o.label == label]
        if len(rows) < 30:
            continue
        # reject and break_fail travel AGAINST the approach, break_go with
        # it -- so the total each one is measured against differs.
        totals = [(o.mae_ticks if label != "break_go" else o.mfe_ticks)
                  for _, o in rows]
        trigs = [o.dist_ticks for _, o in rows]
        tot = ns._median(totals)
        trg = ns._median(trigs)
        print(f"  {label:<12}{len(rows):>7,}{tot:>13.0f} t"
              f"{trg:>11.0f} t{tot - trg:>10.0f} t")
    print(f"\n  A taker round trip on one ES contract is "
          f"${17.50:,.2f} = 1.4 ticks.")
    print("  That comparison is yours to make; this script does not make it.")

    n_thin = len({c.key for c in cells if c.thin})
    n_total = len({c.key for c in cells})
    print(f"\n  {n_thin} of {n_total} conditions are under {THIN} "
          f"observations and marked THIN.")
    print("  A frequency without its n is a decoration. Read the n first.")

    if a.hist:
        print("\n=== distance distributions (ticks) ===")
        print("  A median of 8 hides whether the move is reliably 8, or is")
        print("  2 half the time and 30 the rest. Those are different")
        print("  markets and only the shape tells them apart.\n")
        for label in ns.LABELS:
            vals = [o.dist_ticks for _, o in pairs if o.label == label]
            if len(vals) < 10:
                continue
            print(f"  {label}  (n={len(vals)})")
            for lo, hi, n in ns.histogram(vals, 0.0, 60.0, buckets=12):
                bar = "#" * min(46, n * 46 // max(1, max(
                    c for _, _, c in ns.histogram(vals, 0.0, 60.0, 12))))
                print(f"    {lo:>3.0f}-{hi:<3.0f} {bar:<46} {n}")
            print()

    if a.json:
        out = Path(a.json)
        write_json(out, pairs, len(ordered),
                   (ordered[0] if ordered else None,
                    ordered[-1] if ordered else None))
        size = out.stat().st_size / 1000.0
        print(f"\n  wrote {out}  ({size:,.0f} KB, "
              f"{len(JSON_SPLITS)} conditioning splits)")

    print("\n  Descriptive only. No costs applied, no expectancy computed,")
    print("  no view offered on whether any of this is tradeable.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
