#!/usr/bin/env python3
"""Watch the live footprint bar change, tick by tick.

    python scripts/footprint_live_demo.py
    python scripts/footprint_live_demo.py --rule volume --volume 800 \\
        --min-volume 15 --fps 8

Redraws the live bar in place while a feed thread pushes ticks, so
you can see what the spec asks for: imbalances appearing and shifting
as the bar fills, then transitioning to confirmed the moment it
closes.

THE FEED AND THE DISPLAY ARE SEPARATE THREADS ON PURPOSE. That is the
arrangement the engine has to be safe under, and a demo that polled
from the feed thread would prove nothing about it. The display takes
no lock: it reads a published pulse for the scalars and asks for a
view only at the frame rate.

MOCK DATA. A random walk with planted one-sided bursts. It exercises
the engine; it says nothing about any market.
"""

from __future__ import annotations

import argparse
import random
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liqmap.footprint import (                             # noqa: E402
    AnyRule, DeltaRule, ImbalanceConfig, Tick, TimeRule, VolumeRule)
from liqmap.footprint_live import FootprintStream          # noqa: E402

BAR = "█"


def feed(stream, args, stop):
    """A tick source on its own thread, as a real one would be."""
    rng = random.Random(args.seed)
    ts = time.time()
    px = 5800.0
    burst, side = 0, None
    i = 0
    while not stop.is_set():
        i += 1
        ts += args.tick_gap
        if burst == 0 and i % args.burst_every == 0:
            burst, side = args.burst_len, rng.choice(("buy", "sell"))
        if burst > 0:
            burst -= 1
            aggressor = side
            size = rng.randint(4, 30)
            if rng.random() < 0.6:
                px += 0.25 if side == "buy" else -0.25
        else:
            aggressor = rng.choice(("buy", "sell"))
            size = rng.randint(1, 6)
            px += rng.choice((-0.25, 0.0, 0.0, 0.25))
        try:
            stream.on_tick(Tick(ts=ts, price=round(px, 2), size=size,
                                aggressor=aggressor))
        except Exception as exc:                        # noqa: BLE001
            print(f"feed stopped: {exc}", file=sys.stderr)
            return
        time.sleep(args.sleep)


def meter(value: float, width: int = 21) -> str:
    """A -1..+1 value as a centred bar. Reads faster than a number
    when it is changing several times a second."""
    half = width // 2
    n = int(round(max(-1.0, min(1.0, value)) * half))
    cells = [" "] * width
    cells[half] = "|"
    if n > 0:
        for k in range(half + 1, half + 1 + n):
            cells[k] = BAR
    elif n < 0:
        for k in range(half + n, half):
            cells[k] = BAR
    return "".join(cells)


def draw(stream, args, closed_note) -> str:
    v = stream.view()
    p = v.pulse
    out = []
    out.append(f"  live bar   seq {p.seq:>7,}   age {p.bar_age_s:>6.1f}s"
               f"   bars closed {p.bars_closed:,}")
    out.append(f"  O {p.open:>9.2f}  H {p.high:>9.2f}  "
               f"L {p.low:>9.2f}  last {p.last:>9.2f}")
    r = "  --  " if p.ratio is None else f"{p.ratio:>6.2f}"
    out.append(f"  volume {p.total_volume:>8,}   "
               f"buyers {p.ask_volume:>7,}  sellers {p.bid_volume:>7,}"
               f"   ratio {r}")
    out.append(f"  delta  {p.delta:>+8,}  [{meter(p.pressure)}] "
               f"{p.pressure:>+6.2f}   bar")
    out.append(f"  window {p.window_delta:>+8,}  "
               f"[{meter(p.window_pressure)}] {p.window_pressure:>+6.2f}"
               f"   last {args.window:.0f}s")
    out.append(f"  unconfirmed {p.unconfirmed:>3}  "
               f"({p.buy_flags} buy / {p.sell_flags} sell)"
               f"   faded {len(v.faded):>3}")
    out.append("")
    out.append(f"  {'price':>9} {'bid':>7} {'ask':>7} {'delta':>7}  state")

    rows = v.rows[:args.rows]
    for r_ in rows:
        tag = ""
        if r_["imbalance"]:
            tag = "  " + "/".join(x.upper() for x in r_["imbalance"]) \
                  + "  unconfirmed"
        out.append(f"  {r_['price']:>9.2f} {r_['bid_volume']:>7,}"
                   f" {r_['ask_volume']:>7,} {r_['delta']:>+7,}{tag}")
    if len(v.rows) > args.rows:
        out.append(f"  ... {len(v.rows) - args.rows} more levels")

    if closed_note:
        out.append("")
        out.append(closed_note)
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rule", default="volume",
                    choices=("time", "volume", "delta", "any"))
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--volume", type=int, default=800)
    ap.add_argument("--delta", type=int, default=300)
    ap.add_argument("--mult", type=float, default=4.0)
    ap.add_argument("--min-volume", type=int, default=15)
    ap.add_argument("--window", type=float, default=10.0)
    ap.add_argument("--fps", type=float, default=6.0)
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--run", type=float, default=20.0,
                    help="seconds to run for; 0 means until Ctrl-C")
    ap.add_argument("--sleep", type=float, default=0.004,
                    help="pause between generated ticks")
    ap.add_argument("--tick-gap", type=float, default=0.05,
                    help="simulated seconds between ticks")
    ap.add_argument("--burst-every", type=int, default=150)
    ap.add_argument("--burst-len", type=int, default=25)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    rule = {"time": lambda: TimeRule(args.seconds),
            "volume": lambda: VolumeRule(args.volume),
            "delta": lambda: DeltaRule(args.delta),
            "any": lambda: AnyRule(TimeRule(args.seconds),
                                   VolumeRule(args.volume))}[args.rule]()

    stream = FootprintStream(
        tick_size=0.25, rule=rule, window_s=args.window,
        imbalance=ImbalanceConfig(multiplier=args.mult,
                                  min_volume=args.min_volume))

    stop = threading.Event()
    t = threading.Thread(target=feed, args=(stream, args, stop),
                         daemon=True)
    t.start()

    print(f"rule {args.rule}   multiplier {args.mult:g}x   "
          f"min volume {args.min_volume}   window {args.window:.0f}s")
    print("MOCK DATA -- a random walk with planted bursts.\n")

    seen_bars = 0
    note = ""
    t0 = time.time()
    lines_drawn = 0
    try:
        while args.run <= 0 or time.time() - t0 < args.run:
            p = stream.pulse()
            if p.bars_closed > seen_bars:
                seen_bars = p.bars_closed
                bar = stream.last_closed()
                conf = stream.confirmed_flags(-1)
                buys = sum(1 for f in conf if f.side == "buy")
                note = (f"  CLOSED bar {seen_bars} by "
                        f"{bar.close_reason}: {len(conf)} imbalances "
                        f"CONFIRMED ({buys} buy / {len(conf) - buys} "
                        f"sell), delta {bar.delta:+,}")
            frame = draw(stream, args, note)
            if lines_drawn:
                sys.stdout.write(f"\033[{lines_drawn}A")
            body = frame.split("\n")
            for line in body:
                sys.stdout.write("\033[2K" + line + "\n")
            lines_drawn = len(body)
            sys.stdout.flush()
            time.sleep(1.0 / max(1.0, args.fps))
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        t.join(timeout=1)

    p = stream.pulse()
    print(f"\n{'=' * 62}")
    print(f"{p.seq:,} ticks, {p.bars_closed:,} bars closed")
    total = sum(len(stream.confirmed_flags(i))
                for i in range(len(stream.history)))
    print(f"{total:,} confirmed imbalances across them")
    if stream.history:
        print(f"{total / len(stream.history):.1f} a bar")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
