"""Command line interface."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


def cmd_check(args) -> int:
    """Confirm the API is reachable and still shaped the way we expect."""
    from .hl import InfoClient

    client = InfoClient()
    print(client.check())

    if args.wallet:
        raw = client.clearinghouse_state(args.wallet)
        from .bucket import parse_clearinghouse_state
        positions = parse_clearinghouse_state(args.wallet, raw)
        print(f"\n{args.wallet}: {len(positions)} open positions")
        for p in positions[:10]:
            liq = f"{p.liquidation_px:,.4f}" if p.liquidation_px else "none"
            side = "LONG " if p.is_long else "SHORT"
            print(f"  {side} {p.coin:<8} ${p.notional:>12,.0f}  "
                  f"{p.leverage:>3.0f}x {p.leverage_type:<8} liq {liq}")
    return 0


def cmd_harvest(args) -> int:
    """Build a wallet universe off the public trades feed."""
    from .hl import TradeHarvester

    coins = [c.strip().upper() for c in args.coins.split(",")]
    h = TradeHarvester(coins, min_notional=args.min_notional)

    print(f"listening on {coins} for {args.minutes:.0f} min, "
          f"fills over ${args.min_notional:,.0f}")
    print("this is building the wallet universe; leave it running\n")

    h.run(seconds=args.minutes * 60,
          on_progress=lambda n: print(f"  {n} wallets seen"))

    wallets = h.top_wallets(args.top)
    Path(args.out).write_text(json.dumps(wallets, indent=2))
    print(f"\nwrote {len(wallets)} wallets to {args.out}")

    if wallets:
        ranked = sorted(h.wallets.values(), key=lambda w: -w.notional)[:5]
        print("\nbusiest addresses seen:")
        for w in ranked:
            print(f"  {w.wallet}  ${w.notional:>14,.0f} across {w.fills} fills")
    return 0


def cmd_snapshot(args) -> int:
    """Sweep wallets, build the map, store it with its placebos."""
    from .bucket import build_map, render
    from .hl import InfoClient, sweep_positions
    from .store import Store

    wallets = json.loads(Path(args.wallets).read_text())
    if not wallets:
        print("wallet file is empty -- run `harvest` first")
        return 1

    client = InfoClient()
    store = Store(args.db)

    mids = client.all_mids()
    spot = mids.get(args.coin)
    if not spot:
        print(f"no mid price for {args.coin}")
        return 1
    store.log_price(args.coin, spot)

    print(f"sweeping {len(wallets)} wallets for {args.coin} at {spot:,.4f}")
    print(f"  (~{len(wallets) / 600:.1f} min at the rate limit)")

    positions, failed = sweep_positions(
        client, wallets,
        on_progress=lambda i, n: print(f"  {i}/{n}"))

    lm = build_map(positions, args.coin, spot, bucket_bps=args.bucket_bps)
    print("\n" + render(lm))
    if failed:
        print(f"\n  {failed} wallets failed and were skipped")

    if args.sigma is None:
        print("\nno --sigma given, so nothing was stored. The baseline needs "
              "the volatility\nas it was known AT SNAPSHOT TIME; computing it "
              "later from the window\nbeing tested would leak the answer into "
              "the question.")
        return 0

    snap_id = store.save_snapshot(lm, args.sigma, min_notional=args.min_notional,
                                  top=args.top)
    print(f"\nstored snapshot {snap_id}")
    print(json.dumps(store.counts(), indent=2))
    return 0


def cmd_positions(args) -> int:
    """Score open positions: survivability, carry, commitment, fragility."""
    from .bucket import build_map, render
    from .hl import InfoClient, sweep_positions
    from .strength import fragility_weighter, score_all, summarise

    wallets = json.loads(Path(args.wallets).read_text())
    client = InfoClient()

    mids = client.all_mids()
    spot = mids.get(args.coin)
    if not spot:
        print(f"no mid price for {args.coin}")
        return 1

    positions, failed = sweep_positions(client, wallets[:args.limit])
    on_coin = [p for p in positions if p.coin == args.coin]
    if not on_coin:
        print(f"no open {args.coin} positions across {len(wallets[:args.limit])} wallets")
        return 0

    strengths = sorted(score_all(on_coin, spot, args.sigma, args.horizon),
                       key=lambda s: -s.position.notional)

    print(f"\n{args.coin} positions, largest first "
          f"({len(on_coin)} of {len(positions)} total, {failed} wallets failed)\n")
    for s in strengths[:args.top]:
        print("  " + s.render())

    print("\n" + summarise(strengths, args.coin, spot).render())

    print("\n\nRAW MAP  (where leverage sits)")
    print(render(build_map(on_coin, args.coin, spot, bucket_bps=args.bucket_bps)))

    print("\n\nFRAGILITY-WEIGHTED MAP  (where pressure is)")
    print(render(build_map(on_coin, args.coin, spot, bucket_bps=args.bucket_bps,
                           weight_fn=fragility_weighter(spot, args.sigma, args.horizon))))

    print("\n  Compare the two. Levels that shrink between them are held by "
          "traders\n  who can sit through the move; they are not fuel.")
    return 0


def cmd_track(args) -> int:
    """Poll and store the mid price. Needed to resolve touches later."""
    from .hl import InfoClient
    from .store import Store

    client = InfoClient()
    store = Store(args.db)
    coins = [c.strip().upper() for c in args.coins.split(",")]
    deadline = time.time() + args.minutes * 60
    n = 0

    print(f"tracking {coins} every {args.interval:.0f}s for {args.minutes:.0f} min")
    try:
        while time.time() < deadline:
            started = time.monotonic()
            try:
                mids = client.all_mids()
                for coin in coins:
                    if coin in mids:
                        store.log_price(coin, mids[coin])
                        n += 1
            except Exception as exc:
                print(f"  poll failed: {exc}")
            time.sleep(max(0.0, args.interval - (time.monotonic() - started)))
    except KeyboardInterrupt:
        print("\nstopped.")

    print(f"stored {n} prices")
    return 0


def cmd_resolve(args) -> int:
    """Score every cluster whose horizon has elapsed."""
    from .store import Store, resolve_touch

    store = Store(args.db)
    pending = store.unresolved_clusters(args.horizon)
    if not pending:
        print("nothing ready to resolve yet")
        return 0

    resolved = skipped = 0
    for row in pending:
        start = datetime.fromisoformat(row["ts"])
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)

        prices = store.prices_between(
            row["coin"], start,
            start + timedelta(minutes=args.horizon + args.follow))
        if len(prices) < 5:
            skipped += 1
            continue

        touched, when, move = resolve_touch(
            prices, start, float(row["spot"]), float(row["price"]),
            float(row["sigma_per_min"]), args.horizon, args.follow)
        store.record_outcome(row["cluster_id"], args.horizon, touched, when, move)
        resolved += 1

    print(f"resolved {resolved}, skipped {skipped} for want of price history")
    print(json.dumps(store.counts(), indent=2))
    return 0


def cmd_report(args) -> int:
    from .store import Store
    from .validate import full_report

    store = Store(args.db)
    obs = store.observations(horizon_minutes=args.horizon)
    if not obs:
        print("no resolved observations yet -- run `resolve` first")
        return 1
    print(full_report(obs))
    return 0


def cmd_demo(args) -> int:
    """Run the whole validation on synthetic data. No API access needed.

    Two worlds are generated: one where clusters are inert, and one where they
    genuinely pull price. The point is to show what the report looks like when
    there is nothing there, so a real null result is recognisable.
    """
    import numpy as np

    from .validate import ClusterObservation, full_report

    SPOT, SIGMA, HORIZON = 100_000.0, 0.0006, 240.0
    sd = SIGMA * math.sqrt(HORIZON)

    def build(magnetism: float, seed: int):
        rng = np.random.default_rng(seed)
        obs = []
        for i in range(220):
            ts = datetime(2026, 9, 1, tzinfo=timezone.utc) + timedelta(hours=i)
            k = 5
            dists = rng.uniform(0.3, 2.5, k) * rng.choice([-1, 1], k)
            barriers = SPOT * np.exp(dists * sd)

            # one shared path per snapshot, as in reality
            steps = 480
            dt = HORIZON / steps
            step_sd = SIGMA * math.sqrt(dt)
            logp = math.log(SPOT)
            up = np.log(barriers) > logp
            touched = np.zeros(k, dtype=bool)
            for _ in range(steps):
                logp += -0.5 * SIGMA ** 2 * dt + step_sd * rng.standard_normal()
                touched |= np.where(up, logp >= np.log(barriers),
                                    logp <= np.log(barriers))

            notionals = rng.lognormal(13, 1.2, k)
            big = notionals > np.exp(13.5)
            touched = touched | ((~touched) & big & (rng.random(k) < magnetism))

            for j in range(k):
                obs.append(ClusterObservation(
                    snapshot_id=f"s{i}", ts=ts, coin="BTC", spot=SPOT,
                    sigma_per_min=SIGMA, cluster_price=float(barriers[j]),
                    cluster_notional=float(notionals[j]),
                    side="above" if dists[j] > 0 else "below",
                    horizon_minutes=HORIZON, touched=bool(touched[j]),
                    move_after_touch_sigmas=(
                        float(rng.standard_normal() * 0.5
                              + (0.35 if magnetism > 0 else 0.0))
                        if touched[j] else None),
                ))
        return obs

    print("\n" + "#" * 64)
    print("#  WORLD A: clusters are inert. This is what NOTHING looks like.")
    print("#" * 64)
    print(full_report(build(0.0, seed=1)))

    print("\n\n" + "#" * 64)
    print("#  WORLD B: clusters genuinely pull price.")
    print("#" * 64)
    print(full_report(build(0.45, seed=2)))

    print("\n\nIf your real data reads like World A, the map is decoration.")
    print("Most of the time, on most assets, it will.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="liqmap",
        description="Liquidation cluster mapping for Hyperliquid, with the "
                    "validation that decides whether the map means anything.")
    p.add_argument("--db", default="liqmap.db")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("check", help="verify API reachability and shape")
    c.add_argument("--wallet", help="optionally dump one wallet's positions")
    c.set_defaults(func=cmd_check)

    c = sub.add_parser("harvest", help="build a wallet universe from the trade feed")
    c.add_argument("--coins", default="BTC,ETH")
    c.add_argument("--minutes", type=float, default=60.0)
    c.add_argument("--min-notional", type=float, default=25_000.0)
    c.add_argument("--top", type=int, default=2000)
    c.add_argument("--out", default="wallets.json")
    c.set_defaults(func=cmd_harvest)

    c = sub.add_parser("positions", help="score open positions and compare maps")
    c.add_argument("--coin", default="BTC")
    c.add_argument("--wallets", default="wallets.json")
    c.add_argument("--sigma", type=float, required=True,
                   help="per-minute volatility, e.g. 0.0006")
    c.add_argument("--horizon", type=float, default=240.0)
    c.add_argument("--limit", type=int, default=500)
    c.add_argument("--top", type=int, default=25)
    c.add_argument("--bucket-bps", type=float, default=25.0)
    c.set_defaults(func=cmd_positions)

    c = sub.add_parser("track", help="poll and store mid prices")
    c.add_argument("--coins", default="BTC")
    c.add_argument("--minutes", type=float, default=600.0)
    c.add_argument("--interval", type=float, default=15.0)
    c.set_defaults(func=cmd_track)

    c = sub.add_parser("snapshot", help="sweep wallets and store a map")
    c.add_argument("--coin", default="BTC")
    c.add_argument("--wallets", default="wallets.json")
    c.add_argument("--bucket-bps", type=float, default=25.0)
    c.add_argument("--min-notional", type=float, default=250_000.0)
    c.add_argument("--top", type=int, default=12)
    c.add_argument("--sigma", type=float, default=None,
                   help="per-minute volatility as known NOW; required to store")
    c.set_defaults(func=cmd_snapshot)

    c = sub.add_parser("resolve", help="score clusters whose horizon has elapsed")
    c.add_argument("--horizon", type=float, default=240.0)
    c.add_argument("--follow", type=float, default=60.0)
    c.set_defaults(func=cmd_resolve)

    c = sub.add_parser("report", help="the validation report")
    c.add_argument("--horizon", type=float, default=240.0)
    c.set_defaults(func=cmd_report)

    c = sub.add_parser("demo", help="run the validation on synthetic data")
    c.set_defaults(func=cmd_demo)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
