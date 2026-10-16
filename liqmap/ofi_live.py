"""The same OFI engine, driven by the Databento live feed.

    from liqmap.ofi_live import LiveRace
    race = LiveRace(symbols=["ES.c.0"], window=100, baseline=1000)
    for reading in race.run():
        ...

Or from the command line:

    python -m liqmap.ofi_live --symbols ES.c.0 --window 100 \\
        --baseline 1000 --z 0.5 --redis redis://localhost:6379

ONE ENGINE, TWO SOURCES. The records a live subscription delivers are
the same MBP-10 records a file holds, so `ofi.BookState` and
`ofi.Race` are used unchanged. Nothing here reimplements the
measurement -- if the live numbers ever disagreed with the replayed
ones, the cause would be the feed or the clock, not a second copy of
the formula that drifted.

CREDENTIALS COME FROM THE ENVIRONMENT. `DATABENTO_API_KEY`, read by
the client itself. It is never passed as an argument, never logged and
never written to Redis, because anything that appears in a process
listing or a log eventually appears somewhere else.

REDIS IS OPTIONAL. Without it the readings are yielded to the caller
and nothing is published. The publish is fire-and-forget: a Redis that
is down must not stop the engine measuring, so a failed publish is
counted and reported, never raised.

LIVE DATA IS A SEPARATE ENTITLEMENT from historical on a Databento
account. If the subscription is refused, that is what the error will
say, and it is not something this module can work around.

NOTHING HERE PLACES, SIZES OR TIMES AN ORDER. It publishes a number.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field

from . import ofi

DATASET = "GLBX.MDP3"          # CME Globex MDP 3.0
SCHEMA = "mbp-10"


@dataclass
class Publisher:
    """Writes readings to Redis, or nowhere.

    Failures are counted rather than raised: the engine's job is to
    keep measuring, and a transport that drops is not a reason to stop
    reading the book.
    """

    url: str = ""
    channel: str = "ofi"
    key: str = "ofi:last"
    _client: object = None
    sent: int = 0
    failed: int = 0
    last_error: str = ""

    def connect(self) -> bool:
        if not self.url:
            return False
        try:
            import redis
        except ImportError:
            self.last_error = "redis not installed (pip install redis)"
            return False
        try:
            self._client = redis.Redis.from_url(self.url)
            self._client.ping()
            return True
        except Exception as exc:                        # noqa: BLE001
            self._client = None
            self.last_error = f"{type(exc).__name__}: {exc}"
            return False

    def publish(self, payload: dict) -> None:
        if self._client is None:
            return
        try:
            blob = json.dumps(payload, separators=(",", ":"))
            self._client.set(self.key, blob)
            self._client.publish(self.channel, blob)
            self.sent += 1
        except Exception as exc:                        # noqa: BLE001
            self.failed += 1
            self.last_error = f"{type(exc).__name__}: {exc}"


def reading_payload(ev: ofi.Event, rd: ofi.Reading,
                    z_enter: float | None) -> dict:
    """What goes on the wire.

    `side` is present only when the caller named a line. With no line
    there is no verdict to publish -- the subscriber gets the
    measurement and draws its own.
    """
    t = ev.touch
    out = {
        "ts": ev.ts,
        "instrument": ev.instrument,
        "sequence": ev.sequence,
        "e_n": ev.e_n,
        "e_multi": round(ev.e_multi, 4),
        "window": rd.window,
        "z": rd.z,
        "warm": rd.warm,
        "bid_px": t.bid_px,
        "bid_sz": t.bid_sz,
        "ask_px": t.ask_px,
        "ask_sz": t.ask_sz,
        "spread": t.spread,
        "obi": round(ev.obi, 6),
        "bid_depth": ev.bid_depth,
        "ask_depth": ev.ask_depth,
    }
    if z_enter is not None:
        out["side"] = rd.side(z_enter)
    return out


@dataclass
class LiveRace:
    """Databento live MBP-10 -> BookState -> Race -> readings."""

    symbols: list = field(default_factory=lambda: ["ES.c.0"])
    dataset: str = DATASET
    window: int = 100
    baseline: int = 1000
    z_enter: float | None = None
    stype_in: str = "continuous"
    publisher: Publisher | None = None

    def __post_init__(self) -> None:
        self.state = ofi.BookState()
        self.race = ofi.Race(window=self.window, baseline=self.baseline)
        self.feats = ofi.Features()

    def client(self):
        """The Databento live client. Key from the environment only."""
        # The key is checked FIRST. It costs nothing, it is the same
        # answer on every machine, and "you have not set your key" is
        # the mistake people actually make -- an import error reported
        # ahead of it hides the real problem behind a second one.
        if not os.environ.get("DATABENTO_API_KEY"):
            raise RuntimeError(
                "DATABENTO_API_KEY is not set. The client reads it "
                "from the environment; do not pass it as an argument.")
        try:
            import databento as db
        except ImportError as exc:                      # noqa: BLE001
            raise RuntimeError(
                "databento is not installed (pip install databento)"
            ) from exc
        live = db.Live()
        live.subscribe(dataset=self.dataset, schema=SCHEMA,
                       stype_in=self.stype_in, symbols=self.symbols)
        return live

    def consume(self, records):
        """Drive the engine from any iterable of MBP-10 records.

        Separate from `run` so the live path and a replayed file go
        through exactly the same code -- which is what makes the two
        comparable.
        """
        for rec in records:
            ev = self.state.feed(rec)
            if ev is None:
                continue
            self.feats.observe(ev.ts, ev.action)
            rd = self.race.push(ev.e_n, ts=ev.ts)
            if self.publisher is not None:
                self.publisher.publish(
                    reading_payload(ev, rd, self.z_enter))
            yield ev, rd

    def run(self):
        yield from self.consume(self.client())


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--symbols", default="ES.c.0",
                    help="comma separated")
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--stype-in", default="continuous")
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--baseline", type=int, default=1000)
    ap.add_argument("--z", type=float, default=None,
                    help="your line; omitted means publish the "
                         "measurement with no verdict")
    ap.add_argument("--redis", default="",
                    help="redis:// url; omitted means publish nowhere")
    ap.add_argument("--channel", default="ofi")
    ap.add_argument("--key", default="ofi:last")
    ap.add_argument("--print-every", type=int, default=100)
    args = ap.parse_args(argv)

    pub = Publisher(url=args.redis, channel=args.channel, key=args.key)
    if args.redis:
        ok = pub.connect()
        print(f"redis      {args.redis}  "
              f"{'connected' if ok else 'NOT connected: ' + pub.last_error}",
              file=sys.stderr)
    else:
        print("redis      not configured -- readings print only",
              file=sys.stderr)

    engine = LiveRace(symbols=[s.strip() for s in args.symbols.split(",")
                               if s.strip()],
                      dataset=args.dataset, window=args.window,
                      baseline=args.baseline, z_enter=args.z,
                      stype_in=args.stype_in,
                      publisher=pub if args.redis else None)

    print(f"dataset    {args.dataset}  {SCHEMA}", file=sys.stderr)
    print(f"symbols    {args.symbols}", file=sys.stderr)
    print(f"window     {args.window}   baseline {args.baseline}",
          file=sys.stderr)
    line = ("none -- measurement only" if args.z is None
            else f"|z| >= {args.z}")
    print(f"your line  {line}", file=sys.stderr)
    print(file=sys.stderr)

    n = 0
    try:
        for ev, rd in engine.run():
            n += 1
            if args.print_every and n % args.print_every == 0:
                z = "   --  " if rd.z is None else f"{rd.z:+7.2f}"
                side = ("" if args.z is None
                        else f"  {rd.side(args.z)}")
                print(f"{ev.ts:.3f}  e_n {ev.e_n:+6d}  "
                      f"win {rd.window:+8d}  z {z}{side}")
    except KeyboardInterrupt:
        print("\nstopped", file=sys.stderr)
    finally:
        r = engine.state.refused
        print(f"\n{engine.state.seen:,} records, {n:,} readings, "
              f"{r.total:,} refused {r.as_dict()}", file=sys.stderr)
        if pub.url:
            print(f"redis: {pub.sent:,} sent, {pub.failed:,} failed"
                  + (f" ({pub.last_error})" if pub.failed else ""),
                  file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
