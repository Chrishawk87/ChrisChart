#!/usr/bin/env python3
"""Find out what Ironbeam's market data actually contains. Read only.

    export IRONBEAM_USERNAME='...'
    export IRONBEAM_PASSWORD='...'
    export IRONBEAM_API_KEY='...'
    python scripts/ironbeam_probe.py --symbol MES

    python scripts/ironbeam_probe.py --symbol MES --seconds 180 --raw 8

WHAT IT IS FOR

One question decides whether this feed can drive the engine at all:

    does a trade message say WHICH SIDE CROSSED THE SPREAD?

Delta, CVD, intent, absorption, the efficiency check and the whole
trajectory count start from that field. Without it the side has to be
inferred from the quote, which is a different and worse thing, and this
prints how much worse: what share of prints are off the mid and could be
assigned at all.

If the field IS there, the second question is whether it means what it
says, and the tape settles that by itself -- a crossing buyer lifts the
offer, so buyer-marked trades print above the mid. The labels are treated
as two opaque buckets with no meaning attached, because `Bid`/`Ask` is
genuinely ambiguous and documentation is sometimes wrong about its own
encoder.

IT DOES NOT KNOW THEIR SCHEMA, AND DOES NOT PRETEND TO

Ironbeam's API documentation could not be read when this was written, so
nothing here assumes a field name. It prints the first few raw messages of
each kind exactly as they arrive, then works out which fields look like a
price, a size, a side and a book by their shape. Where it guesses, it says
so, and the raw dump is there to correct it from.

So the first run is a conversation, not a verdict. Paste the short report
back and the guesses become knowledge.

WHY IT CANNOT PLACE AN ORDER

Ironbeam is a broker. Every request goes through one function that checks
the path against an allowlist of read-only endpoints and raises on anything
else, so order placement is not one typo away -- it is not reachable. The
credentials are read from the environment and never written anywhere.

Use a market-data-only key if Ironbeam issues one. Then a leak exposes
quotes rather than an account.

NOTHING HERE PLACES, SIZES, MODIFIES OR CANCELS ANYTHING.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from liqmap.aggressor import Convention          # noqa: E402

DEMO = "https://demo.ironbeamapi.com/v1"
LIVE = "https://live.ironbeamapi.com/v1"

# THE ONLY PATHS THIS PROGRAM MAY REQUEST. Checked before every call, so
# a broker API cannot turn into an order path through a typo or a later
# edit. Anything not matching one of these raises.
ALLOWED = ("/auth", "/info/", "/market/", "/stream", "/quote", "/depth",
           "/trade/history", "/security", "/symbol")

# Paths that must never be touched whatever else changes. Belt and braces
# with ALLOWED, and the thing a reviewer looks for first.
FORBIDDEN = ("/order", "/position", "/cancel", "/modify", "/place",
             "/execute", "/account/withdraw")

MAX_RAW = 5
MAX_SECONDS = 600


def _check(path: str) -> str:
    low = path.lower()
    for bad in FORBIDDEN:
        if bad in low:
            raise RuntimeError(f"refusing to touch {path}: this program "
                               f"reads market data and nothing else")
    if not any(ok in low for ok in ALLOWED):
        raise RuntimeError(f"{path} is not on the read-only allowlist")
    return path


def _need(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        print(f"{name} is not set. Export it in your shell profile -- "
              f"never in the repo.", file=sys.stderr)
        raise SystemExit(2)
    return v


# ---------------------------------------------------------------- shapes
#
# Nothing below knows Ironbeam's field names. Each of these asks what a
# value LOOKS like, which survives a schema this file has never seen.


PRICE_HINTS = ("price", "px", "last", "trade", "fill", "value")
SIZE_HINTS = ("size", "qty", "quantity", "volume", "amount", "lots")
SIDE_HINTS = ("side", "aggressor", "initiator", "taker", "direction",
              "buysell", "buy_sell", "tickdirection", "action")
BID_HINTS = ("bid",)
ASK_HINTS = ("ask", "offer")


def _flat(obj: Any, prefix: str = "") -> dict:
    """Nested message to {dotted.key: value}, so a field two levels down
    is as findable as one at the top."""
    out: dict = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flat(v, f"{prefix}{k}."))
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:4]):
            out.update(_flat(v, f"{prefix}{i}."))
    else:
        out[prefix.rstrip(".")] = obj
    return out


def _pick(flat: dict, hints, numeric: bool = True) -> tuple[str, Any] | None:
    """The field whose NAME suggests it, preferring the shortest match.

    Shortest because `price` beats `priceFormatted` and `lastPrice` beats
    `lastPriceChangePercent` -- a longer name is usually a derived field,
    and a derived field is the wrong one to audit against.
    """
    best = None
    for k, v in flat.items():
        tail = k.split(".")[-1].lower()
        if not any(h in tail for h in hints):
            continue
        if numeric and not isinstance(v, (int, float)):
            continue
        if best is None or len(k) < len(best[0]):
            best = (k, v)
    return best


def _as_float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------- report


class Probe:
    """Everything learned, and nothing concluded that was not measured."""

    def __init__(self, raw_keep: int = MAX_RAW):
        self.raw_keep = raw_keep
        self.kinds: Counter = Counter()
        self.raw: dict[str, list] = defaultdict(list)
        self.fields: dict[str, Counter] = defaultdict(Counter)
        self.conv = Convention()
        self.bid = 0.0
        self.ask = 0.0
        self.book_seen = 0
        self.trades_seen = 0
        self.side_values: Counter = Counter()
        self.guesses: dict[str, str] = {}

    def kind_of(self, flat: dict) -> str:
        """What sort of message this is, by shape rather than by name."""
        has_bid = _pick(flat, BID_HINTS) is not None
        has_ask = _pick(flat, ASK_HINTS) is not None
        has_px = _pick(flat, PRICE_HINTS) is not None
        has_sz = _pick(flat, SIZE_HINTS) is not None
        if has_bid and has_ask:
            return "book"
        if has_px and has_sz:
            return "trade"
        if has_px:
            return "price-ish"
        return "other"

    def observe(self, msg: Any) -> None:
        flat = _flat(msg)
        # A top-level scalar -- a bare heartbeat, a stray None -- flattens
        # to one entry under an empty key. It is not a message and
        # counting it puts noise in the only report anybody reads.
        if not flat or set(flat) == {""}:
            return
        kind = self.kind_of(flat)
        self.kinds[kind] += 1
        if len(self.raw[kind]) < self.raw_keep:
            self.raw[kind].append(msg)
        for k, v in flat.items():
            self.fields[kind][k] += 1

        if kind == "book":
            b = _pick(flat, BID_HINTS)
            a = _pick(flat, ASK_HINTS)
            if b and a:
                self.guesses.setdefault("bid", b[0])
                self.guesses.setdefault("ask", a[0])
                nb, na = _as_float(b[1]), _as_float(a[1])
                if nb > 0 and na > 0:
                    self.bid, self.ask = nb, na
                    self.book_seen += 1
            return

        if kind in ("trade", "price-ish"):
            p = _pick(flat, PRICE_HINTS)
            if not p:
                return
            self.guesses.setdefault("price", p[0])
            s = _pick(flat, SIDE_HINTS, numeric=False)
            if s is None:
                s = _pick(flat, SIDE_HINTS, numeric=True)
            if s is not None:
                self.guesses.setdefault("side", s[0])
                self.side_values[str(s[1])] += 1
            self.trades_seen += 1
            # THE BOOK AS IT STOOD BEFORE THIS PRINT. Whatever the last
            # book message said, which is the closest thing a websocket
            # ordering gives to "immediately before".
            self.conv.observe(_as_float(p[1]), s[1] if s else None,
                              self.bid, self.ask)

    # ------------------------------------------------------------ output

    def print_report(self, symbol: str) -> None:
        r = self.conv.report()
        say = print
        say("")
        say("=" * 68)
        say(f"  IRONBEAM PROBE -- {symbol}")
        say("=" * 68)

        say("\nMESSAGES SEEN")
        if not self.kinds:
            say("  none. Either the subscription did not take, or the "
                "market is shut.")
            return
        for k, n in self.kinds.most_common():
            say(f"  {k:12} {n}")

        say("\nFIELDS IT GUESSED (correct these from the raw dump below)")
        for role in ("price", "side", "bid", "ask"):
            say(f"  {role:6} -> {self.guesses.get(role) or '(not found)'}")

        say("\nTHE QUESTION THAT DECIDES EVERYTHING")
        if "side" not in self.guesses:
            say("  NO SIDE FIELD FOUND on any trade message.")
            say("  That does not settle it -- it may be named something "
                "this script")
            say("  did not think of. Check the raw trade message below "
                "by eye.")
        else:
            say(f"  A side field is present: {self.guesses['side']}")
            vals = ", ".join(f"{v!r} x{n}"
                             for v, n in self.side_values.most_common(6))
            say(f"  Values seen: {vals}")

        say("\nWHAT THE TAPE SAYS ABOUT ITS OWN CONVENTION")
        say(f"  {r.describe()}")
        say(f"  labelled prints off the mid: {r.n}   "
            f"book updates seen: {self.book_seen}")
        if r.settled:
            say(f"  => '{r.buyer_label()}' is the aggressive buyer.")
        elif r.verdict == "no label":
            say(f"  => the side would have to be INFERRED. At best "
                f"{r.assignable:.0%} of")
            say("     prints could be assigned from the quote; the rest "
                "are at the mid.")
        else:
            say("  => not settled. Run it longer, in a busier session.")

        say("\nRAW MESSAGES (first few of each kind, verbatim)")
        for kind, msgs in self.raw.items():
            say(f"\n  --- {kind} ---")
            for m in msgs[:self.raw_keep]:
                try:
                    say("  " + json.dumps(m)[:400])
                except (TypeError, ValueError):
                    say("  " + repr(m)[:400])
        say("")
        say("Paste the section above back into the chat. It is a few "
            "hundred bytes")
        say("and it is everything needed to write the translator.")
        say("")


# ---------------------------------------------------------------- the wire


def connect_and_listen(base: str, symbol: str, seconds: float,
                       probe: Probe) -> None:
    """Authenticate, subscribe, and read for `seconds`.

    Written against an API whose documentation could not be reached, so
    every step says what it tried and what came back rather than failing
    with a stack trace. Expect to adjust the two marked lines once the
    first run shows what the endpoints actually are.
    """
    try:
        import requests
    except ImportError:
        raise SystemExit("pip install requests")

    user = _need("IRONBEAM_USERNAME")
    pw = _need("IRONBEAM_PASSWORD")
    key = _need("IRONBEAM_API_KEY")

    url = base + _check("/auth")
    print(f"authenticating at {url} ...")
    resp = requests.post(url, json={"username": user, "password": pw,
                                    "apikey": key}, timeout=20)
    print(f"  HTTP {resp.status_code}")
    if resp.status_code != 200:
        print(f"  {resp.text[:300]}")
        raise SystemExit("auth failed -- check the three environment "
                         "variables, and whether this is the demo or live "
                         "host")
    body = resp.json()
    token = (body.get("token") or body.get("accessToken")
             or body.get("access_token") or "")
    if not token:
        print("  authenticated, but no token field recognised. Response "
              "keys:")
        print("  " + ", ".join(sorted(body.keys()))[:300])
        raise SystemExit("adjust the token field and run again")
    print("  got a token")

    try:
        from websocket import WebSocketApp      # websocket-client
    except ImportError:
        raise SystemExit("pip install websocket-client")

    ws_url = (base.replace("https://", "wss://").replace("http://", "ws://")
              + _check("/stream") + f"?token={token}")
    stop_at = time.time() + seconds

    def on_open(ws):
        print(f"socket open, subscribing to {symbol} ...")
        # ADJUST AFTER THE FIRST RUN. Two plausible shapes, sent both --
        # an unrecognised subscribe is ignored, a missing one is silence,
        # and silence is the expensive failure here.
        for payload in (
            {"type": "subscribe", "symbols": [symbol],
             "feeds": ["trades", "depth", "quotes"]},
            {"action": "subscribe", "symbol": symbol,
             "channels": ["trade", "depth"]},
        ):
            ws.send(json.dumps(payload))

    def on_message(ws, raw):
        try:
            msg = json.loads(raw)
        except (TypeError, ValueError):
            return
        for one in (msg if isinstance(msg, list) else [msg]):
            probe.observe(one)
        if time.time() > stop_at:
            ws.close()

    def on_error(ws, err):
        print(f"  socket error: {err}")

    app = WebSocketApp(ws_url, on_open=on_open, on_message=on_message,
                       on_error=on_error)
    print(f"listening for {seconds:.0f}s ...")
    app.run_forever(ping_interval=20, ping_timeout=10)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--symbol", default="MES",
                    help="contract to listen to, e.g. MES or MESZ6")
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--raw", type=int, default=MAX_RAW,
                    help="raw messages of each kind to print")
    ap.add_argument("--live", action="store_true",
                    help="use the live host instead of demo")
    args = ap.parse_args(argv)

    if args.seconds > MAX_SECONDS:
        print(f"capped at {MAX_SECONDS}s", file=sys.stderr)
        args.seconds = MAX_SECONDS

    probe = Probe(raw_keep=max(1, min(args.raw, 20)))
    base = LIVE if args.live else DEMO
    try:
        connect_and_listen(base, args.symbol, args.seconds, probe)
    except KeyboardInterrupt:
        print("\nstopped early")
    except SystemExit:
        raise
    except Exception as exc:                      # noqa: BLE001
        print(f"\n{type(exc).__name__}: {exc}")
        print("Printing whatever was collected before that.")
    probe.print_report(args.symbol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
