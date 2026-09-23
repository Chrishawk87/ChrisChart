"""Hyperliquid data access.

NOT EXERCISED AGAINST THE LIVE API. Written to the published documentation
(endpoint shapes, field names and rate-limit weights verified against
Hyperliquid's docs plus two independent mirrors), but the sandbox this was
built in cannot reach exchange APIs. Expect to fix something on first run, and
treat `check()` as the thing to run before anything else.

Three jobs:

  WALLET DISCOVERY. There is no endpoint that lists positions across users, and
  no official leaderboard. But the public `trades` WebSocket carries
  `users: [buyer, seller]` on every fill, so you can build your own universe
  off the tape. Filter by size and the universe self-selects toward the
  participants whose liquidations actually move price.

  POSITION SWEEPING. `clearinghouseState` per wallet, which carries
  `liquidationPx` per position. Weight 2 against a 1200/minute budget, so 600
  wallets a minute is the ceiling. The limiter below enforces it rather than
  discovering it through 429s.

  LIQUIDITY AND FLOW. `l2Book` polled for resting depth, the same trades
  WebSocket read for aggression, both fed into `flow.LevelWatch`. Two silent
  failure modes here rather than loud ones, so there is a smoke test for each:
  `check_book()` catches the bid/ask groups arriving swapped, and
  `LevelWatcher.side_report` re-derives the aggressor convention from the tape
  instead of trusting the documented codes.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Sequence

import requests

MAINNET = "https://api.hyperliquid.xyz"
WS_MAINNET = "wss://api.hyperliquid.xyz/ws"

# Published request weights. Everything not listed costs 20.
WEIGHTS = {
    "clearinghouseState": 2,
    "allMids": 2,
    "l2Book": 2,
    "metaAndAssetCtxs": 20,
}
WEIGHT_BUDGET_PER_MIN = 1200


class RateLimiter:
    """Sliding-window weight limiter.

    The budget is per IP and shared across every call, so this is deliberately
    conservative: it blocks until the request fits rather than firing and
    handling the rejection. Getting throttled mid-sweep leaves you with a
    half-built map, which is worse than a slow one.
    """

    def __init__(self, budget_per_min: int = WEIGHT_BUDGET_PER_MIN,
                 safety: float = 0.85):
        self.budget = budget_per_min * safety
        self._events: deque[tuple[float, int]] = deque()
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        while self._events and now - self._events[0][0] > 60.0:
            self._events.popleft()

    def acquire(self, weight: int) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._prune(now)
                used = sum(w for _, w in self._events)
                if used + weight <= self.budget:
                    self._events.append((now, weight))
                    return
                oldest = self._events[0][0]
            time.sleep(max(0.05, 60.0 - (time.monotonic() - oldest)))

    @property
    def used_last_minute(self) -> int:
        with self._lock:
            self._prune(time.monotonic())
            return sum(w for _, w in self._events)


class HyperliquidError(RuntimeError):
    pass


class InfoClient:
    """POST /info. No authentication needed for any of this."""

    def __init__(self, base_url: str = MAINNET, timeout: float = 10.0,
                 max_retries: int = 3, limiter: RateLimiter | None = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.limiter = limiter or RateLimiter()
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})
        self._mids_cache: dict[str, tuple[float, dict[str, float]]] = {}
        self._dexes_cache: tuple[float, list[dict]] | None = None

    def post(self, body: dict) -> Any:
        weight = WEIGHTS.get(body.get("type", ""), 20)
        last: Exception | None = None

        for attempt in range(self.max_retries):
            self.limiter.acquire(weight)
            try:
                r = self._session.post(f"{self.base_url}/info", json=body,
                                       timeout=self.timeout)
                if r.status_code == 429:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                if r.status_code >= 400:
                    raise HyperliquidError(
                        f"{r.status_code} on {body.get('type')}: {r.text[:200]}")
                return r.json()
            except requests.RequestException as exc:
                last = exc
                time.sleep(0.5 * (2 ** attempt))

        raise HyperliquidError(f"{body.get('type')} failed after "
                               f"{self.max_retries} attempts: {last}")

    # -- endpoints --------------------------------------------------------

    # Mids move constantly but a price fetched a second ago is still a price.
    # Without this, a dashboard polling every two seconds multiplies by the
    # number of perp DEXes and spends the whole rate-limit budget on quotes.
    MIDS_TTL_S = 2.0
    # The set of deployed DEXes changes when a builder ships a market, which
    # is not something to re-ask about every few seconds.
    DEXES_TTL_S = 600.0

    def all_mids(self, dex: str = "", max_age: float | None = None
                 ) -> dict[str, float]:
        ttl = self.MIDS_TTL_S if max_age is None else max_age
        hit = self._mids_cache.get(dex)
        if hit and time.time() - hit[0] < ttl:
            return hit[1]

        body: dict[str, Any] = {"type": "allMids"}
        if dex:
            body["dex"] = dex
        raw = self.post(body)
        out = {}
        for k, v in (raw or {}).items():
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                continue
        self._mids_cache[dex] = (time.time(), out)
        return out

    def mid(self, coin: str) -> float | None:
        """One symbol's price in ONE request.

        `all_mids_everywhere` costs a request per perp DEX plus two, and it
        was being called on every price poll and every read. On a venue with
        a dozen builder DEXes that is fourteen requests for one number,
        several times a second -- which does not fail loudly, it just queues
        behind the rate limiter until the whole dashboard feels broken.

        The symbol already says which DEX it lives on, so ask that one.
        """
        dex, _ = split_symbol(coin)
        mids = self.all_mids(dex)
        if coin in mids:
            return mids[coin]

        # HIP-3 responses may or may not carry the prefix. Try the bare form.
        bare = coin.split(":", 1)[-1]
        if bare in mids:
            return mids[bare]

        if dex:                       # namespaced but absent: try canonical
            canon = self.all_mids()
            return canon.get(coin) or canon.get(bare)
        return None

    def all_mids_everywhere(self, dexes: Sequence[str] | None = None
                            ) -> dict[str, float]:
        """Mids for the canonical perps AND every HIP-3 DEX.

        HIP-3 symbols come back namespaced as `dex:COIN` so two builders can
        both list GOLD without colliding. A plain `allMids` returns only the
        canonical crypto markets, which is why gold, oil, FX and the stock
        perps look like they do not exist: the sweep asks for a mid, gets
        nothing, and skips the market as unlisted.

        One DEX failing does not stop the others.
        """
        out = dict(self.all_mids())
        if dexes is None:
            try:
                dexes = [d["name"] for d in self.perp_dexs() if d["name"]]
            except HyperliquidError:
                return out

        for name in dexes:
            try:
                for coin, px in self.all_mids(name).items():
                    out[coin if ":" in coin else join_symbol(name, coin)] = px
            except HyperliquidError:
                continue
        return out

    def clearinghouse_state(self, wallet: str, dex: str = "") -> dict:
        """One wallet's perp state.

        `dex` selects which perp DEX. Empty string is the canonical Hyperliquid
        perps -- the crypto majors. HIP-3 builder-deployed DEXes, which is
        where gold, oil, FX, the index and single-stock perps live, each have
        their own name. "ALL_DEXES" asks for every one of them at once.
        """
        body = {"type": "clearinghouseState", "user": wallet}
        if dex:
            body["dex"] = dex
        return self.post(body)

    def perp_dexs(self) -> list[dict]:
        """Every perp DEX on the exchange.

        The canonical crypto DEX usually comes back as a null entry or under
        the name "native"; HIP-3 DEXes are named. Shapes differ between
        documentation sources, so this normalises to a list of dicts with at
        least a `name`.
        """
        if (self._dexes_cache
                and time.time() - self._dexes_cache[0] < self.DEXES_TTL_S):
            return self._dexes_cache[1]

        raw = self.post({"type": "perpDexs"})
        out: list[dict] = []
        for entry in raw or []:
            if entry is None:
                out.append({"name": "", "full_name": "Hyperliquid perps",
                            "native": True})
                continue
            if isinstance(entry, str):
                out.append({"name": entry, "full_name": entry, "native": False})
                continue
            if isinstance(entry, dict):
                name = str(entry.get("name") or entry.get("dex") or "")
                out.append({
                    "name": name,
                    "full_name": str(entry.get("fullName")
                                     or entry.get("full_name") or name),
                    "deployer": entry.get("deployer"),
                    "oracle_updater": entry.get("oracleUpdater"),
                    "native": not name,
                })
        self._dexes_cache = (time.time(), out)
        return out

    def l2_book(self, coin: str) -> dict:
        return self.post({"type": "l2Book", "coin": coin})

    # Intervals the exchange accepts. Anything else is rejected outright.
    INTERVALS = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
                 "1h": 3600, "2h": 7200, "4h": 14400, "8h": 28800,
                 "12h": 43200, "1d": 86400, "3d": 259200, "1w": 604800}

    def candles(self, coin: str, interval: str = "15m", bars: int = 200):
        """OHLCV as `structure.Candle`s, oldest first.

        `bars` is how many of the most recent candles to ask for; the window
        is computed from the interval rather than passed in, because getting
        those two out of step returns a silently truncated series.
        """
        from .structure import parse_candles

        step = self.INTERVALS.get(interval)
        if step is None:
            raise HyperliquidError(
                f"unknown interval {interval!r} -- "
                f"use one of {', '.join(self.INTERVALS)}")

        end = int(time.time() * 1000)
        start = end - step * 1000 * max(bars, 2)
        raw = self.post({"type": "candleSnapshot",
                         "req": {"coin": coin, "interval": interval,
                                 "startTime": start, "endTime": end}})
        return parse_candles(raw)

    def book(self, coin: str):
        """An L2 snapshot as a `flow.Book`."""
        return parse_book(coin, self.l2_book(coin))

    def check(self) -> str:
        """Smoke test. Run this first -- it is the cheapest way to find out
        whether the API shape has moved since this was written."""
        mids = self.all_mids()
        if not mids:
            return "allMids returned nothing -- endpoint or shape has changed"
        sample = ", ".join(f"{k}={v:,.4f}" for k, v in list(mids.items())[:3])
        return f"ok -- {len(mids)} markets. {sample}"

    def check_dexes(self) -> str:
        """Third smoke test: what markets can this account actually see.

        The canonical perp DEX carries the crypto majors and nothing else.
        Gold, oil, FX, the index perps and the single stocks are HIP-3
        builder-deployed markets on SEPARATE perp DEXes, and a
        `clearinghouseState` call without a `dex` never returns any of them.
        That is not an empty market, it is a query pointed at the wrong venue,
        and the two are indistinguishable from the dashboard.
        """
        try:
            dexes = self.perp_dexs()
        except HyperliquidError as exc:
            return (f"perpDexs failed ({exc}). This build can still read the "
                    f"canonical crypto perps; HIP-3 markets will be invisible.")
        if not dexes:
            return "perpDexs returned nothing -- only canonical perps available"

        named = [d["name"] for d in dexes if d["name"]]
        return (f"ok -- {len(dexes)} perp DEXes. canonical + "
                f"{len(named)} HIP-3: {', '.join(named[:12])}"
                + (" …" if len(named) > 12 else ""))

    def check_book(self, coin: str = "BTC") -> str:
        """Second smoke test, for the liquidity side.

        `l2Book`'s `levels` field is documented as [bids, asks]. If that ever
        flips, every depth and slippage number inverts silently, so this
        checks the invariant that actually matters rather than the field name:
        the first group must sit BELOW the second.
        """
        raw = self.l2_book(coin)
        b = parse_book(coin, raw)
        if b.empty:
            return f"l2Book returned no levels for {coin} -- shape has changed"
        if not (b.bids and b.asks):
            return f"l2Book gave only one side for {coin}"
        if b.best_bid >= b.best_ask:
            return (f"BOOK IS CROSSED: bid {b.best_bid} >= ask {b.best_ask}. "
                    f"The bid/ask groups are probably swapped -- do not trust "
                    f"any depth or slippage number until this is fixed.")
        return (f"ok -- {coin} {b.best_bid:,.2f} / {b.best_ask:,.2f}, "
                f"spread {b.spread_bps:.2f}bps, "
                f"{len(b.bids)}x{len(b.asks)} levels")


# --------------------------------------------------------------------------
# wallet discovery
# --------------------------------------------------------------------------

@dataclass
class WalletSighting:
    wallet: str
    notional: float = 0.0
    fills: int = 0
    last_seen: float = field(default_factory=time.time)


class TradeHarvester:
    """Collect wallet addresses off the public trades feed.

    Each trade carries both counterparties, so a few hours of listening on the
    coins you care about yields a universe weighted toward size -- which is
    exactly the population whose forced exits matter.

    `min_notional` is the filter that makes this useful rather than a list of
    every address that ever touched the exchange.
    """

    def __init__(self, coins: Sequence[str], min_notional: float = 25_000.0,
                 ws_url: str = WS_MAINNET):
        self.coins = list(coins)
        self.min_notional = min_notional
        self.ws_url = ws_url
        self.wallets: dict[str, WalletSighting] = {}
        self._stop = threading.Event()

    def _handle(self, msg: dict) -> None:
        if msg.get("channel") != "trades":
            return
        for trade in msg.get("data") or []:
            try:
                px = float(trade.get("px", 0))
                sz = float(trade.get("sz", 0))
            except (TypeError, ValueError):
                continue
            notional = px * sz
            if notional < self.min_notional:
                continue
            for addr in trade.get("users") or []:
                if not isinstance(addr, str) or not addr.startswith("0x"):
                    continue
                w = self.wallets.get(addr)
                if w is None:
                    w = WalletSighting(wallet=addr)
                    self.wallets[addr] = w
                w.notional += notional
                w.fills += 1
                w.last_seen = time.time()

    def run(self, seconds: float = 3600.0,
            on_progress: Callable[[int], None] | None = None) -> dict[str, WalletSighting]:
        """Listen for `seconds`, then return what was gathered.

        Requires the `websocket-client` package.
        """
        try:
            import websocket
        except ImportError as exc:
            raise ImportError(
                "pip install websocket-client for wallet harvesting") from exc

        deadline = time.time() + seconds
        last_report = 0.0

        def on_message(_ws, raw):
            try:
                self._handle(json.loads(raw))
            except (json.JSONDecodeError, TypeError):
                pass

        def on_open(ws):
            for coin in self.coins:
                ws.send(json.dumps({
                    "method": "subscribe",
                    "subscription": {"type": "trades", "coin": coin},
                }))

        while time.time() < deadline and not self._stop.is_set():
            try:
                ws = websocket.WebSocketApp(
                    self.ws_url, on_open=on_open, on_message=on_message)
                # run_forever blocks; cap it so the deadline is respected.
                t = threading.Thread(
                    target=ws.run_forever,
                    kwargs={"ping_interval": 30, "ping_timeout": 10},
                    daemon=True)
                t.start()

                while time.time() < deadline and t.is_alive() and not self._stop.is_set():
                    time.sleep(1.0)
                    if on_progress and time.time() - last_report > 20:
                        last_report = time.time()
                        on_progress(len(self.wallets))
                ws.close()
                t.join(timeout=5)
            except Exception:
                time.sleep(3.0)      # reconnect and carry on

        return self.wallets

    def stop(self) -> None:
        self._stop.set()

    def top_wallets(self, n: int = 2000) -> list[str]:
        ranked = sorted(self.wallets.values(), key=lambda w: -w.notional)
        return [w.wallet for w in ranked[:n]]


# --------------------------------------------------------------------------
# position sweeping
# --------------------------------------------------------------------------

def split_symbol(coin: str) -> tuple[str, str]:
    """`"para:CRDO"` -> `("para", "CRDO")`; `"BTC"` -> `("", "BTC")`.

    HIP-3 namespaces every asset as `{dex}:{coin}` so two builders can list
    the same ticker without colliding. Two deployers can both list GOLD.
    """
    if ":" in coin:
        dex, _, base = coin.partition(":")
        return dex, base
    return "", coin


def join_symbol(dex: str, coin: str) -> str:
    return f"{dex}:{coin}" if dex else coin


def positions_from_state(wallet: str, payload: Any) -> list:
    """Parse a `clearinghouseState` response in either shape.

    A single-DEX call returns the flat object with `assetPositions` and
    `marginSummary`. An ALL_DEXES call returns a mapping of DEX name to that
    object, with the canonical perps under "native".

    Which shape you get back is not something this code can know in advance,
    and guessing wrong loses every position silently rather than raising. So
    it detects instead: if the payload has `assetPositions` it is flat, and
    otherwise anything that looks like a state object is parsed and its
    symbols namespaced by the key it arrived under.
    """
    from .bucket import Position, parse_clearinghouse_state

    if not isinstance(payload, dict):
        return []

    if "assetPositions" in payload or "marginSummary" in payload:
        return parse_clearinghouse_state(wallet, payload)

    out: list[Position] = []
    for key, sub in payload.items():
        if not isinstance(sub, dict):
            continue
        if "assetPositions" not in sub and "marginSummary" not in sub:
            continue
        dex = "" if key in ("native", "", None) else str(key)
        for p in parse_clearinghouse_state(wallet, sub):
            # Namespace the symbol unless the venue already did it, so a GOLD
            # on one builder's DEX never merges with a GOLD on another's.
            if dex and ":" not in p.coin:
                p = replace(p, coin=join_symbol(dex, p.coin))
            out.append(p)
    return out


def sweep_positions(client: InfoClient, wallets: Iterable[str],
                    on_progress: Callable[[int, int], None] | None = None,
                    dex: str = "ALL_DEXES") -> tuple[list, int]:
    """Pull open positions for each wallet, across every perp DEX.

    Returns (positions, wallets_failed). A failed wallet is skipped rather
    than aborting the sweep -- a map missing one address is fine, a map that
    never completes is not.

    `dex` defaults to ALL_DEXES so gold, oil, FX, index and stock perps come
    back alongside the crypto majors. If the exchange rejects that (older
    node, or the parameter is not supported on this endpoint), it falls back
    to the canonical DEX once and stays there rather than failing every
    wallet in turn.

    At weight 2 per call against a 1200/minute budget this runs at roughly 600
    wallets per minute, so a 5,000-wallet universe takes about eight minutes.
    A wildcard call may cost more than a single-DEX one; if sweeps start
    getting throttled, that is the first thing to suspect.
    """
    wallets = list(wallets)
    positions = []
    failed = 0
    use_dex = dex

    for i, w in enumerate(wallets):
        try:
            positions.extend(
                positions_from_state(w, client.clearinghouse_state(w, use_dex)))
        except HyperliquidError as exc:
            if use_dex and _looks_like_bad_dex(exc):
                # One retry, then give up on the wildcard for the whole sweep.
                use_dex = ""
                try:
                    positions.extend(
                        positions_from_state(w, client.clearinghouse_state(w, "")))
                    continue
                except HyperliquidError:
                    pass
            failed += 1
        if on_progress and (i + 1) % 100 == 0:
            on_progress(i + 1, len(wallets))

    return positions, failed


def _looks_like_bad_dex(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(s in text for s in ("dex", "unknown field", "invalid", "400"))


# --------------------------------------------------------------------------
# liquidity and order flow
# --------------------------------------------------------------------------

def parse_book(coin: str, raw: dict) -> "Book":
    """`l2Book` -> `flow.Book`.

    Documented shape is {"coin", "time", "levels": [bids, asks]} with each
    level {"px", "sz", "n"}. Anything unparseable is dropped rather than
    raising: a book missing one bad level is usable, a book that throws is not.
    """
    from .flow import Book, Level

    groups = (raw or {}).get("levels") or []
    out: list[list[Level]] = [[], []]

    for i in (0, 1):
        if i >= len(groups):
            continue
        for lv in groups[i] or []:
            try:
                px = float(lv["px"])
                sz = float(lv["sz"])
            except (KeyError, TypeError, ValueError):
                continue
            if px <= 0 or sz <= 0:
                continue
            try:
                n = int(lv.get("n", 0) or 0)
            except (TypeError, ValueError):
                n = 0
            out[i].append(Level(px=px, sz=sz, n=n))

    return Book(coin=coin, ts=_ms_to_s((raw or {}).get("time")),
                bids=out[0], asks=out[1])


def _ms_to_s(raw_ms: Any) -> float:
    """Epoch milliseconds -> seconds, falling back to now only when the field
    is genuinely absent.

    `float(x or 0) / 1000 or time.time()` looks equivalent and is not: a
    timestamp of 0 is falsy, so it silently becomes the wall clock. One trade
    stamped with the wall clock among trades stamped with the feed's clock
    puts them billions of seconds apart, and every window and bucket
    downstream stops working.
    """
    if raw_ms is None or raw_ms == "":
        return time.time()
    try:
        return float(raw_ms) / 1000.0
    except (TypeError, ValueError):
        return time.time()


# Which raw `side` code means the aggressor was buying. Hyperliquid documents
# "B" for buy, but see `InfoClient`'s note -- none of this has been run
# against the live feed, so `LevelWatcher` re-derives it from the tape and
# reports what it found rather than trusting this constant.
BUY_CODES = {"B", "b", "buy", "Buy", "BUY", "bid"}


def parse_trade(raw: dict, buy_codes: Iterable[str] = BUY_CODES):
    """One entry from the trades feed -> `flow.Trade`. None if unusable."""
    from .flow import Trade

    try:
        px = float(raw["px"])
        sz = float(raw["sz"])
    except (KeyError, TypeError, ValueError):
        return None
    if px <= 0 or sz <= 0:
        return None

    ts = _ms_to_s(raw.get("time"))
    code = str(raw.get("side", ""))
    return Trade(px=px, sz=sz,
                 aggressor="buy" if code in set(buy_codes) else "sell",
                 ts=ts)


class LevelWatcher:
    """Drive a `flow.LevelWatch` from the live feeds.

    Trades stream over the WebSocket; the book is polled, because there is no
    reason to process every book delta when what you need is a depth reading
    every few seconds. At weight 2 a poll every 3 seconds costs 40/minute
    against a 1200 budget, so it coexists with a position sweep.

    NOT EXERCISED AGAINST THE LIVE API. Two things to check on the first run,
    both of which produce plausible-looking wrong numbers rather than errors:

      1. `side_report` after a minute. It re-derives the aggressor convention
         from where prints land relative to mid. If it disagrees with
         BUY_CODES, every flow number is inverted.
      2. `InfoClient.check_book`, which catches the bid/ask groups arriving
         swapped.
    """

    def __init__(self, client: "InfoClient", coin: str, level: float,
                 band_bps: float = 10.0, window_s: float = 300.0,
                 book_every_s: float = 3.0, ws_url: str = WS_MAINNET):
        from .flow import LevelWatch

        self.client = client
        self.coin = coin
        self.watch = LevelWatch(coin, level, band_bps=band_bps,
                                window_s=window_s)
        self.book_every_s = book_every_s
        self.ws_url = ws_url

        self.last_book = None
        self.last_book_ts = 0.0
        self.trades_seen = 0
        self.books_seen = 0
        self.errors: list[str] = []
        self.side_report = "not checked yet"

        self._side_samples: list[tuple[str, float]] = []
        self._stop = threading.Event()
        self._lock = threading.Lock()

    # -- ingest -----------------------------------------------------------

    def _on_trades(self, entries: list[dict]) -> None:
        from .flow import check_side_convention

        for raw in entries or []:
            t = parse_trade(raw)
            if t is None:
                continue
            with self._lock:
                self.watch.on_trade(t)
                self.trades_seen += 1
                if len(self._side_samples) < 400:
                    self._side_samples.append((str(raw.get("side", "")), t.px))

        if self.last_book is not None and len(self._side_samples) >= 40:
            mid = self.last_book.mid
            if mid > 0:
                self.side_report = check_side_convention(self._side_samples, mid)

    def _poll_book(self) -> None:
        try:
            b = self.client.book(self.coin)
        except (HyperliquidError, requests.RequestException) as exc:
            self._note(f"book poll failed: {exc}")
            return
        if b.empty:
            self._note("book poll returned no levels")
            return
        with self._lock:
            self.watch.on_book(b)
            self.last_book = b
            self.last_book_ts = time.time()
            self.books_seen += 1

    def _note(self, msg: str) -> None:
        self.errors.append(f"{time.strftime('%H:%M:%S')} {msg}")
        del self.errors[:-20]

    # -- run --------------------------------------------------------------

    def run(self, seconds: float = 900.0,
            on_progress: Callable[[int, int], None] | None = None) -> None:
        try:
            import websocket
        except ImportError as exc:
            raise ImportError(
                "pip install websocket-client to watch a level") from exc

        deadline = time.time() + seconds

        def on_message(_ws, payload):
            try:
                msg = json.loads(payload)
            except (json.JSONDecodeError, TypeError):
                return
            if msg.get("channel") == "trades":
                self._on_trades(msg.get("data") or [])

        def on_open(ws):
            ws.send(json.dumps({
                "method": "subscribe",
                "subscription": {"type": "trades", "coin": self.coin},
            }))

        self._poll_book()      # one immediately, so there is a reading at once
        next_book = time.time() + self.book_every_s
        last_report = 0.0

        while time.time() < deadline and not self._stop.is_set():
            try:
                ws = websocket.WebSocketApp(self.ws_url, on_open=on_open,
                                            on_message=on_message)
                t = threading.Thread(
                    target=ws.run_forever,
                    kwargs={"ping_interval": 30, "ping_timeout": 10},
                    daemon=True)
                t.start()

                while (time.time() < deadline and t.is_alive()
                       and not self._stop.is_set()):
                    time.sleep(0.25)
                    now = time.time()
                    if now >= next_book:
                        next_book = now + self.book_every_s
                        self._poll_book()
                    if on_progress and now - last_report > 15:
                        last_report = now
                        on_progress(self.trades_seen, self.books_seen)
                ws.close()
                t.join(timeout=5)
            except Exception as exc:        # reconnect rather than die
                self._note(f"websocket: {exc}")
                time.sleep(3.0)

    def stop(self) -> None:
        self._stop.set()

    # -- read -------------------------------------------------------------

    def snapshot(self, size: float = 0.0) -> dict:
        with self._lock:
            a = self.watch.absorption()
            b = self.last_book
            div = self.watch.tape.divergence(self.watch.window_s)
            cvd = self.watch.tape.cvd

        now = time.time()
        out: dict[str, Any] = {
            "coin": self.coin,
            "level": self.watch.level,
            "book_age_s": (now - self.last_book_ts) if self.last_book_ts else None,
            "last_trade_age_s": ((now - self.watch.tape.last_ts)
                                 if self.watch.tape.last_ts else None),
            "trades_seen": self.trades_seen,
            "books_seen": self.books_seen,
            "side_report": self.side_report,
            "errors": self.errors[-5:],
            "cvd": cvd,
            "divergence": {
                "price_change_bps": div.price_change_bps,
                "delta_notional": div.delta_notional,
                "disagrees": div.disagrees,
                "note": div.note,
            },
            "absorption": {
                "direction": a.direction,
                "aggressive_notional": a.aggressive_notional,
                "observed_bps": a.observed_bps,
                "expected_bps": a.expected_bps,
                "impact_ratio": a.impact_ratio,
                "confident": a.confident,
                "absorbing": a.absorbing,
                "thin": a.thin,
                "verdict": a.verdict(),
                "trades": a.trades,
            },
            "band": {
                "consumed": a.band.consumed,
                "replenished": a.band.replenished,
                "refill_events": a.band.refill_events,
                "replenish_ratio": a.band.replenish_ratio,
                "defended": a.band.defended,
                "last_notional": a.band.last_notional,
            },
        }

        if b is not None and not b.empty:
            out["book"] = {
                "bid": b.best_bid, "ask": b.best_ask, "mid": b.mid,
                "spread_bps": b.spread_bps,
                "depth_bid_25bps": b.depth(25.0, "buy"),
                "depth_ask_25bps": b.depth(25.0, "sell"),
                "imbalance_25bps": b.imbalance(25.0),
                "shelves": [{"px": s.px, "notional": s.notional,
                             "multiple": s.multiple, "side": s.side}
                            for s in b.shelves()],
            }
            if size > 0:
                buy, sell = b.walk(size, "buy"), b.walk(size, "sell")
                out["cost"] = {
                    "size": size,
                    "entry_bps": buy.slippage_bps,
                    "exit_bps": sell.slippage_bps,
                    "round_trip_bps": buy.slippage_bps + sell.slippage_bps,
                    "entry_avg_px": buy.avg_px,
                    "exit_avg_px": sell.avg_px,
                    "exhausted": buy.exhausted or sell.exhausted,
                    "levels": max(buy.levels_consumed, sell.levels_consumed),
                }
        return out
