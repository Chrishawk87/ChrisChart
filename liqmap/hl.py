"""Hyperliquid data access.

NOT EXERCISED AGAINST THE LIVE API. Written to the published documentation
(endpoint shapes, field names and rate-limit weights verified against
Hyperliquid's docs plus two independent mirrors), but the sandbox this was
built in cannot reach exchange APIs. Expect to fix something on first run, and
treat `check()` as the thing to run before anything else.

Two jobs:

  WALLET DISCOVERY. There is no endpoint that lists positions across users, and
  no official leaderboard. But the public `trades` WebSocket carries
  `users: [buyer, seller]` on every fill, so you can build your own universe
  off the tape. Filter by size and the universe self-selects toward the
  participants whose liquidations actually move price.

  POSITION SWEEPING. `clearinghouseState` per wallet, which carries
  `liquidationPx` per position. Weight 2 against a 1200/minute budget, so 600
  wallets a minute is the ceiling. The limiter below enforces it rather than
  discovering it through 429s.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
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

    def all_mids(self) -> dict[str, float]:
        raw = self.post({"type": "allMids"})
        out = {}
        for k, v in (raw or {}).items():
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                continue
        return out

    def clearinghouse_state(self, wallet: str) -> dict:
        return self.post({"type": "clearinghouseState", "user": wallet})

    def check(self) -> str:
        """Smoke test. Run this first -- it is the cheapest way to find out
        whether the API shape has moved since this was written."""
        mids = self.all_mids()
        if not mids:
            return "allMids returned nothing -- endpoint or shape has changed"
        sample = ", ".join(f"{k}={v:,.4f}" for k, v in list(mids.items())[:3])
        return f"ok -- {len(mids)} markets. {sample}"


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

def sweep_positions(client: InfoClient, wallets: Iterable[str],
                    on_progress: Callable[[int, int], None] | None = None
                    ) -> tuple[list, int]:
    """Pull open positions for each wallet.

    Returns (positions, wallets_failed). A failed wallet is skipped rather
    than aborting the sweep -- a map missing one address is fine, a map that
    never completes is not.

    At weight 2 per call against a 1200/minute budget this runs at roughly 600
    wallets per minute, so a 5,000-wallet universe takes about eight minutes.
    """
    from .bucket import parse_clearinghouse_state

    wallets = list(wallets)
    positions = []
    failed = 0

    for i, w in enumerate(wallets):
        try:
            positions.extend(parse_clearinghouse_state(w, client.clearinghouse_state(w)))
        except HyperliquidError:
            failed += 1
        if on_progress and (i + 1) % 100 == 0:
            on_progress(i + 1, len(wallets))

    return positions, failed
