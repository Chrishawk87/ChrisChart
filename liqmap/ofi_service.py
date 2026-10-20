"""The OFI engine as a service the site can read.

Holds the latest reading in memory and serves it. Starts itself only
when it is actually configured, so the module can sit in the deploy
doing nothing until the Databento key arrives -- at which point it is
two environment variables and a restart, with no code change.

    LIQMAP_OFI           "true" to run it at all
    LIQMAP_OFI_SYMBOLS   default "ES.c.0", comma separated
    LIQMAP_OFI_DATASET   default "GLBX.MDP3"
    LIQMAP_OFI_STYPE     default "continuous"
    LIQMAP_OFI_WINDOW    events summed into the current OFI
    LIQMAP_OFI_BASELINE  window sums the z-score is measured against
    LIQMAP_OFI_Z         your line; unset means publish no verdict
    DATABENTO_API_KEY    read by the client, never by this module

WHY IT REPORTS WHY IT IS OFF

A service that is simply absent looks identical to one that is broken.
`snapshot()` always answers, and when it is not running it says which
of the conditions was not met -- no key, not enabled, no symbol, the
client failed to connect. That is the difference between "waiting for
your subscription" and "silently dead", and from the dashboard those
are otherwise the same blank panel.

NOTHING IS THRESHOLDED UNLESS YOU SET A LINE. With `LIQMAP_OFI_Z`
unset the snapshot carries the measurement and no `side`.

NOTHING HERE PLACES, SIZES OR TIMES AN ORDER. It holds a number.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from . import ofi
from .cme_tape import Tape

# The names the gateway accepts. A typo here fails at connect time with
# a message from the far end that does not say which of your settings
# was wrong, so it is checked before any socket is opened.
VALID_SCHEMAS = frozenset((
    "mbo", "mbp-1", "mbp-10", "cmbp-1", "bbo-1s", "bbo-1m",
    "cbbo-1s", "cbbo-1m", "tbbo", "tcbbo", "trades",
    "ohlcv-1s", "ohlcv-1m", "ohlcv-1h", "ohlcv-1d", "ohlcv-eod",
    "definition", "imbalance", "statistics", "status"))

DEFAULT_SYMBOL = "ES.c.0"
DEFAULT_WINDOW = 100
DEFAULT_BASELINE = 1000
HISTORY = 600               # readings kept for the panel to draw


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default=None):
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


@dataclass
class Config:
    enabled: bool = False
    symbols: tuple = (DEFAULT_SYMBOL,)
    dataset: str = "GLBX.MDP3"
    stype_in: str = "continuous"
    schema: str = "mbp-10"
    window: int = DEFAULT_WINDOW
    baseline: int = DEFAULT_BASELINE
    z_enter: float | None = None

    @classmethod
    def from_env(cls) -> "Config":
        syms = tuple(s.strip() for s in
                     os.environ.get("LIQMAP_OFI_SYMBOLS",
                                    DEFAULT_SYMBOL).split(",")
                     if s.strip())
        return cls(
            enabled=_env_bool("LIQMAP_OFI", False),
            symbols=syms or (DEFAULT_SYMBOL,),
            dataset=os.environ.get("LIQMAP_OFI_DATASET", "GLBX.MDP3"),
            stype_in=os.environ.get("LIQMAP_OFI_STYPE", "continuous"),
            schema=os.environ.get("LIQMAP_OFI_SCHEMA", "mbp-10"),
            # LIQMAP_OFI_SCHEMA takes a comma-separated list, because
            # an L1 entitlement needs `bbo-1s` for the book and
            # `trades` for the prints.
            window=_env_int("LIQMAP_OFI_WINDOW", DEFAULT_WINDOW),
            baseline=_env_int("LIQMAP_OFI_BASELINE", DEFAULT_BASELINE),
            z_enter=_env_float("LIQMAP_OFI_Z", None),
        )

    @property
    def schema_list(self) -> list:
        """The schemas, split and de-duplicated, as the engine will
        subscribe to them."""
        out, seen = [], set()
        for x in str(self.schema).split(","):
            name = x.strip().lower()
            if name and name not in seen:
                seen.add(name)
                out.append(name)
        return out

    @property
    def market(self) -> str:
        """The name the picker shows. 'ES.c.0' -> 'ES'.

        The subscription symbol carries Databento's continuous-contract
        suffix, which is noise in a dropdown and would not match
        anything a person types.
        """
        raw = (self.symbols or ("ES",))[0]
        return raw.split(".")[0].upper()

    def blocked_by(self) -> str:
        """The first unmet condition, in words. '' when ready."""
        if not self.enabled:
            return "LIQMAP_OFI is not set to true"
        if not os.environ.get("DATABENTO_API_KEY"):
            return "DATABENTO_API_KEY is not set"
        if not self.symbols:
            return "LIQMAP_OFI_SYMBOLS is empty"
        if self.window < 1 or self.baseline < 2:
            return "window must be >= 1 and baseline >= 2"
        bad = [n for n in self.schema_list if n not in VALID_SCHEMAS]
        if bad:
            return (f"unknown schema {', '.join(bad)} -- one of "
                    f"{', '.join(sorted(VALID_SCHEMAS))}")
        if not self.schema_list:
            return "LIQMAP_OFI_SCHEMA is empty"
        if os.environ.get("PYTEST_CURRENT_TEST"):
            # Every Runtime() builds one of these, and the suite builds
            # many. With LIQMAP_OFI exported in a shell, a test run
            # would open that many live market-data subscriptions --
            # billable, and against a feed the tests have no business
            # touching. Checked last so the reasons a person actually
            # needs still take precedence in the message.
            return "not started under pytest"
        return ""


class OfiService:
    """Runs the live engine on a background thread; serves the latest.

    The latest reading is published by rebinding one attribute, which
    is atomic under the GIL, so readers never see a half-written dict
    and no lock sits between the feed and the dashboard. The history
    deque is append-only from one thread and snapshotted by copy.
    """

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config.from_env()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last: dict | None = None
        self._history: deque = deque(maxlen=HISTORY)
        self.started_at: float | None = None
        self.updated_at: float | None = None
        self.readings = 0
        self.error = ""
        self.engine = None
        # Candles from the same stream. Present whether or not the
        # engine ever connects, so a panel reading it gets an empty
        # tape rather than an attribute error.
        self.tape = Tape(symbol=(self.config.symbols or ["ES"])[0])

    # ------------------------------------------------------- lifecycle

    def start(self) -> bool:
        """Start if configured. Returns whether it is now running."""
        if self._thread is not None and self._thread.is_alive():
            return True
        why = self.config.blocked_by()
        if why:
            self.error = why
            return False
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="ofi", daemon=True)
        self._thread.start()
        self.started_at = time.time()
        # Seed the chart with recent history so a newly-started engine
        # does not open to a blank chart. Runs on a background thread
        # so it does not block the main thread or the live stream.
        seed_thread = threading.Thread(
            target=self._seed_history, name="ofi-seed", daemon=True)
        seed_thread.start()
        return True

    def _seed_history(self) -> None:
        """Fetch recent OHLCV bars from Databento and pre-populate the
        tape's candle builders.

        The live MBP subscription delivers prints from the moment it
        connects; a chart opened at session start has no prior context.
        One historical OHLCV query per interval fills the visible window
        immediately rather than waiting for bars to accumulate.

        Uses the Historical client (same API key, separate billing item).
        Failures are recorded in `self.error` but never raise -- a chart
        that loads without seed history is still correct, just shallow.
        """
        if not os.environ.get("DATABENTO_API_KEY"):
            return
        try:
            import databento as db
        except ImportError:
            return

        from datetime import datetime, timezone, timedelta
        from .cme_tape import seconds_of
        from .structure import Candle

        # Each interval to seed and how many hours of history to fetch.
        # Kept short: this is paid data and the chart shows ~200 bars.
        SEED: list[tuple[str, int]] = [
            ("1m",  4),   # 4h of 1-minute bars  (~240)
            ("5m",  24),  # 24h of 5-minute bars  (~288)
            ("15m", 48),  # 48h of 15-minute bars (~192)
            ("1h",  200), # 200h of hourly bars   (~200)
        ]

        symbol = (self.config.symbols or ["ES.c.0"])[0]
        dataset = self.config.dataset
        stype_in = self.config.stype_in

        try:
            client = db.Historical()
        except Exception as exc:
            self.error = f"historical seed: {exc}"
            return

        for interval, hours in SEED:
            # Skip intervals whose builder does not exist yet and that
            # the tape was not configured to build.
            if interval not in self.tape.builders:
                # Ensure the builder exists so seed_interval can add it.
                secs = seconds_of(interval)
                if not secs:
                    continue

            schema = f"ohlcv-{interval}"
            now_utc = datetime.now(timezone.utc)
            start = now_utc - timedelta(hours=hours)
            try:
                store = client.timeseries.get_range(
                    dataset=dataset,
                    schema=schema,
                    symbols=[symbol],
                    stype_in=stype_in,
                    start=start,
                    end=now_utc,
                )
                candles: list[Candle] = []
                for rec in store:
                    ts = getattr(rec, "ts_event", None)
                    if ts is None:
                        continue
                    ts_s = int(ts) * 1e-9
                    o = getattr(rec, "open", 0) * 1e-9
                    h = getattr(rec, "high", 0) * 1e-9
                    lo = getattr(rec, "low", 0) * 1e-9
                    c = getattr(rec, "close", 0) * 1e-9
                    v = float(getattr(rec, "volume", 0) or 0)
                    if o <= 0 or h <= 0 or lo <= 0 or c <= 0:
                        continue
                    candles.append(
                        Candle(ts=ts_s, open=o, high=h, low=lo,
                               close=c, volume=v))
                if candles:
                    self.tape.seed_interval(interval, candles)
            except Exception as exc:               # noqa: BLE001
                # A failed seed for one interval should not block the rest.
                self.error = f"historical seed ({interval}): {exc}"

    def stop(self) -> None:
        self._stop.set()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ---------------------------------------------------------- the run

    def _run(self) -> None:
        from .ofi_live import LiveRace, reading_payload
        try:
            self.engine = LiveRace(
                symbols=list(self.config.symbols),
                dataset=self.config.dataset,
                window=self.config.window,
                baseline=self.config.baseline,
                z_enter=self.config.z_enter,
                stype_in=self.config.stype_in,
                schema=self.config.schema,
                tape=self.tape)
            for ev, rd in self.engine.run():
                if self._stop.is_set():
                    break
                payload = reading_payload(ev, rd, self.config.z_enter)
                payload["at"] = time.time()
                self._last = payload           # atomic rebind
                self._history.append(
                    (payload["at"], rd.window, rd.z))
                self.readings += 1
                self.updated_at = payload["at"]
        except Exception as exc:               # noqa: BLE001
            # The thread dying silently is the failure mode that looks
            # exactly like a quiet market. Record it so `snapshot`
            # can say what happened.
            self.error = f"{type(exc).__name__}: {exc}"

    # ----------------------------------------------------------- read

    def snapshot(self, history: int = 0) -> dict:
        """Always answers, running or not."""
        cfg = self.config
        out = {
            "running": self.running,
            "enabled": cfg.enabled,
            "blocked_by": cfg.blocked_by(),
            "error": self.error,
            "symbols": list(cfg.symbols),
            "market": cfg.market,
            "dataset": cfg.dataset,
            "schema": cfg.schema,
            "schemas": cfg.schema_list,
            "window": cfg.window,
            "baseline": cfg.baseline,
            "z_enter": cfg.z_enter,
            "readings": self.readings,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "age_s": (None if self.updated_at is None
                      else round(time.time() - self.updated_at, 3)),
            "last": self._last,
        }
        if self.engine is not None:
            out["refused"] = self.engine.state.refused.as_dict()
            out["records"] = self.engine.state.seen
        out["tape"] = {"trades": self.tape.trades,
                       "records": self.tape.records,
                       "last_px": self.tape.last_px,
                       "last_side": self.tape.last_side,
                       "closed": {k: len(v)
                                  for k, v in self.tape.closed.items()}}
        if history:
            rows = list(self._history)[-history:]
            out["history"] = [{"at": a, "window": w, "z": z}
                              for a, w, z in rows]
        return out

    def feed_records(self, records) -> int:
        """Drive the service from an iterable instead of the socket.

        The replay path, used to fill the panel from a file before any
        subscription exists -- and the reason the dashboard can be
        built and looked at today.
        """
        from .ofi_live import LiveRace, reading_payload
        if self.engine is None:
            self.engine = LiveRace(
                symbols=list(self.config.symbols),
                dataset=self.config.dataset,
                window=self.config.window,
                baseline=self.config.baseline,
                z_enter=self.config.z_enter,
                stype_in=self.config.stype_in,
                schema=self.config.schema,
                tape=self.tape)
        n = 0
        for ev, rd in self.engine.consume(records):
            payload = reading_payload(ev, rd, self.config.z_enter)
            payload["at"] = time.time()
            self._last = payload
            self._history.append((payload["at"], rd.window, rd.z))
            self.readings += 1
            self.updated_at = payload["at"]
            n += 1
        return n
