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
            window=_env_int("LIQMAP_OFI_WINDOW", DEFAULT_WINDOW),
            baseline=_env_int("LIQMAP_OFI_BASELINE", DEFAULT_BASELINE),
            z_enter=_env_float("LIQMAP_OFI_Z", None),
        )

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
        return True

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
                schema=self.config.schema)
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
            "dataset": cfg.dataset,
            "schema": cfg.schema,
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
                schema=self.config.schema)
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
