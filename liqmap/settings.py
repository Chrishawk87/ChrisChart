"""Runtime settings, stored in the database so they survive a redeploy.

On Railway the container filesystem is rebuilt on every deploy. Anything kept
in a config file on disk is gone the moment you push. Environment variables
survive but changing one triggers a restart, which is a heavy way to adjust a
bucket width.

So settings live in SQLite, on the mounted volume, editable over HTTP. The
environment still supplies the defaults for a fresh database and always wins
for secrets, which have no business being editable from a web form.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (
    key     TEXT PRIMARY KEY,
    value   TEXT NOT NULL,
    updated TEXT NOT NULL
);
"""


def default_db_path() -> str:
    """Where the database lives.

    Railway volumes are mounted at a path you choose -- /data is the
    convention. Falling back to the working directory keeps local runs
    working, but on Railway WITHOUT a mounted volume this silently writes to
    ephemeral disk and every deploy wipes your history. `web.startup_report()`
    warns about exactly that.
    """
    env = os.environ.get("LIQMAP_DB")
    if env:
        return env
    if Path("/data").is_dir():
        return "/data/liqmap.db"
    return "liqmap.db"


@dataclass
class Settings:
    """Everything adjustable at runtime. Secrets are NOT here."""

    coins: list[str]
    sigma_per_min: float          # volatility estimate driving every baseline
    horizon_minutes: float
    bucket_bps: float
    min_cluster_notional: float
    max_distance_pct: float
    sweep_interval_minutes: float
    price_interval_seconds: float
    wallet_limit: int
    min_wallet_notional: float
    auto_run: bool

    @classmethod
    def defaults(cls) -> "Settings":
        return cls(
            coins=[c.strip().upper() for c in
                   os.environ.get("LIQMAP_COINS", "BTC,ETH").split(",") if c.strip()],
            sigma_per_min=float(os.environ.get("LIQMAP_SIGMA", "0.0006")),
            horizon_minutes=float(os.environ.get("LIQMAP_HORIZON", "240")),
            bucket_bps=float(os.environ.get("LIQMAP_BUCKET_BPS", "25")),
            min_cluster_notional=float(os.environ.get("LIQMAP_MIN_CLUSTER", "250000")),
            max_distance_pct=float(os.environ.get("LIQMAP_MAX_DISTANCE", "0.25")),
            sweep_interval_minutes=float(os.environ.get("LIQMAP_SWEEP_MIN", "30")),
            price_interval_seconds=float(os.environ.get("LIQMAP_PRICE_SEC", "30")),
            wallet_limit=int(os.environ.get("LIQMAP_WALLET_LIMIT", "600")),
            min_wallet_notional=float(os.environ.get("LIQMAP_MIN_WALLET", "25000")),
            auto_run=os.environ.get("LIQMAP_AUTO", "false").lower() in ("1", "true", "yes"),
        )

    def validate(self) -> list[str]:
        problems = []
        if not self.coins:
            problems.append("at least one coin is required")
        if self.sigma_per_min <= 0:
            problems.append("sigma_per_min must be positive")
        if self.sigma_per_min > 0.01:
            problems.append(
                f"sigma_per_min of {self.sigma_per_min} is 100 bp/min, which "
                "is a crash, not a regime. Check the units -- it is a "
                "PER-MINUTE standard deviation as a decimal.")
        if self.horizon_minutes <= 0:
            problems.append("horizon_minutes must be positive")
        if self.bucket_bps <= 0:
            problems.append("bucket_bps must be positive")
        if not 0 < self.max_distance_pct <= 1:
            problems.append("max_distance_pct must be between 0 and 1")
        if self.wallet_limit > 5000:
            problems.append(
                f"wallet_limit {self.wallet_limit} takes over "
                f"{self.wallet_limit / 600:.0f} minutes per sweep at the rate "
                "limit, which is longer than most sweep intervals")
        if self.sweep_interval_minutes * 60 < self.wallet_limit / 10:
            problems.append(
                "sweep interval is shorter than a sweep takes; sweeps will "
                "overlap and the rate limiter will throttle both")
        return problems


class SettingsStore:
    def __init__(self, path: str | None = None):
        self.path = path or default_db_path()
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def load(self) -> Settings:
        """Stored values layered over environment defaults."""
        base = asdict(Settings.defaults())
        rows = self._conn.execute("SELECT key, value FROM settings").fetchall()
        known = {f.name for f in fields(Settings)}

        for r in rows:
            if r["key"] not in known:
                continue
            try:
                base[r["key"]] = json.loads(r["value"])
            except json.JSONDecodeError:
                continue

        return Settings(**base)

    def update(self, changes: dict[str, Any]) -> tuple[Settings, list[str]]:
        """Apply changes and return the new settings plus any complaints.

        Invalid values are REJECTED rather than stored -- a bad sigma silently
        persisted would corrupt every baseline computed afterwards, and the
        damage would not be visible until the validation report came back
        nonsense.
        """
        from datetime import datetime, timezone

        current = asdict(self.load())
        known = {f.name for f in fields(Settings)}
        applied = {k: v for k, v in changes.items() if k in known}

        candidate_dict = {**current, **applied}
        try:
            candidate = Settings(**candidate_dict)
        except TypeError as exc:
            return self.load(), [f"bad settings payload: {exc}"]

        problems = candidate.validate()
        if problems:
            return self.load(), problems

        now = datetime.now(timezone.utc).isoformat()
        with self._tx() as c:
            c.executemany(
                """INSERT INTO settings (key, value, updated) VALUES (?,?,?)
                   ON CONFLICT(key) DO UPDATE SET
                       value = excluded.value, updated = excluded.updated""",
                [(k, json.dumps(v), now) for k, v in applied.items()])

        return candidate, []

    def reset(self) -> Settings:
        with self._tx() as c:
            c.execute("DELETE FROM settings")
        return self.load()

    def close(self) -> None:
        self._conn.close()
