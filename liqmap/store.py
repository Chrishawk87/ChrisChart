"""Persistence for snapshots, clusters, prices and outcomes.

The validation needs three things recorded at the moment a snapshot is taken,
and one thing recorded later:

    at snapshot time   spot, volatility, and every cluster with its price,
                       notional and side
    afterwards         whether each cluster's price was touched inside the
                       horizon, and what price did next

Recording the volatility AT SNAPSHOT TIME matters and is easy to get wrong.
The baseline probability has to be computed from what was knowable then. Using
the realised volatility of the window you are testing would leak the answer
into the question, and the map would look predictive because the test was
rigged.
"""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    id              TEXT PRIMARY KEY,
    ts              TEXT NOT NULL,
    coin            TEXT NOT NULL,
    spot            REAL NOT NULL,
    sigma_per_min   REAL NOT NULL,
    wallets_swept   INTEGER,
    positions_used  INTEGER,
    total_notional  REAL,
    concentration   REAL
);
CREATE INDEX IF NOT EXISTS idx_snap_coin_ts ON snapshots(coin, ts);

CREATE TABLE IF NOT EXISTS clusters (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id     TEXT NOT NULL,
    price           REAL NOT NULL,
    notional        REAL NOT NULL,
    long_notional   REAL,
    short_notional  REAL,
    n_positions     INTEGER,
    side            TEXT NOT NULL,
    is_placebo      INTEGER DEFAULT 0,
    FOREIGN KEY (snapshot_id) REFERENCES snapshots(id)
);
CREATE INDEX IF NOT EXISTS idx_cluster_snap ON clusters(snapshot_id);

CREATE TABLE IF NOT EXISTS prices (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      TEXT NOT NULL,
    coin    TEXT NOT NULL,
    price   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_price_coin_ts ON prices(coin, ts);

CREATE TABLE IF NOT EXISTS outcomes (
    cluster_id              INTEGER PRIMARY KEY,
    horizon_minutes         REAL NOT NULL,
    touched                 INTEGER NOT NULL,
    minutes_to_touch        REAL,
    move_after_touch_sigmas REAL,
    resolved_ts             TEXT NOT NULL,
    FOREIGN KEY (cluster_id) REFERENCES clusters(id)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse(ts: str) -> datetime:
    d = datetime.fromisoformat(ts)
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


class Store:
    def __init__(self, path: str | Path = "liqmap.db"):
        self.path = str(path)
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

    # -- writes -----------------------------------------------------------

    def log_price(self, coin: str, price: float,
                  ts: datetime | None = None) -> None:
        with self._tx() as c:
            c.execute("INSERT INTO prices (ts, coin, price) VALUES (?,?,?)",
                      ((ts or datetime.now(timezone.utc)).isoformat(), coin, price))

    def save_snapshot(self, lm, sigma_per_min: float, min_notional: float,
                      top: int = 12, placebo_per_cluster: int = 1,
                      ts: datetime | None = None) -> str:
        """Persist a map plus distance-matched placebo levels.

        The placebos are the control group: levels at the same distance from
        spot as a real cluster, but where no cluster sits. Without them the
        only benchmark is the analytic one, and any way the random-walk model
        misdescribes the asset shows up as fake lift.
        """
        import random

        snap_id = uuid.uuid4().hex[:16]
        stamp = (ts or datetime.now(timezone.utc)).isoformat()

        with self._tx() as c:
            c.execute(
                """INSERT INTO snapshots (id, ts, coin, spot, sigma_per_min,
                       wallets_swept, positions_used, total_notional, concentration)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (snap_id, stamp, lm.coin, lm.spot, sigma_per_min,
                 lm.wallets_seen, lm.positions_used, lm.total_notional(),
                 lm.concentration()))

            occupied = {round(b.mid, 8) for b in lm.buckets}

            for b in lm.clusters(min_notional=min_notional, top=top):
                side = "above" if b.mid > lm.spot else "below"
                c.execute(
                    """INSERT INTO clusters (snapshot_id, price, notional,
                           long_notional, short_notional, n_positions, side, is_placebo)
                       VALUES (?,?,?,?,?,?,?,0)""",
                    (snap_id, b.mid, b.total_notional, b.long_notional,
                     b.short_notional, b.count, side))

                # Matched placebo: same absolute distance, opposite side, or
                # jittered nearby -- chosen so it is NOT on a real cluster.
                for _ in range(placebo_per_cluster):
                    dist = abs(b.mid - lm.spot)
                    for _attempt in range(12):
                        sign = random.choice([-1, 1])
                        jitter = 1.0 + random.uniform(-0.08, 0.08)
                        price = lm.spot + sign * dist * jitter
                        if price <= 0:
                            continue
                        if any(abs(price - m) / lm.spot < 0.0015 for m in occupied):
                            continue
                        c.execute(
                            """INSERT INTO clusters (snapshot_id, price, notional,
                                   long_notional, short_notional, n_positions,
                                   side, is_placebo)
                               VALUES (?,?,0,0,0,0,?,1)""",
                            (snap_id, price, "above" if price > lm.spot else "below"))
                        break

        return snap_id

    def record_outcome(self, cluster_id: int, horizon_minutes: float,
                       touched: bool, minutes_to_touch: float | None,
                       move_after_touch_sigmas: float | None) -> None:
        with self._tx() as c:
            c.execute(
                """INSERT INTO outcomes (cluster_id, horizon_minutes, touched,
                       minutes_to_touch, move_after_touch_sigmas, resolved_ts)
                   VALUES (?,?,?,?,?,?)
                   ON CONFLICT(cluster_id) DO UPDATE SET
                       horizon_minutes = excluded.horizon_minutes,
                       touched = excluded.touched,
                       minutes_to_touch = excluded.minutes_to_touch,
                       move_after_touch_sigmas = excluded.move_after_touch_sigmas,
                       resolved_ts = excluded.resolved_ts""",
                (cluster_id, horizon_minutes, int(touched), minutes_to_touch,
                 move_after_touch_sigmas, _now()))

    # -- reads ------------------------------------------------------------

    def prices_between(self, coin: str, start: datetime, end: datetime
                       ) -> list[tuple[datetime, float]]:
        rows = self._conn.execute(
            """SELECT ts, price FROM prices
               WHERE coin = ? AND ts >= ? AND ts <= ? ORDER BY ts""",
            (coin, start.isoformat(), end.isoformat())).fetchall()
        return [(_parse(r["ts"]), float(r["price"])) for r in rows]

    def unresolved_clusters(self, horizon_minutes: float) -> list[dict[str, Any]]:
        """Clusters whose horizon has fully elapsed but have no outcome yet."""
        cutoff = (datetime.now(timezone.utc)
                  - timedelta(minutes=horizon_minutes)).isoformat()
        rows = self._conn.execute(
            """SELECT cl.id AS cluster_id, cl.price, cl.notional, cl.side,
                      cl.is_placebo, s.id AS snapshot_id, s.ts, s.coin,
                      s.spot, s.sigma_per_min
               FROM clusters cl
               JOIN snapshots s ON s.id = cl.snapshot_id
               LEFT JOIN outcomes o ON o.cluster_id = cl.id
               WHERE o.cluster_id IS NULL AND s.ts <= ?
               ORDER BY s.ts""", (cutoff,)).fetchall()
        return [dict(r) for r in rows]

    def observations(self, horizon_minutes: float | None = None) -> list:
        """Everything resolved, as ClusterObservation objects for the validator."""
        from .validate import ClusterObservation

        rows = self._conn.execute(
            """SELECT cl.id, cl.price, cl.notional, cl.side, cl.is_placebo,
                      s.id AS snapshot_id, s.ts, s.coin, s.spot, s.sigma_per_min,
                      o.horizon_minutes, o.touched, o.minutes_to_touch,
                      o.move_after_touch_sigmas
               FROM clusters cl
               JOIN snapshots s ON s.id = cl.snapshot_id
               JOIN outcomes o ON o.cluster_id = cl.id
               ORDER BY s.ts""").fetchall()

        out = []
        for r in rows:
            if horizon_minutes is not None and r["horizon_minutes"] != horizon_minutes:
                continue
            out.append(ClusterObservation(
                snapshot_id=r["snapshot_id"],
                ts=_parse(r["ts"]),
                coin=r["coin"],
                spot=float(r["spot"]),
                sigma_per_min=float(r["sigma_per_min"]),
                cluster_price=float(r["price"]),
                cluster_notional=float(r["notional"]),
                side=r["side"],
                horizon_minutes=float(r["horizon_minutes"]),
                touched=bool(r["touched"]),
                minutes_to_touch=r["minutes_to_touch"],
                move_after_touch_sigmas=r["move_after_touch_sigmas"],
                is_placebo=bool(r["is_placebo"]),
            ))
        return out

    def counts(self) -> dict[str, int]:
        def one(sql: str) -> int:
            return int(self._conn.execute(sql).fetchone()[0])
        return {
            "snapshots": one("SELECT COUNT(*) FROM snapshots"),
            "clusters": one("SELECT COUNT(*) FROM clusters WHERE is_placebo = 0"),
            "placebos": one("SELECT COUNT(*) FROM clusters WHERE is_placebo = 1"),
            "resolved": one("SELECT COUNT(*) FROM outcomes"),
            "prices": one("SELECT COUNT(*) FROM prices"),
        }

    def close(self) -> None:
        self._conn.close()


# --------------------------------------------------------------------------

def resolve_touch(prices: Sequence[tuple[datetime, float]], start: datetime,
                  spot: float, cluster_price: float, sigma_per_min: float,
                  horizon_minutes: float, follow_minutes: float = 60.0
                  ) -> tuple[bool, float | None, float | None]:
    """Work out whether a level was touched, and what happened next.

    The follow-up move is signed toward CONTINUATION -- through an overhead
    level is positive when price keeps rising, through a level below is
    positive when price keeps falling. That makes cascade and reversal one
    readable number regardless of side, and it is measured in sigma so it
    compares across assets and regimes.
    """
    if not prices:
        return False, None, None

    up = cluster_price > spot
    end = start + timedelta(minutes=horizon_minutes)

    touch_idx = None
    for i, (ts, px) in enumerate(prices):
        if ts < start or ts > end:
            continue
        if (px >= cluster_price) if up else (px <= cluster_price):
            touch_idx = i
            break

    if touch_idx is None:
        return False, None, None

    touch_ts, _ = prices[touch_idx]
    minutes_to_touch = (touch_ts - start).total_seconds() / 60.0

    follow_end = touch_ts + timedelta(minutes=follow_minutes)
    after = [px for ts, px in prices[touch_idx:] if ts <= follow_end]
    if len(after) < 2:
        return True, minutes_to_touch, None

    sd = sigma_per_min * (follow_minutes ** 0.5)
    if sd <= 0:
        return True, minutes_to_touch, None

    import math
    move = math.log(after[-1] / cluster_price) / sd
    return True, minutes_to_touch, (move if up else -move)
