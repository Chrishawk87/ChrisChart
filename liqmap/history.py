"""Sweep history and change detection over time.

`strength.diff_positions` can compare two sweeps, but until something stores
consecutive sweeps it can never actually fire. This is that something.

Every sweep is persisted, and each new one is diffed against the previous
sweep of the same coin. The changes are recorded as events, which is what
makes DEFENDED observable at all -- it is defined entirely by a difference
between two points in time and is invisible in any single snapshot.

What the events are worth, in rough order:

    DEFENDED    size held, liquidation pushed away. They posted margin. The
                strongest conviction signal on-chain, and the one nobody
                watching a live dashboard can see.
    LIQUIDATED  the position vanished with spot sitting on its liquidation
                price. Confirms fuel actually burned, which is the ground
                truth for whether a cluster meant anything.
    WEAKENED    liquidation drifted closer without a size change. Margin
                pulled out, or cross-margin pressure from another position.
    ADDED       size up. Conviction, or averaging into a loser -- the PnL at
                the time tells you which.
    REDUCED     size down.
    CLOSED      gone, and not near liquidation. A decision, not a forced exit.

Sweeps are stored whole rather than just their diffs. Storage is cheap and
re-deriving history after a scoring change is not possible if the raw rows
were thrown away.
"""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from .bucket import Position
from .strength import PositionChange, diff_positions

SCHEMA = """
CREATE TABLE IF NOT EXISTS sweeps (
    id          TEXT PRIMARY KEY,
    ts          TEXT NOT NULL,
    coin        TEXT NOT NULL,
    spot        REAL NOT NULL,
    n_positions INTEGER NOT NULL,
    n_wallets   INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sweep_coin_ts ON sweeps(coin, ts);

CREATE TABLE IF NOT EXISTS sweep_positions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    sweep_id        TEXT NOT NULL,
    wallet          TEXT NOT NULL,
    coin            TEXT NOT NULL,
    szi             REAL NOT NULL,
    entry_px        REAL,
    liquidation_px  REAL,
    position_value  REAL,
    leverage        REAL,
    leverage_type   TEXT,
    unrealized_pnl  REAL,
    margin_used     REAL,
    funding_since_open REAL,
    account_value   REAL,
    FOREIGN KEY (sweep_id) REFERENCES sweeps(id)
);
CREATE INDEX IF NOT EXISTS idx_sp_sweep ON sweep_positions(sweep_id);
CREATE INDEX IF NOT EXISTS idx_sp_wallet ON sweep_positions(wallet, coin);

CREATE TABLE IF NOT EXISTS position_changes (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           TEXT NOT NULL,
    coin         TEXT NOT NULL,
    wallet       TEXT NOT NULL,
    kind         TEXT NOT NULL,
    size_change_pct REAL,
    liq_move_pct REAL,
    prev_notional REAL,
    new_notional REAL,
    spot         REAL,
    from_sweep   TEXT,
    to_sweep     TEXT
);
CREATE INDEX IF NOT EXISTS idx_change_ts ON position_changes(ts);
CREATE INDEX IF NOT EXISTS idx_change_kind ON position_changes(coin, kind, ts);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse(ts: str) -> datetime:
    d = datetime.fromisoformat(ts)
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


class History:
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

    # -- writing ----------------------------------------------------------

    def record_sweep(self, coin: str, spot: float, positions: Sequence[Position],
                     ts: datetime | None = None) -> tuple[str, list[PositionChange]]:
        """Store a sweep and diff it against the previous one.

        Returns the new sweep id and the changes detected. On the very first
        sweep there is nothing to compare against, so the change list is
        empty -- that is expected, not a failure.
        """
        stamp = (ts or datetime.now(timezone.utc)).isoformat()
        sweep_id = uuid.uuid4().hex[:16]

        on_coin = [p for p in positions if p.coin == coin]
        previous = self.latest_sweep_positions(coin)

        with self._tx() as c:
            c.execute(
                """INSERT INTO sweeps (id, ts, coin, spot, n_positions, n_wallets)
                   VALUES (?,?,?,?,?,?)""",
                (sweep_id, stamp, coin, spot, len(on_coin),
                 len({p.wallet for p in on_coin})))

            c.executemany(
                """INSERT INTO sweep_positions (
                       sweep_id, wallet, coin, szi, entry_px, liquidation_px,
                       position_value, leverage, leverage_type, unrealized_pnl,
                       margin_used, funding_since_open, account_value
                   ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                [(sweep_id, p.wallet, p.coin, p.szi, p.entry_px,
                  p.liquidation_px, p.position_value, p.leverage,
                  p.leverage_type, p.unrealized_pnl, p.margin_used,
                  p.funding_since_open, p.account_value) for p in on_coin])

        changes: list[PositionChange] = []
        if previous:
            prev_id, prev_positions = previous
            changes = diff_positions(prev_positions, on_coin, spot)
            self._record_changes(changes, coin, spot, stamp, prev_id, sweep_id)

        return sweep_id, changes

    def _record_changes(self, changes: Sequence[PositionChange], coin: str,
                        spot: float, stamp: str, from_sweep: str,
                        to_sweep: str) -> None:
        # HELD is the overwhelming majority and carries no information, so it
        # is counted but not stored. Keeping it would bury the signal.
        interesting = [c for c in changes if c.kind != "HELD"]
        if not interesting:
            return

        with self._tx() as c:
            c.executemany(
                """INSERT INTO position_changes (
                       ts, coin, wallet, kind, size_change_pct, liq_move_pct,
                       prev_notional, new_notional, spot, from_sweep, to_sweep
                   ) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                [(stamp, coin, ch.wallet, ch.kind, ch.size_change_pct,
                  ch.liq_move_pct, ch.prev_notional, ch.new_notional, spot,
                  from_sweep, to_sweep) for ch in interesting])

    # -- reading ----------------------------------------------------------

    def latest_sweep_positions(self, coin: str
                               ) -> tuple[str, list[Position]] | None:
        row = self._conn.execute(
            "SELECT id FROM sweeps WHERE coin = ? ORDER BY ts DESC LIMIT 1",
            (coin,)).fetchone()
        if not row:
            return None
        return row["id"], self.sweep_positions(row["id"])

    def sweep_positions(self, sweep_id: str) -> list[Position]:
        rows = self._conn.execute(
            "SELECT * FROM sweep_positions WHERE sweep_id = ?",
            (sweep_id,)).fetchall()
        return [Position(
            wallet=r["wallet"], coin=r["coin"], szi=r["szi"],
            entry_px=r["entry_px"] or 0.0, liquidation_px=r["liquidation_px"],
            position_value=r["position_value"] or 0.0,
            leverage=r["leverage"] or 0.0,
            leverage_type=r["leverage_type"] or "cross",
            unrealized_pnl=r["unrealized_pnl"] or 0.0,
            margin_used=r["margin_used"] or 0.0,
            funding_since_open=r["funding_since_open"] or 0.0,
            account_value=r["account_value"] or 0.0,
        ) for r in rows]

    def recent_changes(self, coin: str | None = None, kinds: Sequence[str] | None = None,
                       hours: float = 24.0, limit: int = 200) -> list[dict[str, Any]]:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        sql = "SELECT * FROM position_changes WHERE ts >= ?"
        params: list[Any] = [cutoff]
        if coin:
            sql += " AND coin = ?"
            params.append(coin)
        if kinds:
            sql += f" AND kind IN ({','.join('?' * len(kinds))})"
            params.extend(kinds)
        sql += " ORDER BY ts DESC LIMIT ?"
        params.append(limit)
        return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def conviction_flow(self, coin: str, hours: float = 24.0) -> dict[str, Any]:
        """Net notional behind confidence-increasing versus decreasing moves.

        ADDED and DEFENDED are traders committing more. REDUCED, WEAKENED and
        CLOSED are traders stepping back. LIQUIDATED is separate -- it was not
        a decision at all, and lumping forced exits in with voluntary ones is
        how this kind of tally gets misread.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        rows = self._conn.execute(
            """SELECT kind, COUNT(*) AS n,
                      SUM(MAX(prev_notional, new_notional)) AS notional
               FROM position_changes
               WHERE coin = ? AND ts >= ?
               GROUP BY kind""", (coin, cutoff)).fetchall()

        by_kind = {r["kind"]: {"n": r["n"], "notional": r["notional"] or 0.0}
                   for r in rows}

        def total(kinds: Sequence[str], field: str) -> float:
            return sum(by_kind.get(k, {}).get(field, 0) for k in kinds)

        committing = total(("ADDED", "DEFENDED"), "notional")
        retreating = total(("REDUCED", "WEAKENED", "CLOSED"), "notional")

        return {
            "coin": coin,
            "hours": hours,
            "by_kind": by_kind,
            "committing_notional": committing,
            "retreating_notional": retreating,
            "net_conviction": committing - retreating,
            "liquidated_notional": total(("LIQUIDATED",), "notional"),
            "liquidated_count": int(total(("LIQUIDATED",), "n")),
        }

    def sweeps(self, coin: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        sql = "SELECT * FROM sweeps"
        params: list[Any] = []
        if coin:
            sql += " WHERE coin = ?"
            params.append(coin)
        sql += " ORDER BY ts DESC LIMIT ?"
        params.append(limit)
        return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def counts(self) -> dict[str, int]:
        def one(sql: str) -> int:
            return int(self._conn.execute(sql).fetchone()[0])
        return {
            "sweeps": one("SELECT COUNT(*) FROM sweeps"),
            "sweep_positions": one("SELECT COUNT(*) FROM sweep_positions"),
            "changes": one("SELECT COUNT(*) FROM position_changes"),
            "defended": one("SELECT COUNT(*) FROM position_changes WHERE kind='DEFENDED'"),
            "liquidated": one("SELECT COUNT(*) FROM position_changes WHERE kind='LIQUIDATED'"),
        }

    def prune(self, keep_days: int = 30) -> int:
        """Drop old sweeps. Changes are kept -- they are small and are the
        part worth having a long history of."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=keep_days)).isoformat()
        with self._tx() as c:
            old = [r["id"] for r in c.execute(
                "SELECT id FROM sweeps WHERE ts < ?", (cutoff,)).fetchall()]
            if not old:
                return 0
            marks = ",".join("?" * len(old))
            c.execute(f"DELETE FROM sweep_positions WHERE sweep_id IN ({marks})", old)
            c.execute(f"DELETE FROM sweeps WHERE id IN ({marks})", old)
            return len(old)

    def close(self) -> None:
        self._conn.close()


def render_changes(changes: Sequence[dict[str, Any]], top: int = 25) -> str:
    """Text view, most notable first."""
    if not changes:
        return "no position changes recorded in this window"

    # Order by how much the event is worth looking at, then by size.
    priority = {"LIQUIDATED": 0, "DEFENDED": 1, "WEAKENED": 2,
                "ADDED": 3, "REDUCED": 4, "CLOSED": 5, "OPENED": 6}
    ranked = sorted(changes,
                    key=lambda c: (priority.get(c["kind"], 9),
                                   -max(c.get("prev_notional") or 0,
                                        c.get("new_notional") or 0)))

    lines = [f"  {'when':<17} {'kind':<11} {'wallet':<14} {'notional':>13}  detail",
             "  " + "-" * 76]
    for c in ranked[:top]:
        notional = max(c.get("prev_notional") or 0, c.get("new_notional") or 0)
        when = c["ts"][5:16].replace("T", " ")
        wallet = c["wallet"][:10] + ".." if len(c["wallet"]) > 12 else c["wallet"]

        if c["kind"] in ("DEFENDED", "WEAKENED"):
            detail = f"liq moved {c.get('liq_move_pct') or 0:+.2%}"
        elif c["kind"] in ("ADDED", "REDUCED"):
            detail = f"size {c.get('size_change_pct') or 0:+.1%}"
        elif c["kind"] == "LIQUIDATED":
            detail = "forced out"
        else:
            detail = ""

        lines.append(f"  {when:<17} {c['kind']:<11} {wallet:<14} "
                     f"${notional:>12,.0f}  {detail}")
    return "\n".join(lines)
