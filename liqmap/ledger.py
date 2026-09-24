"""The agent's own book. Positions it took, decisions it made, and why.

WHY A SEPARATE BOOK FROM YOURS

`history.py` already records suggestions and what you did with them. That
answers "was the tool's advice any good given that Chris took some of it and
ignored the rest", which is a question about the two of you together. It
cannot answer "is the agent any good", because the sample is filtered by
your judgement -- and your judgement is the thing the agent is supposed to
be measured against, not blended into.

So the agent keeps its own book. Every candle it commits: long, short, or
stand aside. It manages what it opened and it closes it for a stated reason.
Nothing here places an order anywhere; it is a written record of what the
agent WOULD have done, kept honestly enough to be scored.

The two books run side by side on the same signals. When they disagree
about a market, that disagreement is the most interesting row in the
database.

THREE TABLES

    paper_decisions   every decision, including the ones to do nothing.
                      Standing aside is a decision and it has an outcome;
                      leaving those rows out is how a strategy gets a hit
                      rate it never had.

    paper_positions   what it opened, what happened, and the reason it came
                      out. MAE and MFE are tracked because "stopped out"
                      and "stopped out after being 14bps in front" are
                      different failures with different fixes.

    tuning_proposals  a change the agent wants to make to itself, with the
                      evidence attached. Nothing here takes effect until
                      it is adopted.

    param_overrides   the knobs that have actually been adopted. Empty is
                      the normal state; the shipped defaults are the
                      starting point and stay in force until displaced.

WHAT IS DELIBERATELY NOT STORED

No order ids, no fills, no venue. This is a measurement device. If a future
version of this file grows a `place_order`, the thing being measured has
changed into something else.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal, Sequence

from .db import ThreadedDB

Side = Literal["long", "short"]
Action = Literal["enter_long", "enter_short", "hold", "exit", "stand_aside"]
ExitReason = Literal["target", "stop", "invalidated", "time", "candle_end",
                     "flip", "manual"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_decisions (
    id           TEXT PRIMARY KEY,
    ts           REAL NOT NULL,
    coin         TEXT NOT NULL,
    interval     TEXT NOT NULL,
    candle_ts    REAL NOT NULL,
    action       TEXT NOT NULL,
    side         TEXT,
    price        REAL NOT NULL,
    position_id  TEXT,
    gate         TEXT,
    grade        TEXT,
    agreeing     INTEGER DEFAULT 0,
    conviction   REAL DEFAULT 0,
    reason       TEXT NOT NULL DEFAULT '',
    features     TEXT
);
CREATE INDEX IF NOT EXISTS ix_pd_coin ON paper_decisions(coin, interval, ts);
CREATE INDEX IF NOT EXISTS ix_pd_action ON paper_decisions(action, ts);

CREATE TABLE IF NOT EXISTS paper_positions (
    id           TEXT PRIMARY KEY,
    coin         TEXT NOT NULL,
    interval     TEXT NOT NULL,
    candle_ts    REAL NOT NULL,
    opened_at    REAL NOT NULL,
    side         TEXT NOT NULL,
    entry        REAL NOT NULL,
    target_px    REAL NOT NULL,
    stop_px      REAL NOT NULL,
    target_bps   REAL NOT NULL,
    risk_bps     REAL NOT NULL,
    cost_bps     REAL NOT NULL DEFAULT 0,
    breakeven    REAL NOT NULL DEFAULT 0,
    grade        TEXT,
    grade_3way   TEXT,
    agreeing     INTEGER DEFAULT 0,
    conviction   REAL DEFAULT 0,
    runway_bps   REAL DEFAULT 0,
    size_usd     REAL DEFAULT 0,
    features     TEXT,
    status       TEXT NOT NULL DEFAULT 'open',
    closed_at    REAL,
    exit_px      REAL,
    exit_reason  TEXT,
    gross_bps    REAL,
    net_bps      REAL,
    r_multiple   REAL,
    mae_bps      REAL DEFAULT 0,
    mfe_bps      REAL DEFAULT 0,
    held_s       REAL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_pp_status ON paper_positions(status, coin);
CREATE INDEX IF NOT EXISTS ix_pp_closed ON paper_positions(closed_at);
CREATE UNIQUE INDEX IF NOT EXISTS ix_pp_candle
    ON paper_positions(coin, interval, candle_ts);

CREATE TABLE IF NOT EXISTS tuning_proposals (
    id           TEXT PRIMARY KEY,
    created_at   REAL NOT NULL,
    param        TEXT NOT NULL,
    current_val  REAL NOT NULL,
    proposed_val REAL NOT NULL,
    n_fit        INTEGER NOT NULL,
    n_test       INTEGER NOT NULL,
    fit_metric   REAL NOT NULL,
    test_metric  REAL NOT NULL,
    base_metric  REAL NOT NULL,
    rationale    TEXT NOT NULL DEFAULT '',
    status       TEXT NOT NULL DEFAULT 'pending',
    decided_at   REAL,
    -- A paired change. Target and stop are one decision: adopting a wider
    -- target without the stop it was measured with is a setting nobody
    -- tested.
    param2       TEXT,
    current_val2 REAL,
    proposed_val2 REAL
);
CREATE INDEX IF NOT EXISTS ix_tp_status ON tuning_proposals(status, created_at);

CREATE TABLE IF NOT EXISTS param_overrides (
    param        TEXT PRIMARY KEY,
    value        REAL NOT NULL,
    adopted_at   REAL NOT NULL,
    proposal_id  TEXT
);
"""


def _uid() -> str:
    return uuid.uuid4().hex[:16]


def _row(r: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(r) if r is not None else None


@dataclass
class OpenPosition:
    """A live paper position, in memory, while it is being managed."""

    id: str
    coin: str
    interval: str
    candle_ts: float
    side: Side
    entry: float
    target_px: float
    stop_px: float
    target_bps: float
    risk_bps: float
    cost_bps: float
    breakeven: float
    opened_at: float
    grade: str = ""
    grade_3way: str = ""
    agreeing: int = 0
    conviction: float = 0.0
    runway_bps: float = 0.0
    size_usd: float = 0.0
    mae_bps: float = 0.0
    mfe_bps: float = 0.0
    features: dict[str, Any] = field(default_factory=dict)
    # How long the read has been against this position. Exits are not taken
    # on a single flickering poll -- see autopilot.py.
    against_s: float = 0.0
    last_seen: float = 0.0

    def signed_bps(self, px: float) -> float:
        """Move in this position's favour, in basis points."""
        if self.entry <= 0:
            return 0.0
        raw = (px - self.entry) / self.entry * 10_000.0
        return raw if self.side == "long" else -raw

    def mark(self, px: float, now: float) -> None:
        """Update the excursion high-water marks."""
        b = self.signed_bps(px)
        self.mfe_bps = max(self.mfe_bps, b)
        self.mae_bps = min(self.mae_bps, b)
        self.last_seen = now

    def touched(self, high: float, low: float) -> ExitReason | None:
        """Did price reach a level? Stop wins ties -- bars cannot order them.

        A bar that spans both the target and the stop is unreadable: nothing
        in OHLC says which came first. Taking the good one is how a paper
        book manufactures a hit rate it will never reproduce with money.
        """
        if high <= 0 or low <= 0 or high < low:
            return None
        if self.side == "long":
            if low <= self.stop_px:
                return "stop"
            if high >= self.target_px:
                return "target"
        else:
            if high >= self.stop_px:
                return "stop"
            if low <= self.target_px:
                return "target"
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "coin": self.coin, "interval": self.interval,
            "side": self.side, "entry": self.entry,
            "target_px": self.target_px, "stop_px": self.stop_px,
            "target_bps": round(self.target_bps, 2),
            "risk_bps": round(self.risk_bps, 2),
            "breakeven": round(self.breakeven, 4),
            "grade": self.grade, "grade_3way": self.grade_3way,
            "agreeing": self.agreeing,
            "conviction": round(self.conviction, 4),
            "runway_bps": round(self.runway_bps, 2),
            "opened_at": self.opened_at,
            "mae_bps": round(self.mae_bps, 2),
            "mfe_bps": round(self.mfe_bps, 2),
            "against_s": round(self.against_s, 1),
            "size_usd": self.size_usd,
        }


class Ledger(ThreadedDB):
    """The agent's own record of what it did and what came of it."""

    def __init__(self, path: str = "liqmap.db"):
        super().__init__(path, SCHEMA)

    def _bootstrap(self, conn: sqlite3.Connection) -> None:
        conn.executescript(SCHEMA)
        self._migrate(conn)
        conn.commit()

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Add later columns to a database created by an earlier build."""
        have = {r["name"] for r in
                conn.execute("PRAGMA table_info(tuning_proposals)")}
        for col, decl in (("param2", "TEXT"), ("current_val2", "REAL"),
                          ("proposed_val2", "REAL")):
            if col not in have:
                conn.execute(
                    f"ALTER TABLE tuning_proposals ADD COLUMN {col} {decl}")

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        conn = self._conn
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    # ---------------------------------------------------------------- log

    def record_decision(self, *, coin: str, interval: str, candle_ts: float,
                        action: Action, price: float, side: str | None = None,
                        position_id: str | None = None,
                        gate: str | None = None, grade: str = "",
                        agreeing: int = 0, conviction: float = 0.0,
                        reason: str = "",
                        features: dict[str, Any] | None = None,
                        now: float | None = None) -> str:
        """Every decision, including standing aside.

        The stand-aside rows are the ones that make the gate audit possible.
        A strategy that only records its trades cannot tell you what its
        filters cost it.
        """
        did = _uid()
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO paper_decisions (id, ts, coin, interval, "
                "candle_ts, action, side, price, position_id, gate, grade, "
                "agreeing, conviction, reason, features) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (did, now if now is not None else time.time(), coin, interval,
                 candle_ts, action, side, price, position_id, gate, grade,
                 int(agreeing), float(conviction), reason,
                 json.dumps(features or {})))
        return did

    def decisions(self, coin: str | None = None, interval: str | None = None,
                  action: str | None = None, limit: int = 200
                  ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM paper_decisions WHERE 1=1"
        args: list[Any] = []
        if coin:
            sql += " AND coin = ?"; args.append(coin)
        if interval:
            sql += " AND interval = ?"; args.append(interval)
        if action:
            sql += " AND action = ?"; args.append(action)
        sql += " ORDER BY ts DESC LIMIT ?"
        args.append(int(limit))
        out = []
        for r in self._conn.execute(sql, args).fetchall():
            d = dict(r)
            d["features"] = json.loads(d.get("features") or "{}")
            out.append(d)
        return out

    # ----------------------------------------------------------- positions

    def open_position(self, *, coin: str, interval: str, candle_ts: float,
                      side: Side, entry: float, target_px: float,
                      stop_px: float, target_bps: float, risk_bps: float,
                      cost_bps: float = 0.0, breakeven: float = 0.0,
                      grade: str = "", grade_3way: str = "",
                      agreeing: int = 0, conviction: float = 0.0,
                      runway_bps: float = 0.0, size_usd: float = 0.0,
                      features: dict[str, Any] | None = None,
                      now: float | None = None) -> OpenPosition | None:
        """Open one. Returns None if this candle already has a position.

        One per candle is enforced in the schema, not in the caller. A retry
        or a double poll must not be able to produce two entries on the same
        bar and double-count the result.
        """
        pid = _uid()
        ts = now if now is not None else time.time()
        try:
            with self._tx() as conn:
                conn.execute(
                    "INSERT INTO paper_positions (id, coin, interval, "
                    "candle_ts, opened_at, side, entry, target_px, stop_px, "
                    "target_bps, risk_bps, cost_bps, breakeven, grade, "
                    "grade_3way, agreeing, conviction, runway_bps, size_usd, "
                    "features, status) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'open')",
                    (pid, coin, interval, candle_ts, ts, side, entry,
                     target_px, stop_px, target_bps, risk_bps, cost_bps,
                     breakeven, grade, grade_3way, int(agreeing),
                     float(conviction), runway_bps, size_usd,
                     json.dumps(features or {})))
        except sqlite3.IntegrityError:
            return None
        return OpenPosition(
            id=pid, coin=coin, interval=interval, candle_ts=candle_ts,
            side=side, entry=entry, target_px=target_px, stop_px=stop_px,
            target_bps=target_bps, risk_bps=risk_bps, cost_bps=cost_bps,
            breakeven=breakeven, opened_at=ts, grade=grade,
            grade_3way=grade_3way, agreeing=agreeing, conviction=conviction,
            runway_bps=runway_bps, size_usd=size_usd,
            features=dict(features or {}), last_seen=ts)

    def close_position(self, pos: OpenPosition, *, exit_px: float,
                       reason: ExitReason, now: float | None = None
                       ) -> dict[str, Any] | None:
        """Close it and compute the result NET of the round trip.

        Gross is the move; net is what would have landed. Scoring gross is
        how a scalping strategy looks profitable at four ticks a trade and
        is not.
        """
        ts = now if now is not None else time.time()
        gross = pos.signed_bps(exit_px)
        net = gross - pos.cost_bps
        r = (gross / pos.risk_bps) if pos.risk_bps > 0 else 0.0
        with self._tx() as conn:
            cur = conn.execute(
                "UPDATE paper_positions SET status='closed', closed_at=?, "
                "exit_px=?, exit_reason=?, gross_bps=?, net_bps=?, "
                "r_multiple=?, mae_bps=?, mfe_bps=?, held_s=? "
                "WHERE id=? AND status='open'",
                (ts, exit_px, reason, gross, net, r, pos.mae_bps, pos.mfe_bps,
                 max(0.0, ts - pos.opened_at), pos.id))
            if cur.rowcount == 0:
                return None
        return {"id": pos.id, "exit_px": exit_px, "reason": reason,
                "gross_bps": round(gross, 2), "net_bps": round(net, 2),
                "r_multiple": round(r, 3),
                "held_s": round(max(0.0, ts - pos.opened_at), 1)}

    def load_open(self, coin: str | None = None) -> list[OpenPosition]:
        """Rehydrate open positions after a restart.

        Without this a redeploy silently abandons every position the agent
        was managing, and the ledger fills with trades that have no exit --
        which then quietly drop out of every average.
        """
        sql = "SELECT * FROM paper_positions WHERE status='open'"
        args: list[Any] = []
        if coin:
            sql += " AND coin = ?"; args.append(coin)
        out = []
        for r in self._conn.execute(sql, args).fetchall():
            d = dict(r)
            out.append(OpenPosition(
                id=d["id"], coin=d["coin"], interval=d["interval"],
                candle_ts=d["candle_ts"], side=d["side"], entry=d["entry"],
                target_px=d["target_px"], stop_px=d["stop_px"],
                target_bps=d["target_bps"], risk_bps=d["risk_bps"],
                cost_bps=d["cost_bps"] or 0.0,
                breakeven=d["breakeven"] or 0.0, opened_at=d["opened_at"],
                grade=d["grade"] or "", grade_3way=d["grade_3way"] or "",
                agreeing=d["agreeing"] or 0, conviction=d["conviction"] or 0.0,
                runway_bps=d["runway_bps"] or 0.0,
                size_usd=d["size_usd"] or 0.0,
                mae_bps=d["mae_bps"] or 0.0, mfe_bps=d["mfe_bps"] or 0.0,
                features=json.loads(d.get("features") or "{}"),
                last_seen=d["opened_at"]))
        return out

    def closed(self, coin: str | None = None, interval: str | None = None,
               since: float | None = None, limit: int = 5000
               ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM paper_positions WHERE status='closed'"
        args: list[Any] = []
        if coin:
            sql += " AND coin = ?"; args.append(coin)
        if interval:
            sql += " AND interval = ?"; args.append(interval)
        if since is not None:
            sql += " AND closed_at >= ?"; args.append(since)
        sql += " ORDER BY closed_at ASC LIMIT ?"
        args.append(int(limit))
        out = []
        for r in self._conn.execute(sql, args).fetchall():
            d = dict(r)
            d["features"] = json.loads(d.get("features") or "{}")
            out.append(d)
        return out

    def positions_between(self, coin: str | None = None,
                          interval: str | None = None,
                          since: float | None = None,
                          limit: int = 500) -> list[dict[str, Any]]:
        """Open and closed together, for drawing on the chart.

        Open ones matter as much as closed: a position still running is the
        one you are looking at the chart to think about.
        """
        sql = "SELECT * FROM paper_positions WHERE 1=1"
        args: list[Any] = []
        if coin:
            sql += " AND coin = ?"; args.append(coin)
        if interval:
            sql += " AND interval = ?"; args.append(interval)
        if since is not None:
            sql += " AND opened_at >= ?"; args.append(since)
        sql += " ORDER BY opened_at ASC LIMIT ?"
        args.append(int(limit))
        out = []
        for r in self._conn.execute(sql, args).fetchall():
            d = dict(r)
            d["features"] = json.loads(d.get("features") or "{}")
            out.append(d)
        return out

    def position(self, pid: str) -> dict[str, Any] | None:
        d = _row(self._conn.execute(
            "SELECT * FROM paper_positions WHERE id = ?", (pid,)).fetchone())
        if d:
            d["features"] = json.loads(d.get("features") or "{}")
        return d

    # ----------------------------------------------------------- proposals

    def propose(self, *, param: str, current: float, proposed: float,
                n_fit: int, n_test: int, fit_metric: float,
                test_metric: float, base_metric: float, rationale: str,
                param2: str | None = None, current2: float | None = None,
                proposed2: float | None = None,
                now: float | None = None) -> str:
        """File a change for approval. Supersedes any pending one for the
        same knob, so the list never grows two competing answers."""
        pid = _uid()
        ts = now if now is not None else time.time()
        with self._tx() as conn:
            conn.execute(
                "UPDATE tuning_proposals SET status='superseded', decided_at=? "
                "WHERE param=? AND status='pending'", (ts, param))
            if param2:
                conn.execute(
                    "UPDATE tuning_proposals SET status='superseded', "
                    "decided_at=? WHERE param2=? AND status='pending'",
                    (ts, param2))
            conn.execute(
                "INSERT INTO tuning_proposals (id, created_at, param, "
                "current_val, proposed_val, n_fit, n_test, fit_metric, "
                "test_metric, base_metric, rationale, status, param2, "
                "current_val2, proposed_val2) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,'pending',?,?,?)",
                (pid, ts, param, current, proposed, int(n_fit), int(n_test),
                 fit_metric, test_metric, base_metric, rationale,
                 param2, current2, proposed2))
        return pid

    def proposals(self, status: str | None = "pending", limit: int = 50
                  ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM tuning_proposals"
        args: list[Any] = []
        if status:
            sql += " WHERE status = ?"; args.append(status)
        sql += " ORDER BY created_at DESC LIMIT ?"
        args.append(int(limit))
        return [dict(r) for r in self._conn.execute(sql, args).fetchall()]

    def decide_proposal(self, pid: str, adopt: bool,
                        now: float | None = None) -> dict[str, Any] | None:
        """Adopt or reject. Adoption writes the override; nothing else does.

        A rejected proposal stays in the table. The record of what the agent
        wanted to do and was told no is part of its history.
        """
        ts = now if now is not None else time.time()
        row = _row(self._conn.execute(
            "SELECT * FROM tuning_proposals WHERE id = ? AND status='pending'",
            (pid,)).fetchone())
        if row is None:
            return None
        with self._tx() as conn:
            conn.execute(
                "UPDATE tuning_proposals SET status=?, decided_at=? WHERE id=?",
                ("adopted" if adopt else "rejected", ts, pid))
            if adopt:
                pairs = [(row["param"], row["proposed_val"])]
                # Both halves of a paired change, or neither. Half of a
                # tested pair is an untested setting.
                if row.get("param2") and row.get("proposed_val2") is not None:
                    pairs.append((row["param2"], row["proposed_val2"]))
                for name, value in pairs:
                    conn.execute(
                        "INSERT INTO param_overrides (param, value, "
                        "adopted_at, proposal_id) VALUES (?,?,?,?) "
                        "ON CONFLICT(param) DO UPDATE SET "
                        "value=excluded.value, "
                        "adopted_at=excluded.adopted_at, "
                        "proposal_id=excluded.proposal_id",
                        (name, value, ts, pid))
        row["status"] = "adopted" if adopt else "rejected"
        row["decided_at"] = ts
        return row

    def overrides(self) -> dict[str, float]:
        return {r["param"]: r["value"] for r in
                self._conn.execute("SELECT * FROM param_overrides").fetchall()}

    def clear_override(self, param: str) -> bool:
        with self._tx() as conn:
            cur = conn.execute("DELETE FROM param_overrides WHERE param = ?",
                               (param,))
        return cur.rowcount > 0

    def override_history(self, limit: int = 50) -> list[dict[str, Any]]:
        return [dict(r) for r in self._conn.execute(
            "SELECT * FROM tuning_proposals WHERE status IN "
            "('adopted','rejected') ORDER BY decided_at DESC LIMIT ?",
            (int(limit),)).fetchall()]

    # --------------------------------------------------------------- misc

    def counts(self) -> dict[str, int]:
        def one(sql: str, args: Sequence[Any] = ()) -> int:
            return int(self._conn.execute(sql, args).fetchone()[0])
        return {
            "decisions": one("SELECT COUNT(*) FROM paper_decisions"),
            "positions": one("SELECT COUNT(*) FROM paper_positions"),
            "open": one("SELECT COUNT(*) FROM paper_positions "
                        "WHERE status='open'"),
            "closed": one("SELECT COUNT(*) FROM paper_positions "
                          "WHERE status='closed'"),
            "pending_proposals": one("SELECT COUNT(*) FROM tuning_proposals "
                                     "WHERE status='pending'"),
        }

    def prune(self, keep_days: int = 60) -> int:
        cutoff = time.time() - keep_days * 86400
        with self._tx() as conn:
            cur = conn.execute("DELETE FROM paper_decisions WHERE ts < ?",
                               (cutoff,))
            n = cur.rowcount
            conn.execute("DELETE FROM paper_positions WHERE status='closed' "
                         "AND closed_at < ?", (cutoff,))
        return n
