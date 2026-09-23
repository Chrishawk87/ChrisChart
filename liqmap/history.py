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
from .db import ThreadedDB
from .strength import PositionChange, diff_positions

SCHEMA = """
-- Live candle reads, and how they turned out.
--
-- The historical replay in backtest.py can only see what candles provide.
-- Flow, absorption, book depth and the liquidation magnet were never recorded
-- by the exchange, so a backtest over past candles is permanently blind to
-- the four strongest live signals. The only way to measure those is to write
-- down what the read said at the time and come back when the candle closes.
--
-- That is what this table is: a forward test that accumulates while you use
-- the thing. It answers the question a historical replay cannot.
CREATE TABLE IF NOT EXISTS reads (
    id            TEXT PRIMARY KEY,
    ts            TEXT NOT NULL,
    coin          TEXT NOT NULL,
    interval      TEXT NOT NULL,
    candle_ts     REAL NOT NULL,
    candle_end    REAL NOT NULL,
    score         REAL NOT NULL,
    lean          TEXT NOT NULL,
    confidence    REAL NOT NULL,
    elapsed_frac  REAL NOT NULL,
    price         REAL NOT NULL,
    had_flow      INTEGER NOT NULL DEFAULT 0,
    signals       TEXT,
    resolved      INTEGER NOT NULL DEFAULT 0,
    outcome_px    REAL,
    correct       INTEGER,
    move_bps      REAL
);
CREATE INDEX IF NOT EXISTS idx_reads_open ON reads(resolved, candle_end);
CREATE INDEX IF NOT EXISTS idx_reads_market ON reads(coin, interval, ts);

-- Suggestions, and what was done about them.
--
-- A read is an opinion about direction. A suggestion is a trade: an entry, a
-- target, an invalidation and a cost. Those settle differently -- a read is
-- right if the candle closed the correct side, a suggestion is right only if
-- the target came before the stop -- so they are scored separately.
--
-- `decision` is the part that does not exist anywhere else. Every suggestion
-- is stored whether it was taken or ignored, which makes two measurements
-- possible that a hit rate alone cannot give:
--
--     the tool's edge     how the suggestions did, all of them
--     the operator's edge how the TAKEN ones did against the IGNORED ones
--
-- If the ignored suggestions win more often than the taken ones, the filter
-- being applied is costing money, and there is no way to discover that
-- without writing down the ones that were passed on.
CREATE TABLE IF NOT EXISTS suggestions (
    id            TEXT PRIMARY KEY,
    ts            TEXT NOT NULL,
    coin          TEXT NOT NULL,
    interval      TEXT NOT NULL,
    candle_ts     REAL NOT NULL,
    candle_end    REAL NOT NULL,
    -- Wall clock when the suggestion was made, which is NOT the candle's
    -- open. A 15m bar can spike 300bps in its first two minutes and come
    -- back; settling a suggestion made at minute ten against that bar's full
    -- high would record a target hit that happened eight minutes before the
    -- trade existed. Resolution measures from here forward, never from the
    -- candle's start.
    made_at       REAL NOT NULL DEFAULT 0,
    side          TEXT NOT NULL,
    entry         REAL NOT NULL,
    target_px     REAL NOT NULL,
    stop_px       REAL NOT NULL,
    target_bps    REAL NOT NULL,
    risk_bps      REAL NOT NULL,
    rr            REAL NOT NULL,
    cost_bps      REAL NOT NULL,
    conviction    REAL NOT NULL,
    score         REAL NOT NULL,
    reason        TEXT,
    decision      TEXT NOT NULL DEFAULT 'pending',
    decided_at    TEXT,
    resolved      INTEGER NOT NULL DEFAULT 0,
    outcome       TEXT,
    outcome_px    REAL,
    pnl_bps       REAL,      -- NET of cost_bps, not gross
    gross_bps     REAL
);
CREATE INDEX IF NOT EXISTS idx_sugg_open ON suggestions(resolved, candle_end);
CREATE INDEX IF NOT EXISTS idx_sugg_decision ON suggestions(decision, resolved);
-- Unique rather than a plain index. `record_suggestion` checks before it
-- inserts, but the dashboard polls this route every few seconds and FastAPI
-- runs sync handlers on a threadpool, so two overlapping polls can both pass
-- the check and write two rows for one trade -- which then settle
-- identically and double-count that candle in the hit rate.
CREATE UNIQUE INDEX IF NOT EXISTS idx_sugg_candle
    ON suggestions(coin, interval, candle_ts);

-- Every candle's agreement state, tradeable or not.
--
-- The suggestions table only holds trades. This holds the READ on every
-- candle, including all the ones that were refused, which is the only way to
-- answer the questions that matter about the confirmation rule itself:
--
--     when book and price agree, how far does price actually go?
--     when they disagree, which one turns out to be right?
--     how often is each state even reached?
--
-- None of those can be answered from the trades alone, because the trades are
-- exactly the cases that passed the filter. Measuring only what you took and
-- concluding the filter works is the oldest mistake there is.
--
-- This is recorded LIVE and cannot be backfilled. No exchange serves
-- historical order book depth, so `book_dir` has no historical equivalent --
-- see `/api/agreement-replay` for what CAN be measured from bars, which is
-- the price-action half on its own.
CREATE TABLE IF NOT EXISTS agreement_states (
    id            TEXT PRIMARY KEY,
    ts            TEXT NOT NULL,
    coin          TEXT NOT NULL,
    interval      TEXT NOT NULL,
    candle_ts     REAL NOT NULL,
    candle_end    REAL NOT NULL,
    made_at       REAL NOT NULL DEFAULT 0,
    elapsed_frac  REAL NOT NULL DEFAULT 0,
    book_dir      TEXT NOT NULL,
    book_strength REAL NOT NULL DEFAULT 0,
    price_dir     TEXT NOT NULL,
    price_strength REAL NOT NULL DEFAULT 0,
    verdict       TEXT NOT NULL,
    price         REAL NOT NULL,
    -- The RAW readings behind the verdict, as JSON: microprice tilt, near
    -- imbalance, replenishment, depletion, aggression, spread, and the
    -- price-action components.
    --
    -- Storing only the verdict would mean every change to a threshold or a
    -- weight starts the measurement over from zero. Storing what the book
    -- and the tape actually SAID lets any new rule be re-scored against
    -- history that has already been collected — which is the difference
    -- between tuning in an afternoon and tuning in a quarter.
    features      TEXT,
    resolved      INTEGER NOT NULL DEFAULT 0,
    close_px      REAL,
    move_bps      REAL,     -- signed: + is up, from `price` to the close
    mfe_bps       REAL,     -- furthest it went UP after the read
    mae_bps       REAL      -- furthest it went DOWN after the read
);
CREATE INDEX IF NOT EXISTS idx_state_open ON agreement_states(resolved, candle_end);
CREATE INDEX IF NOT EXISTS idx_state_market
    ON agreement_states(coin, interval, verdict);
CREATE UNIQUE INDEX IF NOT EXISTS idx_state_candle
    ON agreement_states(coin, interval, candle_ts);

-- Uploaded chart history.
--
-- Kept in the database rather than on the filesystem so it lands on the same
-- mounted volume as everything else. On Railway the container disk is
-- rebuilt every deploy; a dataset written beside the code would be gone on
-- the next push, which is exactly the surprise this avoids.
--
-- Bars are stored as a JSON array rather than a row per bar. They are only
-- ever read whole, to be replayed, and a hundred thousand rows of five
-- floats each would make the database larger and the replay slower.
CREATE TABLE IF NOT EXISTS datasets (
    id          TEXT PRIMARY KEY,
    ts          TEXT NOT NULL,
    name        TEXT NOT NULL,
    coin        TEXT NOT NULL,
    interval    TEXT NOT NULL,
    interval_s  REAL NOT NULL,
    bars        INTEGER NOT NULL,
    start_ts    REAL NOT NULL,
    end_ts      REAL NOT NULL,
    report      TEXT,
    candles     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_datasets_market ON datasets(coin, interval);

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


class History(ThreadedDB):
    """Sweep, read and suggestion history.

    One connection per thread -- see `db.py`. The background worker writes
    while the dashboard reads, and a shared connection corrupts both.
    """

    def __init__(self, path: str | Path = "liqmap.db"):
        super().__init__(path, SCHEMA)

    def _bootstrap(self, conn: sqlite3.Connection) -> None:
        super()._bootstrap(conn)
        self._migrate(conn)
        conn.commit()

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Add columns that later versions introduced.

        `CREATE TABLE IF NOT EXISTS` does nothing to a table that already
        exists, so a database written by an earlier build keeps the old
        shape and every INSERT naming a new column fails at runtime. These
        are additive only -- nothing is dropped or rewritten.
        """
        wanted = {
            "suggestions": [("made_at", "REAL NOT NULL DEFAULT 0"),
                            ("gross_bps", "REAL")],
        }
        for table, columns in wanted.items():
            try:
                have = {r["name"] for r in conn.execute(
                    f"PRAGMA table_info({table})").fetchall()}
            except sqlite3.Error:
                continue
            if not have:
                continue
            for name, decl in columns:
                if name not in have:
                    conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN {name} {decl}")

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

    # -- forward testing ---------------------------------------------------

    # One read per candle per market. Polling every ten seconds would
    # otherwise write ninety rows for a single candle and let one lucky
    # fifteen minutes dominate the measured hit rate.
    def record_read(self, coin: str, interval: str, candle_ts: float,
                    candle_end: float, score: float, lean: str,
                    confidence: float, elapsed_frac: float, price: float,
                    had_flow: bool, signals: str = "") -> str | None:
        """Write down what the read said, unless this candle already has one.

        Returns the row id, or None when a read for this candle exists. Later
        reads in the same candle are deliberately dropped rather than
        updating: the honest question is what it said at a given point, not
        what it settled on once the answer was nearly known.
        """
        existing = self._conn.execute(
            "SELECT id FROM reads WHERE coin=? AND interval=? AND candle_ts=?",
            (coin, interval, candle_ts)).fetchone()
        if existing:
            return None

        rid = uuid.uuid4().hex[:16]
        with self._tx() as c:
            c.execute(
                """INSERT INTO reads (id, ts, coin, interval, candle_ts,
                       candle_end, score, lean, confidence, elapsed_frac,
                       price, had_flow, signals)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (rid, datetime.now(timezone.utc).isoformat(), coin, interval,
                 candle_ts, candle_end, score, lean, confidence, elapsed_frac,
                 price, 1 if had_flow else 0, signals))
        return rid

    def pending_reads(self, now: float, limit: int = 500) -> list[dict[str, Any]]:
        """Reads whose candle has closed but which have no outcome yet."""
        rows = self._conn.execute(
            """SELECT * FROM reads WHERE resolved = 0 AND candle_end <= ?
               ORDER BY candle_end LIMIT ?""", (now, limit)).fetchall()
        return [dict(r) for r in rows]

    def resolve_read(self, read_id: str, outcome_px: float) -> bool:
        """Settle one read against the price at its candle's close.

        Correct means price finished on the side the read leaned, measured
        FROM THE PRICE AT THE TIME OF THE READ -- not from the candle's open.
        That is the question a scalper is actually asking: if I had acted on
        this, would I be up when the candle ended?
        """
        row = self._conn.execute(
            "SELECT lean, price FROM reads WHERE id = ?", (read_id,)).fetchone()
        if row is None:
            return False

        entry = float(row["price"] or 0)
        if entry <= 0 or outcome_px <= 0:
            return False
        move = (outcome_px - entry) / entry * 10_000.0

        lean = row["lean"]
        if lean == "up":
            correct = 1 if move > 0 else 0
        elif lean == "down":
            correct = 1 if move < 0 else 0
        else:
            correct = None          # a flat read makes no claim to score

        with self._tx() as c:
            c.execute(
                """UPDATE reads SET resolved = 1, outcome_px = ?, correct = ?,
                       move_bps = ? WHERE id = ?""",
                (outcome_px, correct, move, read_id))
        return True

    def read_stats(self, coin: str | None = None, interval: str | None = None
                   ) -> dict[str, Any]:
        """Measured performance of the live read, by score band.

        Splits on whether flow and absorption were available, because those
        are exactly the signals a historical backtest cannot see -- and the
        difference between the two is the value of having a level watch
        running.
        """
        where = ["resolved = 1", "correct IS NOT NULL"]
        params: list[Any] = []
        if coin:
            where.append("coin = ?")
            params.append(coin)
        if interval:
            where.append("interval = ?")
            params.append(interval)
        clause = " AND ".join(where)

        rows = [dict(r) for r in self._conn.execute(
            f"SELECT score, correct, move_bps, had_flow FROM reads "
            f"WHERE {clause}", params).fetchall()]

        def summarise(subset: list[dict[str, Any]]) -> dict[str, Any]:
            if not subset:
                return {"n": 0, "accuracy": None, "avg_move_bps": None}
            right = sum(1 for r in subset if r["correct"])
            moves = [abs(float(r["move_bps"] or 0)) for r in subset]
            return {"n": len(subset), "accuracy": right / len(subset),
                    "avg_move_bps": sum(moves) / len(moves)}

        bands = []
        for lo, hi, name in ((0.15, 0.30, "weak"), (0.30, 0.50, "moderate"),
                             (0.50, 1.01, "strong")):
            sub = [r for r in rows if lo <= abs(float(r["score"])) < hi]
            bands.append({"band": name, **summarise(sub)})

        pending = int(self._conn.execute(
            "SELECT COUNT(*) FROM reads WHERE resolved = 0").fetchone()[0])

        return {
            "overall": summarise(rows),
            "with_flow": summarise([r for r in rows if r["had_flow"]]),
            "without_flow": summarise([r for r in rows if not r["had_flow"]]),
            "bands": bands,
            "pending": pending,
        }

    # -- suggestions and decisions -----------------------------------------

    # One suggestion per candle per market, same reasoning as reads: polling
    # every few seconds would otherwise write ninety rows for one trade.
    def record_suggestion(self, coin: str, interval: str, candle_ts: float,
                          candle_end: float, side: str, entry: float,
                          target_px: float, stop_px: float, target_bps: float,
                          risk_bps: float, rr: float, cost_bps: float,
                          conviction: float, score: float,
                          reason: str = "",
                          made_at: float | None = None) -> str | None:
        """Write down a suggestion. Returns its id, or None if this candle
        already has one.

        The INSERT relies on the unique index rather than only on the check
        above it: two overlapping polls can both find nothing and both write.
        """
        import time as _time

        existing = self._conn.execute(
            "SELECT id FROM suggestions WHERE coin=? AND interval=? AND candle_ts=?",
            (coin, interval, candle_ts)).fetchone()
        if existing:
            return None

        sid = uuid.uuid4().hex[:16]
        try:
            with self._tx() as c:
                c.execute(
                    """INSERT INTO suggestions (id, ts, coin, interval,
                           candle_ts, candle_end, made_at, side, entry,
                           target_px, stop_px, target_bps, risk_bps, rr,
                           cost_bps, conviction, score, reason)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (sid, _now(), coin, interval, candle_ts, candle_end,
                     made_at if made_at is not None else _time.time(), side,
                     entry, target_px, stop_px, target_bps, risk_bps, rr,
                     cost_bps, conviction, score, reason))
        except sqlite3.IntegrityError:
            # Another thread won the race for this candle. Its row is the
            # one that counts.
            return None
        return sid

    def decide(self, suggestion_id: str, taken: bool,
               now: float | None = None) -> bool:
        """Record that it was taken or ignored, while the candle is still open.

        A decision can be changed until the candle closes -- a misclick
        should be fixable. After that it is frozen, because choosing whether
        you "took" a trade once you can see how it went is not a decision,
        it is a record of hindsight, and a table full of those measures
        nothing.

        The freeze is on THE CLOCK, not on whether the row happens to have
        been settled yet. An earlier version checked `resolved`, which meant
        that any delay in the resolver -- or a resolver that was never wired
        up at all -- left every past suggestion editable indefinitely. The
        UPDATE carries the same condition so a settle landing mid-call
        cannot be overtaken.
        """
        import time as _time

        stamp = now if now is not None else _time.time()
        with self._tx() as c:
            cur = c.execute(
                """UPDATE suggestions SET decision = ?, decided_at = ?
                   WHERE id = ? AND resolved = 0 AND candle_end > ?""",
                ("taken" if taken else "ignored", _now(), suggestion_id,
                 stamp))
        return cur.rowcount > 0

    def pending_suggestions(self, now: float, limit: int = 500
                            ) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """SELECT * FROM suggestions WHERE resolved = 0 AND candle_end <= ?
               ORDER BY candle_end LIMIT ?""", (now, limit)).fetchall()
        return [dict(r) for r in rows]

    def resolve_suggestion(self, suggestion_id: str, high: float, low: float,
                           close: float) -> bool:
        """Settle one suggestion against what price actually did.

        Target or stop, whichever came first. When a single bar touched both,
        `suggest.settle` returns the stop -- bar data cannot order two touches
        inside one bar, and assuming the good one manufactures a hit rate that
        will never be reproduced live.
        """
        from .suggest import settle

        row = self._conn.execute(
            """SELECT side, entry, target_px, stop_px, cost_bps
               FROM suggestions WHERE id = ?""", (suggestion_id,)).fetchone()
        if row is None:
            return False

        entry = float(row["entry"] or 0)
        if entry <= 0:
            return False

        outcome = settle(row["side"], float(row["target_px"]),
                         float(row["stop_px"]), high, low)

        if outcome == "target":
            exit_px = float(row["target_px"])
        elif outcome == "stop":
            exit_px = float(row["stop_px"])
        else:
            exit_px = close        # still open at the close: mark it to market

        if exit_px <= 0:
            return False
        move = (exit_px - entry) / entry * 10_000.0
        gross = move if row["side"] == "long" else -move

        # Net of the round trip this suggestion was priced against. Without
        # this, four trades at plus and minus 100bps with a 2bp cost each
        # report a total of zero when the truth is minus eight -- and the
        # entire reason the cost gate exists is that a move which does not
        # clear the round trip is not a win. A scoreboard that forgets the
        # cost turns a losing log into a break-even one.
        pnl = gross - float(row["cost_bps"] or 0.0)

        with self._tx() as c:
            c.execute(
                """UPDATE suggestions SET resolved = 1, outcome = ?,
                       outcome_px = ?, gross_bps = ?, pnl_bps = ?
                   WHERE id = ?""",
                (outcome, exit_px, gross, pnl, suggestion_id))
        return True

    def abandon_suggestion(self, suggestion_id: str) -> bool:
        """Mark a suggestion as permanently unsettleable.

        `resolved = 2` is deliberately neither state: `pending_suggestions`
        looks for 0 and `suggestion_stats` looks for 1, so an abandoned row
        leaves the queue without ever entering the scoreboard. Counting it
        as an outcome would be inventing one; leaving it pending would block
        the queue behind it forever.
        """
        with self._tx() as c:
            cur = c.execute(
                "UPDATE suggestions SET resolved = 2, outcome = 'expired' "
                "WHERE id = ? AND resolved = 0", (suggestion_id,))
        return cur.rowcount > 0

    def suggestion_stats(self, coin: str | None = None,
                         interval: str | None = None) -> dict[str, Any]:
        """How the suggestions did, and how the decisions about them did.

        The second half is the one worth reading. `taken` versus `ignored`
        compares the operator's filter against the tool: if the ignored
        suggestions are winning, the filter is subtracting value, and no
        amount of looking at an equity curve reveals that.
        """
        where = ["resolved = 1"]
        params: list[Any] = []
        if coin:
            where.append("coin = ?")
            params.append(coin)
        if interval:
            where.append("interval = ?")
            params.append(interval)
        clause = " AND ".join(where)

        rows = [dict(r) for r in self._conn.execute(
            f"SELECT * FROM suggestions WHERE {clause}", params).fetchall()]

        def summarise(subset: list[dict[str, Any]]) -> dict[str, Any]:
            if not subset:
                return {"n": 0, "hit_rate": None, "avg_pnl_bps": None,
                        "total_pnl_bps": None, "targets": 0, "stops": 0,
                        "open_at_close": 0}
            targets = sum(1 for r in subset if r["outcome"] == "target")
            stops = sum(1 for r in subset if r["outcome"] == "stop")
            pnls = [float(r["pnl_bps"] or 0.0) for r in subset]
            gross = [float(r["gross_bps"] or 0.0) for r in subset]
            decided = targets + stops
            return {
                "n": len(subset),
                # Hit rate counts only suggestions that reached a level. One
                # that expired mid-range was neither right nor wrong.
                "hit_rate": (targets / decided) if decided else None,
                # Net of cost. `gross` is kept alongside so the difference
                # between "the read was right" and "the trade made money" is
                # visible rather than hidden in one number.
                "avg_pnl_bps": sum(pnls) / len(pnls),
                "total_pnl_bps": sum(pnls),
                "avg_gross_bps": sum(gross) / len(gross),
                "cost_drag_bps": sum(gross) - sum(pnls),
                "targets": targets, "stops": stops,
                "open_at_close": len(subset) - decided,
            }

        taken = [r for r in rows if r["decision"] == "taken"]
        ignored = [r for r in rows if r["decision"] == "ignored"]
        undecided = [r for r in rows if r["decision"] == "pending"]

        overall = summarise(rows)
        t, i = summarise(taken), summarise(ignored)

        verdict = _decision_verdict(t, i, len(undecided))

        # Filtered the same way as everything else in this payload. An
        # unfiltered global count printed beside per-market figures reads as
        # a backlog on this market that is not there.
        pend_where = " AND ".join(["resolved = 0"] + where[1:])
        pending = int(self._conn.execute(
            f"SELECT COUNT(*) FROM suggestions WHERE {pend_where}",
            params).fetchone()[0])

        return {
            "overall": overall,
            "taken": t,
            "ignored": i,
            "undecided": summarise(undecided),
            "verdict": verdict,
            "pending": pending,
        }

    def open_suggestion(self, coin: str, interval: str, candle_ts: float
                        ) -> dict[str, Any] | None:
        row = self._conn.execute(
            """SELECT * FROM suggestions
               WHERE coin=? AND interval=? AND candle_ts=?""",
            (coin, interval, candle_ts)).fetchone()
        return dict(row) if row else None

    def recent_suggestions(self, coin: str | None = None, limit: int = 50
                           ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM suggestions"
        params: list[Any] = []
        if coin:
            sql += " WHERE coin = ?"
            params.append(coin)
        sql += " ORDER BY ts DESC LIMIT ?"
        params.append(limit)
        return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    # -- agreement states, on every candle ---------------------------------

    def record_state(self, coin: str, interval: str, candle_ts: float,
                     candle_end: float, book_dir: str, book_strength: float,
                     price_dir: str, price_strength: float, verdict: str,
                     price: float, elapsed_frac: float = 0.0,
                     made_at: float | None = None,
                     features: dict[str, Any] | None = None) -> str | None:
        """One row per candle per market, tradeable or not."""
        import json as _json
        import time as _time

        if price <= 0:
            return None
        sid = uuid.uuid4().hex[:16]
        try:
            with self._tx() as c:
                c.execute(
                    """INSERT INTO agreement_states (
                           id, ts, coin, interval, candle_ts, candle_end,
                           made_at, elapsed_frac, book_dir, book_strength,
                           price_dir, price_strength, verdict, price,
                           features)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (sid, _now(), coin, interval, candle_ts, candle_end,
                     made_at if made_at is not None else _time.time(),
                     elapsed_frac, book_dir, book_strength, price_dir,
                     price_strength, verdict, price,
                     _json.dumps(features) if features else None))
        except sqlite3.IntegrityError:
            return None            # this candle already has its read
        return sid

    def pending_states(self, now: float, limit: int = 500
                       ) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """SELECT * FROM agreement_states
               WHERE resolved = 0 AND candle_end <= ?
               ORDER BY candle_end LIMIT ?""", (now, limit)).fetchall()
        return [dict(r) for r in rows]

    def resolve_state(self, state_id: str, close_px: float, high_px: float,
                      low_px: float) -> bool:
        """Settle one read against what the candle actually did.

        Three numbers, because the average move alone hides the shape of it.
        A state that drifts to +5bps having first gone to +40 and back is a
        different animal from one that walks quietly to +5, and only the
        excursions tell them apart.
        """
        row = self._conn.execute(
            "SELECT price FROM agreement_states WHERE id = ?",
            (state_id,)).fetchone()
        if row is None:
            return False
        entry = float(row["price"] or 0)
        if entry <= 0 or close_px <= 0 or high_px < low_px:
            return False

        move = (close_px - entry) / entry * 10_000.0
        mfe = (high_px - entry) / entry * 10_000.0
        mae = (low_px - entry) / entry * 10_000.0

        with self._tx() as c:
            c.execute(
                """UPDATE agreement_states SET resolved = 1, close_px = ?,
                       move_bps = ?, mfe_bps = ?, mae_bps = ?
                   WHERE id = ? AND resolved = 0""",
                (close_px, move, mfe, mae, state_id))
        return True

    def feature_slice(self, coin: str | None = None,
                      interval: str | None = None,
                      feature: str = "tilt", edges: Sequence[float] = (),
                      verdict: str = "confirmed") -> dict[str, Any]:
        """How the outcome varies with one stored reading.

        This is the tuning surface. Ask "what does the confirmed state look
        like when microprice tilt is above 0.5 versus below it" and the
        answer comes from candles already recorded, rather than from three
        more weeks of waiting.

        Buckets are half-open [lo, hi), so a value never lands in two.
        """
        import json as _json

        where = ["resolved = 1", "move_bps IS NOT NULL", "features IS NOT NULL"]
        params: list[Any] = []
        if coin:
            where.append("coin = ?")
            params.append(coin)
        if interval:
            where.append("interval = ?")
            params.append(interval)
        if verdict:
            where.append("verdict = ?")
            params.append(verdict)

        rows = [dict(r) for r in self._conn.execute(
            f"SELECT * FROM agreement_states WHERE {' AND '.join(where)}",
            params).fetchall()]

        cuts = list(edges) or [-1.0, -0.5, -0.2, 0.2, 0.5, 1.01]
        buckets: list[dict[str, Any]] = []

        for lo, hi in zip(cuts, cuts[1:]):
            called, hits = [], 0
            for r in rows:
                try:
                    feats = _json.loads(r["features"] or "{}")
                except (ValueError, TypeError):
                    continue
                v = feats.get(feature)
                if v is None or not (lo <= float(v) < hi):
                    continue
                s = 1 if r["book_dir"] == "up" else -1 if r["book_dir"] == "down" else 0
                if not s:
                    continue
                m = float(r["move_bps"] or 0.0) * s
                called.append(m)
                if m > 0:
                    hits += 1

            buckets.append({
                "from": lo, "to": hi, "n": len(called),
                "avg_called_bps": (sum(called) / len(called)) if called else None,
                "hit_rate": (hits / len(called)) if called else None,
            })

        return {"feature": feature, "verdict": verdict, "n": len(rows),
                "buckets": buckets,
                "note": ("Each bucket is the average move in the direction "
                         "the book called, for candles whose "
                         f"{feature} fell in that range. A feature that "
                         "matters shows a rising line; one that does not "
                         "shows noise.")}

    def stored_features(self, limit: int = 200) -> list[str]:
        """Which readings are available to slice on."""
        import json as _json

        rows = self._conn.execute(
            "SELECT features FROM agreement_states "
            "WHERE features IS NOT NULL LIMIT ?", (limit,)).fetchall()
        names: set[str] = set()
        for r in rows:
            try:
                names.update(_json.loads(r["features"] or "{}").keys())
            except (ValueError, TypeError):
                continue
        return sorted(names)

    def agreement_table(self, coin: str | None = None,
                        interval: str | None = None,
                        min_strength: float = 0.0) -> dict[str, Any]:
        """What actually happens in each state.

        The headline is `avg_called_bps`: the average move IN THE DIRECTION
        THE STATE CALLED. Averaging raw signed moves across up-calls and
        down-calls would cancel a perfectly good signal to zero, which is
        the easiest way to conclude a working rule does nothing.

        On conflicts the two sides are scored separately, because "which one
        is right when they disagree" is the question that decides whether
        standing aside is correct or whether one side should simply be
        followed.
        """
        where = ["resolved = 1", "move_bps IS NOT NULL"]
        params: list[Any] = []
        if coin:
            where.append("coin = ?")
            params.append(coin)
        if interval:
            where.append("interval = ?")
            params.append(interval)
        if min_strength > 0:
            where.append("(book_strength >= ? OR price_strength >= ?)")
            params.extend([min_strength, min_strength])

        rows = [dict(r) for r in self._conn.execute(
            f"SELECT * FROM agreement_states WHERE {' AND '.join(where)}",
            params).fetchall()]

        total = len(rows)

        def sign(d: str) -> int:
            return 1 if d == "up" else -1 if d == "down" else 0

        def summarise(subset: list[dict[str, Any]], called: str) -> dict[str, Any]:
            """`called` picks which direction counts as 'the call'."""
            if not subset:
                return {"n": 0, "share": 0.0, "avg_called_bps": None,
                        "median_called_bps": None, "hit_rate": None,
                        "book_right": None, "price_right": None,
                        "avg_mfe_bps": None, "avg_mae_bps": None}

            called_moves, favourable, adverse = [], [], []
            book_hits = price_hits = scored = 0

            for r in subset:
                s = sign(r.get(called) or "flat")
                move = float(r["move_bps"] or 0.0)
                if s:
                    called_moves.append(move * s)
                    # Excursions flip with the call, so a short's favourable
                    # excursion is the low, not the high.
                    up, down = float(r["mfe_bps"] or 0.0), float(r["mae_bps"] or 0.0)
                    favourable.append(up if s > 0 else -down)
                    adverse.append(down if s > 0 else -up)

                bs, ps = sign(r["book_dir"]), sign(r["price_dir"])
                if move != 0 and (bs or ps):
                    scored += 1
                    if bs and (move > 0) == (bs > 0):
                        book_hits += 1
                    if ps and (move > 0) == (ps > 0):
                        price_hits += 1

            def mid(xs):
                if not xs:
                    return None
                ys = sorted(xs)
                n = len(ys)
                return ys[n // 2] if n % 2 else (ys[n // 2 - 1] + ys[n // 2]) / 2

            return {
                "n": len(subset),
                "share": len(subset) / total if total else 0.0,
                "avg_called_bps": (sum(called_moves) / len(called_moves)
                                   if called_moves else None),
                "median_called_bps": mid(called_moves),
                "hit_rate": (sum(1 for m in called_moves if m > 0)
                             / len(called_moves)) if called_moves else None,
                "book_right": book_hits / scored if scored else None,
                "price_right": price_hits / scored if scored else None,
                "avg_mfe_bps": (sum(favourable) / len(favourable)
                                if favourable else None),
                "avg_mae_bps": (sum(adverse) / len(adverse)
                                if adverse else None),
            }

        def pick(**kw) -> list[dict[str, Any]]:
            return [r for r in rows
                    if all(r.get(k) == v for k, v in kw.items())]

        confirmed = pick(verdict="confirmed")
        conflict = pick(verdict="conflict")

        table = [
            {"state": "CONFIRMED up", "detail": "book up, price up",
             **summarise(pick(verdict="confirmed", book_dir="up"), "book_dir")},
            {"state": "CONFIRMED down", "detail": "book down, price down",
             **summarise(pick(verdict="confirmed", book_dir="down"), "book_dir")},
            {"state": "CONFIRMED (both)", "detail": "either direction",
             **summarise(confirmed, "book_dir")},
            {"state": "CONFLICT, book up", "detail": "book up, price down",
             **summarise(pick(verdict="conflict", book_dir="up"), "book_dir")},
            {"state": "CONFLICT, book down", "detail": "book down, price up",
             **summarise(pick(verdict="conflict", book_dir="down"), "book_dir")},
            {"state": "CONFLICT (both)", "detail": "they disagree",
             **summarise(conflict, "book_dir")},
            {"state": "UNCONFIRMED", "detail": "book leans, price flat",
             **summarise(pick(verdict="unconfirmed"), "book_dir")},
            {"state": "NO SIGNAL", "detail": "book balanced",
             **summarise(pick(verdict="no signal"), "price_dir")},
        ]

        pending = int(self._conn.execute(
            "SELECT COUNT(*) FROM agreement_states WHERE resolved = 0"
        ).fetchone()[0])

        return {
            "n": total, "pending": pending, "table": table,
            "verdict": _agreement_verdict(table, conflict),
            "source": "live order book plus live tape — recorded as it happened",
        }

    # -- uploaded chart history --------------------------------------------

    def save_dataset(self, name: str, coin: str, interval: str,
                     interval_s: float, candles: Sequence[Any],
                     report: str = "", merge_into: str | None = None) -> str:
        """Store uploaded bars. Returns the dataset id.

        With `merge_into`, an overlapping upload EXTENDS an existing dataset
        rather than creating a second copy of the overlap -- uploading March
        and then February-to-April should leave one continuous series, not
        two that double-count the middle.
        """
        import json as _json

        from .ingest import merge as _merge

        rows = list(candles)
        did = merge_into

        if merge_into:
            existing = self.dataset(merge_into)
            if existing is None:
                did = None
            elif (existing["coin"] != coin
                  or existing["interval"] != interval):
                # Merging a 1m ETH export into a 1h BTC dataset produces one
                # series that is neither, and the replay would then label the
                # result with whichever metadata happened to survive.
                raise ValueError(
                    f"cannot merge {coin} {interval} into a dataset holding "
                    f"{existing['coin']} {existing['interval']} — upload it "
                    f"as a new dataset instead")
            else:
                rows = _merge(existing["candles"], rows)

        if not rows:
            raise ValueError("refusing to store a dataset with no bars")

        payload = _json.dumps([[c.ts, c.open, c.high, c.low, c.close, c.volume]
                               for c in rows])
        if did is None:
            did = uuid.uuid4().hex[:16]

        with self._tx() as c:
            c.execute(
                """INSERT INTO datasets (id, ts, name, coin, interval,
                       interval_s, bars, start_ts, end_ts, report, candles)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                       ts = excluded.ts, name = excluded.name,
                       coin = excluded.coin, interval = excluded.interval,
                       interval_s = excluded.interval_s,
                       bars = excluded.bars,
                       start_ts = excluded.start_ts, end_ts = excluded.end_ts,
                       report = excluded.report, candles = excluded.candles""",
                (did, _now(), name, coin, interval, interval_s, len(rows),
                 rows[0].ts, rows[-1].ts, report, payload))
        return did

    def dataset(self, dataset_id: str) -> dict[str, Any] | None:
        import json as _json

        from .structure import Candle as _Candle

        row = self._conn.execute(
            "SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
        if row is None:
            return None
        out = dict(row)
        try:
            raw = _json.loads(out.pop("candles"))
        except (ValueError, TypeError):
            return None
        out["candles"] = [
            _Candle(ts=r[0], open=r[1], high=r[2], low=r[3], close=r[4],
                    volume=r[5] if len(r) > 5 else 0.0) for r in raw]
        return out

    def datasets(self, coin: str | None = None) -> list[dict[str, Any]]:
        """Dataset metadata, without the bars. The candle payload is
        deliberately excluded: a listing should not load megabytes."""
        sql = ("SELECT id, ts, name, coin, interval, interval_s, bars, "
               "start_ts, end_ts, report FROM datasets")
        params: list[Any] = []
        if coin:
            sql += " WHERE coin = ?"
            params.append(coin)
        sql += " ORDER BY ts DESC"
        return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def delete_dataset(self, dataset_id: str) -> bool:
        with self._tx() as c:
            cur = c.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
        return cur.rowcount > 0

    def coins_with_data(self) -> list[dict[str, Any]]:
        """Every coin that has at least one recorded sweep, newest first.

        The dashboard needs this to tell the difference between "that coin is
        quiet" and "that coin has never been swept", which look identical from
        an empty panel and have completely different fixes.
        """
        rows = self._conn.execute(
            """SELECT coin, COUNT(*) AS sweeps, MAX(ts) AS last_ts,
                      SUM(n_positions) AS positions
               FROM sweeps GROUP BY coin ORDER BY last_ts DESC""").fetchall()
        return [dict(r) for r in rows]

    def counts(self) -> dict[str, int]:
        def one(sql: str) -> int:
            return int(self._conn.execute(sql).fetchone()[0])
        return {
            "sweeps": one("SELECT COUNT(*) FROM sweeps"),
            "sweep_positions": one("SELECT COUNT(*) FROM sweep_positions"),
            "changes": one("SELECT COUNT(*) FROM position_changes"),
            "defended": one("SELECT COUNT(*) FROM position_changes WHERE kind='DEFENDED'"),
            "liquidated": one("SELECT COUNT(*) FROM position_changes WHERE kind='LIQUIDATED'"),
            "reads": one("SELECT COUNT(*) FROM reads"),
            "suggestions": one("SELECT COUNT(*) FROM suggestions"),
            "taken": one("SELECT COUNT(*) FROM suggestions WHERE decision='taken'"),
            "ignored": one("SELECT COUNT(*) FROM suggestions WHERE decision='ignored'"),
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


# How many settled trades before a difference between two hit rates means
# anything. Below this the gap is sampling noise, and reporting it as a
# finding is how someone talks themselves out of a filter that works.
MIN_FOR_COMPARISON = 20


def _agreement_verdict(table: Sequence[dict[str, Any]],
                       conflicts: Sequence[dict[str, Any]]) -> str:
    """Plain words on what the table is saying, or that it is too early.

    Refuses to draw a conclusion under `MIN_FOR_COMPARISON` in a row. The
    whole point of the table is to replace an impression with a measurement,
    and a measurement of twelve candles is another impression.
    """
    by_state = {r["state"]: r for r in table}
    both = by_state.get("CONFIRMED (both)", {})
    clash = by_state.get("CONFLICT (both)", {})

    lines = []
    n = both.get("n") or 0
    if n < MIN_FOR_COMPARISON:
        lines.append(f"Only {n} confirmed candles settled — too few to "
                     f"conclude anything. Needs {MIN_FOR_COMPARISON}+.")
    else:
        avg = both.get("avg_called_bps")
        hit = both.get("hit_rate")
        lines.append(
            f"When book and price agree, price goes {avg:+.1f}bps their way "
            f"on average and closes that way {hit:.0%} of the time "
            f"({n} candles).")

    cn = clash.get("n") or 0
    if cn < MIN_FOR_COMPARISON:
        lines.append(f"{cn} conflicts settled — not enough to say which side "
                     f"wins yet.")
    else:
        br, pr = clash.get("book_right"), clash.get("price_right")
        if br is not None and pr is not None:
            gap = (br - pr) * 100
            if gap > 5:
                lines.append(
                    f"When they disagree the BOOK is right more often "
                    f"({br:.0%} vs {pr:.0%} on {cn}) — worth testing whether "
                    f"following the book through a conflict beats standing "
                    f"aside.")
            elif gap < -5:
                lines.append(
                    f"When they disagree PRICE ACTION is right more often "
                    f"({pr:.0%} vs {br:.0%} on {cn}) — the book is being "
                    f"absorbed and price is telling the truth, which is the "
                    f"case the confirmation rule was built for.")
            else:
                lines.append(
                    f"When they disagree neither side wins ({br:.0%} book "
                    f"vs {pr:.0%} price on {cn}) — a coin flip, so standing "
                    f"aside is the right call.")
    return " ".join(lines)


def _decision_verdict(taken: dict[str, Any], ignored: dict[str, Any],
                      undecided: int) -> str:
    """Plain words on whether the filtering is helping.

    This refuses to draw a conclusion from a handful of trades. The whole
    point of the table is to replace an impression with a measurement, and a
    measurement of twelve trades is another impression.
    """
    tn, ig = taken["n"], ignored["n"]
    if tn + ig == 0:
        return ("Nothing settled yet. Take or ignore suggestions as they "
                "appear and this fills in on its own.")

    if tn < MIN_FOR_COMPARISON or ig < MIN_FOR_COMPARISON:
        return (f"{tn} taken, {ig} ignored, {undecided} never decided. "
                f"Needs {MIN_FOR_COMPARISON} of each before the comparison "
                f"means anything — right now the difference between them is "
                f"noise.")

    th, ih = taken["hit_rate"], ignored["hit_rate"]
    if th is None or ih is None:
        return f"{tn} taken, {ig} ignored, but too few reached a level to score."

    gap = (th - ih) * 100
    head = (f"Taken: {th:.0%} on {tn}. Ignored: {ih:.0%} on {ig}.")

    if gap > 5:
        return (f"{head} Your filter is adding {gap:.0f} points — you are "
                f"passing on the worse ones.")
    if gap < -5:
        return (f"{head} The ones you passed on did {abs(gap):.0f} points "
                f"BETTER. Whatever you are filtering on is costing you; "
                f"worth finding out what the skipped ones had in common.")
    return (f"{head} No real difference — your filter is neither helping nor "
            f"hurting, so the tool's own selectivity is doing the work.")


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
