"""The trade journal for the four-must-agree rule.

WHAT THIS IS, AND WHAT IT IS NOT.

It is a record of trades Chris takes, and a score of how they went in
ticks and in dollars. It opens and closes rows in a table.

IT DOES NOT PLACE ORDERS. There is no broker in this module, no
credential, no network call, and no function that could be mistaken
for one. "Auto" here means the journal writes its own rows when the
four line up, so a signal does not have to be clicked to be recorded;
the order at Ironbeam is still Chris's to place by hand. A test greps
this file for the shapes an execution path would take, and the point
of that test is to fail loudly if anyone -- including me -- ever
starts down that road.

WHAT A TICK IS WORTH. ES moves in quarter-points and a tick is
$12.50, so the dollar column is ticks times 12.5 and nothing else. It
is not a P&L: commission is not in it, and the price recorded is the
live mid at the moment the button was pressed, which is not a price
anybody is filled at. Crossing the spread costs about a tick a round
trip, so a journal averaging under a tick a trade is describing a
loss. The panel says that out loud rather than leaving it to be
remembered.

THE ROWS SURVIVE A RESTART, because a trade log that forgets is worse
than no trade log -- it looks complete and is not.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .db import ThreadedDB

TICK = 0.25
TICK_VALUE = 12.50
# What a round trip costs in ticks, near enough, when you cross the
# spread on both sides. Used ONLY to tell the reader whether the
# numbers above it mean anything -- never subtracted from them.
ROUND_TRIP_TICKS = 1.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS journal_trades (
    id         INTEGER PRIMARY KEY,
    coin       TEXT    NOT NULL,
    side       TEXT    NOT NULL,
    mode       TEXT    NOT NULL DEFAULT 'manual',
    opened_ts  REAL    NOT NULL,
    open_px    REAL    NOT NULL,
    closed_ts  REAL,
    close_px   REAL,
    why        TEXT    NOT NULL DEFAULT '',
    tp         REAL    NOT NULL DEFAULT 0,
    sl         REAL    NOT NULL DEFAULT 0,
    note       TEXT    NOT NULL DEFAULT '',
    -- WHICH OFI CANDLE THIS TRADE BELONGS TO. The cap is two per
    -- candle and one per direction, and a cap held only in memory
    -- resets on every restart -- which on a poll-driven column is
    -- indistinguishable from no cap at all.
    bar_ts     REAL    NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS journal_open
    ON journal_trades (coin, closed_ts);

CREATE TABLE IF NOT EXISTS journal_settings (
    coin       TEXT PRIMARY KEY,
    auto       INTEGER NOT NULL DEFAULT 0,
    tp_on      INTEGER NOT NULL DEFAULT 0,
    sl_on      INTEGER NOT NULL DEFAULT 0,
    tp_ticks   REAL    NOT NULL DEFAULT 16,
    sl_ticks   REAL    NOT NULL DEFAULT 8
);
"""


@dataclass
class Settings:
    coin: str = "ES"
    auto: bool = False
    tp_on: bool = False
    sl_on: bool = False
    tp_ticks: float = 16.0
    sl_ticks: float = 8.0

    def to_dict(self) -> dict:
        return {"coin": self.coin, "auto": self.auto,
                "tp_on": self.tp_on, "sl_on": self.sl_on,
                "tp_ticks": self.tp_ticks, "sl_ticks": self.sl_ticks,
                "tick_value": TICK_VALUE, "tick": TICK}


@dataclass
class Trade:
    id: int = 0
    coin: str = "ES"
    side: str = "up"
    mode: str = "manual"
    opened_ts: float = 0.0
    open_px: float = 0.0
    closed_ts: float | None = None
    close_px: float | None = None
    why: str = ""
    tp: float = 0.0
    sl: float = 0.0
    note: str = ""
    bar_ts: float = 0.0

    @property
    def open(self) -> bool:
        return self.closed_ts is None

    def ticks_at(self, px: float) -> float:
        """Signed ticks from the open, in this trade's direction."""
        if not px or not self.open_px:
            return 0.0
        d = (px - self.open_px) / TICK
        return d if self.side == "up" else -d

    @property
    def ticks(self) -> float:
        if self.close_px is None:
            return 0.0
        return round(self.ticks_at(float(self.close_px)), 2)

    @property
    def dollars(self) -> float:
        return round(self.ticks * TICK_VALUE, 2)

    @property
    def won(self) -> bool:
        return self.ticks > 0

    def to_dict(self, live_px: float = 0.0) -> dict:
        d = {"id": self.id, "coin": self.coin, "side": self.side,
             "mode": self.mode, "opened_ts": self.opened_ts,
             "open_px": self.open_px, "closed_ts": self.closed_ts,
             "close_px": self.close_px, "why": self.why,
             "tp": self.tp, "sl": self.sl, "note": self.note,
             "open": self.open}
        if self.open:
            # UNREALISED, AND LABELLED AS SUCH. An open trade showing a
            # number in the same column as the closed ones is how a
            # journal flatters itself: the figure is whatever the last
            # print happened to be and it is not money.
            t = round(self.ticks_at(live_px), 2) if live_px else 0.0
            d["live_ticks"] = t
            d["live_dollars"] = round(t * TICK_VALUE, 2)
        else:
            d["ticks"] = self.ticks
            d["dollars"] = self.dollars
            d["won"] = self.won
            d["held_s"] = round(max(0.0, (self.closed_ts or 0)
                                    - self.opened_ts), 1)
        return d


@dataclass
class Stats:
    trades: list = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.trades)

    @property
    def wins(self) -> int:
        return sum(1 for t in self.trades if t.ticks > 0)

    @property
    def losses(self) -> int:
        return sum(1 for t in self.trades if t.ticks < 0)

    @property
    def flat(self) -> int:
        return sum(1 for t in self.trades if t.ticks == 0)

    @property
    def ticks(self) -> float:
        return round(sum(t.ticks for t in self.trades), 2)

    @property
    def dollars(self) -> float:
        return round(self.ticks * TICK_VALUE, 2)

    @property
    def rate(self) -> float:
        return (self.wins / self.n) if self.n else 0.0

    @property
    def won_ticks(self) -> float:
        return round(sum(t.ticks for t in self.trades if t.ticks > 0), 2)

    @property
    def lost_ticks(self) -> float:
        return round(sum(t.ticks for t in self.trades if t.ticks < 0), 2)

    @property
    def spread_ticks(self) -> float:
        """How much the trades scatter, as a standard deviation.

        Needed because the average on its own cannot say whether it
        means anything. Twenty trades averaging +0.4 ticks with a
        spread of 8 is indistinguishable from a coin.
        """
        if self.n < 2:
            return 0.0
        m = self.ticks / self.n
        var = sum((t.ticks - m) ** 2 for t in self.trades) / (self.n - 1)
        return var ** 0.5

    def verdict(self) -> dict:
        """Whether this record is long enough to mean anything yet.

        THE QUESTION CHRIS IS ACTUALLY ASKING: let it log trades, then
        decide whether to trust it. The totals alone answer that
        badly, because the thing that fools you at small sample size
        is a run of wins -- and a run of wins is exactly what a
        promising start looks like.

        So this reports the average per trade against its own standard
        ERROR, and against the cost of trading. `t` is how many
        standard errors the average sits above a round trip: under 2
        the record cannot tell this rule from a coin that happens to
        be ahead. It is a rough normal approximation and nothing more
        -- trades are not independent, a trending session correlates
        them, so 2 here is weaker evidence than 2 would be in a
        textbook. It is a floor to clear, not a result.
        """
        if self.n < 2:
            return {"enough": False, "t": 0.0, "se": 0.0,
                    "need": max(0, 30 - self.n),
                    "say": f"{self.n} trade"
                           f"{'' if self.n == 1 else 's'} so far. "
                           "Nothing can be read from this yet."}
        m = self.ticks / self.n
        se = self.spread_ticks / (self.n ** 0.5)
        t = ((m - ROUND_TRIP_TICKS) / se) if se > 0 else 0.0
        enough = self.n >= 30 and t >= 2.0
        if self.n < 30:
            say = (f"{self.n} trades. Under thirty, a good run and a "
                   "real edge look the same -- keep logging.")
        elif t >= 2.0:
            say = (f"{self.n} trades averaging {m:+.2f} ticks, which "
                   f"is {t:.1f} standard errors above what a round "
                   "trip costs. That is a floor cleared, not a result: "
                   "trades in one session are not independent, so this "
                   "is weaker than the number looks. Check it against "
                   "a stretch it has not seen.")
        elif m <= ROUND_TRIP_TICKS:
            say = (f"{self.n} trades averaging {m:+.2f} ticks, which "
                   "does not cover the ~1 tick a round trip costs. On "
                   "this record the rule is not paying for itself.")
        else:
            say = (f"{self.n} trades averaging {m:+.2f} ticks, but it "
                   f"scatters {self.spread_ticks:.1f} ticks either "
                   "way -- that average is within noise of the cost. "
                   "Not yet distinguishable from a coin.")
        return {"enough": enough, "t": round(t, 2),
                "se": round(se, 3), "need": max(0, 30 - self.n),
                "say": say}

    def streaks(self) -> dict:
        """Longest run of wins and of losses, oldest first.

        A record is read in order or not at all: ten wins then ten
        losses and an alternating twenty have the same totals and are
        not the same thing.
        """
        best = {"win": 0, "loss": 0}
        run, kind = 0, ""
        for t in reversed(self.trades):          # `closed` is newest-first
            k = "win" if t.ticks > 0 else "loss" if t.ticks < 0 else ""
            if k and k == kind:
                run += 1
            else:
                kind, run = k, 1 if k else 0
            if k:
                best[k] = max(best[k], run)
        return best

    def to_dict(self) -> dict:
        per = (self.ticks / self.n) if self.n else 0.0
        return {
            "trades": self.n, "wins": self.wins, "losses": self.losses,
            "flat": self.flat, "rate": round(self.rate, 4),
            "ticks": self.ticks, "dollars": self.dollars,
            "won_ticks": self.won_ticks, "lost_ticks": self.lost_ticks,
            "won_dollars": round(self.won_ticks * TICK_VALUE, 2),
            "lost_dollars": round(self.lost_ticks * TICK_VALUE, 2),
            "per_trade_ticks": round(per, 3),
            "per_trade_dollars": round(per * TICK_VALUE, 2),
            # NOT SUBTRACTED, just stated. Whether these numbers
            # survive costs is the first question to ask of them.
            "round_trip_ticks": ROUND_TRIP_TICKS,
            "beats_costs": bool(self.n and per > ROUND_TRIP_TICKS),
            "avg_win_ticks": round(self.won_ticks / self.wins, 2)
                             if self.wins else 0.0,
            "avg_loss_ticks": round(self.lost_ticks / self.losses, 2)
                              if self.losses else 0.0,
            "spread_ticks": round(self.spread_ticks, 2),
            "verdict": self.verdict(),
            "streaks": self.streaks(),
        }


def _trade_of(r: sqlite3.Row) -> Trade:
    return Trade(id=int(r["id"]), coin=r["coin"], side=r["side"],
                 mode=r["mode"], opened_ts=float(r["opened_ts"]),
                 open_px=float(r["open_px"]),
                 closed_ts=(float(r["closed_ts"])
                            if r["closed_ts"] is not None else None),
                 close_px=(float(r["close_px"])
                           if r["close_px"] is not None else None),
                 why=r["why"] or "", tp=float(r["tp"]),
                 sl=float(r["sl"]), note=r["note"] or "",
                 bar_ts=float(r["bar_ts"] or 0.0))


# Columns added after the table first shipped. CREATE TABLE IF NOT
# EXISTS does NOT alter a table that already exists, so a database
# written by an earlier build keeps its old shape and every insert
# fails with "no column named ...". Chris hit exactly that on a
# deploy: the trade button returned OperationalError and nothing
# could be logged at all.
_ADDED = (
    ("bar_ts", "REAL NOT NULL DEFAULT 0"),
)


class Journal(ThreadedDB):
    """Trades and their settings. One row open per market at a time."""

    def __init__(self, path: str | Path = "liqmap.db"):
        super().__init__(path, SCHEMA)
        self._lock = threading.Lock()

    def _bootstrap(self, conn: sqlite3.Connection) -> None:
        """Create the tables, then add any column a later build grew.

        Rows written before a column existed keep their rows and read
        back with its default, which for `bar_ts` is zero -- and zero
        means "no candle recorded", which is what those trades
        honestly are. Nothing is rewritten: a logged trade is a fact,
        and inventing a candle id for one that never had one would
        put it into a cap it was never subject to.
        """
        super()._bootstrap(conn)
        have = {r[1] for r in
                conn.execute("PRAGMA table_info(journal_trades)")}
        for name, decl in _ADDED:
            if name not in have:
                conn.execute(
                    f"ALTER TABLE journal_trades ADD COLUMN {name} {decl}")
        conn.commit()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # -- settings --------------------------------------------------------

    def settings(self, coin: str = "ES") -> Settings:
        r = self._conn.execute(
            "SELECT * FROM journal_settings WHERE coin=?",
            (str(coin),)).fetchone()
        if r is None:
            return Settings(coin=str(coin))
        return Settings(coin=r["coin"], auto=bool(r["auto"]),
                        tp_on=bool(r["tp_on"]), sl_on=bool(r["sl_on"]),
                        tp_ticks=float(r["tp_ticks"]),
                        sl_ticks=float(r["sl_ticks"]))

    def save_settings(self, s: Settings) -> Settings:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO journal_settings"
                " (coin, auto, tp_on, sl_on, tp_ticks, sl_ticks)"
                " VALUES (?,?,?,?,?,?)"
                " ON CONFLICT(coin) DO UPDATE SET"
                " auto=excluded.auto, tp_on=excluded.tp_on,"
                " sl_on=excluded.sl_on, tp_ticks=excluded.tp_ticks,"
                " sl_ticks=excluded.sl_ticks",
                (s.coin, int(s.auto), int(s.tp_on), int(s.sl_on),
                 max(0.0, float(s.tp_ticks)),
                 max(0.0, float(s.sl_ticks))))
        return self.settings(s.coin)

    # -- trades ----------------------------------------------------------

    def open_trade(self, coin: str, side: str, px: float,
                   mode: str = "manual", ts: float | None = None,
                   tp: float = 0.0, sl: float = 0.0,
                   bar_ts: float = 0.0) -> Trade | None:
        """Open one. Returns None when one is already open.

        ONE AT A TIME, enforced here rather than in the page. A double
        click, a retried request and the auto rule firing on the same
        poll are three ways to end up with two rows for one position,
        and a journal that can hold two is a journal whose total is
        wrong in a way nobody will spot.
        """
        side = "up" if str(side).lower() in ("up", "buy", "long") \
            else "down"
        if not px or px <= 0:
            return None
        with self._lock:
            if self.open_trade_for(coin) is not None:
                return None
            with self._tx() as conn:
                cur = conn.execute(
                    "INSERT INTO journal_trades"
                    " (coin, side, mode, opened_ts, open_px, tp, sl,"
                    "  bar_ts) VALUES (?,?,?,?,?,?,?,?)",
                    (str(coin), side, str(mode),
                     float(ts if ts is not None else time.time()),
                     float(px), float(tp), float(sl), float(bar_ts)))
                new_id = int(cur.lastrowid)
        return self.get(new_id)

    def close_trade(self, coin: str, px: float, why: str = "manual",
                    ts: float | None = None) -> Trade | None:
        if not px or px <= 0:
            return None
        with self._lock:
            t = self.open_trade_for(coin)
            if t is None:
                return None
            with self._tx() as conn:
                conn.execute(
                    "UPDATE journal_trades SET closed_ts=?, close_px=?,"
                    " why=? WHERE id=?",
                    (float(ts if ts is not None else time.time()),
                     float(px), str(why), t.id))
        return self.get(t.id)

    def open_trade_for(self, coin: str) -> Trade | None:
        r = self._conn.execute(
            "SELECT * FROM journal_trades WHERE coin=? AND closed_ts IS"
            " NULL ORDER BY id DESC LIMIT 1", (str(coin),)).fetchone()
        return _trade_of(r) if r else None

    def get(self, trade_id: int) -> Trade | None:
        r = self._conn.execute(
            "SELECT * FROM journal_trades WHERE id=?",
            (int(trade_id),)).fetchone()
        return _trade_of(r) if r else None

    def sides_in_bar(self, coin: str, bar_ts: float) -> tuple:
        """Which directions have already been opened in this candle.

        FROM THE TABLE, NOT FROM MEMORY. The column is driven by a
        poll and the process restarts; a cap counted in a variable
        would reset on both, which is indistinguishable from having
        no cap.
        """
        if not bar_ts:
            return ()
        rows = self._conn.execute(
            "SELECT side FROM journal_trades WHERE coin=? AND bar_ts=?"
            " ORDER BY id", (str(coin), float(bar_ts))).fetchall()
        return tuple(r["side"] for r in rows)

    def closed(self, coin: str, limit: int = 200) -> list:
        rows = self._conn.execute(
            "SELECT * FROM journal_trades WHERE coin=? AND closed_ts IS"
            " NOT NULL ORDER BY id DESC LIMIT ?",
            (str(coin), max(1, int(limit)))).fetchall()
        return [_trade_of(r) for r in rows]

    def stats(self, coin: str, limit: int = 1000) -> Stats:
        return Stats(trades=self.closed(coin, limit))

    def delete(self, trade_id: int) -> bool:
        with self._tx() as conn:
            cur = conn.execute("DELETE FROM journal_trades WHERE id=?",
                               (int(trade_id),))
            return bool(cur.rowcount)

    def clear(self, coin: str) -> int:
        """Wipe this market's trades. Only ever on an explicit ask."""
        with self._tx() as conn:
            cur = conn.execute("DELETE FROM journal_trades WHERE coin=?",
                               (str(coin),))
            return int(cur.rowcount or 0)


def should_close(t: Trade, s: Settings, px: float,
                 agree: str) -> str:
    """Why this open trade should end now, or "" to hold it.

    LOSING AGREEMENT IS NOT AN EXIT. ONLY A FLIP IS.

    It used to exit the moment the four stopped agreeing, and Chris
    watched it close for -1 tick over and over: the read is price
    against the OFI bar open and it crosses that open constantly, so
    agreement breaks for an instant, the trade closes, and the next
    look re-opens it. Each of those round trips gives back about a
    tick, and none of them was a decision about the market -- the
    exit was firing on noise.

    So a trade is held through anything that is not an opposite
    signal. Neutral holds. One of the four going quiet holds. Two
    against two holds. Only three of the four pointing the OTHER way
    ends it, and then the position turns rather than flattening.

    The stop is still checked before the target, because a move
    through both is recorded as stopped out -- the pessimistic
    reading, and the one that does not assume a fill.
    """
    if not t.open or not px:
        return ""
    ticks = t.ticks_at(px)
    if s.sl_on and s.sl_ticks > 0 and ticks <= -s.sl_ticks:
        return "sl"
    if s.tp_on and s.tp_ticks > 0 and ticks >= s.tp_ticks:
        return "tp"
    other = "down" if t.side == "up" else "up"
    if agree == other:
        return "flip"
    return ""


def decide(t: Trade | None, s: Settings, px: float,
           agree: str, taken: tuple = ()) -> tuple[str, str]:
    """What the journal should do right now.

    ("open", side), ("close", why), or ("", reason) -- and the reason
    is never blank when nothing happens. A column that goes quiet
    without saying why is the failure that has cost the most time on
    this panel: it reads as a market with nothing in it, and it has
    meant a missing read, a stale cap and a switch left off.

    THE POLICY LIVES HERE, NOT IN THE ROUTE. It was in a route
    closure, where the only way to reach it was an HTTP request
    through a fixture that never produced agreement -- so a test that
    auto-off opens nothing passed while the check was deleted. A
    policy that cannot be called directly cannot be tested directly.

    AUTO ONLY EVER OPENS. Closing on a target or a stop happens
    whether auto is on or not, because those are settings Chris
    turned on: a journal that ignored them until he refreshed would
    be scoring trades he had already said he was out of. Closing on a
    FLIP is for auto rows only -- a trade he took by hand is his to
    end.

    RE-ENTRY IS IMMEDIATE AND THAT HAS A CONSEQUENCE. His rule: open
    whenever three of the four agree. After a target, the signal that
    opened the trade is usually still there, so the next look opens it
    again at about the price it just closed. The two pieces sum to the
    one move, which means the take profit changes the win RATE and not
    the total. Measured on the backtest before this was written:
    4/2, 2/2 and no levels at all came out at +216.00 to the cent.
    It is his call, it is stated here, and the column counts the
    re-entries so it can be seen happening rather than inferred.
    """
    if not px or px <= 0:
        return "", "no live price -- the tape has not printed"
    if t is not None and t.open:
        why = should_close(t, s, px, agree)
        if why == "flip" and t.mode != "auto":
            why = ""
        return ("close", why) if why else ("", "a trade is already open")
    if not s.auto:
        return "", "AUTO is off -- the button says MANUAL"
    if agree not in ("up", "down"):
        return "", "no signal"
    # ONE TRADE PER DIRECTION, TWO PER CANDLE AT MOST.
    #
    # Chris: "its trading the same ofi candle over and over again it
    # gets out of the trade and immediately gets back in". Holding to
    # a flip stopped the exit churning; it did not stop the entry.
    # With re-entry immediate, one candle could be traded the same
    # way again and again -- and each of those is the same read, the
    # same bar and the same idea, charged a round trip every time.
    #
    # `taken` is the sides already opened in THIS candle.
    if agree in taken:
        return "", f"this candle has already taken its {agree}"
    # THE TWO RULES COINCIDE AT TWO DIRECTIONS, so this one never
    # fires while the one above stands -- deleting it broke no test
    # until one was written that reaches it directly. Kept because
    # the cap is the stated rule and the per-direction rule is the
    # one most likely to be relaxed later.
    if len(taken) >= MAX_PER_BAR:
        return "", "this candle has had its two trades"
    return "open", agree


STACK = ("15m", "5m", "1m")


def signal(says: dict, read: str, delta: str = "") -> tuple[str, str]:
    """The direction to trade, and what stopped it.

    Returns (side, blocked). `blocked` is "" when nothing stopped it,
    otherwise "imbalance" or "stack", and the side it would have
    taken is recoverable from `read`.

    THE OFI AND THE DELTA MUST AGREE. Chris: "take trades when the
    delta and OFI agree". The OFI is the book leaning -- the
    accumulator that decides whether this candle closes green or red.
    The delta is size that actually printed, by the side CME stated,
    and cannot be pulled back. They are genuinely different
    measurements, which is the only reason agreeing with each other
    is worth anything.

    `read` here is the OFI side, which is the candle's colour: green
    is up, red is down. It is NOT price against the bar's open --
    that was measured and called the read for a while, and it is a
    different fact that disagrees often.

    THE STACK IS STILL A VETO AND ONLY A VETO, and only when all
    three point the other way. Two against is not a veto; a silent
    row is not opposed.

    FLAT IS NOT A DIRECTION, on either of them.
    """
    side = read if read in ("up", "down") else ""
    if not side:
        return "", ""
    if delta != side:
        # Includes a flat or missing delta: nothing has printed, so
        # there is nothing for the candle to agree WITH.
        return "", "delta"
    other = "down" if side == "up" else "up"
    if all(says.get(t) == other for t in STACK):
        return "", "stack"
    return side, ""


# ONE TRADE PER OFI CANDLE. Chris: "we only ever use the current
# candle and thats all taking one trade per OFI candle". It was two;
# one is what he asked for after watching it run.
MAX_PER_BAR = 1
