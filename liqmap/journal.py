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
    note       TEXT    NOT NULL DEFAULT ''
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
                 sl=float(r["sl"]), note=r["note"] or "")


class Journal(ThreadedDB):
    """Trades and their settings. One row open per market at a time."""

    def __init__(self, path: str | Path = "liqmap.db"):
        super().__init__(path, SCHEMA)
        self._lock = threading.Lock()

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
                   tp: float = 0.0, sl: float = 0.0) -> Trade | None:
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
                    " (coin, side, mode, opened_ts, open_px, tp, sl)"
                    " VALUES (?,?,?,?,?,?,?)",
                    (str(coin), side, str(mode),
                     float(ts if ts is not None else time.time()),
                     float(px), float(tp), float(sl)))
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

    THE ORDER IS THE ANSWER TO A QUESTION CHRIS ALREADY SETTLED:
    whichever comes first. The stop is checked before the target
    because a bar that traded through both is recorded as stopped out,
    which is the pessimistic reading and the right one -- the
    alternative assumes a fill that may not have happened, and
    assuming a fill is how a journal invents money.
    """
    if not t.open or not px:
        return ""
    ticks = t.ticks_at(px)
    if s.sl_on and s.sl_ticks > 0 and ticks <= -s.sl_ticks:
        return "sl"
    if s.tp_on and s.tp_ticks > 0 and ticks >= s.tp_ticks:
        return "tp"
    if agree != t.side:
        return "disagree"
    return ""


def decide(t: Trade | None, s: Settings, px: float,
           agree: str) -> tuple[str, str]:
    """What the journal should do right now: ("", "") to do nothing.

    ("close", why) or ("open", side).

    THE POLICY LIVES HERE, NOT IN THE ROUTE. It was in a route
    closure, where the only way to reach it was an HTTP request
    through a fixture that never produced agreement -- so a test that
    auto-off opens nothing passed while the check was deleted. A
    policy that cannot be called directly cannot be tested directly.

    AUTO ONLY EVER OPENS. Closing on a target or a stop happens
    whether auto is on or not, because those are settings Chris
    turned on: a journal that ignored them until he refreshed would
    be scoring trades he had already said he was out of. Closing on
    disagreement is for auto rows only -- a trade he took by hand is
    his to end.
    """
    if not px or px <= 0:
        return "", ""
    if t is not None and t.open:
        why = should_close(t, s, px, agree)
        if why == "disagree" and t.mode != "auto":
            why = ""
        return ("close", why) if why else ("", "")
    if s.auto and agree in ("up", "down"):
        return "open", agree
    return "", ""
