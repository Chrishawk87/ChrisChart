"""The imbalance rule as trades, so entry levels can be compared.

THE RULE

    enter   the running imbalance reaches +level (buy) or -level (sell)
    exit    the first of: the target in points, the opposite level,
            or the bar closing

A bar may hold several trades. Nothing survives the bar.

TWO EXIT RULES, MEASURED SIDE BY SIDE
-----------------------------------------------------------------------
They differ on one question -- what happens while a trade is open and
the imbalance falls back inside the band -- and the answer changes
which entry level looks best, so neither is assumed:

    HOLD    nothing. The trade runs to the target, the opposite
            level, or the bar close. Falling back inside the band is
            not an exit.
    BAND    coming back through the level flattens the trade, the way
            a reversal does.

RE-ARMING
-----------------------------------------------------------------------
After a trade exits, the same side cannot be taken again until the
imbalance has come back inside the band. Without that, a trade that
exits at target with the imbalance still far past the level re-enters
on the very next update at the same unchanged reading, and one move
is counted as a dozen trades.

THE CONTROL IS THE SAME RULE, BACKWARDS
-----------------------------------------------------------------------
Every level also runs with the side flipped: buy where the rule sells
and sell where it buys. Same entries, same exits, same bars, opposite
direction. It is an exactly paired control, and it is the only number
that says whether the signal carries anything:

    rule 45%, flipped 55%   the signal is pointing the wrong way
    both near 50%           there is nothing in it
    rule 58%, flipped 42%   something is there

A win rate on its own cannot distinguish those, because most of what
drives it is how long a bar runs and how far price travels in one --
which is identical for both.

WHAT THIS IS NOT
-----------------------------------------------------------------------
It is not a backtest and it models no fills, costs, size or risk. It
answers "did price travel the target distance the called way before
the bar ended", and refuses to be read as more than that.

Nothing here predicts, and nothing here places, sizes or times an
order.
"""

from __future__ import annotations

from dataclasses import dataclass, field

UP = "up"
DOWN = "down"

# What happens while a trade is open and the imbalance comes back
# inside the band.
HOLD = "hold"
BAND = "band"
POLICIES: tuple[str, ...] = (HOLD, BAND)

# One ES point in raw feed units.
POINT = 1_000_000_000

# The entry levels worth comparing, in contracts of imbalance.
LEVELS: tuple[int, ...] = (100, 150, 200, 250, 300, 400, 500)

TARGET_POINTS = 3.0


@dataclass(slots=True)
class Trade:
    """One trade, with everything needed to argue about it afterwards."""

    side: str = UP
    level: int = 0
    bar: int = 0
    entry_px: float = 0.0
    entry_ts: float = 0.0
    entry_acc: int = 0
    exit_px: float = 0.0
    exit_ts: float = 0.0
    why: str = ""               # target | reverse | band | bar
    updates: int = 0
    mfe: float = 0.0            # furthest it went the trade's way, raw
    mae: float = 0.0            # furthest against, as a positive

    @property
    def points(self) -> float:
        d = self.exit_px - self.entry_px
        return (d if self.side == UP else -d) / POINT

    @property
    def won(self) -> bool:
        """Reached the target. Nothing else counts as a win."""
        return self.why == "target"

    @property
    def seconds(self) -> float:
        return self.exit_ts - self.entry_ts

    def as_dict(self) -> dict:
        return {"side": self.side, "level": self.level, "bar": self.bar,
                "entry_px": self.entry_px, "entry_ts": self.entry_ts,
                "entry_acc": self.entry_acc, "exit_px": self.exit_px,
                "exit_ts": self.exit_ts, "why": self.why,
                "updates": self.updates,
                "points": round(self.points, 4),
                "won": self.won,
                "mfe_pts": round(self.mfe / POINT, 4),
                "mae_pts": round(self.mae / POINT, 4),
                "seconds": round(self.seconds, 3)}


class ImbSim:
    """One entry level, one exit rule, one direction convention."""

    __slots__ = ("level", "target", "policy", "flip", "trades", "bar",
                 "_side", "_t", "_armed_up", "_armed_down", "_per_bar")

    def __init__(self, level: int, target_points: float = TARGET_POINTS,
                 policy: str = HOLD, flip: bool = False) -> None:
        if level < 1:
            raise ValueError("level must be at least 1")
        if target_points <= 0:
            raise ValueError("target_points must be positive")
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}")
        self.level = int(level)
        self.target = float(target_points) * POINT
        self.policy = policy
        # The control: take the opposite side of every signal. Same
        # entries, same exits, same bars.
        self.flip = bool(flip)
        self.trades: list[Trade] = []
        self.bar = 0
        self._side = ""
        self._t: Trade | None = None
        self._armed_up = True
        self._armed_down = True
        self._per_bar: list[int] = []

    # -- the loop --------------------------------------------------------

    def tick(self, ts: float, acc: int, mid: float) -> None:
        """One book update: the imbalance now, and the mid now."""
        t = self._t
        if t is not None:
            t.updates += 1
            d = mid - t.entry_px
            fav = d if t.side == UP else -d
            if fav > t.mfe:
                t.mfe = fav
            if -fav > t.mae:
                t.mae = -fav
            if fav >= self.target:
                self._close("target", ts, mid)
                t = None

        if t is not None and self.policy == BAND and abs(acc) < self.level:
            self._close("band", ts, mid)
            t = None

        # The opposite level ends this trade and starts the next one.
        if t is not None:
            want = self._signal(acc)
            if want and want != t.side:
                self._close("reverse", ts, mid)
                t = None

        # RE-ARM. A side becomes available again once the imbalance has
        # come back inside the band on that side; without it a trade
        # that exits at target with the imbalance unchanged re-enters
        # on the next update and one move is counted many times.
        if acc < self.level:
            self._armed_up = True
        if acc > -self.level:
            self._armed_down = True

        if self._t is None:
            want = self._signal(acc)
            if want == UP and self._armed_up:
                self._open(UP, ts, mid, acc)
            elif want == DOWN and self._armed_down:
                self._open(DOWN, ts, mid, acc)

    def bar_close(self, ts: float, px: float) -> None:
        """The bar ended. Nothing survives it."""
        if self._t is not None:
            self._close("bar", ts, px)
        self._per_bar.append(self.bar)
        self.bar = 0

    # -- internals -------------------------------------------------------

    def _signal(self, acc: int) -> str:
        if acc >= self.level:
            side = UP
        elif acc <= -self.level:
            side = DOWN
        else:
            return ""
        if self.flip:
            side = DOWN if side == UP else UP
        return side

    def _open(self, side: str, ts: float, mid: float, acc: int) -> None:
        self._t = Trade(side=side, level=self.level, entry_px=mid,
                        entry_ts=ts, entry_acc=int(acc))
        self._side = side
        self.bar += 1

    def _close(self, why: str, ts: float, px: float) -> None:
        t = self._t
        if t is None:
            return
        t.why = why
        t.exit_px = px
        t.exit_ts = ts
        self.trades.append(t)
        # The side just taken is disarmed until the imbalance comes
        # back inside the band.
        if t.side == UP:
            self._armed_up = False
        else:
            self._armed_down = False
        self._t = None
        self._side = ""

    # -- reading it ------------------------------------------------------

    @property
    def trades_per_bar(self) -> float:
        bars = len(self._per_bar)
        return (sum(self._per_bar) / bars) if bars else 0.0

    @property
    def multi_bars(self) -> int:
        """Bars that held more than one trade."""
        return sum(1 for n in self._per_bar if n > 1)

    def summary(self) -> dict:
        ts_ = self.trades
        n = len(ts_)
        won = sum(1 for t in ts_ if t.won)
        why: dict = {}
        for t in ts_:
            why[t.why] = why.get(t.why, 0) + 1
        pts = [t.points for t in ts_]
        pts_sorted = sorted(pts)
        total = sum(pts)
        return {
            "level": self.level, "policy": self.policy, "flip": self.flip,
            "trades": n,
            "won": won,
            "win_rate": round(won / n, 4) if n else 0.0,
            "why": why,
            "mean_points": round(total / n, 4) if n else 0.0,
            "median_points": round(
                pts_sorted[n // 2], 4) if n else 0.0,
            "total_points": round(total, 2),
            "trades_per_bar": round(self.trades_per_bar, 2),
            "multi_bars": self.multi_bars,
            "mean_updates": round(
                sum(t.updates for t in ts_) / n, 1) if n else 0.0,
        }


@dataclass
class Sweep:
    """Every level, both exit rules, and the flipped control of each.

    All of them are fed from one pass over the data: each is a handful
    of comparisons per update, and running them together is the only
    way the comparison is of the same tape rather than of several
    different readings of it.
    """

    levels: tuple = LEVELS
    target_points: float = TARGET_POINTS
    policies: tuple = POLICIES
    sims: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sims:
            for lv in self.levels:
                for pol in self.policies:
                    for flip in (False, True):
                        self.sims[(lv, pol, flip)] = ImbSim(
                            lv, self.target_points, pol, flip)

    def tick(self, ts: float, acc: int, mid: float) -> None:
        for s in self.sims.values():
            s.tick(ts, acc, mid)

    def bar_close(self, ts: float, px: float) -> None:
        for s in self.sims.values():
            s.bar_close(ts, px)

    def rows(self) -> list[dict]:
        out = []
        for lv in self.levels:
            for pol in self.policies:
                real = self.sims[(lv, pol, False)].summary()
                ctl = self.sims[(lv, pol, True)].summary()
                real["control_win_rate"] = ctl["win_rate"]
                real["control_mean_points"] = ctl["mean_points"]
                real["control_trades"] = ctl["trades"]
                # The only number that says whether the signal carries
                # anything: how far it beats itself run backwards.
                real["edge"] = round(real["win_rate"] - ctl["win_rate"], 4)
                real["edge_points"] = round(
                    real["mean_points"] - ctl["mean_points"], 4)
                out.append(real)
        return out

    def trades(self, policy: str = HOLD) -> list[Trade]:
        out: list[Trade] = []
        for lv in self.levels:
            out.extend(self.sims[(lv, policy, False)].trades)
        out.sort(key=lambda t: (t.entry_ts, t.level))
        return out
