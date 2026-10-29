"""The imbalance rule as trades, so entry levels can be compared.

BOTH SIDES OF THE IMBALANCE, NOT JUST THE NET

The net cannot tell a one-sided push from a brawl that happened to
tilt: +300 from 350 buying and 50 selling is the same number as +300
from 9,000 buying and 8,700 selling. The second is where a trade gets
taken just before price turns over, because the opposing flow is
sitting right there and the net has hidden it.

So every trade records the SHARE at entry -- how much of the bar's
gross flow had been buying, 0.5 for an even fight and 1.0 for one-way
traffic -- and the sweep bands the outcomes by it. If the trades that
fail are the contested ones, that shows up as a gradient across the
bands and a filter is one parameter away. If it does not, the net was
not hiding anything worth having.

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

TARGET_POINTS = 1.0

# Entry-share bands the outcomes are grouped into. The edges are what
# "how one-sided was it" looks like in practice: a bar past the level
# is typically 0.51, so the interesting question is whether the few
# that reach 0.6 and beyond behave differently.
SHARE_BANDS: tuple[float, ...] = (0.50, 0.52, 0.55, 0.60, 0.70)


@dataclass(slots=True)
class Trade:
    """One trade, with everything needed to argue about it afterwards."""

    side: str = UP
    level: int = 0
    bar: int = 0
    entry_px: float = 0.0
    entry_ts: float = 0.0
    entry_acc: int = 0
    # Where the CALL fired, which is not where the trade was taken
    # once confirmation is on. `slip` is the difference, and it is a
    # cost paid on every trade in exchange for the ones not taken.
    call_px: float = 0.0
    call_ts: float = 0.0
    # How one-sided the bar's flow was at the moment of entry, and how
    # much flow there had been. The net alone cannot say either.
    entry_share: float = 0.5
    entry_gross: int = 0
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
    def slip(self) -> float:
        """Ticks given up waiting for price to confirm the call."""
        if not self.call_px:
            return 0.0
        d = self.entry_px - self.call_px
        return (d if self.side == UP else -d) / (POINT / 4.0)

    @property
    def seconds(self) -> float:
        return self.exit_ts - self.entry_ts

    def as_dict(self) -> dict:
        return {"side": self.side, "level": self.level, "bar": self.bar,
                "entry_share": round(self.entry_share, 4),
                "entry_gross": self.entry_gross,
                "dominance": round(abs(self.entry_share - 0.5) * 2.0, 4),
                "entry_px": self.entry_px, "entry_ts": self.entry_ts,
                "call_px": self.call_px, "call_ts": self.call_ts,
                "slip_ticks": round(self.slip, 3),
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

    __slots__ = ("level", "target", "policy", "flip", "min_share",
                 "confirm", "max_calls", "skipped", "trades", "bar",
                 "_calls", "_side", "_t",
                 "_armed_up", "_armed_down", "_per_bar", "_share",
                 "_gross", "_call_side", "_call_px", "_call_ts")

    def __init__(self, level: int, target_points: float = TARGET_POINTS,
                 policy: str = HOLD, flip: bool = False,
                 min_share: float = 0.0,
                 confirm_ticks: float = 0.0,
                 max_calls: int = 0) -> None:
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
        # 0.0 means off: take every signal and let the bands do the
        # talking. Set it only once the bands say there is something
        # to filter on.
        self.min_share = float(min_share)
        # CONFIRMATION BEFORE ENTRY. The call fires on the imbalance;
        # the trade waits until price has gone that way by this many
        # ticks. A call that lapses first produced NO TRADE -- it is
        # counted in `skipped` and never lands in `trades`, because a
        # trade nobody took cannot win or lose.
        self.confirm = max(0.0, float(confirm_ticks)) * (POINT / 4.0)
        # ONE CALL PER CANDLE, when set. The imbalance crossing the
        # level twenty times is one push; counting it as twenty
        # trades measures how finely it oscillates around a line.
        # Counts CALLS, not fills -- counting fills lets a candle keep
        # trying after each unconfirmed call, which is a cap in name
        # only.
        self.max_calls = max(0, int(max_calls))
        self._calls = 0
        self.skipped = 0
        self._call_side = ""
        self._call_px = 0.0
        self._call_ts = 0.0
        self._share = 0.5
        self._gross = 0
        self.trades: list[Trade] = []
        self.bar = 0
        self._side = ""
        self._t: Trade | None = None
        self._armed_up = True
        self._armed_down = True
        self._per_bar: list[int] = []

    # -- the loop --------------------------------------------------------

    def tick(self, ts: float, acc: int, mid: float,
             share: float = 0.5, gross: int = 0) -> None:
        """One book update: the imbalance, the mid, and both sides.

        `share` is how much of the bar's gross flow has been buying so
        far. It never gates anything unless `min_share` is set; it is
        recorded on every trade so the outcomes can be banded by it.
        """
        self._share = float(share)
        self._gross = int(gross)
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
            ok = (want == UP and self._armed_up and self._one_sided(UP)) or \
                 (want == DOWN and self._armed_down
                  and self._one_sided(DOWN))
            if not ok:
                # The call is gone. Anything waiting on it was never
                # taken, and that is a skip, not a loss.
                if self._call_side:
                    self.skipped += 1
                    self._call_side = ""
                return
            if self.max_calls and self._calls >= self.max_calls:
                return
            if self.confirm <= 0:
                self._calls += 1
                self._open(want, ts, mid, acc, mid, ts)
                return
            if self._call_side != want:
                # A fresh call. Note where it fired and wait.
                self._calls += 1
                self._call_side = want
                self._call_px = mid
                self._call_ts = ts
                return
            d = mid - self._call_px
            if (d if want == UP else -d) >= self.confirm:
                self._open(want, ts, mid, acc, self._call_px,
                           self._call_ts)
                self._call_side = ""

    def bar_close(self, ts: float, px: float) -> None:
        """The bar ended. Nothing survives it."""
        if self._t is not None:
            self._close("bar", ts, px)
        self._per_bar.append(self.bar)
        self.bar = 0
        self._calls = 0
        self._call_side = ""

    # -- internals -------------------------------------------------------

    def _one_sided(self, side: str) -> bool:
        """Is the flow one-sided enough to take this side?

        Read on the RULE's side, not the flipped one: the control has
        to face the same filter as the rule or it is not a control of
        the same thing.
        """
        if self.min_share <= 0.0:
            return True
        raw = side
        if self.flip:
            raw = DOWN if side == UP else UP
        return (self._share >= self.min_share if raw == UP
                else self._share <= 1.0 - self.min_share)

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

    def _open(self, side: str, ts: float, mid: float, acc: int,
              call_px: float = 0.0, call_ts: float = 0.0) -> None:
        self._t = Trade(side=side, level=self.level, entry_px=mid,
                        entry_ts=ts, entry_acc=int(acc),
                        call_px=call_px or mid, call_ts=call_ts or ts,
                        entry_share=self._share, entry_gross=self._gross)
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
    min_share: float = 0.0
    # Ticks of confirmation to fan out over. The chart runs one of
    # these; the sweep runs all of them on the same tape so the choice
    # can be checked rather than argued about.
    confirms: tuple = (0.0,)
    max_calls: int = 0
    sims: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sims:
            for lv in self.levels:
                for pol in self.policies:
                    for cf in self.confirms:
                        for flip in (False, True):
                            self.sims[(lv, pol, cf, flip)] = ImbSim(
                                lv, self.target_points, pol, flip,
                                self.min_share, cf, self.max_calls)

    def tick(self, ts: float, acc: int, mid: float,
             share: float = 0.5, gross: int = 0) -> None:
        for s in self.sims.values():
            s.tick(ts, acc, mid, share, gross)

    def bar_close(self, ts: float, px: float) -> None:
        for s in self.sims.values():
            s.bar_close(ts, px)

    def rows(self) -> list[dict]:
        out = []
        for lv, pol, cf in [(a, b, c) for a in self.levels
                            for b in self.policies
                            for c in self.confirms]:
                real = self.sims[(lv, pol, cf, False)].summary()
                ctl = self.sims[(lv, pol, cf, True)].summary()
                real["confirm"] = cf
                real["skipped"] = self.sims[(lv, pol, cf, False)].skipped
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

    def trades(self, policy: str = HOLD, confirm: float | None = None
               ) -> list[Trade]:
        cf = self.confirms[0] if confirm is None else confirm
        out: list[Trade] = []
        for lv in self.levels:
            out.extend(self.sims[(lv, policy, cf, False)].trades)
        out.sort(key=lambda t: (t.entry_ts, t.level))
        return out


# ----------------------------------------- did one-sidedness separate them

def _band_of(share: float, side: str, edges=SHARE_BANDS) -> float:
    """Which one-sidedness band a trade entered in.

    Measured on the SIDE IT TOOK, so a short entered at share 0.38 is
    as one-sided as a long entered at 0.62 and lands in the same band.
    Grouping on the raw share instead would split every band in two
    and compare shorts against longs.
    """
    own = share if side == UP else 1.0 - share
    pick = edges[0]
    for e in edges:
        if own >= e:
            pick = e
    return pick


def bands(trades, edges=SHARE_BANDS) -> list[dict]:
    """Outcomes grouped by how one-sided the flow was at entry.

    THE QUESTION THIS ANSWERS. The net cannot tell a one-sided push
    from a brawl that tilted, and the brawl is where a trade gets
    taken just before price turns. If that is true, the clear rate
    rises across these bands. If it does not rise, the two sides were
    not hiding anything worth having and no filter will help.

    `why` is kept per band, because HOW a trade failed is a different
    fact from how often: a trade that got most of the way and ran out
    of bar is not the same failure as one that went straight against.
    """
    by: dict = {}
    for t in trades:
        b = _band_of(t.entry_share, t.side, edges)
        row = by.setdefault(b, {"band": b, "trades": 0, "cleared": 0,
                                "why": {}, "mfe_missed": [],
                                "points": 0.0})
        row["trades"] += 1
        row["why"][t.why] = row["why"].get(t.why, 0) + 1
        row["points"] += t.points
        if t.won:
            row["cleared"] += 1
        else:
            # How far the ones that did not clear actually got. A
            # failure at nine tenths of the distance is a different
            # animal from one that never left the entry.
            row["mfe_missed"].append(t.mfe / POINT)
    out = []
    for b in sorted(by):
        r = by[b]
        n = r["trades"]
        missed = sorted(r["mfe_missed"])
        out.append({
            "band": b, "trades": n,
            "cleared": r["cleared"],
            "missed": n - r["cleared"],
            "clear_rate": round(r["cleared"] / n, 4) if n else 0.0,
            "mean_points": round(r["points"] / n, 4) if n else 0.0,
            "why": r["why"],
            "median_missed_mfe": round(
                missed[len(missed) // 2], 4) if missed else 0.0,
            "missed_near": sum(1 for m in missed if m >= 0.5),
        })
    return out
