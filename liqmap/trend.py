"""Direction that carries, instead of restarting every five minutes.

Chris, after 109 hours of tape said his rule was a coin flip:

    "the direction is a continuation and not resetting at ever 5
    minute interval"

WHAT THE RULE WAS DOING, AND WHY IT COULD NOT WORK.

    `direction = close vs open of the CURRENT 5-minute candle`.

    That anchor is thrown away and replaced every 300 seconds. A
    move that has run for twenty minutes is re-measured from
    scratch at each boundary, and in the first seconds of a new
    candle price sits ON its open -- so the direction there is one
    tick of noise deciding a trade. Worse, a strong trend prints a
    pullback candle sooner or later, and the moment it does the
    rule reads the trend as reversed because its only reference is
    that one candle.

    Measured: -0.379 ticks a trade BEFORE costs over 1,194 trades,
    confirmed at -0.308 on a held-back day. A coin flip, which is
    what a direction measured from an arbitrary recent anchor
    should produce.

FOUR DEFINITIONS, AND THE TAPE PICKS.

    BAR       what it does now: this candle's close against this
              candle's open. Resets every boundary. The control.

    CLOSED    the last CLOSED candle's direction, held until
              another closed candle disagrees. One bar of memory --
              the smallest possible continuation.

    ANCHOR    price against the open of the candle where the move
              BEGAN, not the current one. The anchor only moves
              when the MOVE turns -- it hangs off RUN's counter,
              so a pullback candle does not reset it. This is the
              most literal reading of "not resetting at every 5
              minute interval".

              It first moved on any direction-changing close,
              which made one pullback candle reset it: most of the
              way back to the per-bar anchor it exists to replace.
              ANCHOR and RUN therefore turn on the same counter
              and differ in what they report -- RUN gives the
              move's direction, ANCHOR gives price against where
              that move started, which can be the other way during
              a deep pullback.

    RUN       the direction persists until `flip_bars` consecutive
              CLOSED candles disagree with it. A pullback does not
              end a trend; a sequence does.

    None of these is obviously right and this module asserts
    nothing about which is. They exist to be measured against each
    other on real tape, which is the only thing that has been able
    to settle anything here.

Nothing in this module places, sizes, routes or times an order.
"""

from __future__ import annotations

MODES = ("bar", "closed", "anchor", "run")


def _dir(open_px: float, close_px: float) -> str:
    if open_px <= 0 or close_px <= 0:
        return "flat"
    if close_px > open_px:
        return "up"
    if close_px < open_px:
        return "down"
    return "flat"


class Trend:
    """One definition of "which way is the 5-minute going".

    Fed the forming candle on every print and told when a candle
    closes. `direction` is whatever the chosen mode says, and the
    caller never needs to know which mode it asked for.
    """

    __slots__ = ("mode", "flip_bars", "_cur_open", "_cur_close",
                 "_last_closed_dir", "_anchor", "_anchor_dir",
                 "_run_dir", "_against", "closes", "flips")

    def __init__(self, mode: str = "bar", flip_bars: int = 2) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self.flip_bars = max(1, int(flip_bars))
        self._cur_open = 0.0
        self._cur_close = 0.0
        self._last_closed_dir = ""
        # The price the ANCHOR mode measures against, and the
        # direction that anchor established.
        self._anchor = 0.0
        self._anchor_dir = ""
        # RUN KEEPS ITS OWN DIRECTION, and that is not tidiness.
        #
        # It shared `_anchor_dir` at first, and ANCHOR moves that on
        # the FIRST disagreeing close -- so the counter it was
        # supposed to be protected by reset at the same instant and
        # RUN flipped on one pullback candle, exactly like the mode
        # it exists to improve on. Two modes, two memories.
        self._run_dir = ""
        self._against = 0          # consecutive closes against the run
        self.closes = 0
        self.flips = 0

    # -- feeding ---------------------------------------------------------

    def on_price(self, px: float, bar_open: float) -> None:
        """The forming candle, as it stands right now."""
        try:
            self._cur_close = float(px or 0.0)
            self._cur_open = float(bar_open or 0.0)
        except (TypeError, ValueError):
            return
        if self._anchor <= 0 and self._cur_open > 0:
            # Nothing has established a direction yet, so the first
            # candle's open is the only anchor available.
            self._anchor = self._cur_open

    def on_close(self, bar_open: float, bar_close: float) -> None:
        """A 5-minute candle just ended. This is where memory moves."""
        d = _dir(bar_open, bar_close)
        self.closes += 1
        if d in ("up", "down"):
            if self._last_closed_dir and d != self._last_closed_dir:
                self.flips += 1
            self._last_closed_dir = d

        # RUN first: a pullback is one candle, a reversal is a
        # sequence. The anchor hangs off this.
        turned = False
        if d in ("up", "down"):
            if not self._run_dir:
                self._run_dir = d
                self._against = 0
                turned = True
            elif d == self._run_dir:
                self._against = 0
            else:
                self._against += 1
                if self._against >= self.flip_bars:
                    self._run_dir = d
                    self._against = 0
                    turned = True

        # ANCHOR: the reference moves when the MOVE turns, not when
        # a candle does.
        #
        # It was moving on any direction-changing close, so a single
        # pullback candle reset it -- which is most of the way back
        # to the per-bar anchor this mode exists to replace. Hanging
        # it off the run means it stays where the move began for as
        # long as the move is still running, which is what Chris
        # described.
        if turned:
            self._anchor_dir = self._run_dir
            self._anchor = bar_open

    # -- reading ---------------------------------------------------------

    @property
    def direction(self) -> str:
        m = self.mode
        if m == "bar":
            return _dir(self._cur_open, self._cur_close)
        if m == "closed":
            return self._last_closed_dir or "flat"
        if m == "anchor":
            if self._anchor <= 0 or self._cur_close <= 0:
                return "flat"
            return _dir(self._anchor, self._cur_close)
        # RUN: the direction only turns once `flip_bars` closed
        # candles have gone against it.
        return self._run_dir or "flat"

    @property
    def anchor(self) -> float:
        return self._anchor

    def to_dict(self) -> dict:
        return {"mode": self.mode, "dir": self.direction,
                "run_dir": self._run_dir,
                "anchor": round(self._anchor, 4),
                "anchor_dir": self._anchor_dir,
                "last_closed": self._last_closed_dir,
                "against": self._against,
                "closes": self.closes, "flips": self.flips}
