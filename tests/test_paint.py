"""THE TWO ROWS ARE A TRANSCRIPTION OF THE RENDERER.

Chris, after watching three different readings sit under rows that
were supposed to be his charts:

    "i dont care about cutting in halves i want this to actually
    look at the footprint OFI chart and say what its doing for the
    outer OFI and the inner tick to look at the tick chart and
    represent what its saying visually up or down its not
    calculating anything just saying what it see the bar doing
    immediately ... always looking at it red down green up same for
    the tick chart it reads whatever is there green up red down a
    continuous reading"

So the rows are no longer a reading OF the charts. They are the
COLOUR ON SCREEN, in words. That makes the drawing code the
specification, and the whole risk is that somebody changes the
drawing code and nothing here notices -- the rows would go on
agreeing with a rule the chart stopped using. The first class
below is that alarm: it reads the renderer's own expressions out
of the served page and fails if they move.

THERE IS NO FLAT IN EITHER ROW, because there is no third colour
on either chart. Both renderers paint green at zero. A row that
reported flat there would blank at the open of every bar, which is
the complaint this whole arc started from: "these goes flat when
there is actual movement in the OFI and the tick chart".
"""

from __future__ import annotations

import pathlib

from liqmap.ofibar import CHOP, GREEN, RED, OfiBarBuilder, paint_of
from liqmap.tickbar import TickBars

SRC = pathlib.Path(__file__).resolve().parents[1] / "liqmap"
WEB = (SRC / "web.py").read_text()

POINT = 1_000_000_000
TICK = POINT // 4
PX = 6700 * POINT


def ofi(target: int = 2000) -> OfiBarBuilder:
    return OfiBarBuilder(tick_size=0.25, threshold=target, max_s=900.0,
                         min_updates=2)


def book(b: OfiBarBuilder, ts: float, bid: int, bsz: int = 100,
         asz: int = 100):
    return b.on_book(ts, bid, bsz, bid + TICK, asz)


def seed(b: OfiBarBuilder) -> None:
    """THE FIRST TOUCH IS A REFERENCE, NOT AN EVENT.

    OFI is a comparison of two book states, so one update produces
    no flow, no bar and nothing on screen. The chart does the same
    thing -- its forming candle needs `updates > 0` -- so a blank
    here is faithful rather than a gap to paper over.
    """
    book(b, 0.0, PX)


class TestTheRendererIsStillWhatWeTranscribed:
    """IF THIS FAILS, THE CHART CHANGED AND THE ROWS DID NOT.

    Every assertion here is a line of drawing code quoted exactly.
    They are quoted rather than described because a description is
    what `paint_of` already is, and two descriptions of the same
    rule is how the rows drifted from the charts three times.
    """

    def test_the_ofi_candle_still_colours_by_direction_then_sign(self):
        i = WEB.index("const bodyColour = b => {")
        fn = WEB[i:WEB.index("};", i)]
        assert "if (b.direction === 'green') return '#35c27a';" in fn
        assert "if (b.direction === 'red') return '#e2574c';" in fn
        # THE FORMING CANDLE TAKES THIS BRANCH. It has no direction
        # until it closes, so its colour is the bare sign of the
        # imbalance it has so far -- and zero is drawn GREEN.
        assert "return (b.ofi || 0) < 0 ? '#e2574c' : '#35c27a';" in fn

    def test_the_forming_candle_is_drawn_from_the_accumulator(self):
        """`b.ofi` on the forming bar IS the accumulator, so the
        sign the renderer tests is the one `paint` reads."""
        i = WEB.index('forming = {"call_side": _cs')
        seg = WEB[i:WEB.index('"seconds": round(cur.seconds', i)]
        assert '"ofi": b.accumulated,' in seg

    def test_a_plain_candle_still_colours_close_against_open(self):
        """The tick chart's candles carry no direction, so they
        take the last branch -- and `>=` paints a doji GREEN."""
        i = WEB.index("const fused = (b.dir !== undefined")
        seg = WEB[i:i + 400]
        assert ": (b.c >= b.o ? up : down);" in seg

    def test_paint_of_matches_the_three_branches(self):
        assert paint_of(GREEN, -9999) == "up", \
            "a closed green bar is green whatever its imbalance says"
        assert paint_of(RED, 9999) == "down"
        assert paint_of(CHOP, -1) == "down"
        assert paint_of(CHOP, 1) == "up"
        assert paint_of(CHOP, 0) == "up", \
            "the renderer draws zero green; this does not"

    def test_paint_of_never_says_flat(self):
        for d in (GREEN, RED, CHOP, "", "nonsense"):
            for v in (-5000, -1, 0, 1, 5000):
                assert paint_of(d, v) in ("up", "down")

    def test_paint_of_survives_junk(self):
        for v in (None, "", "12", 0.0):
            assert paint_of(CHOP, v) in ("up", "down")


class TestTheOfiBuilderPaintsWhatIsOnScreen:
    def test_nothing_drawn_yet_is_no_colour(self):
        """A BLANK, NOT A GUESS. Before the first book update
        there is no candle on the chart to mirror."""
        assert ofi().paint == ""

    def test_a_forming_bar_follows_its_accumulator(self):
        b = ofi()
        seed(b)
        book(b, 1.0, PX, bsz=400, asz=100)
        assert b.accumulated > 0
        assert b.paint == "up"
        book(b, 2.0, PX, bsz=100, asz=900)
        assert b.accumulated < 0
        assert b.paint == "down", \
            "the candle went red and the row did not"

    def test_it_flips_with_the_candle_every_time(self):
        """NO HOLD, NO BAND, NO HYSTERESIS. Those were all tried
        and each one left the row saying one thing while the chart
        said the other."""
        b = ofi()
        seed(b)
        seen = []
        for n, (bsz, asz) in enumerate(
                [(400, 100), (100, 400), (400, 100), (100, 400)]):
            book(b, float(n + 1), PX, bsz=bsz, asz=asz)
            seen.append(b.paint)
        assert seen == ["up", "down", "up", "down"], seen

    def test_between_bars_it_shows_the_last_closed_candle(self):
        """THE CHART STILL HAS A CANDLE ON IT. `current` is None
        between bars -- for that moment the rightmost candle is
        the one that just closed, and the row says its colour."""
        b = ofi(target=200)
        seed(b)
        t = 0.0
        while not b.bars and t < 200:
            t += 1.0
            book(b, t, PX, bsz=100 + int(t) * 300, asz=100)
        assert b.bars[-1].direction == GREEN
        assert b.paint == "up"

    def test_the_reading_carries_the_colour(self):
        b = ofi()
        seed(b)
        book(b, 1.0, PX, bsz=400, asz=100)
        r = b.bar_read
        assert r["paint"] == b.paint == "up"
        # AND THE GATED READINGS SURVIVE BESIDE IT. They are
        # measurements somebody made; they are simply not the row.
        assert "read" in r and "raw_side" in r

    def test_the_empty_reading_carries_it_too(self):
        """ONE SHAPE EITHER WAY. A missing key and a real value
        look identical to the panel's `|| ''` guard."""
        assert ofi().bar_read["paint"] == ""

    def test_the_gate_can_be_flat_while_the_candle_is_not(self):
        """THE ARITHMETIC BEHIND THE COMPLAINT. At a call level of
        500 a visibly green 2,000-contract candle reads FLAT for
        most of its life -- which is what the row used to say."""
        b = ofi()
        seed(b)
        book(b, 1.0, PX, bsz=180, asz=100)
        r = b.bar_read
        assert 0 < r["imbalance"] < r["level"]
        assert r["read"] == "flat"
        assert r["paint"] == "up"


class TestTheTickBuilderPaintsWhatIsOnScreen:
    def test_nothing_drawn_yet_is_no_colour(self):
        assert TickBars(per_bar=5000).paint == ""

    def test_up_and_down_follow_the_body(self):
        b = TickBars(per_bar=5000)
        b.on_trade(0.0, 7000.0, 1.0, "B")
        b.on_trade(1.0, 7000.25, 1.0, "B")
        assert b.paint == "up"
        b.on_trade(2.0, 6999.5, 1.0, "S")
        assert b.paint == "down"

    def test_the_first_print_of_a_bar_is_green_not_flat(self):
        """CLOSE == OPEN IS WHERE THE ROW USED TO BLANK. The
        chart draws that candle green -- `b.c >= b.o` -- so the
        row says UP. Every bar begins in this state."""
        b = TickBars(per_bar=5000)
        b.on_trade(0.0, 7000.0, 1.0, "B")
        assert b.current.direction == "flat"
        assert b.paint == "up"

    def test_it_never_goes_quiet_once_a_print_has_landed(self):
        b = TickBars(per_bar=5000)
        px = 7000.0
        b.on_trade(0.0, px, 1.0, "B")
        for n in range(200):
            px += 0.25 if n % 3 else -0.25
            b.on_trade(float(n + 1), px, 1.0, "B")
            assert b.paint in ("up", "down")

    def test_between_bars_it_shows_the_last_closed_candle(self):
        b = TickBars(per_bar=4)
        px = 7000.0
        for n in range(4):
            px += 0.25
            b.on_trade(float(n), px, 1.0, "B")
        assert b.current is None, "this test no longer tests that"
        assert b.last_closed is not None
        assert b.paint == "up"

    def test_the_path_vote_can_disagree_and_the_row_follows_the_body(self):
        """TWO REAL MEASUREMENTS, ONE OF THEM ON SCREEN. The path
        vote is what the rest of the engine reads; the candle is
        what Chris reads. The row is the candle."""
        b = TickBars(per_bar=5000)
        px = 7000.0
        b.on_trade(0.0, px, 1.0, "B")
        for n in range(30):
            px += 0.25
            b.on_trade(float(n + 1), px, 1.0, "B")
        px -= 12.0
        b.on_trade(100.0, px, 1.0, "S")
        assert b.current.close < b.current.open
        assert b.paint == "down"


class TestTheColourReachesTheGate:
    """SOURCE, BECAUSE THE ENGINE DOES NOT RUN UNDER THE TESTS.

    A key dropped from `_tick_now` leaves the builder correct, the
    unit tests above green, and the row blank on the screen --
    which is the exact failure Chris reported and the one shape
    this suite cannot catch by executing.
    """

    @staticmethod
    def _fn(name: str) -> str:
        i = WEB.index(f"    def {name}(")
        return WEB[i:WEB.index("\n    def ", i + 10)]

    def test_the_forming_tick_reading_carries_it(self):
        fn = self._fn("_tick_now")
        i = fn.index('"body_dir": body_dir,')
        assert '"paint": bld.paint,' in fn[i:], \
            "the forming tick bar's colour never leaves the helper"

    def test_the_between_bars_tick_reading_carries_it_too(self):
        """ONE SHAPE EITHER WAY. The early return is a whole
        second payload and has been missing a key before."""
        fn = self._fn("_tick_now")
        early = fn[fn.index("if cur is None:"):fn.index('"dir": (sm.get')]
        assert '"paint": bld.paint,' in early

    def test_the_ofi_reading_carries_it_through_the_helper(self):
        """`_ofi_bar_read` copies `bar_read` wholesale, so the key
        rides along -- pinned because a filtered copy was proposed
        here once and would drop it silently."""
        fn = self._fn("_ofi_bar_read")
        assert "out = dict(b.bar_read or {})" in fn

    def test_the_two_rows_read_nothing_else(self):
        fn = self._fn("_fused_gate")
        seg = fn[fn.index("if _src:"):fn.index("out = _fr.state(")]
        code = "\n".join(ln for ln in seg.splitlines()
                         if not ln.strip().startswith("#"))
        assert '_ofi_row = str(_ob.get("paint") or "")' in code
        assert '_tick_row = str(_tb.get("paint") or "")' in code
        # THE ASSIGNMENTS ONLY. `body_dir` and `raw_side` are
        # still carried further down as evidence beside the rows,
        # which is the point -- they are measurements, they are
        # just not what the row says.
        rows = [ln.strip() for ln in code.splitlines()
                if ln.strip().startswith(("_ofi_row =", "_tick_row ="))]
        assert len(rows) == 2, rows
        for ln in rows:
            for gone in ("ofi_side(", "hold_dir(", "body_dir",
                         "raw_side", "_was", "_band"):
                assert gone not in ln, \
                    f"the chart-sourced row is computing with {gone}"
