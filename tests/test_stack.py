"""The confirmation ladder.

    5m    1m + 5m
    15m   1m + 5m + 15m
    1h    5m + 15m + 1h
    4h    15m + 1h + 4h

Four things would quietly break it, and each has a test that fails if the
code takes the easy path.

AN ABSTENTION IS NOT AN AGREEMENT. The 1m sitting at -0.52 against a bar of
1.0 means there is no trade, not that the 1m is neutral about it. This is
the case Chris gave by name.

A MISSING TIMEFRAME IS NOT A SILENT YES. A set that cannot see one of its
members has not been confirmed.

THE SIGN OF THE MOVE IS THE VOTE, not the winning side. Those two disagree
whenever absorption inverts a row, and for this ladder the move decides.

THE THRESHOLD SCALES. A flat 1.0bps is a real move on a minute and noise on
four hours, so a flat bar would let the slowest row -- the one carrying the
most weight -- be waved through on the least evidence. That exact mistake,
with a different constant, inverted this whole panel a few days ago.
"""

from __future__ import annotations

import pytest

from liqmap import stack as st


def rows(**kw):
    """Timeframes as the bps already on their row. Nothing else reaches
    this module, so nothing else can quietly start mattering."""
    secs = {"1m": 60.0, "5m": 300.0, "15m": 900.0, "1h": 3600.0,
            "4h": 14400.0}
    return [st.Row(timeframe=tf, interval_s=secs[tf], bps=float(bps))
            for tf, bps in kw.items()]


def side(d, tf):
    return d["verdicts"][tf]["side"]


# --------------------------------------------------------------- the ladder

def test_the_five_minute_needs_the_minute_to_agree():
    d = st.read(rows(**{"1m": -2.0, "5m": -4.0}))
    assert side(d, "5m") == "sell"


def test_the_minute_disagreeing_kills_the_five_minute():
    d = st.read(rows(**{"1m": +2.0, "5m": -4.0}))
    assert side(d, "5m") is None
    assert "disagree" in d["verdicts"]["5m"]["why"]


def test_the_fifteen_needs_all_three_of_the_faster_ones():
    agree = st.read(rows(**{"1m": -2.0, "5m": -4.0, "15m": -9.0}))
    assert side(agree, "15m") == "sell"

    split = st.read(rows(**{"1m": -2.0, "5m": +4.0, "15m": -9.0}))
    assert side(split, "15m") is None


def test_the_minute_drops_out_at_the_hour():
    """It is too fast to have an opinion about an hour, and letting it keep
    voting would mean one noisy minute could silence a signal built from
    hours of agreement."""
    d = st.read(rows(**{"1m": +2.0, "5m": -4.0, "15m": -9.0, "1h": -20.0}))
    assert side(d, "1h") == "sell"
    assert "1m" not in [v["timeframe"] for v in d["verdicts"]["1h"]["votes"]]


def test_the_four_hour_is_confirmed_by_the_fifteen_and_the_hour():
    d = st.read(rows(**{"15m": -9.0, "1h": -20.0, "4h": -40.0}))
    assert side(d, "4h") == "sell"
    assert [v["timeframe"] for v in d["verdicts"]["4h"]["votes"]] == \
        ["15m", "1h", "4h"]


def test_the_five_minute_does_not_vote_on_the_four_hour():
    d = st.read(rows(**{"5m": +4.0, "15m": -9.0, "1h": -20.0, "4h": -40.0}))
    assert side(d, "4h") == "sell"


def test_the_minute_never_prints_a_verdict_of_its_own():
    """It argues, it does not decide."""
    d = st.read(rows(**{"1m": -9.0, "5m": -9.0}))
    assert "1m" not in d["verdicts"]
    assert "1m" in d["votes"]


def test_buys_work_the_same_way_as_sells():
    d = st.read(rows(**{"1m": +2.0, "5m": +4.0, "15m": +9.0}))
    assert side(d, "5m") == "buy" and side(d, "15m") == "buy"


# --------------------------------------------------- abstaining and missing

def test_a_move_inside_the_bar_is_not_a_vote():
    """THE case, in the numbers Chris gave: the 1m at -0.52 against a bar
    of 1.0. There is no trade -- not a neutral minute that waves the set
    through."""
    d = st.read(rows(**{"1m": -0.52, "5m": -4.0}))
    assert d["votes"]["1m"]["side"] is None
    assert d["votes"]["1m"]["abstains"]
    assert side(d, "5m") is None
    assert "no vote" in d["verdicts"]["5m"]["why"]


def test_a_small_positive_move_is_not_a_buy_either():
    """The same guard on the other side. Tested by name because a bar
    checked on one side only is a bar that is not there."""
    d = st.read(rows(**{"1m": +0.52, "5m": +4.0}))
    assert d["votes"]["1m"]["side"] is None
    assert side(d, "5m") is None


def test_a_move_that_barely_misses_in_either_direction_abstains():
    assert st.vote_on("15m", 900.0, bps=+0.9).side is None
    assert st.vote_on("15m", 900.0, bps=-0.9).side is None
    assert st.vote_on("15m", 900.0, bps=+1.1).side == "buy"
    assert st.vote_on("15m", 900.0, bps=-1.1).side == "sell"


def test_exactly_at_the_bar_counts():
    d = st.read(rows(**{"1m": -1.0, "5m": -4.0}))
    assert d["votes"]["1m"]["side"] == "sell"


def test_a_timeframe_that_was_not_read_is_not_a_silent_yes():
    d = st.read(rows(**{"5m": -4.0, "15m": -9.0}))
    assert side(d, "5m") is None
    assert "no read on 1m" in d["verdicts"]["5m"]["why"]


def test_an_empty_ladder_prints_nothing_anywhere():
    d = st.read([])
    assert all(v["side"] is None for v in d["verdicts"].values())


# ------------------------------------------------- the sign is the vote

def test_the_sign_of_the_move_decides_not_the_winning_side():
    """A row reads BUYERS while price is down whenever absorption inverts
    it. This module never sees that word, and that is deliberate -- it is
    handed the move and nothing else."""
    import inspect

    src = inspect.getsource(st)
    assert "winner" not in src
    v = st.vote_on("5m", 300.0, bps=-4.0)
    assert v.side == "sell"


# --------------------------------------------------------- the threshold

def test_the_bar_is_the_same_on_every_timeframe_by_default():
    """FLAT IS THE DEFAULT, and it is not the obvious choice.

    Everywhere else here, a fixed bps threshold across timeframes has been
    a bug. It is right in this one place, and only the real numbers show
    why: these moves are all small. A four-hour at -3.4bps is a quiet four
    hours, so scaled against how far four-hour bars normally travel its
    bar lands near 18bps and it can never vote -- every row abstains and
    the ladder prints nothing, ever.
    """
    assert st.threshold_for(60.0) == st.threshold_for(14400.0)
    assert st.threshold_for(14400.0) == pytest.approx(st.BASE_BPS)


def test_the_module_never_sees_a_per_timeframe_yardstick():
    """Observation only: it is handed the bps that is already on the row
    and one number to compare it against."""
    import inspect

    src = inspect.getsource(st)
    body = "\n".join(ln for ln in src.split("\n")
                     if not ln.strip().startswith("#"))
    for gone in ("typical_bps", "anchor_typical", "scale=", "** 0.5"):
        assert gone not in body


def test_the_four_hour_in_chriss_own_screenshot_can_actually_vote():
    """-3.42bps on the 4h, -1.31 on the 1h. Those are the numbers on the
    screen, and a bar they cannot reach is a panel that never speaks."""
    assert st.vote_on("4h", 14400.0, bps=-3.42).side == "sell"
    assert st.vote_on("1h", 3600.0, bps=-1.31).side == "sell"
    assert st.vote_on("15m", 900.0, bps=-0.64).side is None


def test_the_minute_sits_exactly_on_the_number_you_set():
    assert st.threshold_for(60.0, base=1.0) == pytest.approx(1.0)
    assert st.threshold_for(900.0, base=2.5) == pytest.approx(2.5)


def test_every_verdict_says_what_it_took_or_what_stopped_it():
    d = st.read(rows(**{"1m": -0.2, "5m": -4.0, "15m": -9.0}))
    for tf in ("5m", "15m", "1h", "4h"):
        assert len(d["verdicts"][tf]["why"]) > 10


def test_a_verdict_carries_the_votes_behind_it():
    d = st.read(rows(**{"1m": -2.0, "5m": -4.0}))
    v = d["verdicts"]["5m"]
    assert len(v["votes"]) == 2
    assert all("bps" in x and "threshold" in x for x in v["votes"])


def test_the_ladder_is_stated_once_and_read_from_there():
    assert st.CONFIRMERS["5m"] == ("1m", "5m")
    assert st.CONFIRMERS["15m"] == ("1m", "5m", "15m")
    assert st.CONFIRMERS["1h"] == ("5m", "15m", "1h")
    assert st.CONFIRMERS["4h"] == ("15m", "1h", "4h")


def test_nothing_here_places_an_order():
    import inspect

    src = inspect.getsource(st).lower()
    for word in ("order", "buy_market", "submit", "execute", "position"):
        assert f"def {word}" not in src


# --------------------------------------------- on the row and through the route


def _src():
    import inspect

    from liqmap import web as w
    return inspect.getsource(w)


def test_the_word_sits_next_to_the_timeframe_it_belongs_to():
    src = _src()
    row = src[src.index("function renderLadder() {"):]
    row = row[:row.index("\nfunction paintLadder(")]
    assert "t && t.verdict" in row
    cell = src[src.index("function verdictTag(d) {"):]
    cell = cell[:cell.index("\nfunction fuelCell(")]
    assert "'BUY'" not in cell and "'SELL'" not in cell, (
        "the word should come from the verdict, not be guessed in the page")
    assert "d.side.toUpperCase()" in cell


def test_nothing_prints_when_they_do_not_agree():
    """The silence is the signal. A row that explained itself every time
    would be unreadable in the ninety per cent of moments when there is no
    trade, so the reason lives on the cell."""
    src = _src()
    cell = src[src.index("function verdictTag(d) {"):]
    cell = cell[:cell.index("\nfunction fuelCell(")]
    assert "!d.side" in cell and "return ''" in cell


def test_the_reason_is_always_available_even_when_nothing_prints():
    src = _src()
    cell = src[src.index("function fuelCell(name, v, call) {"):]
    cell = cell[:cell.index("\nfunction renderLadder(")]
    assert "call.why" in cell
    assert "title=" in cell


def test_the_route_feeds_the_ladder_the_move_and_not_the_winner():
    """`stack` is handed the bps and the yardstick and nothing else, so it
    cannot accidentally start reading the absorption-inverted side."""
    src = _src()
    block = src[src.index("out[\"stack\"] = stk.read("):]
    block = block[:block.index("paces: dict[str, Any] = {}")]
    assert "bps=p.move_bps" in block
    assert "winner" not in block


def test_the_threshold_is_settable_from_the_request():
    src = _src()
    assert "vote_bps: float = 1.0" in src
    block = src[src.index("out[\"stack\"] = stk.read("):]
    block = block[:block.index("paces: dict[str, Any] = {}")]
    assert "float(vote_bps)" in block


def test_a_broken_ladder_does_not_take_the_read_down():
    src = _src()
    block = src[src.index("out[\"stack\"] = stk.read("):]
    block = block[:block.index("paces: dict[str, Any] = {}")]
    assert "except Exception" in block


def test_every_row_carries_its_own_verdict_and_vote():
    """Joined up on the server rather than in the page: the row already
    knows which timeframe it is, and a join done twice is a join that can
    disagree with itself."""
    src = _src()
    block = src[src.index('out["timeframes"] = ['):]
    block = block[:block.index('out["confrontation"]')]
    assert '"verdict":' in block and '"vote":' in block
    assert '.get("verdicts")' in block and '.get("votes")' in block

