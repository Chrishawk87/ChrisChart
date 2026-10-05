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
    """Timeframes as (bps, typical_bps), defaulting typical to a flat scale."""
    secs = {"1m": 60.0, "5m": 300.0, "15m": 900.0, "1h": 3600.0,
            "4h": 14400.0}
    out = []
    for tf, v in kw.items():
        bps, typ = v if isinstance(v, tuple) else (v, 0.0)
        out.append(st.Row(timeframe=tf, interval_s=secs[tf], bps=bps,
                          typical_bps=typ))
    return out


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
    assert st.vote_on("15m", 900.0, bps=+3.8).side is None
    assert st.vote_on("15m", 900.0, bps=-3.8).side is None
    assert st.vote_on("15m", 900.0, bps=+3.9).side == "buy"
    assert st.vote_on("15m", 900.0, bps=-3.9).side == "sell"


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

def test_a_slower_timeframe_has_a_higher_bar():
    """1.0bps is a real move on a minute and noise on four hours. A flat
    bar would let the slowest row -- the one carrying the most weight in
    the stack -- be waved through on the least evidence."""
    m1, _ = st.threshold_for(60.0)
    m15, _ = st.threshold_for(900.0)
    h4, _ = st.threshold_for(14400.0)
    assert m1 < m15 < h4


def test_the_minute_sits_exactly_on_the_number_you_set():
    t, _ = st.threshold_for(60.0, base=1.0)
    assert t == pytest.approx(1.0)
    t2, _ = st.threshold_for(60.0, base=2.5)
    assert t2 == pytest.approx(2.5)


def test_the_bar_is_learned_from_this_market_where_it_can_be():
    """Scaled by how far bars of each length actually travel here, which
    is the same figure the flat band learns."""
    calm, learned_c = st.threshold_for(900.0, typical_bps=2.0,
                                       anchor_typical=1.0)
    wild, learned_w = st.threshold_for(900.0, typical_bps=20.0,
                                       anchor_typical=1.0)
    assert learned_c and learned_w
    assert wild > calm
    assert calm == pytest.approx(2.0)


def test_with_nothing_learned_it_falls_back_to_the_square_root():
    t, learned = st.threshold_for(900.0)
    assert not learned
    assert t == pytest.approx(1.0 * (900.0 / 60.0) ** 0.5, rel=0.01)


def test_a_quiet_slow_bar_never_becomes_the_easiest_row():
    """A four-hour that has been unusually still must not end up with a
    LOWER bar than the minute. It carries the most weight in the stack and
    would be waved through on the least evidence."""
    t, _ = st.threshold_for(14400.0, typical_bps=0.2, anchor_typical=5.0)
    assert t >= st.BASE_BPS


def test_the_scaling_uses_the_minute_as_its_anchor():
    d = st.read(rows(**{"1m": (-2.0, 1.0), "5m": (-4.0, 3.0)}))
    assert d["votes"]["5m"]["threshold"] == pytest.approx(3.0)
    assert d["votes"]["5m"]["scaled"]


def test_a_row_says_whether_its_bar_was_learned_or_assumed():
    d = st.read(rows(**{"1m": -2.0, "5m": -4.0}))
    assert not d["votes"]["5m"]["scaled"]


# ---------------------------------------------------------------- reporting

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
    assert "typical_bps=p.typical_bps" in block
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
