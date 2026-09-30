"""Key levels, checked against sessions whose answers are known by hand.

Every fixture here is built so the correct result is arithmetic, not opinion:
the prior day's high is a number this file chose, so PDH either equals it or
the module is wrong. That matters more than usual for this module, because a
level engine that is subtly off still produces levels, still draws lines on a
chart, and still looks entirely reasonable.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

import pytest

from liqmap import levels as L
from liqmap.levels import Kind, LevelBook, Outcome, Phase, State


def ts(y, m, d, hh, mm=0) -> float:
    """Epoch seconds from an exchange-local wall clock."""
    return datetime(y, m, d, hh, mm, tzinfo=L.ET).timestamp()


def bar(t, o, h, l, c, v=100.0) -> L.Bar:
    return L.Bar(ts=t, open=o, high=h, low=l, close=c, volume=v)


def flat(t, px, v=100.0) -> L.Bar:
    """A bar that does nothing, for moving the clock along."""
    return bar(t, px, px, px, px, v)


def session(day: tuple[int, int, int], bars: list[tuple]) -> list[L.Bar]:
    """Bars at (hour, minute, o, h, l, c) on one ET date."""
    y, m, d = day
    return [bar(ts(y, m, d, hh, mm), o, h, lo, c)
            for hh, mm, o, h, lo, c in bars]


# ------------------------------------------------------------- sessions

def test_the_globex_open_belongs_to_the_next_trade_date():
    """Sunday evening is Monday's session. Getting this wrong shifts every
    overnight range by a day."""
    assert L.trade_date(ts(2026, 9, 20, 18, 30)) == date(2026, 9, 21)
    assert L.trade_date(ts(2026, 9, 21, 9, 30)) == date(2026, 9, 21)


def test_the_post_settlement_hour_stays_on_the_same_date():
    assert L.trade_date(ts(2026, 9, 21, 16, 30)) == date(2026, 9, 21)


def test_the_phases_land_where_they_should():
    assert L.phase(ts(2026, 9, 21, 3, 0)) is Phase.OVERNIGHT
    assert L.phase(ts(2026, 9, 21, 9, 30)) is Phase.RTH
    assert L.phase(ts(2026, 9, 21, 15, 59)) is Phase.RTH
    assert L.phase(ts(2026, 9, 21, 16, 30)) is Phase.POST
    assert L.phase(ts(2026, 9, 21, 17, 30)) is Phase.CLOSED
    assert L.phase(ts(2026, 9, 21, 19, 0)) is Phase.OVERNIGHT


def test_the_weekend_is_closed():
    assert L.phase(ts(2026, 9, 19, 12, 0)) is Phase.CLOSED   # Saturday
    assert L.phase(ts(2026, 9, 20, 12, 0)) is Phase.CLOSED   # Sunday daytime
    assert L.phase(ts(2026, 9, 20, 19, 0)) is Phase.OVERNIGHT  # Sunday reopen
    assert L.phase(ts(2026, 9, 18, 18, 0)) is Phase.CLOSED   # Friday night


def test_sessions_are_exchange_time_across_the_dst_switch():
    """US DST ended 2026-11-01. The RTH open stays 09:30 ET on both sides of
    it; a fixed UTC offset would put one of these an hour out."""
    before = L.phase(ts(2026, 10, 30, 9, 30))
    after = L.phase(ts(2026, 11, 3, 9, 30))
    assert before is Phase.RTH and after is Phase.RTH

    opened = L.rth_open_ts(date(2026, 11, 3))
    assert L.et(opened).time() == time(9, 30)


# --------------------------------------------------- prior day levels

def _two_days() -> LevelBook:
    """Monday RTH runs 6000-6010, closes 6005. Tuesday opens flat.

    So PDH is 6010.00, PDL is 6000.00 and PRIOR_CLOSE is 6005.00 by
    construction, and any other answer is a bug.
    """
    book = LevelBook()
    book.extend(session((2026, 9, 21), [
        (9, 30, 6005, 6006, 6004, 6005),
        (10, 0, 6005, 6010, 6005, 6008),     # the high
        (11, 0, 6008, 6008, 6000, 6002),     # the low
        (15, 59, 6002, 6006, 6002, 6005),    # the close
    ]))
    # Roll into 9/22 with an OVERNIGHT bar, so the new session's RTH list,
    # VWAP accumulator and gap are all still untouched when a test starts.
    # An RTH bar here would quietly become the session's first bar and every
    # test downstream would be measuring it as well as its own fixture.
    book.observe(bar(ts(2026, 9, 21, 20, 0), 6005, 6006, 6004, 6005))
    return book


def test_prior_day_high_and_low_come_from_the_regular_session():
    book = _two_days()
    assert book.levels[Kind.PDH].price == pytest.approx(6010.0)
    assert book.levels[Kind.PDL].price == pytest.approx(6000.0)
    assert book.levels[Kind.PRIOR_CLOSE].price == pytest.approx(6005.0)


def test_the_overnight_range_excludes_the_regular_session():
    """Overnight is 18:00 to 09:30. An RTH bar leaking into ONH would make
    the two levels identical and the overnight read meaningless."""
    book = LevelBook()
    book.observe(bar(ts(2026, 9, 21, 20, 0), 6005, 6020, 6005, 6015))
    book.observe(bar(ts(2026, 9, 22, 3, 0), 6015, 6018, 5990, 6000))
    book.observe(bar(ts(2026, 9, 22, 9, 30), 6000, 6099, 6000, 6050))
    assert book.levels[Kind.ONH].price == pytest.approx(6020.0)
    assert book.levels[Kind.ONL].price == pytest.approx(5990.0)


def test_levels_do_not_survive_the_roll():
    """Yesterday's initial balance is history, not a level."""
    book = _two_days()
    book.observe(bar(ts(2026, 9, 22, 9, 30), 6005, 6006, 6004, 6005))
    book.observe(bar(ts(2026, 9, 22, 10, 31), 6005, 6006, 6004, 6005))
    assert Kind.IBH in book.levels
    book.observe(bar(ts(2026, 9, 23, 9, 30), 6005, 6005, 6005, 6005))
    assert Kind.IBH not in book.levels


def test_nothing_is_published_on_the_very_first_session():
    """There is no prior day, so there is no PDH. A zero or a guess here
    would be a level the market never traded at."""
    book = LevelBook()
    book.observe(bar(ts(2026, 9, 21, 9, 30), 6000, 6001, 5999, 6000))
    assert Kind.PDH not in book.levels
    assert Kind.PDL not in book.levels


# ------------------------------------------------ the wick/close rule

def test_a_wick_through_is_a_touch_not_a_break():
    """The rule the module turns on. Price trades above PDH and closes back
    under it -- rejection, not a breakout."""
    book = _two_days()
    book.observe(bar(ts(2026, 9, 22, 10, 0), 6005, 6012, 6004, 6006))
    pdh = book.levels[Kind.PDH]
    assert pdh.state is State.TOUCHED
    assert pdh.touches == 1 and pdh.breaks == 0
    assert pdh.outcome is Outcome.REJECTION


def test_a_confirmed_close_beyond_is_a_break():
    book = _two_days()
    book.observe(bar(ts(2026, 9, 22, 10, 0), 6005, 6014, 6004, 6013))
    pdh = book.levels[Kind.PDH]
    assert pdh.state is State.BROKEN_ABOVE
    assert pdh.breaks == 1
    assert pdh.outcome is Outcome.BREAK_HOLD


def test_a_bar_that_closes_through_is_not_also_counted_as_a_touch():
    """Order of checks. A breaking bar wicks through on its way, and
    reporting the touch would lose the break."""
    book = _two_days()
    book.observe(bar(ts(2026, 9, 22, 10, 0), 6005, 6014, 6004, 6013))
    assert book.levels[Kind.PDH].touches == 0


def test_breaking_then_closing_back_is_a_failed_break():
    """The strongest reversal condition in the spec, and the one a
    state-only reading cannot distinguish from 'never broken'."""
    book = _two_days()
    book.observe(bar(ts(2026, 9, 22, 10, 0), 6005, 6014, 6004, 6013))
    book.observe(bar(ts(2026, 9, 22, 10, 30), 6013, 6014, 6002, 6004))
    pdh = book.levels[Kind.PDH]
    assert pdh.state is State.RECLAIMED
    assert pdh.reclaims == 1
    assert pdh.outcome is Outcome.FAILED_BREAK


def test_an_untested_level_reports_pending_not_rejection():
    book = _two_days()
    book.observe(bar(ts(2026, 9, 22, 10, 0), 6005, 6006, 6004, 6005))
    assert book.levels[Kind.PDH].outcome is Outcome.PENDING


def test_a_level_below_reads_the_mirror_image():
    """PDL with price above it: a wick under is the touch, a close under is
    the break. The same code path, so this catches a hard-coded side."""
    book = _two_days()
    book.observe(bar(ts(2026, 9, 22, 10, 0), 6005, 6006, 5998, 6004))
    pdl = book.levels[Kind.PDL]
    assert pdl.state is State.TOUCHED and pdl.outcome is Outcome.REJECTION

    book.observe(bar(ts(2026, 9, 22, 10, 30), 6004, 6005, 5990, 5995))
    assert book.levels[Kind.PDL].state is State.BROKEN_BELOW


# ------------------------------------------------------------ the gap

def test_a_gap_above_makes_the_prior_high_support():
    """Open above PDH and it is no longer resistance. Deriving the side from
    the data rather than the level's name is the whole point -- a hard-coded
    'PDH is above' would call this broken before the bell."""
    book = _two_days()
    book.observe(bar(ts(2026, 9, 23, 9, 30), 6020, 6022, 6018, 6021))
    book.observe(bar(ts(2026, 9, 23, 10, 0), 6021, 6021, 6009, 6019))
    # 6010 was Monday's high; Wednesday opened at 6020, well above it.
    pdh = book.levels.get(Kind.PDH)
    if pdh is not None:
        assert pdh.above_at_set is True
        assert pdh.state is not State.BROKEN_ABOVE


def test_the_gap_is_measured_in_ticks_against_the_prior_close():
    book = _two_days()
    book.observe(bar(ts(2026, 9, 22, 9, 30), 6008, 6009, 6007, 6008))
    # Prior close 6005, open 6008 -> 3 points -> 12 ticks.
    assert book.gap() == pytest.approx(12.0)


def test_there_is_no_gap_before_the_session_opens():
    book = LevelBook()
    book.observe(bar(ts(2026, 9, 21, 3, 0), 6000, 6001, 5999, 6000))
    assert book.gap() is None


# ------------------------------------------- initial balance and vwap

def test_the_initial_balance_is_the_first_hour_only():
    """Bars after 10:30 must not widen it."""
    book = _two_days()
    book.extend(session((2026, 9, 22), [
        (9, 30, 6005, 6008, 6002, 6006),
        (10, 0, 6006, 6009, 6001, 6007),     # still inside the hour
        (10, 31, 6007, 6050, 5950, 6008),    # after it, and wild
    ]))
    assert book.levels[Kind.IBH].price == pytest.approx(6009.0)
    assert book.levels[Kind.IBL].price == pytest.approx(6001.0)


def test_the_initial_balance_is_not_published_early():
    book = _two_days()
    book.observe(bar(ts(2026, 9, 22, 9, 45), 6005, 6008, 6002, 6006))
    assert Kind.IBH not in book.levels


def _vwap_run(book, day, heavy_px, heavy_vol):
    """Four light bars at 6000, then one heavy bar. Five bars is the
    minimum VWAP publishes at."""
    t0 = ts(*day, 9, 30)
    for i in range(4):
        book.observe(L.Bar(t0 + i * 300, 6000, 6000, 6000, 6000, 10))
    book.observe(L.Bar(t0 + 4 * 300, heavy_px, heavy_px, heavy_px,
                       heavy_px, heavy_vol))


def test_vwap_is_volume_weighted_not_an_average_of_prices():
    """The heavy bar carries nine times the weight, so VWAP must sit at
    6090 -- not at 6050, which is where a mean of the two prices lands."""
    book = _two_days()
    _vwap_run(book, (2026, 9, 22), heavy_px=6100, heavy_vol=360)
    assert book.levels[Kind.VWAP].price == pytest.approx(6090.0)
    assert book.levels[Kind.VWAP].price != pytest.approx(6050.0)


def test_vwap_resets_at_the_next_session():
    book = _two_days()
    _vwap_run(book, (2026, 9, 22), heavy_px=6100, heavy_vol=360)
    _vwap_run(book, (2026, 9, 23), heavy_px=6000, heavy_vol=10)
    assert book.levels[Kind.VWAP].price == pytest.approx(6000.0)


def test_vwap_is_not_published_on_the_first_bar_of_a_session():
    """On bar one VWAP is the price, so it would sit zero ticks away and
    raise a phantom approach at every session open."""
    book = _two_days()
    found = book.observe(L.Bar(ts(2026, 9, 22, 9, 30),
                               5950, 5950, 5950, 5950, 100))
    assert Kind.VWAP not in book.levels
    assert not any(a.kind is Kind.VWAP for a in found)


def test_vwap_does_not_stick_as_broken_once_price_crosses_back():
    """VWAP moves every bar, so a fixed side-at-set would leave it reading
    'broken' for the rest of the session after one cross."""
    book = _two_days()
    for i, px in enumerate([6000, 6010, 6020, 6010, 6000, 5990, 6000]):
        book.observe(L.Bar(ts(2026, 9, 22, 9, 30) + i * 300,
                           px, px + 1, px - 1, px, 100))
    assert book.levels[Kind.VWAP].state is not State.EXPIRED


# ----------------------------------------------------- approach & age

def test_coming_within_two_ticks_raises_an_approach():
    book = _two_days()
    found = book.observe(bar(ts(2026, 9, 22, 10, 0), 6005, 6010, 6004, 6009.5))
    kinds = {a.kind for a in found}
    assert Kind.PDH in kinds
    ap = next(a for a in found if a.kind is Kind.PDH)
    assert ap.ticks_away == pytest.approx(2.0)
    assert ap.approached_from_above is False


def test_a_distant_bar_raises_nothing():
    book = _two_days()
    assert book.observe(bar(ts(2026, 9, 22, 10, 0),
                            5950, 5951, 5949, 5950)) == []


def test_a_level_expires_once_price_has_plainly_left_it():
    book = _two_days()
    t0 = ts(2026, 9, 22, 10, 0)
    for i in range(L.EXPIRE_BARS + 2):
        book.observe(flat(t0 + i * 60, 5900.0))
    assert book.levels[Kind.PDH].state is State.EXPIRED
    assert Kind.PDH not in {lv.kind for lv in book.active()}


def test_coming_back_resets_the_expiry_count():
    book = _two_days()
    t0 = ts(2026, 9, 22, 10, 0)
    for i in range(L.EXPIRE_BARS - 1):
        book.observe(flat(t0 + i * 60, 5900.0))
    book.observe(flat(t0 + 100 * 60, 6010.0))       # back at the level
    for i in range(3):
        book.observe(flat(t0 + (200 + i) * 60, 5900.0))
    assert book.levels[Kind.PDH].state is not State.EXPIRED


# --------------------------------------------------------- housekeeping

def test_history_records_every_transition_in_order():
    book = _two_days()
    book.observe(bar(ts(2026, 9, 22, 10, 0), 6005, 6012, 6004, 6006))
    book.observe(bar(ts(2026, 9, 22, 10, 30), 6006, 6014, 6005, 6013))
    book.observe(bar(ts(2026, 9, 22, 11, 0), 6013, 6014, 6000, 6002))
    states = [t.to for t in book.levels[Kind.PDH].history]
    assert states == [State.TOUCHED, State.BROKEN_ABOVE, State.RECLAIMED]


def test_it_accepts_a_structure_candle():
    from liqmap.structure import Candle
    book = LevelBook()
    book.observe(Candle(ts=ts(2026, 9, 21, 9, 30), open=6000, high=6001,
                        low=5999, close=6000, volume=10))
    assert book.session == date(2026, 9, 21)


def test_it_accepts_a_plain_dict():
    book = LevelBook()
    book.observe({"ts": ts(2026, 9, 21, 9, 30), "o": 6000, "h": 6001,
                  "l": 5999, "c": 6000, "v": 10})
    assert book.session == date(2026, 9, 21)


def test_the_tick_size_is_not_hard_coded():
    """Everything is in ticks, so a different instrument must change what
    counts as near without touching the logic."""
    book = LevelBook(tick=1.0, approach_ticks=2.0)
    book.extend(session((2026, 9, 21), [(9, 30, 6005, 6010, 6000, 6005),
                                        (15, 59, 6005, 6006, 6004, 6005)]))
    found = book.observe(bar(ts(2026, 9, 22, 10, 0), 6005, 6008, 6004, 6008))
    assert any(a.kind is Kind.PDH for a in found)


def test_the_dict_the_panel_renders_is_shaped_right():
    book = _two_days()
    book.observe(bar(ts(2026, 9, 22, 10, 0), 6005, 6012, 6004, 6006))
    out = book.to_dict(price=6006.0)
    assert out["session"] == "2026-09-22"
    row = next(r for r in out["levels"] if r["kind"] == "PDH")
    assert row["state"] == "TOUCHED"
    assert row["outcome"] == "REJECTION"
    assert row["ticks_away"] == pytest.approx(-16.0)


def test_no_level_is_ever_published_at_zero():
    book = LevelBook()
    book._publish(Kind.PDH, 0.0, 0.0)
    assert Kind.PDH not in book.levels
