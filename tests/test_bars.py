"""OHLCV loading, resampling and the roll.

Two things here would silently corrupt every downstream result: a
resampler that loses extremes, and a contract roll treated as a price move.
Neither raises. The first shrinks every bar's range so stops look safer than
they are; the second manufactures a breakout out of a bookkeeping change.
"""

from __future__ import annotations

import pytest

from liqmap import bars as B
from liqmap.levels import ET

import datetime as dt


def rec(ts, o, h, l, c, v=100):
    return {"ts_event": int(ts * 1e9), "open": int(o * 1e9),
            "high": int(h * 1e9), "low": int(l * 1e9),
            "close": int(c * 1e9), "volume": v}


def rth(y, m, d, hh=9, mm=30):
    return dt.datetime(y, m, d, hh, mm, tzinfo=ET).timestamp()


def bar(ts, o, h, l, c, v=100.0):
    return B.Bar(ts=ts, open=o, high=h, low=l, close=c, volume=v)


# --------------------------------------------------------- reading

def test_prices_come_through_the_same_scaling_as_the_book_schemas():
    out = list(B.from_records([rec(1_700_000_000, 7000.25, 7001.0,
                                   7000.0, 7000.75, 42)]))
    assert len(out) == 1
    assert out[0].open == pytest.approx(7000.25)
    assert out[0].high == pytest.approx(7001.0)
    assert out[0].volume == 42


def test_a_malformed_bar_is_dropped_rather_than_carried():
    """High below low, or an open outside its own range, means the record
    was misread -- and a profile built from one puts volume at prices that
    never traded."""
    bad = [rec(1, 7000, 6999, 7001, 7000),          # high < low
           rec(2, 7050, 7001, 7000, 7000.5),        # open above high
           rec(3, 7000, 7001, 7000, 7099)]          # close above high
    assert list(B.from_records(bad)) == []


def test_a_zero_priced_record_is_dropped():
    assert list(B.from_records([rec(1, 0, 0, 0, 0)])) == []


def test_a_valid_bar_passes_its_own_check():
    assert bar(1.0, 7000.0, 7001.0, 6999.0, 7000.5).valid


# ------------------------------------------------------- resampling

def test_resampling_keeps_the_extremes():
    """The failure that matters: a resampler that takes the first and last
    prices instead of the true high and low shrinks every range, and every
    stop in the simulator then looks safer than it was."""
    ones = [bar(0.0, 7000.0, 7000.25, 7000.0, 7000.25),
            bar(1.0, 7000.25, 7005.0, 6995.0, 7001.0),   # the real extremes
            bar(2.0, 7001.0, 7001.25, 7000.75, 7001.0),
            bar(3.0, 7001.0, 7001.5, 7000.5, 7001.25),
            bar(4.0, 7001.25, 7001.5, 7001.0, 7001.5)]
    out = B.resample(ones, 5.0)
    assert len(out) == 1
    assert out[0].high == pytest.approx(7005.0)
    assert out[0].low == pytest.approx(6995.0)


def test_resampling_takes_the_first_open_and_the_last_close():
    ones = [bar(float(i), 7000.0 + i, 7000.0 + i, 7000.0 + i, 7000.0 + i)
            for i in range(5)]
    out = B.resample(ones, 5.0)[0]
    assert out.open == pytest.approx(7000.0)
    assert out.close == pytest.approx(7004.0)


def test_resampling_sums_volume():
    ones = [bar(float(i), 7000.0, 7000.0, 7000.0, 7000.0, v=10.0)
            for i in range(5)]
    assert B.resample(ones, 5.0)[0].volume == pytest.approx(50.0)


def test_buckets_are_aligned_to_the_grid_not_to_the_first_bar():
    """Otherwise two files parsed from different start times disagree about
    where every bucket begins, and their profiles cannot be compared."""
    ones = [bar(1003.0 + i, 7000.0, 7000.0, 7000.0, 7000.0)
            for i in range(10)]
    out = B.resample(ones, 5.0)
    assert out[0].ts % 5.0 == 0


def test_a_gap_in_the_tape_does_not_merge_across_it():
    ones = [bar(0.0, 7000.0, 7000.0, 7000.0, 7000.0),
            bar(10_000.0, 7100.0, 7100.0, 7100.0, 7100.0)]
    assert len(B.resample(ones, 5.0)) == 2


def test_resampling_nothing_gives_nothing():
    assert B.resample([], 5.0) == []


# ------------------------------------------------------------ the roll

def _two_days(second_open):
    """Mid-July: deliberately clear of every quarterly expiry.

    The first version used 18-19 June, which IS the June roll window,
    so every session came back suspect for the right reason and these
    tests measured the calendar instead of the gap.
    """
    day1 = [bar(rth(2026, 7, 15) + i * 5, 7000.0, 7001.0, 6999.0, 7000.0)
            for i in range(60)]
    day2 = [bar(rth(2026, 7, 16) + i * 5, second_open, second_open + 1,
                second_open - 1, second_open) for i in range(60)]
    return day1 + day2


def test_an_ordinary_overnight_gap_is_not_flagged():
    out = B.sessions(_two_days(7002.0))
    assert not any(s.suspect for s in out)


def test_a_huge_gap_is_flagged_as_a_data_break():
    """Fifty points session-to-session is a discontinuity, not a market.

    This is the safety net, NOT the roll detector -- rolls are caught by
    the calendar below, and the two carry different reasons because they
    are different problems.
    """
    out = B.sessions(_two_days(7100.0))
    assert out[-1].suspect
    assert "prior close" in out[-1].reason


def test_an_ordinary_ten_point_overnight_move_is_not_flagged():
    """The bug that cost 69% of two years of data.

    The first threshold was 40 ticks -- ten ES points -- chosen on the
    reasoning that it sat above ordinary overnight drift. At ES 6000-7800
    a ten point overnight move is routine, and the detector dropped 352 of
    507 sessions on consecutive weekdays, none of them rolls.
    """
    assert not B.sessions(_two_days(7010.0))[-1].suspect


def test_the_roll_calendar_finds_the_quarterly_expiries():
    """Third Friday of March, June, September and December."""
    rolls = B.roll_dates(dt.date(2024, 9, 30), dt.date(2026, 9, 29))
    assert len(rolls) == 8
    assert all(r.weekday() == 4 for r in rolls)
    assert all(r.month in (3, 6, 9, 12) for r in rolls)
    assert dt.date(2026, 9, 18) in rolls


def test_a_session_in_the_roll_window_is_dropped_by_the_calendar():
    """Caught by DATE, not by price.

    A roll's price discontinuity can be small enough to pass any gap
    test while its effect on the prior session's profile is total --
    the level refers to a different contract either way.
    """
    day1 = [bar(rth(2026, 9, 17) + i * 5, 7000.0, 7001.0, 6999.0, 7000.0)
            for i in range(30)]
    day2 = [bar(rth(2026, 9, 18) + i * 5, 7000.25, 7001.0, 6999.0, 7000.0)
            for i in range(30)]
    out = B.sessions(day1 + day2)
    assert all(s.suspect for s in out)
    assert "roll" in out[-1].reason


def test_the_first_session_is_never_suspect():
    """There is no prior close to gap from."""
    assert not B.sessions(_two_days(7002.0))[0].suspect


def test_the_gap_is_measured_against_the_prior_close_not_the_prior_open():
    day1 = [bar(rth(2026, 7, 15) + i * 5, 7000.0, 7000.0, 7000.0, 7000.0)
            for i in range(30)]
    day1 += [bar(rth(2026, 7, 15) + 200 + i * 5, 7060.0, 7060.0, 7060.0,
                 7060.0) for i in range(30)]          # drifted up intraday
    day2 = [bar(rth(2026, 7, 16) + i * 5, 7061.0, 7061.0, 7061.0, 7061.0)
            for i in range(30)]
    out = B.sessions(day1 + day2)
    assert not out[-1].suspect          # 4 ticks from the CLOSE, not 244


def test_the_threshold_is_configurable_not_hard_coded():
    assert B.sessions(_two_days(7010.0), roll_gap_ticks=200.0)[-1].suspect \
        is False
    assert B.sessions(_two_days(7010.0), roll_gap_ticks=10.0)[-1].suspect


# --------------------------------------------------------- sessions

def test_only_the_regular_session_is_kept_by_default():
    overnight = [bar(dt.datetime(2026, 6, 18, 3, 0, tzinfo=ET).timestamp()
                     + i * 5, 7000.0, 7000.0, 7000.0, 7000.0)
                 for i in range(20)]
    day = [bar(rth(2026, 7, 15) + i * 5, 7000.0, 7000.0, 7000.0, 7000.0)
           for i in range(20)]
    out = B.sessions(overnight + day)
    assert sum(len(s.bars) for s in out) == 20


def test_overnight_can_be_kept_when_asked():
    overnight = [bar(dt.datetime(2026, 6, 18, 3, 0, tzinfo=ET).timestamp()
                     + i * 5, 7000.0, 7000.0, 7000.0, 7000.0)
                 for i in range(20)]
    out = B.sessions(overnight, rth_only=False)
    assert sum(len(s.bars) for s in out) == 20


def test_sessions_come_back_in_date_order_with_bars_sorted():
    shuffled = list(reversed(_two_days(7002.0)))
    out = B.sessions(shuffled)
    assert [s.day for s in out] == sorted(s.day for s in out)
    for s in out:
        assert [b.ts for b in s.bars] == sorted(b.ts for b in s.bars)


def test_the_session_range_is_reported_in_ticks():
    day = [bar(rth(2026, 7, 15) + i * 5, 7000.0, 7010.0, 6990.0, 7000.0)
           for i in range(10)]
    assert B.sessions(day)[0].range_ticks == pytest.approx(80.0)


def test_an_empty_input_gives_no_sessions():
    assert B.sessions([]) == []
