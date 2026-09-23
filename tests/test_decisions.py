"""Suggestions, the decisions taken about them, and uploaded datasets.

The point of storing IGNORED suggestions is the whole reason this table
exists. A record of only the trades that were taken can tell you how those
did; it can never tell you whether the filter that rejected the others was
worth anything. Those two questions have opposite answers often enough that
guessing is expensive.

Pinned here:

IGNORED SUGGESTIONS ARE SETTLED TOO. Otherwise the comparison is between
taken trades and nothing.

A DECISION FREEZES WHEN THE CANDLE CLOSES. Not when the row happens to have
been settled -- keying the freeze off `resolved` meant that any delay in the
resolver left every past suggestion editable, and choosing whether you "took"
a trade once you can see how it went is not a decision.

P&L IS NET OF COST. The gate that refuses a trade whose target does not clear
the round trip is the centre of this tool; a scoreboard that then forgets the
round trip turns a losing log into a break-even one.

A SMALL SAMPLE REFUSES TO DRAW A CONCLUSION. Twelve trades is an impression,
and replacing an impression with a measurement was the point.
"""

import tempfile
import time
from pathlib import Path

import pytest

from liqmap.history import History
from liqmap.ingest import parse
from liqmap.structure import Candle


@pytest.fixture
def hist():
    with tempfile.TemporaryDirectory() as tmp:
        h = History(Path(tmp) / "h.db")
        yield h
        h.close()


COST = 2.0

# Candles are dated relative to now, because `decide` freezes on the candle
# clock. A fixture timestamped in 1970 is a closed candle and every decision
# against it is correctly refused.
NOW = time.time()
OPEN_TS = NOW - 300          # a 15m candle that opened five minutes ago


def add(hist, candle_ts=OPEN_TS, side="long", entry=100.0, target=101.0,
        stop=99.5, coin="BTC", interval="15m"):
    return hist.record_suggestion(
        coin=coin, interval=interval, candle_ts=candle_ts,
        candle_end=candle_ts + 900, side=side, entry=entry,
        target_px=target, stop_px=stop, target_bps=100.0, risk_bps=50.0,
        rr=2.0, cost_bps=COST, conviction=0.7, score=0.55, reason="tilt",
        made_at=candle_ts + 60)


def fill(hist, n, decision, outcome, start=OPEN_TS):
    """n settled suggestions, all taken or all ignored, all won or all lost."""
    for i in range(n):
        sid = add(hist, candle_ts=start + i * 900)
        assert hist.decide(sid, decision == "taken", now=start + i * 900)
        if outcome == "target":
            hist.resolve_suggestion(sid, high=102.0, low=99.9, close=101.5)
        else:
            hist.resolve_suggestion(sid, high=100.1, low=99.0, close=99.2)


# --------------------------------------------------------------------------
# recording
# --------------------------------------------------------------------------

def test_a_suggestion_round_trips(hist):
    sid = add(hist)
    assert sid
    row = hist.open_suggestion("BTC", "15m", OPEN_TS)
    assert row["side"] == "long" and row["decision"] == "pending"


def test_one_suggestion_per_candle(hist):
    """Polling every few seconds must not write ninety rows for one trade."""
    assert add(hist) is not None
    assert add(hist) is None
    assert len(hist.recent_suggestions()) == 1


def test_different_candles_and_markets_are_separate(hist):
    assert add(hist, candle_ts=OPEN_TS) is not None
    assert add(hist, candle_ts=OPEN_TS + 900) is not None
    assert add(hist, candle_ts=OPEN_TS, coin="ETH") is not None
    assert add(hist, candle_ts=OPEN_TS, interval="5m") is not None
    assert len(hist.recent_suggestions()) == 4


# --------------------------------------------------------------------------
# decisions
# --------------------------------------------------------------------------

def test_taking_and_ignoring_are_both_recorded(hist):
    a, b = add(hist, candle_ts=OPEN_TS), add(hist, candle_ts=OPEN_TS + 900)
    assert hist.decide(a, True) and hist.decide(b, False)
    rows = {r["id"]: r["decision"] for r in hist.recent_suggestions()}
    assert rows[a] == "taken" and rows[b] == "ignored"


def test_a_decision_can_be_changed_while_the_trade_is_open(hist):
    """A misclick should be fixable."""
    sid = add(hist)
    hist.decide(sid, True)
    assert hist.decide(sid, False)
    assert hist.open_suggestion("BTC", "15m", OPEN_TS)["decision"] == "ignored"


def test_a_decision_freezes_once_the_outcome_is_known(hist):
    """Otherwise every losing trade becomes one that was ignored."""
    sid = add(hist)
    hist.decide(sid, True)
    hist.resolve_suggestion(sid, high=100.1, low=99.0, close=99.2)
    assert hist.decide(sid, False) is False
    assert hist.open_suggestion("BTC", "15m", OPEN_TS)["decision"] == "taken"


def test_deciding_on_a_suggestion_that_does_not_exist_is_false_not_a_crash(hist):
    assert hist.decide("nope", True) is False


# --------------------------------------------------------------------------
# settling
# --------------------------------------------------------------------------

def test_a_long_that_reached_target_settles_positive(hist):
    sid = add(hist)
    assert hist.resolve_suggestion(sid, high=102.0, low=99.9, close=101.5)
    row = hist.open_suggestion("BTC", "15m", OPEN_TS)
    assert row["outcome"] == "target" and row["pnl_bps"] > 0


def test_a_long_that_hit_the_stop_settles_negative(hist):
    sid = add(hist)
    hist.resolve_suggestion(sid, high=100.1, low=99.0, close=99.2)
    row = hist.open_suggestion("BTC", "15m", OPEN_TS)
    assert row["outcome"] == "stop" and row["pnl_bps"] < 0


def test_a_short_settles_with_the_sign_the_right_way_round(hist):
    """The single most dangerous sign error available: a winning short
    recorded as a loss inverts every conclusion drawn afterwards."""
    sid = add(hist, side="short", entry=100.0, target=99.0, stop=100.5)
    hist.resolve_suggestion(sid, high=100.2, low=98.5, close=98.8)
    row = hist.open_suggestion("BTC", "15m", OPEN_TS)
    assert row["outcome"] == "target" and row["pnl_bps"] > 0


def test_a_trade_still_open_at_the_close_is_marked_to_market(hist):
    sid = add(hist)
    hist.resolve_suggestion(sid, high=100.5, low=99.8, close=100.4)
    row = hist.open_suggestion("BTC", "15m", OPEN_TS)
    assert row["outcome"] == "neither"
    # Gross is the move; pnl is net of the round trip this suggestion was
    # priced against.
    assert row["gross_bps"] == pytest.approx(40.0, abs=1.0)
    assert row["pnl_bps"] == pytest.approx(40.0 - COST, abs=1.0)


def test_both_levels_touched_settles_as_the_stop(hist):
    sid = add(hist)
    hist.resolve_suggestion(sid, high=102.0, low=99.0, close=101.0)
    assert hist.open_suggestion("BTC", "15m", OPEN_TS)["outcome"] == "stop"


def test_pending_suggestions_only_lists_closed_candles(hist):
    add(hist, candle_ts=OPEN_TS)         # ends at 1900
    add(hist, candle_ts=OPEN_TS + 8000)         # ends at 9900
    assert len(hist.pending_suggestions(now=OPEN_TS + 2000)) == 1
    assert len(hist.pending_suggestions(now=OPEN_TS + 99999)) == 2


def test_resolving_twice_does_not_double_count(hist):
    sid = add(hist)
    hist.resolve_suggestion(sid, high=102.0, low=99.9, close=101.5)
    assert hist.pending_suggestions(now=OPEN_TS + 99999) == []


# --------------------------------------------------------------------------
# the comparison that is the point of all this
# --------------------------------------------------------------------------

def test_hit_rate_ignores_trades_that_reached_neither_level(hist):
    """One that expired mid-range was neither right nor wrong. Counting it
    as a loss makes a selective tool look worse the more patient it is."""
    sid = add(hist, candle_ts=OPEN_TS)
    hist.resolve_suggestion(sid, high=102.0, low=99.9, close=101.5)
    sid2 = add(hist, candle_ts=OPEN_TS + 900)
    hist.resolve_suggestion(sid2, high=100.2, low=99.9, close=100.1)

    s = hist.suggestion_stats()["overall"]
    assert s["n"] == 2 and s["targets"] == 1 and s["open_at_close"] == 1
    assert s["hit_rate"] == 1.0


def test_a_small_sample_refuses_to_conclude(hist):
    fill(hist, 5, "taken", "target")
    fill(hist, 5, "ignored", "stop", start=OPEN_TS + 100_000)
    v = hist.suggestion_stats()["verdict"]
    assert "noise" in v and "20 of each" in v


def test_a_filter_that_helps_is_reported_as_helping(hist):
    fill(hist, 25, "taken", "target")
    fill(hist, 25, "ignored", "stop", start=OPEN_TS + 100_000)
    v = hist.suggestion_stats()["verdict"]
    assert "adding" in v and "passing on the worse ones" in v


def test_a_filter_that_hurts_is_reported_bluntly(hist):
    """The finding nobody wants and everybody needs."""
    fill(hist, 25, "taken", "stop")
    fill(hist, 25, "ignored", "target", start=OPEN_TS + 100_000)
    v = hist.suggestion_stats()["verdict"]
    assert "BETTER" in v and "costing you" in v


def test_no_difference_is_reported_as_no_difference(hist):
    for i in range(30):
        sid = add(hist, candle_ts=OPEN_TS + i * 900)
        hist.decide(sid, i % 2 == 0)
        if i % 3:
            hist.resolve_suggestion(sid, high=102.0, low=99.9, close=101.5)
        else:
            hist.resolve_suggestion(sid, high=100.1, low=99.0, close=99.2)
    v = hist.suggestion_stats()["verdict"]
    assert "No real difference" in v or "Needs 20" in v


def test_nothing_settled_says_so(hist):
    assert "Nothing settled yet" in hist.suggestion_stats()["verdict"]


def test_stats_split_taken_from_ignored(hist):
    fill(hist, 3, "taken", "target")
    fill(hist, 2, "ignored", "stop", start=OPEN_TS + 100_000)
    s = hist.suggestion_stats()
    assert s["taken"]["n"] == 3 and s["ignored"]["n"] == 2
    assert s["taken"]["hit_rate"] == 1.0 and s["ignored"]["hit_rate"] == 0.0


def test_stats_can_be_filtered_by_market(hist):
    fill(hist, 3, "taken", "target")
    sid = hist.record_suggestion(
        coin="ETH", interval="15m", candle_ts=OPEN_TS,
        candle_end=OPEN_TS + 900,
        side="long", entry=100.0, target_px=101.0, stop_px=99.5,
        target_bps=100.0, risk_bps=50.0, rr=2.0, cost_bps=2.0,
        conviction=0.7, score=0.55)
    hist.resolve_suggestion(sid, high=102.0, low=99.9, close=101.5)
    assert hist.suggestion_stats(coin="BTC")["overall"]["n"] == 3
    assert hist.suggestion_stats(coin="ETH")["overall"]["n"] == 1


def test_counts_include_the_decision_tallies(hist):
    fill(hist, 2, "taken", "target")
    fill(hist, 3, "ignored", "stop", start=OPEN_TS + 100_000)
    c = hist.counts()
    assert c["suggestions"] == 5 and c["taken"] == 2 and c["ignored"] == 3


# --------------------------------------------------------------------------
# uploaded datasets
# --------------------------------------------------------------------------

def rows(n=20, start=1_700_000_000, step=900):
    head = "time,open,high,low,close,volume"
    body = [f"{start + i * step},100,101,99,100.5,10" for i in range(n)]
    return "\n".join([head] + body)


def test_a_dataset_round_trips_through_storage(hist):
    ing = parse(rows(50))
    did = hist.save_dataset("btc.csv", "BTC", ing.interval_name,
                            ing.interval_s, ing.candles, ing.describe())
    out = hist.dataset(did)
    assert len(out["candles"]) == 50
    assert isinstance(out["candles"][0], Candle)
    assert out["candles"][0].ts == ing.candles[0].ts
    assert out["interval"] == "15m"


def test_the_listing_does_not_carry_the_bars(hist):
    """A listing that loads every candle of every dataset gets slow exactly
    when there is enough history to be useful."""
    ing = parse(rows(50))
    hist.save_dataset("a.csv", "BTC", "15m", 900.0, ing.candles)
    listed = hist.datasets()
    assert len(listed) == 1 and "candles" not in listed[0]
    assert listed[0]["bars"] == 50


def test_uploading_an_overlapping_file_extends_rather_than_duplicates(hist):
    first = parse(rows(20)).candles
    did = hist.save_dataset("a.csv", "BTC", "15m", 900.0, first)
    second = parse(rows(20, start=1_700_000_000 + 10 * 900)).candles
    hist.save_dataset("a.csv", "BTC", "15m", 900.0, second, merge_into=did)

    out = hist.dataset(did)
    assert len(out["candles"]) == 30
    assert len({c.ts for c in out["candles"]}) == 30


def test_merging_into_a_dataset_that_is_gone_creates_a_new_one(hist):
    ing = parse(rows(20))
    did = hist.save_dataset("a.csv", "BTC", "15m", 900.0, ing.candles,
                            merge_into="missing")
    assert hist.dataset(did) is not None


def test_an_empty_dataset_is_refused_rather_than_stored(hist):
    with pytest.raises(ValueError):
        hist.save_dataset("empty.csv", "BTC", "15m", 900.0, [])


def test_datasets_can_be_deleted(hist):
    did = hist.save_dataset("a.csv", "BTC", "15m", 900.0, parse(rows()).candles)
    assert hist.delete_dataset(did) is True
    assert hist.dataset(did) is None
    assert hist.delete_dataset(did) is False


def test_datasets_filter_by_coin(hist):
    hist.save_dataset("a.csv", "BTC", "15m", 900.0, parse(rows()).candles)
    hist.save_dataset("b.csv", "ETH", "15m", 900.0, parse(rows()).candles)
    assert len(hist.datasets(coin="BTC")) == 1
    assert len(hist.datasets()) == 2


def test_a_missing_dataset_reads_as_none(hist):
    assert hist.dataset("nope") is None


# --------------------------------------------------------------------------
# what the adversarial review found
# --------------------------------------------------------------------------

def test_a_decision_cannot_be_made_after_the_candle_closed(hist):
    """The hindsight hole. `decide` used to freeze on `resolved`, which
    nothing set until a resolver ran — so with the resolver unwired, every
    past suggestion stayed editable forever and could be marked "taken" an
    hour later with the result visible on any chart."""
    closed = NOW - 10_000                      # ended long ago
    sid = hist.record_suggestion(
        coin="BTC", interval="15m", candle_ts=closed,
        candle_end=closed + 900, side="long", entry=100.0, target_px=101.0,
        stop_px=99.5, target_bps=100.0, risk_bps=50.0, rr=2.0,
        cost_bps=COST, conviction=0.7, score=0.55)
    assert hist.decide(sid, True) is False
    assert hist.open_suggestion("BTC", "15m", closed)["decision"] == "pending"


def test_a_decision_is_still_allowed_while_the_candle_runs(hist):
    sid = add(hist)
    assert hist.decide(sid, True) is True


def test_settling_mid_call_cannot_be_overtaken_by_a_decision(hist):
    """The UPDATE carries the same condition as the check, so a settle that
    lands between them cannot be raced."""
    sid = add(hist)
    hist.resolve_suggestion(sid, high=102.0, low=99.9, close=101.5)
    assert hist.decide(sid, True) is False


def test_cost_is_netted_off_every_outcome(hist):
    """Two winners and two losers at plus and minus 100bps, 2bps of cost
    each, must total minus eight — not zero."""
    # Symmetric levels, so gross nets to exactly zero and anything left is
    # the cost.
    for i in range(2):
        sid = add(hist, candle_ts=OPEN_TS + i * 900, stop=99.0)
        hist.resolve_suggestion(sid, high=102.0, low=99.9, close=101.5)
    for i in range(2, 4):
        sid = add(hist, candle_ts=OPEN_TS + i * 900, stop=99.0)
        hist.resolve_suggestion(sid, high=100.1, low=98.9, close=99.0)

    s = hist.suggestion_stats()["overall"]
    assert s["hit_rate"] == 0.5
    assert s["total_pnl_bps"] == pytest.approx(-4 * COST, abs=0.01)
    assert s["cost_drag_bps"] == pytest.approx(4 * COST, abs=0.01)
    assert s["avg_gross_bps"] == pytest.approx(0.0, abs=0.01)


def test_an_abandoned_suggestion_leaves_the_queue_without_entering_the_score(hist):
    """A row too old to settle must not block the queue behind it, and must
    not be counted as an outcome either."""
    sid = add(hist)
    assert hist.abandon_suggestion(sid) is True
    assert hist.pending_suggestions(now=NOW + 99999) == []
    assert hist.suggestion_stats()["overall"]["n"] == 0
    assert hist.abandon_suggestion(sid) is False


def test_pending_is_filtered_the_same_way_as_everything_beside_it(hist):
    add(hist, coin="BTC")
    add(hist, coin="ETH")
    assert hist.suggestion_stats(coin="BTC")["pending"] == 1
    assert hist.suggestion_stats()["pending"] == 2


def test_merging_a_different_market_into_a_dataset_is_refused(hist):
    """One series that is neither BTC 1h nor ETH 1m, labelled as whichever
    metadata survived."""
    did = hist.save_dataset("btc.csv", "BTC", "1h", 3600.0,
                            parse(rows()).candles)
    with pytest.raises(ValueError):
        hist.save_dataset("eth.csv", "ETH", "1m", 60.0,
                          parse(rows()).candles, merge_into=did)

    kept = hist.datasets()[0]
    assert kept["coin"] == "BTC" and kept["interval"] == "1h"


def test_a_merge_updates_the_metadata_it_should(hist):
    did = hist.save_dataset("a.csv", "BTC", "15m", 900.0, parse(rows(20)).candles)
    hist.save_dataset("a-extended.csv", "BTC", "15m", 900.0,
                      parse(rows(20, start=1_700_000_000 + 10 * 900)).candles,
                      merge_into=did, report="second")
    meta = hist.datasets()[0]
    assert meta["name"] == "a-extended.csv" and meta["bars"] == 30
