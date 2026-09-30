"""The Databento reader, against records built to the published schema.

No real DBN file is needed here, and that is deliberate: these pin the
translation rules that documentation can settle -- price scaling, undefined
levels, enum spellings, ordering -- so that when a real file does arrive the
only open question left is the aggressor convention, which no amount of
reading can settle and which `SideAudit` answers from the tape itself.
"""

from __future__ import annotations

import pytest

from liqmap import dbn
from liqmap.flow import Book, Level


def lvl(bid_px=None, bid_sz=0, ask_px=None, ask_sz=0, bid_ct=0, ask_ct=0):
    """One entry of the levels array, in raw fixed-precision."""
    return {"bid_px": dbn.UNDEF_PRICE if bid_px is None else int(bid_px * 1e9),
            "bid_sz": bid_sz, "bid_ct": bid_ct,
            "ask_px": dbn.UNDEF_PRICE if ask_px is None else int(ask_px * 1e9),
            "ask_sz": ask_sz, "ask_ct": ask_ct}


def rec(action="A", side="N", px=0.0, sz=0, ts=1_700_000_000.0, levels=None):
    return {"action": action, "side": side,
            "price": int(px * 1e9) if px else 0, "size": sz,
            "ts_event": int(ts * 1e9), "levels": levels or []}


def book10(best_bid=6000.00, best_ask=6000.25, sz=50):
    """A tidy ten-deep ES book one tick wide."""
    return [lvl(bid_px=best_bid - i * 0.25, bid_sz=sz,
                ask_px=best_ask + i * 0.25, ask_sz=sz) for i in range(10)]


# ------------------------------------------------------------- scaling

def test_a_price_is_scaled_by_one_billionth():
    # Databento's own worked example.
    assert dbn.price(5411750000000) == pytest.approx(5411.75)


def test_an_undefined_price_is_zero_not_nine_billion():
    """INT64_MAX scaled naively is 9.22e9. As an ask that would sort to the
    top of the book and make every spread and depth figure nonsense."""
    assert dbn.price(dbn.UNDEF_PRICE) == 0.0


def test_junk_and_missing_prices_are_zero_not_an_exception():
    assert dbn.price(None) == 0.0
    assert dbn.price("") == 0.0
    assert dbn.price(-5) == 0.0


def test_timestamps_come_back_in_seconds():
    assert dbn.seconds(1_700_000_000_123_456_789) == pytest.approx(
        1_700_000_000.123457, abs=1e-5)


# ------------------------------------------------------- enum spellings

@pytest.mark.parametrize("raw,want", [
    ("B", dbn.SIDE_BID), ("A", dbn.SIDE_ASK), ("N", dbn.SIDE_NONE),
    ("b", dbn.SIDE_BID), ("Side.BID", dbn.SIDE_BID),
    ("Side.ASK", dbn.SIDE_ASK), ("", dbn.SIDE_NONE),
])
def test_the_side_field_is_read_however_the_client_spells_it(raw, want):
    """The Python client may hand back an enum whose str is 'Side.BID'
    rather than a bare character. Taking the first character of that gives
    'S', which is neither side, and every trade would be dropped."""
    assert dbn._side({"side": raw}) == want


def test_an_enum_object_is_read_through_its_value():
    class FakeEnum:
        value = "B"
    assert dbn._side({"side": FakeEnum()}) == dbn.SIDE_BID


def test_the_action_field_is_read_the_same_way():
    assert dbn._action({"action": "T"}) == "T"
    assert dbn._action({"action": "Action.TRADE"}) == "T"


# ------------------------------------------------------------ the book

def test_ten_levels_come_through_on_both_sides():
    b = dbn.book_from(rec(levels=book10()))
    assert len(b.bids) == 10 and len(b.asks) == 10


def test_the_touch_is_where_it_should_be():
    b = dbn.book_from(rec(levels=book10()))
    assert b.best_bid == pytest.approx(6000.00)
    assert b.best_ask == pytest.approx(6000.25)
    assert b.mid == pytest.approx(6000.125)


def test_unpopulated_levels_are_dropped_rather_than_carried_as_zeros():
    levels = book10()[:3] + [lvl() for _ in range(7)]
    b = dbn.book_from(rec(levels=levels))
    assert len(b.bids) == 3 and len(b.asks) == 3
    assert all(l.px > 0 for l in b.bids + b.asks)


def test_a_level_with_price_but_no_size_is_dropped():
    b = dbn.book_from(rec(levels=[lvl(bid_px=6000.0, bid_sz=0,
                                      ask_px=6000.25, ask_sz=5)]))
    assert b.bids == [] and len(b.asks) == 1


def test_order_counts_are_carried_when_present():
    b = dbn.book_from(rec(levels=[lvl(bid_px=6000.0, bid_sz=50, bid_ct=7,
                                      ask_px=6000.25, ask_sz=40, ask_ct=3)]))
    assert b.bids[0].n == 7 and b.asks[0].n == 3


def test_depth_can_be_capped():
    b = dbn.book_from(rec(levels=book10()), depth=3)
    assert len(b.bids) == 3


def test_an_empty_record_makes_an_empty_book_not_a_crash():
    assert dbn.book_from(rec()).empty


# ----------------------------------------------------------- the trade

def test_a_bid_marked_trade_is_a_buy_aggressor():
    t = dbn.trade_from(rec(action="T", side="B", px=6000.25, sz=3))
    assert t is not None
    assert t.aggressor == "buy"
    assert t.px == pytest.approx(6000.25) and t.sz == 3


def test_an_ask_marked_trade_is_a_sell_aggressor():
    t = dbn.trade_from(rec(action="T", side="A", px=6000.0, sz=2))
    assert t.aggressor == "sell"


def test_the_flip_flag_inverts_the_mapping():
    """The remedy if the audit finds the convention reversed -- a flag,
    not an edit someone later 'corrects' back."""
    t = dbn.trade_from(rec(action="T", side="B", px=6000.25, sz=3), flip=True)
    assert t.aggressor == "sell"


def test_a_fill_counts_as_a_trade():
    assert dbn.trade_from(rec(action="F", side="B", px=6000.0, sz=1))


def test_book_updates_are_not_trades():
    for action in ("A", "C", "M"):
        assert dbn.trade_from(rec(action=action, side="B",
                                  px=6000.0, sz=1)) is None


def test_a_sideless_trade_is_dropped_not_guessed():
    """'N' means the feed did not say who crossed. Assigning a side would
    invent delta that nobody traded."""
    assert dbn.trade_from(rec(action="T", side="N", px=6000.0, sz=5)) is None


def test_a_zero_size_or_zero_price_trade_is_dropped():
    assert dbn.trade_from(rec(action="T", side="B", px=6000.0, sz=0)) is None
    assert dbn.trade_from(rec(action="T", side="B", px=0.0, sz=5)) is None


def test_signed_notional_follows_the_aggressor():
    buy = dbn.trade_from(rec(action="T", side="B", px=6000.0, sz=2))
    sell = dbn.trade_from(rec(action="T", side="A", px=6000.0, sz=2))
    assert buy.signed_notional > 0 > sell.signed_notional


# ------------------------------------------------------------ the audit

def _tape(n=200, correct=True):
    """Trades that lift the offer or hit the bid, book first each time."""
    out = []
    for i in range(n):
        out.append(rec(levels=book10()))
        buy = i % 2 == 0
        marked = ("B" if buy else "A") if correct else ("A" if buy else "B")
        out.append(rec(action="T", side=marked,
                       px=6000.25 if buy else 6000.00, sz=1,
                       levels=book10()))
    return out


def test_a_correctly_labelled_tape_reads_as_documented():
    report = dbn.audit(_tape(correct=True))
    assert report.verdict == "documented"
    assert report.ok and not report.flip
    assert report.agree > 0.9


def test_an_inverted_tape_is_caught():
    """The failure this module exists to catch. Nothing crashes on an
    inverted feed -- every delta simply reads backwards."""
    report = dbn.audit(_tape(correct=False))
    assert report.verdict == "inverted"
    assert report.flip and not report.ok


def test_a_coin_flip_tape_reads_as_unclear_not_as_a_pass():
    import random
    rng = random.Random(4)
    recs = []
    for i in range(300):
        recs.append(rec(levels=book10()))
        recs.append(rec(action="T", side=rng.choice(["B", "A"]),
                        px=rng.choice([6000.0, 6000.25]), sz=1,
                        levels=book10()))
    assert dbn.audit(recs).verdict == "unclear"


def test_too_few_samples_is_reported_rather_than_guessed():
    assert dbn.audit(_tape(n=3)).verdict == "not enough samples"


def test_trades_at_the_mid_are_discarded_not_split():
    """In a one-tick market these are most of the tape and say nothing
    about the convention. Counting them halves the signal."""
    recs = []
    for _ in range(100):
        recs.append(rec(levels=book10()))
        recs.append(rec(action="T", side="B", px=6000.125, sz=1,
                        levels=book10()))
    assert dbn.audit(recs).verdict == "not enough samples"


def test_the_audit_ignores_trades_with_no_prior_book():
    """The first trade in a file has nothing to be compared against."""
    a = dbn.SideAudit()
    a.observe(rec(action="T", side="B", px=6000.25, sz=1), None)
    assert a.n == 0


# ----------------------------------------------------------- streaming

def test_the_stream_yields_trades_and_books_in_order():
    out = list(dbn.stream([rec(levels=book10()),
                           rec(action="T", side="B", px=6000.25, sz=2,
                               levels=book10())]))
    kinds = [k for k, _ in out]
    assert kinds == ["book", "trade", "book"]


def test_a_trade_is_emitted_before_the_book_that_followed_it():
    """MBP-10 carries the book *after* the fill. Emitting it first would
    show a reader the consequence of a trade at the moment it is deciding
    about that trade -- a one-record look-ahead leak."""
    out = list(dbn.stream([rec(action="T", side="B", px=6000.25, sz=2,
                               levels=book10())]))
    assert [k for k, _ in out] == ["trade", "book"]


def test_a_clear_is_surfaced_rather_than_silently_skipped():
    """A book reset mid-session is real -- carrying stale levels across one
    would invent liquidity that no longer exists."""
    out = list(dbn.stream([rec(action="R", ts=1_700_000_000.0)]))
    assert out[0][0] == "clear"


def test_streaming_an_empty_file_yields_nothing():
    assert list(dbn.stream([])) == []
