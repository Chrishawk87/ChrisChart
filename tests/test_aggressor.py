"""Which side crossed, and whether a feed's label can be trusted about it.

This is the check that stands between the engine and being confidently
backwards, so the ways it could quietly pass are what get tested.

THE LABEL MUST NOT BE ASSUMED TO MEAN ANYTHING. 'Ask' can mean the
aggressor lifted the ask or that the resting side was the ask -- the same
event from either end. A dictionary guesses; this has to measure.

THE BOOK MUST BE THE ONE BEFORE THE TRADE. Sampled after, the taken level
is already gone, the mid has drifted towards the aggressor, and EVERY
convention looks correct.

AND MID PRINTS MUST BE DISCARDED, NOT SPLIT. In a one-tick market they are
most of the tape; counting them as half agreement drags every verdict into
'unclear'.
"""

from __future__ import annotations

import pytest

from liqmap import aggressor as ag


def tape(c, n=60, bid=100.0, ask=100.25, buyer="B", seller="S",
         flip=False):
    """`n` prints, half from each side, correctly labelled.

    `flip` inverts the convention: the label that should be lifting the
    offer prints below the mid instead.
    """
    hi, lo = ask, bid
    for i in range(n):
        if i % 2 == 0:
            c.observe(lo if flip else hi, buyer, bid, ask)
        else:
            c.observe(hi if flip else lo, seller, bid, ask)
    return c


# -------------------------------------------------- the convention falls out

def test_a_correctly_labelled_tape_identifies_its_buyer():
    r = tape(ag.Convention()).report()
    assert r.verdict == "b_is_buyer"
    assert r.buyer_label() == "b"
    assert r.settled
    assert "aggressive buyer" in r.describe()


def test_an_inverted_tape_is_caught_rather_than_believed():
    """THE failure this exists for. Nothing crashes; every delta is
    simply sign-flipped."""
    r = tape(ag.Convention(), flip=True).report()
    assert r.verdict == "a_is_buyer"
    assert r.buyer_label() == "a"


def test_the_label_itself_carries_no_meaning():
    """'Bid'/'Ask' is the ambiguous pair, and the tape settles it either
    way round without the strings changing."""
    straight = tape(ag.Convention(), buyer="Bid", seller="Ask").report()
    inverted = tape(ag.Convention(), buyer="Bid", seller="Ask",
                    flip=True).report()
    assert straight.verdict == "b_is_buyer"
    assert inverted.verdict == "a_is_buyer"


@pytest.mark.parametrize("b,s", [("B", "S"), ("BUY", "SELL"), ("0", "1"),
                                 ("Bid", "Ask"), ("buyer", "seller"),
                                 ("+1", "-1")])
def test_the_common_labellings_all_separate_into_two_buckets(b, s):
    r = tape(ag.Convention(), buyer=b, seller=s).report()
    assert r.verdict == "b_is_buyer", (b, s)
    assert r.b_total and r.a_total


def test_a_label_it_does_not_recognise_counts_as_unlabelled():
    """Better than guessing it into a bucket, which would put half the
    tape on the wrong side and read as 'unclear'."""
    c = ag.Convention()
    for _ in range(50):
        c.observe(100.25, "???", 100.0, 100.25)
    r = c.report()
    assert r.unlabelled == 50
    assert r.verdict == "no label"


# --------------------------------------------------------- what is discarded

def test_prints_at_the_mid_decide_nothing():
    """In a one-tick market they are most of the tape."""
    c = ag.Convention()
    for _ in range(100):
        c.observe(100.125, "B", 100.0, 100.25)      # exactly the mid
    r = c.report()
    assert r.seen == 100
    assert r.quotable == 100
    assert r.off_mid == 0
    assert r.n == 0
    assert r.verdict == "not enough samples"


def test_mid_prints_do_not_drag_a_clean_tape_into_unclear():
    """If they were split half-and-half instead of discarded, a tape that
    is two-thirds mid prints could never reach a verdict."""
    c = ag.Convention()
    for _ in range(120):
        c.observe(100.125, "B", 100.0, 100.25)
    tape(c)
    assert c.report().verdict == "b_is_buyer"


def test_a_print_with_no_book_beside_it_is_not_counted():
    c = ag.Convention()
    for _ in range(60):
        c.observe(100.25, "B", 0.0, 0.0)
    r = c.report()
    assert r.seen == 60 and r.quotable == 0 and r.n == 0


def test_a_crossed_or_nonsense_book_is_refused():
    c = ag.Convention()
    for _ in range(60):
        c.observe(100.25, "B", 101.0, 100.0)        # bid above ask
    assert c.report().quotable == 0


def test_a_print_with_no_price_is_not_even_seen():
    c = ag.Convention()
    c.observe(0.0, "B", 100.0, 100.25)
    c.observe(-5.0, "B", 100.0, 100.25)
    assert c.report().seen == 0


# ------------------------------------------------- not enough, and not sure

def test_a_handful_of_prints_settles_nothing():
    c = ag.Convention()
    tape(c, n=10)
    r = c.report()
    assert r.n == 10
    assert r.verdict == "not enough samples"
    assert not r.settled
    assert str(ag.MIN_SAMPLES) in r.describe()


def test_only_one_side_ever_printing_settles_nothing():
    """One bucket being lopsided can be a thin sample. The two
    disagreeing with each other is the convention."""
    c = ag.Convention()
    for _ in range(80):
        c.observe(100.25, "B", 100.0, 100.25)
    r = c.report()
    assert r.b_total == 80 and r.a_total == 0
    assert r.verdict == "not enough samples"


def test_both_labels_printing_high_is_unclear_not_a_verdict():
    """The case the second half of the test exists for, and a realistic
    one: a feed that labels each trade by the book side it occurred on,
    in a market grinding up, puts BOTH buckets above the mid.

    Looking only at whether 'b' is lopsided calls that a clean buyer
    convention. It is a label that means nothing.
    """
    c = ag.Convention()
    for i in range(100):
        c.observe(100.25, "B" if i % 2 else "S", 100.0, 100.25)
    r = c.report()
    assert r.b_share > ag.CLEAR and r.a_share > ag.CLEAR
    assert r.verdict == "unclear"
    assert r.buyer_label() is None


def test_both_labels_printing_low_is_unclear_too():
    c = ag.Convention()
    for i in range(100):
        c.observe(100.0, "B" if i % 2 else "S", 100.0, 100.25)
    r = c.report()
    assert r.b_share < 1.0 - ag.CLEAR and r.a_share < 1.0 - ag.CLEAR
    assert r.verdict == "unclear"


def test_labels_that_do_not_line_up_with_the_book_are_called_unclear():
    """Not quietly rounded to whichever side is ahead."""
    c = ag.Convention()
    for i in range(80):
        # Both labels print above the mid about half the time.
        c.observe(100.25 if i % 2 else 100.0, "B" if i % 3 else "S",
                  100.0, 100.25)
    r = c.report()
    assert r.verdict == "unclear"
    assert not r.settled
    assert "do not trust" in r.describe()


# ------------------------------------- the ceiling on an inferred delta

def test_a_feed_with_no_side_field_reports_how_far_inference_could_go():
    """Not a failure -- a different situation, and the useful number is
    what fraction of prints could be assigned from the quote at all."""
    c = ag.Convention()
    for i in range(100):
        px = 100.125 if i < 40 else (100.25 if i % 2 else 100.0)
        c.observe(px, None, 100.0, 100.25)
    r = c.report()
    assert r.verdict == "no label"
    assert r.unlabelled == 100
    assert r.assignable == pytest.approx(0.6)
    assert "60%" in r.describe()


def test_the_assignable_share_is_reported_even_when_a_label_exists():
    """Worth knowing either way: it bounds what a fallback would give if
    the label ever stopped arriving."""
    r = tape(ag.Convention()).report()
    assert r.assignable == pytest.approx(1.0)


def test_nothing_quotable_is_a_share_of_zero_rather_than_a_crash():
    c = ag.Convention()
    c.observe(100.0, "B", 0.0, 0.0)
    assert c.report().assignable == 0.0


def test_the_ceiling_is_measured_against_prints_that_had_a_book():
    """Against every print seen, a feed whose book arrives late reports a
    ceiling lower than the truth -- the quote could not have been
    consulted for those at all, so they are not evidence either way.

    Forty prints with no book, sixty with one, all sixty off the mid:
    the ceiling is 100%, not 60%.
    """
    c = ag.Convention()
    for _ in range(40):
        c.observe(100.25, None, 0.0, 0.0)
    for _ in range(60):
        c.observe(100.25, None, 100.0, 100.25)
    r = c.report()
    assert r.seen == 100 and r.quotable == 60 and r.off_mid == 60
    assert r.assignable == pytest.approx(1.0)


# ------------------------------------------------------------------ limits

def test_it_stops_accumulating_at_the_cap():
    c = ag.Convention(keep=50)
    tape(c, n=200)
    assert c.report().seen == 50


def test_nothing_here_places_an_order():
    import inspect

    src = inspect.getsource(ag).lower()
    for word in ("def submit", "def execute", "def place", "def order"):
        assert word not in src


def test_the_dict_carries_the_verdict_and_the_counts_behind_it():
    d = tape(ag.Convention()).report().to_dict()
    assert d["verdict"] == "b_is_buyer" and d["buyer"] == "b"
    for k in ("n", "seen", "b_share", "a_share", "assignable", "describe"):
        assert k in d
