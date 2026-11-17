"""A fill is priced at the side you cross, from a book you can trust.

Chris: "when i log a buy or sell hit the tp/sl or close the trade it
definitely dont match the real entry and exit prices", and later
"sometimes the entry price is far below or above where price is".

TWO FAULTS, MONTHS APART, SAME COLUMN OF NUMBERS.

    The engine had ONE price -- `tape.last_px` -- and used it for
    the entry fill, the exit fill, and both the stop and the target.
    A market buy lifts the offer and a market sell hits the bid, so
    that number is the wrong price for at least one side of every
    round trip, and the log never said so.

    Then, once fills were priced off the touch, the BOOK had no
    timestamp. It is replaced whenever a record carries levels and
    nothing recorded when, so the staleness check aged it by
    `last_ts` -- the last TRADE. Trades and book updates are
    separate record streams: prints can keep arriving while the
    depth stalls, and the check passed on the trades' freshness
    while the bid and ask were from a price that had gone.

This file was lost once already: it was written, it caught both
faults, and it never reached the repository because the patch that
shipped the fix covered `liqmap/` and not `tests/`.
"""

from __future__ import annotations

import pytest

from liqmap import quote as q

BID, ASK, LAST = 7477.75, 7478.00, 7477.75


class TestTheFourPrices:
    @pytest.mark.parametrize("side,action,want", [
        ("up", "open", ASK),        # buy to open   -> pay the offer
        ("up", "close", BID),       # sell to close -> hit the bid
        ("down", "open", BID),      # sell to open  -> hit the bid
        ("down", "close", ASK),     # buy to close  -> pay the offer
    ])
    def test_you_cross_to_get_filled(self, side, action, want):
        assert q.fill_px(side, action, BID, ASK, LAST)[0] == want

    def test_a_round_trip_costs_the_spread_before_anything_moves(self):
        """The old reading made entry free. It is not."""
        o = q.fill_px("up", "open", BID, ASK, LAST)[0]
        c = q.fill_px("up", "close", BID, ASK, LAST)[0]
        assert (c - o) / q.TICK == -1.0, "a round trip came out free"
        assert q.round_trip_ticks(BID, ASK) == 2.0

    def test_it_costs_the_same_going_the_other_way(self):
        o = q.fill_px("down", "open", BID, ASK, LAST)[0]
        c = q.fill_px("down", "close", BID, ASK, LAST)[0]
        assert -(c - o) / q.TICK == -1.0

    def test_the_source_is_reported_so_a_fill_can_be_audited(self):
        assert q.fill_px("up", "open", BID, ASK, LAST)[1] == "ask"
        assert q.fill_px("up", "close", BID, ASK, LAST)[1] == "bid"


class TestAnOpenPositionIsMarkedAtItsExit:
    """A long can only leave into the bid. Marking it anywhere else
    lets a price it could not have filled at decide the outcome."""

    def test_a_long_is_marked_at_the_bid(self):
        assert q.mark_px("up", BID, ASK, LAST)[0] == BID

    def test_a_short_is_marked_at_the_ask(self):
        assert q.mark_px("down", BID, ASK, LAST)[0] == ASK

    def test_the_mark_is_the_close_price(self):
        for side in ("up", "down"):
            assert (q.mark_px(side, BID, ASK, LAST)
                    == q.fill_px(side, "close", BID, ASK, LAST))

    def test_a_print_on_the_far_side_does_not_set_a_longs_mark(self):
        """A buyer lifting the offer prints at the ask. A long
        cannot exit there, so it must not be marked there -- that is
        how a target fires at a price nothing would have paid."""
        assert q.mark_px("up", BID, ASK, last=ASK)[0] == BID


class TestItRefusesABookItCannotTrust:
    """And says which, because a silent fallback to the last print
    is the original bug wearing better manners."""

    @pytest.mark.parametrize("bid,ask,age,word", [
        (7478.0, 7477.0, None, "crossed"),     # crossed
        (7478.0, 7478.0, None, "crossed"),     # locked
        (7470.0, 7480.0, None, "spread"),      # a gap, not a quote
        (BID, ASK, 11.0, "old"),               # a memory
        (0.0, ASK, None, "no quote"),
        (BID, 0.0, None, "no quote"),
    ])
    def test_it_falls_back_and_names_the_reason(self, bid, ask, age, word):
        px, src = q.fill_px("up", "open", bid, ask, 7477.5, age)
        assert px == 7477.5, "it priced off a book it should not trust"
        assert src.startswith("last: ")
        assert word in src

    def test_a_good_quote_is_not_refused(self):
        ok, why = q.usable(BID, ASK, 0.3)
        assert ok and why == ""

    @pytest.mark.parametrize("bad", ["", "sideways", None, "UP"])
    def test_an_unknown_side_never_invents_a_price(self, bad):
        assert q.fill_px(bad, "open", BID, ASK, LAST)[0] == LAST

    def test_an_unknown_action_never_invents_a_price(self):
        assert q.fill_px("up", "vibes", BID, ASK, LAST)[0] == LAST


class TestAStaleBookCannotPriceAFill:
    """The second fault. A book with a plausible clock and prices
    that have gone."""

    def test_an_old_quote_is_refused(self):
        px, src = q.fill_px("up", "open", BID, ASK, 7477.5, age_s=9.0)
        assert px == 7477.5 and "9.0s old" in src

    def test_a_fresh_one_is_not(self):
        assert q.fill_px("up", "open", BID, ASK, LAST, age_s=0.4)[1] == "ask"

    @pytest.mark.parametrize("last,word", [
        (7490.00, "above the ask"),
        (7460.00, "below the bid"),
    ])
    def test_a_print_far_outside_the_touch_refuses_it(self, last, word):
        """THE CHECK A TIMESTAMP CANNOT MAKE. A book that stopped
        updating while prints kept arriving has a plausible clock
        and a bid and ask from a price that has gone."""
        px, src = q.fill_px("up", "open", BID, ASK, last, age_s=0.2)
        assert px == last, "it still priced off the dead book"
        assert word in src

    def test_a_print_through_the_touch_is_fine(self):
        """A trade can print a tick through the touch in a fast
        move. That is normal and must not refuse the quote."""
        assert q.fill_px("up", "open", BID, ASK, ASK + 0.25,
                         age_s=0.2)[1] == "ask"
        assert q.fill_px("up", "open", BID, ASK, BID - 0.25,
                         age_s=0.2)[1] == "ask"

    def test_the_bound_is_in_ticks_and_loose_on_purpose(self):
        edge = ASK + q.MAX_AWAY_TICKS * q.TICK
        assert q.usable(BID, ASK, 0.2, last=edge)[0] is True
        assert q.usable(BID, ASK, 0.2, last=edge + 0.25)[0] is False

    def test_no_last_price_does_not_refuse_the_quote(self):
        assert q.usable(BID, ASK, 0.2, last=0.0)[0] is True

    def test_the_reading_reports_it(self):
        r = q.read(type("B", (), {"best_bid": BID, "best_ask": ASK})(),
                   7490.0, 0.2)
        assert r["usable"] is False
        assert "above the ask" in r["why"]


class TestTheTapeTimestampsItsBook:
    def test_the_tape_has_a_book_ts_field(self):
        from liqmap.cme_tape import Tape
        assert hasattr(Tape(), "book_ts")

    def test_it_is_set_from_the_record_not_the_wall_clock(self):
        """A feed that is behind should LOOK behind. Stamping it
        with time.time() would make a late book read as current and
        defeat the whole check."""
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "cme_tape.py").read_text()
        i = src.index("self.book = dbn.book_from(")
        # CODE ONLY. The comment beside this explains why the wall
        # clock is wrong here, and a plain substring check matches
        # that explanation -- the third test in this repo to fail
        # against its own documentation.
        blk = "\n".join(
            ln for ln in src[i:i + 900].splitlines()
            if not ln.lstrip().startswith("#"))
        assert "self.book_ts = _bts" in blk
        assert "ts_event" in blk
        assert "time.time()" not in blk

    def test_the_route_ages_the_quote_by_the_book(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        i = src.index("def _quote_now(")
        blk = src[i:src.index("def _px_age(", i)]
        assert 'getattr(tape, "book_ts", 0.0)' in blk, \
            "the quote is aged by the tape again"

    def test_a_tape_with_no_book_ts_falls_back_rather_than_refusing(self):
        """Refusing every quote would price every fill off the last
        print and silently undo the spread fix."""
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        i = src.index("def _quote_now(")
        blk = src[i:src.index("def _px_age(", i)]
        assert "age = _px_age(rt_)" in blk


class TestTheReadingThePanelGets:
    def test_it_carries_both_prices_and_the_cost_between_them(self):
        r = q.read(type("B", (), {"best_bid": BID, "best_ask": ASK})(),
                   LAST, 0.4)
        assert r["bid"] == BID and r["ask"] == ASK
        assert r["spread_ticks"] == 1.0
        assert r["round_trip_ticks"] == 2.0
        assert r["buy_at"] == ASK and r["sell_at"] == BID
        assert r["usable"] is True

    def test_no_book_reads_as_unusable_rather_than_zero(self):
        r = q.read(None, LAST, None)
        assert r["usable"] is False
        assert r["bid"] is None and r["ask"] is None
        assert r["why"]

    def test_the_panel_draws_the_quote_and_its_age(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        assert "book ' + Number(J.quote.age_s)" in src, \
            "a stalled book is invisible again"
        assert "NOT USED" in src


class TestTheRouteActuallyUsesIt:
    """PINNING THE WIRING. `quote.py` being correct says nothing
    about the engine: the whole first bug was a correct price
    existing somewhere the fill never read."""

    def test_the_step_marks_an_open_trade_at_its_exit(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        i = src.index("def _journal_step_un(")
        blk = src[i:i + 30000]
        assert "jn_quote.mark_px(" in blk, \
            "the open position is still marked at the last print"
        assert "jn_quote.fill_px(" in blk, \
            "the entry still does not cross the spread"
        assert "close_trade(coin_, mark_px" in blk, \
            "the close still books at the last print"


class TestItDoesNotTrade:
    def test_nothing_here_submits_an_order(self):
        import ast
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "quote.py")
        tree = ast.parse(src.read_text())
        bad = [n.name for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
               and n.name in ("submit", "execute", "place", "order")]
        assert not bad, f"quote.py grew {bad}"
