"""Dual-trigger candles: both legs, the cap, and the gray ones.

The rule under test, in Chris's words:

    The candle closes only when both thresholds are hit: 5000 ticks
    and 2000 contracts of OFI accumulated. Direction = OFI sign at
    close. Colour = green/red by OFI sign. Tradeable flag =
    tick_count >= 5000 AND tick-direction agrees with OFI sign. If
    they disagree the candle prints but is marked untradeable (gray
    overlay), and you skip it.

Plus the trap he named: in a market that is all size and few prints
the tick leg never fires, so there is a time cap and a capped bar
says so.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from liqmap import fusedbar as fb
from liqmap.fusedbar import FusedBar, FusedBars

ROOT = Path(__file__).resolve().parents[1]

# Raw integer feed units, as the book arrives.
P = 1_000_000_000
T = P // 4
BID = 6800 * P
ASK = BID + T


def js_func(name: str, path: str = "liqmap/web.py") -> str:
    src = (ROOT / path).read_text()
    i = src.index(f"function {name}(")
    depth, j = 0, src.index("{", i)
    for k in range(j, len(src)):
        if src[k] == "{":
            depth += 1
        elif src[k] == "}":
            depth -= 1
            if depth == 0:
                return src[i:k + 1]
    raise AssertionError(f"{name} has no closing brace")


def py_func(name: str, path: str = "liqmap/web.py") -> str:
    src = (ROOT / path).read_text()
    i = src.index(f"def {name}(")
    start = src.rfind("\n", 0, i) + 1
    indent = i - start
    lines = src[start:].split("\n")
    out = [lines[0]]
    for line in lines[1:]:
        if line.strip() and (len(line) - len(line.lstrip())) <= indent:
            break
        out.append(line)
    return "\n".join(out)


def push_ofi(b: FusedBars, n: int, *, up: bool = True, t0: float = 1000.0,
             dt: float = 0.01, step: int = 10, base: int = 100) -> None:
    """n book updates that walk the bid size, moving OFI one way.

    Growing the resting bid raises OFI; shrinking it lowers OFI. The
    ask is held still so the sign is unambiguous.

    `base` is where the bid size starts. It matters whenever a test
    calls this MORE THAN ONCE on the same builder: leaving it at the
    default restarts the size at 100, which after a run up to 490 is
    a 390-contract DROP and cancels most of what the first call
    built. That is a saw-tooth, not a rise, and a test that wanted a
    rise measures the wrong thing without noticing.
    """
    for i in range(n):
        sz = (base + i * step) if up else max(1, base + n * step - i * step)
        b.on_book(t0 + i * dt, BID, sz, ASK, 100)


def push_prints(b: FusedBars, n: int, *, px: float = 6800.0,
                step: float = 0.0, t0: float = 1000.0, dt: float = 0.01,
                side: str = "B", size: float = 1.0) -> None:
    for i in range(n):
        b.on_trade(t0 + i * dt, px + i * step, size, side)


class TestItDoesNotTrade:
    def test_no_order_entry_points(self):
        tree = ast.parse((ROOT / "liqmap" / "fusedbar.py").read_text())
        names = {n.name for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for banned in ("submit", "execute", "place", "order", "buy",
                       "sell", "send_order"):
            assert banned not in names, banned

    def test_imports_no_broker(self):
        tree = ast.parse((ROOT / "liqmap" / "fusedbar.py").read_text())
        mods = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods |= {a.name.split(".")[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom) and n.module:
                mods.add(n.module.split(".")[0])
        assert not (mods & {"ironbeam", "requests", "httpx", "socket",
                            "urllib", "databento"}), mods


class TestBothLegsAreRequired:
    """The conjunction. Neither leg alone may close a bar."""

    def test_prints_alone_do_not_close_it(self):
        """A quiet burst of tiny trades. Not enough contracts."""
        b = FusedBars(tick_target=20, ofi_target=5_000, max_s=1e9)
        push_prints(b, 200)
        assert b.last_closed is None
        assert b.current.prints == 200
        assert b.current.ticks_met is True
        assert b.current.ofi_met is False

    def test_imbalance_alone_does_not_close_it(self):
        """A single huge block. Not enough prints."""
        b = FusedBars(tick_target=5_000, ofi_target=100, max_s=1e9)
        push_ofi(b, 60)
        b.on_trade(1001.0, 6800.0, 900.0, "B")
        assert b.last_closed is None
        assert b.current.ofi_met is True
        assert b.current.ticks_met is False

    def test_both_together_close_it(self):
        b = FusedBars(tick_target=10, ofi_target=100, max_s=1e9)
        push_prints(b, 10)
        assert b.last_closed is None, "prints first, still open"
        push_ofi(b, 40)
        assert b.last_closed is not None
        assert b.last_closed.close_reason == "both"

    def test_order_does_not_matter(self):
        b = FusedBars(tick_target=10, ofi_target=100, max_s=1e9)
        push_ofi(b, 40)
        assert b.last_closed is None, "book first, still open"
        push_prints(b, 10)
        assert b.last_closed is not None
        assert b.last_closed.close_reason == "both"

    def test_the_closing_event_can_be_either_kind(self):
        a = FusedBars(tick_target=5, ofi_target=100, max_s=1e9)
        push_prints(a, 5)
        push_ofi(a, 40)
        b = FusedBars(tick_target=5, ofi_target=100, max_s=1e9)
        push_ofi(b, 40)
        push_prints(b, 5)
        assert a.last_closed is not None and b.last_closed is not None


class TestTheOfiLegIsSigned:
    def test_a_downward_book_qualifies_too(self):
        """Crossing -target counts. It is a band, not a floor."""
        b = FusedBars(tick_target=5, ofi_target=100, max_s=1e9)
        push_prints(b, 5)
        push_ofi(b, 40, up=False)
        assert b.last_closed is not None
        assert b.last_closed.net_ofi <= -100
        assert b.last_closed.direction == "down"

    def test_churn_that_nets_out_never_qualifies(self):
        """The reason the leg is signed.

        A book worked hard in both directions has committed to
        nothing, and a signed accumulator says so where a gross one
        would call it a strong bar.
        """
        b = FusedBars(tick_target=2, ofi_target=500, max_s=1e9)
        push_prints(b, 50)
        for i in range(60):
            b.on_book(1000.0 + i * 0.01, BID, 100 + (i % 2) * 40,
                      ASK, 100)
        assert b.last_closed is None
        assert abs(b.current.net_ofi) < 500

    def test_the_accumulator_resets_at_every_close(self):
        """Not a latch.

        A global accumulator tracks cumulative price, walks away from
        zero on any directional session and never returns -- the
        construction `ofibar` measured printing 99.9% green.

        THE SAME SEQUENCE drives a control whose tick target can
        never be reached, so it never closes and therefore never
        resets. If the reset were missing, the two would agree.
        """
        b = FusedBars(tick_target=3, ofi_target=100, max_s=1e9)
        latch = FusedBars(tick_target=10**9, ofi_target=10**9, max_s=1e9)
        for r in range(3):
            for bld in (b, latch):
                push_prints(bld, 3)
                # Each round continues the book UP from where the
                # last one left it, so the run really is directional.
                push_ofi(bld, 40, base=100 + r * 500)
        assert len(b.done) >= 2, "the real one closed bars"
        assert latch.current is not None and not latch.done

        # The control holds the whole run in one bar. The real one
        # must hold a fraction of it in its newest.
        whole_run = abs(latch.current.net_ofi)
        newest = abs(b.current.net_ofi) if b.current else 0
        assert whole_run > 1_000, whole_run
        assert newest < whole_run / 2, (newest, whole_run)

        # And no closed bar ran away either: each is its own stretch
        # of book, not a running total of every stretch before it.
        for bar in b.done:
            assert abs(bar.net_ofi) < whole_run / 2, bar.net_ofi

    def test_the_book_carries_across_a_close(self):
        """The accumulator resets; the previous touch does not.

        Treating the first update of a new bar as if there were no
        previous one discards a real move at every boundary.
        """
        b = FusedBars(tick_target=1, ofi_target=50, max_s=1e9)
        push_prints(b, 1)
        push_ofi(b, 20)
        assert b.last_closed is not None
        assert b._prev is not None

    def test_a_crossed_or_invalid_book_is_refused(self):
        b = FusedBars(tick_target=1, ofi_target=1, max_s=1e9)
        assert b.on_book(1.0, ASK, 10, BID, 10) is None      # crossed
        assert b.on_book(1.0, 0, 0, 0, 0) is None            # invalid


class TestTheTimeCap:
    def test_it_closes_a_bar_that_is_all_size_and_few_prints(self):
        """The trap, exactly as named."""
        b = FusedBars(tick_target=5_000, ofi_target=100, max_s=10.0)
        push_ofi(b, 40, t0=1000.0, dt=0.01)
        b.on_trade(1001.0, 6800.0, 500.0, "B")
        assert b.last_closed is None
        b.on_trade(1020.0, 6800.0, 500.0, "B")      # past the cap
        assert b.last_closed is not None
        assert b.last_closed.close_reason == "capped"

    def test_a_capped_bar_is_marked(self):
        b = FusedBars(tick_target=5_000, ofi_target=100, max_s=10.0)
        push_ofi(b, 40)
        b.on_trade(1020.0, 6800.0, 5.0, "B")
        d = b.last_closed.to_dict()
        assert d["capped"] is True
        assert d["close_reason"] == "capped"

    def test_a_capped_bar_is_not_tradeable(self):
        """It did not reach the print count, so it cannot be."""
        b = FusedBars(tick_target=5_000, ofi_target=100, max_s=10.0)
        push_ofi(b, 40)
        b.on_trade(1020.0, 6801.0, 5.0, "B")
        bar = b.last_closed
        assert bar.direction == "up"
        assert bar.tradeable is False
        assert "short of" in bar.why_not

    def test_both_legs_beat_the_cap(self):
        """When both are met the reason is 'both', never 'capped'."""
        b = FusedBars(tick_target=2, ofi_target=100, max_s=10.0)
        push_prints(b, 2, t0=1000.0)
        push_ofi(b, 40, t0=1000.0)
        assert b.last_closed.close_reason == "both"

    def test_the_summary_counts_capped_bars(self):
        b = FusedBars(tick_target=5_000, ofi_target=100, max_s=5.0)
        for k in range(12):
            push_ofi(b, 40, t0=1000.0 + k * 100)
            b.on_trade(1000.0 + k * 100 + 50, 6800.0, 5.0, "B")
        s = b.summary()
        assert s["capped"] >= 10
        assert s["capped_pct"] > 60
        assert "CAP" in s["say"]


class TestDirectionAndColour:
    @staticmethod
    def _bar(net_ofi, o, c, prints=10, target=10):
        return FusedBar(start_ts=0.0, tick_target=target, ofi_target=1,
                        open=o, high=max(o, c), low=min(o, c), close=c,
                        prints=prints, net_ofi=net_ofi)

    def test_direction_is_the_ofi_sign_not_the_price(self):
        up_book_down_price = self._bar(500, 10.0, 9.0)
        assert up_book_down_price.direction == "up"
        assert up_book_down_price.tick_dir == "down"

    def test_colour_follows_direction(self):
        assert self._bar(500, 10.0, 11.0).colour == "green"
        assert self._bar(-500, 10.0, 9.0).colour == "red"
        assert self._bar(0, 10.0, 10.0).colour == "flat"

    def test_tick_direction_is_the_traded_price(self):
        assert self._bar(1, 10.0, 11.0).tick_dir == "up"
        assert self._bar(1, 11.0, 10.0).tick_dir == "down"
        assert self._bar(1, 10.0, 10.0).tick_dir == "flat"

    def test_a_bar_with_no_prints_has_no_tick_direction(self):
        bar = FusedBar(start_ts=0.0, net_ofi=900)
        assert bar.tick_dir == "flat"
        assert bar.tradeable is False
        assert "no prints" in bar.why_not


class TestTradeable:
    @staticmethod
    def _bar(net_ofi, o, c, prints, target=10):
        return FusedBar(start_ts=0.0, tick_target=target, ofi_target=1,
                        open=o, high=max(o, c), low=min(o, c), close=c,
                        prints=prints, net_ofi=net_ofi)

    def test_agreement_and_enough_prints_is_tradeable(self):
        assert self._bar(500, 10.0, 11.0, prints=10).tradeable is True
        assert self._bar(-500, 11.0, 10.0, prints=10).tradeable is True

    def test_disagreement_is_not(self):
        """The bar that looks like a signal and is not."""
        bar = self._bar(500, 11.0, 10.0, prints=10)
        assert bar.tradeable is False
        assert bar.why_not == "the book ended up but price went down"

    def test_the_other_disagreement_too(self):
        bar = self._bar(-500, 10.0, 11.0, prints=10)
        assert bar.tradeable is False
        assert "down" in bar.why_not and "up" in bar.why_not

    def test_too_few_prints_is_not_tradeable_even_when_they_agree(self):
        """The flag is stricter than the close condition.

        Only a capped bar can be here, and a capped bar is not what
        the dual trigger promised however nicely it ended up.
        """
        bar = self._bar(500, 10.0, 11.0, prints=3, target=10)
        assert bar.tick_dir == bar.direction == "up"
        assert bar.tradeable is False
        assert "short of" in bar.why_not

    def test_a_flat_book_is_not_tradeable(self):
        bar = self._bar(0, 10.0, 11.0, prints=10)
        assert bar.tradeable is False
        assert "flat" in bar.why_not

    def test_a_flat_price_is_not_tradeable(self):
        bar = self._bar(500, 10.0, 10.0, prints=10)
        assert bar.tradeable is False
        assert "closed where it opened" in bar.why_not

    def test_why_not_is_never_blank_when_it_is_gray(self):
        """Every refusal says which one it was."""
        cases = [self._bar(500, 11.0, 10.0, 10), self._bar(0, 10.0, 11.0, 10),
                 self._bar(500, 10.0, 10.0, 10), self._bar(500, 10.0, 11.0, 2),
                 FusedBar(start_ts=0.0, net_ofi=5)]
        for bar in cases:
            assert bar.tradeable is False
            assert bar.why_not.strip(), bar.to_dict()

    def test_why_not_is_blank_when_it_is_not(self):
        assert self._bar(500, 10.0, 11.0, 10).why_not == ""


class TestPrintsThatAreNotTrades:
    @pytest.mark.parametrize("px,size", [(0, 5), (-1, 5), (10, 0), (10, -2)])
    def test_they_do_not_advance_the_tick_leg(self, px, size):
        b = FusedBars(tick_target=3, ofi_target=1, max_s=1e9)
        assert b.on_trade(1.0, px, size, "B") is None
        assert b.current is None

    def test_garbage_does_not_raise(self):
        b = FusedBars(tick_target=3, ofi_target=1, max_s=1e9)
        assert b.on_trade("x", "y", "z", "B") is None
        assert b.on_trade(None, None, None, None) is None


class TestUnknownSide:
    def test_n_advances_the_tick_leg(self):
        """Counted, not dropped -- the leg is a count of PRINTS."""
        b = FusedBars(tick_target=4, ofi_target=1, max_s=1e9)
        push_ofi(b, 20)
        push_prints(b, 4, side="N")
        assert b.last_closed is not None
        assert b.last_closed.prints == 4

    def test_n_does_not_move_delta(self):
        b = FusedBars(tick_target=4, ofi_target=1e9, max_s=1e9)
        push_prints(b, 4, side="N", size=3.0)
        assert b.current.delta == 0.0
        assert b.current.unknown == 4
        assert b.current.unknown_size == 12.0

    def test_volume_counts_them(self):
        b = FusedBars(tick_target=99, ofi_target=1e9, max_s=1e9)
        push_prints(b, 4, side="N", size=3.0)
        assert b.current.volume == 12.0


class TestProgress:
    def test_it_says_which_leg_it_is_waiting_on(self):
        b = FusedBars(tick_target=10_000, ofi_target=100, max_s=1e9)
        push_ofi(b, 40)
        push_prints(b, 5)
        p = b.progress()
        assert p["waiting_on"] == "prints"
        assert "waiting on prints" in p["say"]

    def test_and_the_other_way(self):
        b = FusedBars(tick_target=5, ofi_target=100_000, max_s=1e9)
        push_prints(b, 5)
        p = b.progress()
        assert p["waiting_on"] == "the book"
        assert "waiting on the book" in p["say"]

    def test_it_reports_both_percentages_separately(self):
        """One number would hide the thing fusing them is for."""
        b = FusedBars(tick_target=100, ofi_target=100_000, max_s=1e9)
        push_prints(b, 98)
        p = b.progress()
        assert p["tick_pct"] == 98.0
        assert p["ofi_pct"] < 5.0

    def test_no_open_bar_is_not_an_error(self):
        b = FusedBars(tick_target=1, ofi_target=1, max_s=1e9)
        push_ofi(b, 20)
        push_prints(b, 1)
        assert b.current is None
        assert b.progress()["building"] is False


class TestParseInterval:
    def test_the_fused_form_parses(self):
        assert fb.parse_interval("5000x2000") == (5000, 2000)
        assert fb.parse_interval(" 1000X500 ") == (1000, 500)

    @pytest.mark.parametrize("bad", ["15m", "1h", "1000t", "", None, "x",
                                     "5000x", "x2000", "5000", "0x5",
                                     "5000x0", "5.0x2.0", "5000xx2000"])
    def test_everything_else_is_none(self, bad):
        """What keeps the clock and tick paths unchanged."""
        assert fb.parse_interval(bad) is None

    def test_the_boundary_clamps(self):
        assert fb.parse_interval("1x1") == (fb.MIN_TICKS, fb.MIN_OFI)
        assert fb.parse_interval("9999999x9999999") == (fb.MAX_TICKS,
                                                        fb.MAX_OFI)

    def test_the_core_does_not_clamp(self):
        b = FusedBars(tick_target=3, ofi_target=4)
        assert (b.tick_target, b.ofi_target) == (3, 4)

    def test_label_round_trips(self):
        for a, c in fb.PAIRS:
            assert fb.parse_interval(fb.label(a, c)) == (a, c)

    def test_it_does_not_collide_with_the_tick_form(self):
        from liqmap import tickbar as tb
        for s in ("1000t", "233t"):
            assert fb.parse_interval(s) is None
        for s in ("5000x2000", "1000x500"):
            assert tb.parse_interval(s) is None


class TestTheTapeFeedsBothLegs:
    def test_the_tape_has_the_field(self):
        from liqmap.cme_tape import Tape
        assert "fusedbars" in Tape.__dataclass_fields__

    def test_the_tick_leg_is_fed_from_the_raw_side(self):
        src = (ROOT / "liqmap" / "cme_tape.py").read_text()
        feed = src.index("_fb.on_trade(_ts, _rpx, _rsz, _sd)")
        decode = src.index("t = dbn.trade_from(rec)")
        assert feed < decode, "must read the raw record, not the decoded trade"

    def test_the_ofi_leg_is_fed_from_the_raw_levels(self):
        src = (ROOT / "liqmap" / "cme_tape.py").read_text()
        assert "_fb.on_book(bts, bpx, bsz, apx, asz)" in src

    def test_the_trade_is_fed_before_the_book(self):
        """An MBP-10 trade record carries the book AFTER the fill.

        Book first can close the bar on the very update the print
        caused, and the print is then counted against the next bar.
        """
        src = (ROOT / "liqmap" / "cme_tape.py").read_text()
        assert (src.index("_fb.on_trade(_ts, _rpx, _rsz, _sd)")
                < src.index("_fb.on_book(bts, bpx, bsz, apx, asz)"))

    def test_neither_feed_requires_another_builder_to_exist(self):
        """Both guards used to.

        The OFI leg lived under `if self.ofibars:`, so a fused
        builder on a tape with no OFI bars would be fed prints
        forever and never one book update -- a second leg with no way
        to move.
        """
        src = (ROOT / "liqmap" / "cme_tape.py").read_text()
        assert "if self.clockwin is not None or self.tickbars or self.fusedbars:" in src
        assert "if self.ofibars or self.fusedbars:" in src
        assert "for _b in self.ofibars.values():" not in src, \
            "that loop raises when ofibars is None and skips the fused feed"

    def test_a_builder_per_offered_pair_is_created(self):
        src = (ROOT / "liqmap" / "web.py").read_text()
        assert "from .fusedbar import PAIRS as _FPAIRS, FusedBars" in src
        assert "FusedBars(tick_target=int(a)" in src


class TestTheChartRoute:
    def test_it_branches_on_a_fused_interval(self):
        blk = py_func("api_chart")
        assert "_fb.parse_interval(interval)" in blk
        assert "if _pair is not None:" in blk

    def test_the_branch_returns_before_the_other_two_paths(self):
        blk = py_func("api_chart")
        i = blk.index("if _pair is not None:")
        assert i < blk.index("_tb.parse_interval(interval)")
        assert i < blk.index("_candle_rows(coin, interval, n)")

    def test_it_carries_the_progress_and_the_summary(self):
        blk = py_func("api_chart")
        branch = blk[blk.index("if _pair is not None:"):]
        for want in ('"progress": bld.progress()', '"summary": bld.summary()',
                     '"tick_target": _pair[0]', '"ofi_target": _pair[1]'):
            assert want in branch, want

    def test_it_reports_no_interval_seconds(self):
        blk = py_func("api_chart")
        branch = blk[blk.index("if _pair is not None:"):]
        assert '"interval_s": None' in branch

    def test_it_draws_no_trade_markers(self):
        blk = py_func("api_chart")
        branch = blk[blk.index("if _pair is not None:"):]
        assert '"marks": []' in branch


class TestTheRouteActuallyServesThem:
    @pytest.fixture
    def client(self, monkeypatch, tmp_path):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient
        import liqmap.web as web

        monkeypatch.setenv("LIQMAP_DB", str(tmp_path / "web.db"))
        monkeypatch.setenv("LIQMAP_TOKEN", "fused-test-token")
        monkeypatch.setenv("LIQMAP_AUTO", "false")
        monkeypatch.delenv("LIQMAP_WALLETS", raising=False)
        web.RT = None
        with TestClient(web.create_app()) as c:
            c.headers.update({"Authorization": "Bearer fused-test-token"})
            yield c, web
        web.RT = None

    @staticmethod
    def _attach(web, builders):
        class _Tape:
            fusedbars = builders
            last_px = 6800.25

        class _Config:
            enabled = False

        class _Ofi:
            tape = _Tape()
            config = _Config()

        web.RT.ofi = _Ofi()

    def _filled(self):
        b = FusedBars(tick_target=4, ofi_target=100, max_s=1e9)
        for k in range(6):
            push_prints(b, 4, px=6800.0 + k, step=0.25,
                        t0=1_700_000_000.0 + k * 10)
            push_ofi(b, 40, t0=1_700_000_000.0 + k * 10 + 1)
        return b

    def test_a_fused_interval_returns_fused_bars(self, client):
        c, web = client
        self._attach(web, {(4, 100): self._filled()})
        d = c.get("/api/chart",
                  params={"coin": "ES", "interval": "4x100"}).json()
        assert d.get("error") is None, d.get("error")
        assert d["source"] == "fused"
        assert d["interval"] == "4x100"
        assert d["tick_target"] == 4 and d["ofi_target"] == 100
        assert d["interval_s"] is None
        assert d["bars"]
        row = d["bars"][0]
        for k in ("ts", "o", "h", "l", "c", "v", "dir", "tradeable",
                  "why_not", "net_ofi", "prints", "close_reason"):
            assert k in row, k

    def test_a_clock_interval_is_untouched(self, client):
        c, web = client
        self._attach(web, {(4, 100): FusedBars(4, 100)})
        d = c.get("/api/chart",
                  params={"coin": "ES", "interval": "15m"}).json()
        assert d.get("source") != "fused"

    def test_a_tick_interval_is_untouched(self, client):
        c, web = client
        self._attach(web, {(4, 100): FusedBars(4, 100)})
        d = c.get("/api/chart",
                  params={"coin": "ES", "interval": "1000t"}).json()
        assert d.get("source") != "fused"

    def test_an_unoffered_pair_snaps_to_the_nearest(self, client):
        c, web = client
        self._attach(web, {(4, 100): self._filled()})
        d = c.get("/api/chart",
                  params={"coin": "ES", "interval": "7x130"}).json()
        assert d["tick_target"] == 4 and d["ofi_target"] == 100

    def test_nothing_built_yet_explains_itself(self, client):
        c, web = client
        self._attach(web, {(4, 100): FusedBars(4, 100)})
        d = c.get("/api/chart",
                  params={"coin": "ES", "interval": "4x100"}).json()
        assert "nothing built yet" in (d.get("error") or "")

    def test_no_tape_explains_itself(self, client):
        c, web = client
        web.RT.ofi = None
        d = c.get("/api/chart",
                  params={"coin": "ES", "interval": "5000x2000"}).json()
        assert "not running here" in (d.get("error") or "")


class TestThePanel:
    def test_the_dropdown_offers_the_pairs(self):
        src = (ROOT / "liqmap" / "web.py").read_text()
        i = src.index('id="sInt"')
        box = src[i:src.index("</select>", i)]
        for a, c in fb.PAIRS:
            assert f'value="{a}x{c}"' in box, (a, c)

    def test_the_offered_pairs_match_the_builders(self):
        import re
        src = (ROOT / "liqmap" / "web.py").read_text()
        i = src.index('id="sInt"')
        box = src[i:src.index("</select>", i)]
        shown = {(int(a), int(b))
                 for a, b in re.findall(r'value="(\d+)x(\d+)"', box)}
        assert shown == set(fb.PAIRS)

    def test_the_page_knows_the_fused_form(self):
        src = (ROOT / "liqmap" / "web.py").read_text()
        assert "function isFusedTf(v)" in src
        assert "/^[0-9]+x[0-9]+$/" in src

    def test_one_helper_answers_is_this_a_clock_interval(self):
        """Four call sites half-handling two forms is the bug this
        prevents: a form added later gets added in one place."""
        src = (ROOT / "liqmap" / "web.py").read_text()
        assert "function isCountTf(v)" in src
        assert "isTickTf(v) || isFusedTf(v)" in src

    def test_the_read_does_not_follow_the_chart_onto_a_fused_series(self):
        blk = js_func("followChartTf")
        assert "isCountTf(chart.value)" in blk

    def test_the_projection_is_skipped(self):
        blk = js_func("loadChart")
        assert "!isCountTf($('sInt').value)" in blk

    def test_the_note_says_what_the_bar_is_waiting_on(self):
        blk = js_func("loadChart")
        assert "'fused'" in blk
        assert "progress" in blk and "summary" in blk


class TestTheGrayOverlay:
    """Chris's rule: it prints, it is marked, you skip it."""

    @staticmethod
    def _painter() -> str:
        src = (ROOT / "liqmap" / "web.py").read_text()
        i = src.index("bars.forEach((b, i) => {")
        return src[i:src.index("/* ---- the bar that has not closed", i)]

    def test_a_fused_bar_is_coloured_by_the_book_not_the_price(self):
        blk = self._painter()
        assert "b.dir === 'up' ? up : b.dir === 'down' ? down" in blk

    def test_an_untradeable_bar_is_grayed(self):
        blk = self._painter()
        assert "const grayed = fused && !b.tradeable;" in blk
        assert "grayed ? dim" in blk

    def test_it_is_still_drawn(self):
        """It happened, so it is data. It is washed out, not hidden."""
        blk = self._painter()
        assert "return" not in blk.split("const grayed")[1][:200], \
            "a grayed bar must not be skipped by the painter"
        assert "globalAlpha = 0.45" in blk

    def test_the_overlay_covers_the_whole_span(self):
        blk = self._painter()
        assert "y(b.l) - y(b.h)" in blk

    def test_the_alpha_is_restored(self):
        """A leaked globalAlpha washes out every later bar too."""
        blk = self._painter()
        assert blk.count("globalAlpha = 1") >= 2

    def test_a_time_bar_is_still_coloured_by_price(self):
        """Nothing about the clock charts changes."""
        blk = self._painter()
        assert "(b.c >= b.o ? up : down)" in blk


class TestTheIntrabarSignal:
    """Chris: "this is a intrabar strategy not at close".

    `tradeable` is the CLOSE-time flag and it greys a finished candle
    on the chart. `live_signal` is the ENTRY test and it reads the
    bar being built. Using the first as the second would put every
    entry one bar late, which is the lateness this series exists to
    fix.
    """

    def _agreeing(self, ticks=100, ofi=100):
        b = FusedBars(tick_target=ticks, ofi_target=ofi, max_s=1e9)
        push_ofi(b, 40, t0=1000.0)
        push_prints(b, 30, px=6800.0, step=0.25, t0=1001.0)
        return b

    def test_it_fires_long_before_the_bar_closes(self):
        b = self._agreeing()
        assert b.last_closed is None, "the bar is still open"
        side, why = fb.live_signal(b.current)
        assert side == "up" and why == ""
        assert b.current.prints < b.current.tick_target

    def test_the_close_flag_would_still_refuse_it(self):
        """The two tests disagree on purpose, and this pins it.

        If `tradeable` ever becomes the entry test, this fails.
        """
        b = self._agreeing()
        assert b.current.tradeable is False
        assert "short of" in b.current.why_not
        assert fb.live_signal(b.current)[0] == "up"

    def test_disagreement_gives_no_side(self):
        b = FusedBars(tick_target=100, ofi_target=100, max_s=1e9)
        push_ofi(b, 40, t0=1000.0)                      # book up
        push_prints(b, 30, px=6900.0, step=-0.25, t0=1001.0)  # price down
        side, why = fb.live_signal(b.current)
        assert side == ""
        assert why == "the book is up but price is down"

    def test_too_few_prints_is_refused_and_said(self):
        b = FusedBars(tick_target=100, ofi_target=100, max_s=1e9)
        push_ofi(b, 40)
        push_prints(b, 3, px=6800.0, step=0.25, t0=1001.0)
        side, why = fb.live_signal(b.current)
        assert side == ""
        assert "only 3 prints" in why and "10%" in why

    def test_too_little_book_is_refused_and_said(self):
        b = FusedBars(tick_target=10, ofi_target=100_000, max_s=1e9)
        push_prints(b, 30, px=6800.0, step=0.25)
        side, why = fb.live_signal(b.current)
        assert side == ""
        assert "the book has only moved" in why

    def test_the_floor_is_a_fraction_not_the_whole_leg(self):
        """Requiring the full target is requiring the close."""
        b = self._agreeing()
        assert fb.MIN_FRACTION < 0.5
        assert fb.live_signal(b.current, min_fraction=1.0)[0] == ""
        assert fb.live_signal(b.current, min_fraction=0.1)[0] == "up"

    def test_no_bar_is_a_reason_not_a_crash(self):
        side, why = fb.live_signal(None)
        assert side == "" and why.strip()

    @pytest.mark.parametrize("flat", ["book", "price"])
    def test_flat_is_not_agreement(self, flat):
        b = FusedBars(tick_target=10, ofi_target=10, max_s=1e9)
        if flat == "book":
            push_prints(b, 30, px=6800.0, step=0.25)
            for i in range(30):            # book up then back down
                b.on_book(1000 + i * 0.01, BID, 100 + (i % 2) * 50,
                          ASK, 100)
        else:
            push_ofi(b, 40)
            push_prints(b, 30, px=6800.0, step=0.0)
        side, why = fb.live_signal(b.current)
        assert side == ""
        assert why.strip()

    def test_progress_carries_the_live_read(self):
        p = self._agreeing().progress()
        assert p["live_side"] == "up"
        assert p["ofi_dir"] == "up" and p["tick_dir"] == "up"


class TestTheStrategyMovedOut:
    """The fused STRATEGY is no longer this module's dual-trigger bar.

    It was, for about an hour. Chris: "when i said fused i want the
    5000 tick bar to read independent of the OFI ... they should read
    independent but have the same directional interest for a trade to
    happen" -- which is two separate series compared on direction,
    not one bar closing on two conditions.

    So the rule lives in `liqmap.fusion` and is tested in
    test_fusion.py. The dual-trigger CANDLE below stays as a chart
    type, because it is a real way to cut the tape and it is already
    built; it simply is not what the strategy reads.
    """

    def test_the_strategy_reads_the_two_series_module(self):
        from liqmap.journal import STRATEGIES
        assert "fused" in STRATEGIES
        assert "separately" in STRATEGIES["fused"]

    def test_this_module_no_longer_feeds_the_strategy(self):
        src = (ROOT / "liqmap" / "web.py").read_text()
        step = py_func("_journal_step")
        assert "_fusion_now(rt_)" in step
        assert "_fused_now(" not in step

    def test_the_dual_trigger_chart_type_still_works(self):
        b = FusedBars(tick_target=4, ofi_target=100, max_s=1e9)
        push_prints(b, 4)
        push_ofi(b, 40)
        assert b.last_closed is not None
        assert b.last_closed.close_reason == "both"
