"""Does the study's arithmetic find things that are not there?

Three separate times this session a scan handed back a confident result
from randomness, and each was caught by building the control first. So
this never runs on real data until it has been shown to:

  report NOTHING when nothing was planted, on a pure random walk with
  events scattered at random times

  report REAL when something was planted, in BOTH halves of the mirror
  pair, with the sign the planted effect actually had

A scan that only passes the second test is a scan that says yes to
everything.
"""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "l1_study", ROOT / "scripts" / "l1_study.py")
l1_study = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(l1_study)          # type: ignore[union-attr]


class _Sink:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def write(self, s: str) -> None:
        self.lines.append(s)

    @property
    def text(self) -> str:
        return "".join(self.lines)

    def verdict(self, state: str, horizon_ms: int) -> str:
        """The verdict word on `state`'s row under `horizon_ms`."""
        want = f"  {horizon_ms}ms forward"
        block, seen = [], False
        for line in self.text.split("\n"):
            if line.strip().startswith(f"{horizon_ms}ms forward"):
                seen = True
                continue
            if seen and line.strip().endswith("ms forward"):
                break
            if seen:
                block.append(line)
        for line in block:
            if line.strip().startswith(state):
                return line.split()[-1]
        return "missing"


def _track(seed: int, n: int, planted: dict[str, float] | None = None):
    """A synthetic book: random-walk mid on the study's own grid, with
    events scattered through it. `planted` gives a per-state drift in
    ticks applied over the second after each of its events."""
    rng = random.Random(seed)
    tr = l1_study.Track(0.25)
    step = l1_study.GRID_MS / 1000.0

    mid = 5000.0
    ts = 1_700_000_000.0
    bump: list[float] = [0.0] * n          # planted drift per sample

    states = list(l1_study.STATES)
    # Place events on grid indices so the effect lands where the study
    # will look for it.
    for i in range(0, n - 40, 37):
        state = states[(i // 37) % len(states)]
        tr.add_event(ts + i * step, state)
        if planted and state in planted:
            per = planted[state] / 20.0        # spread over ~1s of grid
            for k in range(1, 21):
                if i + k < n:
                    bump[i + k] += per

    for i in range(n):
        mid += rng.choice((-0.25, 0.0, 0.0, 0.25)) + bump[i]
        tr.ts.append(ts + i * step)
        tr.mid.append(mid)
    return tr


class TestItReportsNothingWhenNothingIsThere:

    @pytest.mark.parametrize("seed", [1, 7, 11])
    def test_a_random_walk_yields_no_real_verdicts(self, seed):
        tr = _track(seed, 40_000)
        sink = _Sink()
        l1_study.run(tr, sink)
        for state in l1_study.STATES:
            for ms in (250, 1000, 5000):
                v = sink.verdict(state, ms)
                # "too few to read" is NOT a pass. A row that was never
                # scored cannot demonstrate the scan refuses noise, and
                # an earlier version of this test accepted it and so
                # tested nothing at all.
                assert v == "nothing", \
                    f"{state} at {ms}ms came back {v!r} on pure noise"


class TestItReportsWhatIsThere:

    def test_a_planted_mirror_pair_is_found_in_both_halves(self):
        """permanent_ask up, permanent_bid down -- the shape the thesis
        predicts. Both halves must light up, or the test cannot tell a
        finding from the sample's drift."""
        tr = _track(3, 40_000, planted={
            "permanent_ask": +3.0,      # cleared offer -> price rises
            "permanent_bid": -3.0,      # cleared bid   -> price falls
        })
        sink = _Sink()
        l1_study.run(tr, sink)
        assert sink.verdict("permanent_ask", 1000) == "REAL"
        assert sink.verdict("permanent_bid", 1000) == "REAL"

    def test_the_unplanted_pair_stays_quiet(self):
        """The discriminator. Planting the permanent pair must not light
        up the failed pair, or the scan is responding to event density
        rather than to the events."""
        tr = _track(3, 40_000, planted={
            "permanent_ask": +3.0, "permanent_bid": -3.0,
        })
        sink = _Sink()
        l1_study.run(tr, sink)
        for state in ("failed_ask", "failed_bid"):
            assert sink.verdict(state, 1000) != "REAL", \
                f"{state} lit up with nothing planted in it"

    def test_the_failed_pair_can_be_found_too(self):
        """Fade-the-wall direction: a defended bid precedes UP."""
        tr = _track(5, 40_000, planted={
            "failed_bid": +3.0, "failed_ask": -3.0,
        })
        sink = _Sink()
        l1_study.run(tr, sink)
        assert sink.verdict("failed_bid", 1000) == "REAL"
        assert sink.verdict("failed_ask", 1000) == "REAL"
        assert sink.verdict("permanent_ask", 1000) != "REAL"


class TestTheControlIsTheComparison:

    def test_a_drifting_market_does_not_make_every_state_real(self):
        """THE FAILURE MODE THIS EXISTS FOR. If every state is compared
        to 50% instead of to the control, a market that simply went up
        over the sample makes all four look bullish and real."""
        tr = _track(9, 40_000)
        for i in range(len(tr.mid)):
            tr.mid[i] += i * 0.02          # a strong, uniform uptrend
        sink = _Sink()
        l1_study.run(tr, sink)
        reals = [s for s in l1_study.STATES
                 if sink.verdict(s, 1000) == "REAL"]
        assert not reals, f"drift alone made these REAL: {reals}"

    def test_the_control_row_is_printed_on_every_horizon(self):
        tr = _track(2, 40_000)
        sink = _Sink()
        l1_study.run(tr, sink)
        for ms in l1_study.HORIZONS_MS:
            assert f"{ms}ms forward" in sink.text
        assert sink.text.count("control") >= len(l1_study.HORIZONS_MS)
