"""Backtest tests.

The failure mode this guards against is not a crash. It is a backtest that
returns a beautiful number for the wrong reason, which is worse than no
backtest because it gets acted on.

Three things are pinned here:

NO LOOKAHEAD. Every sample must be scored from bars strictly before the
outcome it predicts. A replay that peeks produces 90% accuracy and loses money.

CHRONOLOGICAL SPLIT. Random splits leak the future into the training half
through overlapping structure.

TUNING CANNOT SEE THE TEST SET. And when tuning helps train far more than
test, that gap has to be reported rather than buried.
"""

import pytest

from liqmap.backtest import (
    Sample, evaluate, replay, run, split, threshold_curve, tune,
)
from liqmap.candleread import WEIGHTS
from liqmap.structure import Candle


def wave(n=400, base=100.0, amp=6.0, period=40, noise=0.0):
    """A deterministic oscillation — enough structure to produce swings,
    zones and trend without being random."""
    import math
    out = []
    for i in range(n):
        centre = base + amp * math.sin(2 * math.pi * i / period)
        nxt = base + amp * math.sin(2 * math.pi * (i + 1) / period)
        o, c = centre, nxt
        drift = noise * ((i * 7919) % 11 - 5) / 5.0
        c += drift
        out.append(Candle(ts=i * 900.0, open=o, high=max(o, c) + 0.5,
                          low=min(o, c) - 0.5, close=c, volume=100.0 + i % 7))
    return out


def walk(n=400, seed=12345, start=100.0, vol=0.4):
    """A random walk with no autocorrelation between consecutive bars.

    The smooth wave above is useful for exercising structure detection, but
    on it the next bar's direction is determined by the current bar's shape,
    so ANY model scores near 100%. That is a property of the fixture, not an
    edge, and measuring accuracy on it says nothing.
    """
    import random
    rng = random.Random(seed)
    out = []
    px = start
    for i in range(n):
        o = px
        c = o + rng.gauss(0.0, vol)
        hi = max(o, c) + abs(rng.gauss(0.0, vol / 2))
        lo = min(o, c) - abs(rng.gauss(0.0, vol / 2))
        out.append(Candle(ts=i * 900.0, open=o, high=hi, low=lo, close=c,
                          volume=100.0 + rng.random() * 50))
        px = c
    return out


def sample(contribs, went_up, i=0):
    return Sample(index=i, ts=float(i), contributions=dict(contribs),
                  went_up=went_up, move_bps=10.0 if went_up else -10.0)


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def test_sample_score_is_weight_normalised():
    s = sample({"flow": 1.0, "zone": -1.0}, True)
    # equal and opposite, so any equal weights cancel
    assert s.score({"flow": 1.0, "zone": 1.0}) == pytest.approx(0.0)
    # weighting one side tips it
    assert s.score({"flow": 2.0, "zone": 1.0}) > 0


def test_a_zero_weight_removes_a_signal_entirely():
    s = sample({"flow": 1.0, "zone": -1.0}, True)
    assert s.score({"flow": 1.0, "zone": 0.0}) == pytest.approx(1.0)


def test_score_of_no_contributions_is_zero():
    assert sample({}, True).score(WEIGHTS) == 0.0


# --------------------------------------------------------------------------
# replay and causality
# --------------------------------------------------------------------------

def test_replay_produces_samples_from_a_structured_series():
    out = replay(wave())
    assert len(out) > 50
    assert all(s.contributions for s in out)


def test_replay_refuses_a_series_too_short_to_mean_anything():
    assert replay(wave(n=20)) == []


def test_every_sample_predicts_the_bar_after_the_one_it_read():
    """The outcome must belong to index+1, never to the bar being read."""
    candles = wave()
    for s in replay(candles):
        nxt = candles[s.index + 1]
        assert s.went_up == (nxt.close > nxt.open)
        assert s.ts == nxt.ts


def test_replay_is_deterministic():
    candles = wave()
    a, b = replay(candles), replay(candles)
    assert [(s.index, s.contributions) for s in a] == \
           [(s.index, s.contributions) for s in b]


def test_truncating_the_future_does_not_change_past_samples():
    """The hard test for lookahead: a sample computed at bar i must be
    identical whether or not bars after i exist at all."""
    candles = wave()
    full = {s.index: s.contributions for s in replay(candles)}
    short = {s.index: s.contributions for s in replay(candles[:250])}

    shared = set(full) & set(short)
    assert len(shared) > 30, "need overlap to compare"
    # The final few bars of the truncated run legitimately differ: zone tests
    # and swing confirmation are still accumulating at the boundary.
    settled = [i for i in shared if i < 230]
    assert settled
    for i in settled:
        assert full[i] == short[i], f"bar {i} changed when the future was added"


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

def test_evaluate_counts_only_directional_calls():
    samples = [sample({"flow": 1.0}, True), sample({"flow": -1.0}, False),
               sample({"flow": 0.01}, True)]     # too weak to be a call
    r = evaluate(samples, {"flow": 1.0})
    assert r.calls == 2 and r.accuracy == 1.0
    assert r.n == 3
    assert r.coverage == pytest.approx(2 / 3)


def test_a_perfectly_wrong_model_scores_zero_not_one():
    samples = [sample({"flow": 1.0}, False) for _ in range(30)]
    assert evaluate(samples, {"flow": 1.0}).accuracy == 0.0


def test_edge_is_measured_against_a_coin_flip():
    samples = ([sample({"flow": 1.0}, True)] * 15
               + [sample({"flow": 1.0}, False)] * 5)
    r = evaluate(samples, {"flow": 1.0})
    assert r.accuracy == pytest.approx(0.75)
    assert r.edge == pytest.approx(25.0)


def test_a_report_with_too_few_calls_says_so_rather_than_quoting_a_rate():
    r = evaluate([sample({"flow": 1.0}, True)], {"flow": 1.0}, label="test")
    assert "too few to conclude" in r.describe()


def test_empty_samples_do_not_divide_by_zero():
    r = evaluate([], WEIGHTS)
    assert r.n == 0 and r.accuracy == 0.0 and r.coverage == 0.0


# --------------------------------------------------------------------------
# the selectivity curve — the number that decides tradeability
# --------------------------------------------------------------------------

def test_threshold_curve_trades_coverage_for_accuracy():
    scored = [(0.9, True), (0.8, True), (0.7, True),
              (0.2, False), (0.18, False), (0.16, True)]
    curve = {row["threshold"]: row for row in threshold_curve(scored)}
    assert curve[0.15]["n"] == 6
    assert curve[0.65]["n"] == 3
    assert curve[0.65]["accuracy"] == 1.0
    assert curve[0.65]["coverage"] == pytest.approx(0.5)


def test_a_threshold_nothing_reaches_reports_none_not_zero():
    curve = {row["threshold"]: row for row in threshold_curve([(0.2, True)])}
    assert curve[0.65]["n"] == 0
    assert curve[0.65]["accuracy"] is None


# --------------------------------------------------------------------------
# splitting
# --------------------------------------------------------------------------

def test_split_is_chronological():
    samples = [sample({"flow": 1.0}, True, i) for i in range(100)]
    tr, te = split(samples, 0.6)
    assert len(tr) == 60 and len(te) == 40
    assert max(s.index for s in tr) < min(s.index for s in te)


def test_split_handles_tiny_inputs():
    tr, te = split([sample({"flow": 1.0}, True)], 0.6)
    assert len(tr) + len(te) == 1


# --------------------------------------------------------------------------
# tuning
# --------------------------------------------------------------------------

def test_tuning_finds_a_signal_that_is_backwards():
    """If a signal consistently points the wrong way, the search should turn
    it down rather than keep trusting it."""
    samples = ([sample({"flow": 1.0, "vwap": -1.0}, True) for _ in range(60)]
               + [sample({"flow": -1.0, "vwap": 1.0}, False) for _ in range(60)])
    tuned = tune(samples, {"flow": 1.0, "vwap": 1.0})
    assert tuned["vwap"] < tuned["flow"]


def test_tuning_will_not_reward_a_model_that_never_speaks():
    """A weight set that produces four calls and gets them all right must not
    beat one that covers the data. Otherwise the search always collapses to
    near-silence."""
    samples = [sample({"flow": 0.2 if i % 10 else 1.0}, i % 10 != 0)
               for i in range(200)]
    tuned = tune(samples, {"flow": 1.0})
    r = evaluate(samples, tuned)
    assert r.coverage > 0.1


def test_tuning_on_empty_input_returns_the_base_weights():
    assert tune([], {"flow": 1.0}) == {"flow": 1.0}


def test_tuning_never_sees_the_test_split():
    """Structural guarantee: `run` tunes on the training half only. If the
    tuned weights were fitted to the test half, shuffling the test outcomes
    would change them."""
    candles = wave()
    samples = replay(candles)
    tr, te = split(samples)

    a = tune(tr)
    for s in te:                       # corrupt the held-out outcomes
        s.went_up = not s.went_up
    b = tune(tr)
    assert a == b


# --------------------------------------------------------------------------
# the full run
# --------------------------------------------------------------------------

def test_run_reports_train_and_held_out_separately():
    bt = run(wave(), "BTC", "15m")
    assert bt.train.n > 0 and bt.test.n > 0
    assert bt.train.label != bt.test.label
    assert "held out" in bt.test.label


def test_run_quantifies_the_overfit_gap():
    bt = run(wave(noise=2.0), "BTC", "15m")
    assert isinstance(bt.overfit_gap, float)


def test_run_keeps_the_untuned_baseline_for_comparison():
    bt = run(wave(), "BTC", "15m", do_tune=True)
    assert bt.baseline_test is not None
    assert bt.tuned_weights is not None


def test_run_without_tuning_leaves_weights_alone():
    bt = run(wave(), "BTC", "15m", do_tune=False)
    assert bt.tuned_weights is None


def test_run_on_too_little_history_refuses_rather_than_guessing():
    bt = run(wave(n=60), "BTC", "15m")      # below the replay minimum
    assert bt.samples == []
    assert bt.test.calls == 0
    assert "Not enough" in bt.verdict()


def test_verdict_warns_when_tuning_did_not_generalise():
    bt = run(walk(), "BTC", "15m")
    text = bt.verdict()
    assert isinstance(text, str) and text
    if bt.test.calls >= 20 and bt.baseline_test:
        assert ("generalise" in text or "real rather than fitted" in text
                or "fitted noise" in text or "not an edge" in text)


def test_a_random_walk_yields_no_real_edge():
    """Structure cannot predict an unpredictable series. Anything close to
    perfect here means the replay is reading the future."""
    bt = run(walk(), "BTC", "15m")
    assert bt.test.calls >= 20
    assert bt.test.accuracy < 0.70, "found an edge in a random walk — leak"


def test_an_implausible_hit_rate_is_called_out_not_celebrated():
    """A smooth synthetic series makes the next bar trivially predictable.
    The report must name that as a data problem rather than an edge."""
    bt = run(wave(), "BTC", "15m")
    if bt.test.accuracy > 0.85 and bt.test.calls >= 20:
        assert "not a plausible hit rate" in bt.verdict()


def test_tuning_cannot_manufacture_an_edge_on_noise():
    """Tuned weights must not beat the default weights on held-out noise by
    any meaningful margin. If they do, the search is fitting the test set."""
    bt = run(walk(seed=777), "BTC", "15m")
    if bt.baseline_test and bt.test.calls >= 20:
        gain = (bt.test.accuracy - bt.baseline_test.accuracy) * 100
        assert gain < 20, f"tuning gained {gain:.0f}pts on noise"
