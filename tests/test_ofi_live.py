"""The live path must be the same engine, and must not leak the key.

1.  ONE ENGINE. `consume` drives BookState and Race directly, so a
    replayed file and the live feed produce identical readings. A
    second copy of the formula on the live side could drift and
    nothing would show it.
2.  THE API KEY. Read from the environment, never an argument, never
    in a payload, never printed.
3.  A DEAD REDIS MUST NOT STOP THE ENGINE. Publishing is
    fire-and-forget; failures are counted, not raised.
4.  NO VERDICT WITHOUT A LINE. With no threshold the payload carries
    the measurement and no `side`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liqmap import ofi, ofi_live                            # noqa: E402

PT = 1_000_000_000
TICK = PT // 4
BASE = 5800 * PT


class Lv:
    def __init__(s, bp, bs, ap, asz):
        s.bid_px, s.bid_sz, s.ask_px, s.ask_sz = bp, bs, ap, asz


class Rec:
    def __init__(s, lv, action="A", ts=1.0, iid=42, seq=1):
        s.levels, s.action = lv, action
        s.ts_event = int(ts * 1e9)
        s.instrument_id, s.sequence = iid, seq


def stream(n=60):
    bid = BASE
    for i in range(n):
        if i % 7 == 0:
            bid += TICK
        yield Rec([Lv(bid, 50 + i, bid + TICK, 40)], seq=i + 1,
                  ts=1.0 + i * 0.01)


class TestSameEngine:

    def test_consume_matches_a_direct_replay(self):
        """The live path and a file must produce identical readings."""
        direct_state = ofi.BookState()
        direct_race = ofi.Race(window=5, baseline=10)
        direct = []
        for rec in stream():
            ev = direct_state.feed(rec)
            if ev is None:
                continue
            direct.append((ev.e_n, direct_race.push(ev.e_n).window))

        eng = ofi_live.LiveRace(window=5, baseline=10)
        live = [(ev.e_n, rd.window) for ev, rd in eng.consume(stream())]
        assert live == direct

    def test_consume_skips_refused_records(self):
        eng = ofi_live.LiveRace(window=5, baseline=10)
        out = list(eng.consume(stream(3)))
        assert len(out) == 2, "the first record should be refused"
        assert eng.state.refused.first == 1

    def test_the_module_reuses_bookstate_and_race(self):
        src = (Path(__file__).resolve().parent.parent
               / "liqmap" / "ofi_live.py").read_text()
        # Matched on the CONSTRUCTOR, not an exact call. Pinning
        # `ofi.BookState()` breaks the moment an argument is added,
        # which says nothing about whether the class was reused.
        assert "ofi.BookState(" in src
        assert "ofi.Race(" in src
        assert "def level_ofi" not in src, "the formula was copied"
        assert "class BookState" not in src, "a second state machine"


class TestCredentials:

    def test_the_key_is_not_a_parameter(self):
        import inspect
        sig = inspect.signature(ofi_live.LiveRace)
        for name in sig.parameters:
            assert "key" not in name.lower() or name == "key", name
        assert "api_key" not in sig.parameters

    def test_a_missing_key_is_refused_before_connecting(self, monkeypatch):
        monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
        eng = ofi_live.LiveRace()
        with pytest.raises(RuntimeError, match="DATABENTO_API_KEY"):
            eng.client()

    def test_no_payload_carries_a_credential(self):
        eng = ofi_live.LiveRace(window=3, baseline=5)
        for ev, rd in eng.consume(stream()):
            p = ofi_live.reading_payload(ev, rd, None)
            flat = str(p).lower()
            for bad in ("key", "token", "secret", "password", "auth"):
                assert bad not in flat, f"payload carries {bad!r}"


class TestPublisher:

    def test_no_url_publishes_nowhere(self):
        p = ofi_live.Publisher()
        assert p.connect() is False
        p.publish({"a": 1})
        assert p.sent == 0 and p.failed == 0

    def test_a_broken_client_counts_rather_than_raises(self):
        class Boom:
            def set(self, *a, **k):
                raise ConnectionError("gone")

            def publish(self, *a, **k):
                raise ConnectionError("gone")

        p = ofi_live.Publisher(url="redis://x")
        p._client = Boom()
        p.publish({"a": 1})
        assert p.failed == 1 and p.sent == 0
        assert "ConnectionError" in p.last_error

    def test_a_working_client_counts_sends(self):
        class Fake:
            def __init__(s):
                s.sets, s.pubs = [], []

            def set(s, k, v):
                s.sets.append((k, v))

            def publish(s, c, v):
                s.pubs.append((c, v))

        p = ofi_live.Publisher(url="redis://x", channel="c", key="k")
        p._client = Fake()
        p.publish({"z": 1.5})
        assert p.sent == 1
        assert p._client.sets[0][0] == "k"
        assert p._client.pubs[0][0] == "c"

    def test_the_payload_is_json(self):
        import json
        class Fake:
            def __init__(s):
                s.blob = None
            def set(s, k, v):
                s.blob = v
            def publish(s, c, v):
                pass
        p = ofi_live.Publisher(url="redis://x")
        p._client = Fake()
        p.publish({"z": 1.5, "e_n": -3})
        assert json.loads(p._client.blob) == {"z": 1.5, "e_n": -3}


class TestPayload:

    def test_no_side_without_a_line(self):
        eng = ofi_live.LiveRace(window=3, baseline=5)
        for ev, rd in eng.consume(stream()):
            assert "side" not in ofi_live.reading_payload(ev, rd, None)

    def test_a_side_appears_when_a_line_is_given(self):
        eng = ofi_live.LiveRace(window=3, baseline=5)
        last = None
        for ev, rd in eng.consume(stream()):
            last = ofi_live.reading_payload(ev, rd, 0.5)
        assert last["side"] in ("buyers", "sellers", "neither")

    def test_the_payload_carries_the_measurement(self):
        eng = ofi_live.LiveRace(window=3, baseline=5)
        for ev, rd in eng.consume(stream()):
            p = ofi_live.reading_payload(ev, rd, None)
            for k in ("e_n", "window", "z", "warm", "obi", "spread"):
                assert k in p


class TestSuggestsOnly:

    def test_the_module_cannot_place_an_order(self):
        src = (Path(__file__).resolve().parent.parent
               / "liqmap" / "ofi_live.py").read_text()
        for bad in ("def submit", "def execute", "def place", "def order"):
            assert bad not in src, f"ofi_live.py contains {bad!r}"

    def test_no_threshold_is_bound_in_the_module(self):
        suspect = ("THRESHOLD", "Z_SCORE", "MIN_OFI", "CUTOFF")
        bound = [n for n, v in vars(ofi_live).items()
                 if any(k in n.upper() for k in suspect)
                 and isinstance(v, (int, float))]
        assert not bound
