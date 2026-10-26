"""Reading pool state, and the ways a reader quietly lies.

In order of how badly each would mislead:

  a half-read batch presented as a book. A pool whose call failed must
  be ABSENT, not present with zeroes -- zeroes read as "no liquidity
  here", which is a fact about the market rather than about the
  request, and the panel would report a dislocation against a pool
  that simply did not answer

  reads spread across blocks. Fetch each pool at "latest" and a new
  block landing mid-batch gives a book half from one block and half
  from the next. The cycle that implies never existed

  a transport error swallowed into an empty result, which looks
  exactly like a quiet market
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from liqmap import dexfeed                                   # noqa: E402
from liqmap.dexarb import V2, V3, Pool                       # noqa: E402

URL = "https://node.example/v2/key"
A = Pool("0xAAAaaaAAaAAAAaAaAAaaAAaaaAAaAAaAAAAAaAaA", V2,
         "0xT0", "0xT1", 3000, "A")
B = Pool("0xBBBbbbBBbBBBBbBbBBbbBBbbbBBbBBbBBBBBbBbB", V3,
         "0xT0", "0xT1", 500, "B")


def word(n: int) -> str:
    return f"{n:064x}"


class FakePost:
    """Records what was sent and replays canned replies."""

    def __init__(self, replies, status=200, raise_exc=None):
        self.replies = replies
        self.status = status
        self.raise_exc = raise_exc
        self.sent = []

    def __call__(self, url, json=None, timeout=None, headers=None):
        if self.raise_exc:
            raise self.raise_exc
        self.sent.append(json)
        payload = self.replies.pop(0) if self.replies else {}

        class R:
            status_code = self.status

            def json(_self):
                if payload is None:
                    raise ValueError("not json")
                return payload
        return R()


def patch(monkeypatch, fake):
    monkeypatch.setattr(dexfeed.requests, "post", fake)


class TestItPinsEveryReadToOneBlock:

    def test_the_block_tag_is_the_same_on_every_call(self):
        fake = FakePost([
            {"jsonrpc": "2.0", "id": 1, "result": "0x1405a10"},
            [{"id": 0, "result": "0x" + word(1000) + word(2000) + word(0)},
             {"id": 1, "result": "0x" + word(5) + word(0) * 6},
             {"id": 2, "result": "0x" + word(7)}],
        ])
        import liqmap.dexfeed as m
        old, m.requests.post = m.requests.post, fake
        try:
            states, block = m.fetch_states([A, B], url=URL)
        finally:
            m.requests.post = old
        assert block == 0x1405a10
        batch = fake.sent[1]
        tags = {c["params"][1] for c in batch}
        assert tags == {hex(0x1405a10)}, f"reads spread across {tags}"
        assert "latest" not in tags

    def test_an_explicit_block_skips_the_lookup(self):
        fake = FakePost([
            [{"id": 0, "result": "0x" + word(1000) + word(2000) + word(0)}],
        ])
        import liqmap.dexfeed as m
        old, m.requests.post = m.requests.post, fake
        try:
            states, block = m.fetch_states([A], url=URL, block=12345)
        finally:
            m.requests.post = old
        assert block == 12345
        assert len(fake.sent) == 1, "no eth_blockNumber call was needed"


class TestAFailedReadIsAbsentNotEmpty:

    def _run(self, replies, pools):
        fake = FakePost(replies)
        import liqmap.dexfeed as m
        old, m.requests.post = m.requests.post, fake
        try:
            return m.fetch_states(pools, url=URL, block=100)
        finally:
            m.requests.post = old

    def test_a_pool_whose_call_errored_is_missing(self):
        states, _ = self._run([[
            {"id": 0, "result": "0x" + word(1000) + word(2000) + word(0)},
            {"id": 1, "error": {"code": -32000, "message": "execution reverted"}},
            {"id": 2, "result": "0x" + word(7)},
        ]], [A, B])
        assert A.key in states
        assert B.key not in states, \
            "a pool that did not answer must not appear as an empty book"

    def test_a_pool_answering_with_zeroes_is_dropped(self):
        states, _ = self._run([[
            {"id": 0, "result": "0x" + word(0) + word(0) + word(0)},
        ]], [A])
        assert states == {}

    def test_a_truncated_result_is_dropped(self):
        states, _ = self._run([[{"id": 0, "result": "0x00"}]], [A])
        assert states == {}

    def test_a_good_read_is_parsed(self):
        states, _ = self._run([[
            {"id": 0, "result": "0x" + word(1234) + word(5678) + word(99)},
        ]], [A])
        st = states[A.key]
        assert st.reserve0 == 1234 and st.reserve1 == 5678
        assert st.block == 100 and st.live

    def test_a_v3_pool_needs_both_halves(self):
        """slot0 without liquidity is not a usable state."""
        states, _ = self._run([[
            {"id": 0, "result": "0x" + word(5) + word(0) * 6},
        ]], [B])
        assert B.key not in states


class TestTransportFailuresAreLoud:

    def _expect(self, fake, pools=(A,)):
        import liqmap.dexfeed as m
        old, m.requests.post = m.requests.post, fake
        try:
            with pytest.raises(dexfeed.FeedError):
                m.fetch_states(list(pools), url=URL, block=1)
        finally:
            m.requests.post = old

    def test_a_connection_error_raises(self):
        """Swallowed into an empty dict it looks like a quiet market."""
        self._expect(FakePost([], raise_exc=dexfeed.requests.RequestException()))

    def test_a_non_200_raises(self):
        self._expect(FakePost([{}], status=500))

    def test_a_non_json_body_raises(self):
        self._expect(FakePost([None]))

    def test_a_rejected_batch_raises(self):
        self._expect(FakePost([{"error": {"message": "batch too large"}}]))

    def test_no_url_raises(self, monkeypatch):
        monkeypatch.delenv("ETH_RPC_URL", raising=False)
        monkeypatch.delenv("RPC_URL", raising=False)
        with pytest.raises(dexfeed.FeedError):
            dexfeed.fetch_states([A])


class TestHousekeeping:

    def test_no_pools_is_not_an_error(self):
        assert dexfeed.fetch_states([], url=URL) == ({}, 0)

    def test_the_selectors_are_the_real_ones(self):
        assert dexfeed.SEL_RESERVES == "0x0902f1ac"
        assert dexfeed.SEL_SLOT0 == "0x3850c7bd"
        assert dexfeed.SEL_LIQUIDITY == "0x1a686502"

    def test_word_reads_the_right_slot(self):
        data = "0x" + word(11) + word(22) + word(33)
        assert dexfeed._word(data, 0) == 11
        assert dexfeed._word(data, 1) == 22
        assert dexfeed._word(data, 2) == 33
        assert dexfeed._word(data, 9) == 0

    def test_the_url_is_read_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("ETH_RPC_URL", "https://a/b")
        assert dexfeed.rpc_url() == "https://a/b"
        monkeypatch.delenv("ETH_RPC_URL")
        monkeypatch.setenv("RPC_URL", "https://c/d")
        assert dexfeed.rpc_url() == "https://c/d"

    def test_the_url_is_never_in_an_error_message(self):
        """Errors reach the panel; the key is in the URL."""
        import liqmap.dexfeed as m
        fake = FakePost([], raise_exc=m.requests.RequestException())
        old, m.requests.post = m.requests.post, fake
        try:
            with pytest.raises(dexfeed.FeedError) as e:
                m.fetch_states([A], url="https://node/v2/SECRETKEY", block=1)
        finally:
            m.requests.post = old
        assert "SECRETKEY" not in str(e.value)

    def test_no_order_code(self):
        src = (ROOT / "liqmap" / "dexfeed.py").read_text()
        for w in ("def submit(", "def execute(", "def place(",
                  "def order(", "eth_sendRawTransaction", "private_key"):
            assert w not in src, w
