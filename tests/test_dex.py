"""Multi-DEX tests.

Hyperliquid's canonical perp DEX carries the crypto majors. Gold, oil, FX, the
index perps and the single stocks are HIP-3 builder-deployed markets, each on
its own perp DEX, and a `clearinghouseState` call without a `dex` parameter
returns NONE of them.

That failure is silent. The response is well-formed and simply has no gold in
it, so the dashboard shows an empty market rather than an error, and an empty
market is indistinguishable from one nobody is trading. Everything here exists
to make sure a symbol that lives on another venue is found rather than quietly
dropped.

The two response shapes matter for the same reason. A single-DEX call returns
a flat object; an ALL_DEXES call returns a mapping keyed by DEX name. Guessing
wrong does not raise -- it returns zero positions.
"""

import pytest

from liqmap.hl import (
    HyperliquidError, join_symbol, positions_from_state, split_symbol,
    sweep_positions,
)


def state(*positions, account="1000000"):
    """A flat single-DEX clearinghouseState payload."""
    return {
        "marginSummary": {"accountValue": account},
        "assetPositions": [
            {"type": "oneWay", "position": {
                "coin": coin, "szi": str(szi), "entryPx": str(entry),
                "liquidationPx": str(liq), "positionValue": str(value),
                "leverage": {"type": "cross", "value": "10"},
                "unrealizedPnl": "500", "marginUsed": "1000",
                "cumFunding": {"sinceOpen": "-12.5"},
            }} for coin, szi, entry, liq, value in positions
        ],
    }


# --------------------------------------------------------------------------
# symbol namespacing
# --------------------------------------------------------------------------

def test_namespaced_symbols_split():
    assert split_symbol("para:CRDO") == ("para", "CRDO")
    assert split_symbol("BTC") == ("", "BTC")


def test_join_is_the_inverse():
    for dex, coin in (("para", "CRDO"), ("", "BTC")):
        assert split_symbol(join_symbol(dex, coin)) == (dex, coin)


def test_a_symbol_with_several_colons_splits_on_the_first():
    assert split_symbol("a:b:c") == ("a", "b:c")


# --------------------------------------------------------------------------
# the two response shapes
# --------------------------------------------------------------------------

def test_flat_single_dex_payload_is_parsed():
    out = positions_from_state("0xa", state(("BTC", 1.0, 99_000, 90_000, 1e6)))
    assert [p.coin for p in out] == ["BTC"]


def test_dex_keyed_payload_is_parsed_and_namespaced():
    """ALL_DEXES returns a mapping. The canonical perps sit under "native" and
    keep their bare symbols; HIP-3 markets get their DEX prefixed so two
    builders listing GOLD never merge into one market."""
    payload = {
        "native": state(("BTC", 1.0, 99_000, 90_000, 1e6)),
        "vntl": state(("GOLD", 2.0, 4_100, 3_800, 800_000)),
        "para": state(("AAPL", -3.0, 230, 260, 500_000)),
    }
    out = positions_from_state("0xa", payload)
    coins = sorted(p.coin for p in out)
    assert coins == ["BTC", "para:AAPL", "vntl:GOLD"]


def test_already_namespaced_symbols_are_not_prefixed_twice():
    payload = {"vntl": state(("vntl:GOLD", 1.0, 4_100, 3_800, 100_000))}
    assert [p.coin for p in positions_from_state("0xa", payload)] == ["vntl:GOLD"]


def test_account_value_survives_the_nesting():
    payload = {"vntl": state(("GOLD", 1.0, 4_100, 3_800, 100_000),
                             account="7500000")}
    assert positions_from_state("0xa", payload)[0].account_value == 7_500_000.0


def test_a_broken_dex_does_not_lose_the_working_ones():
    """One builder's DEX returning junk must not cost you the rest."""
    payload = {
        "native": state(("BTC", 1.0, 99_000, 90_000, 1e6)),
        "broken": "not a dict at all",
        "alsobroken": {"unexpected": True},
        "vntl": state(("GOLD", 1.0, 4_100, 3_800, 100_000)),
    }
    coins = sorted(p.coin for p in positions_from_state("0xa", payload))
    assert coins == ["BTC", "vntl:GOLD"]


def test_junk_payloads_return_nothing_rather_than_raising():
    for bad in (None, [], "text", 42):
        assert positions_from_state("0xa", bad) == []


def test_empty_mapping_is_empty_not_an_error():
    assert positions_from_state("0xa", {}) == []


# --------------------------------------------------------------------------
# the sweep
# --------------------------------------------------------------------------

class FakeClient:
    def __init__(self, responses, reject_dex=False):
        self.responses = responses
        self.reject_dex = reject_dex
        self.calls: list[tuple[str, str]] = []

    def clearinghouse_state(self, wallet, dex=""):
        self.calls.append((wallet, dex))
        if dex and self.reject_dex:
            raise HyperliquidError("422 on clearinghouseState: unknown field `dex`")
        return self.responses.get(wallet, {})


def test_sweep_asks_for_every_dex_by_default():
    c = FakeClient({"0xa": {"native": state(("BTC", 1.0, 99_000, 90_000, 1e6))}})
    positions, failed = sweep_positions(c, ["0xa"])
    assert c.calls == [("0xa", "ALL_DEXES")]
    assert failed == 0
    assert [p.coin for p in positions] == ["BTC"]


def test_sweep_collects_hip3_markets():
    c = FakeClient({
        "0xa": {"native": state(("BTC", 1.0, 99_000, 90_000, 1e6)),
                "vntl": state(("GOLD", 1.0, 4_100, 3_800, 500_000))},
        "0xb": {"vntl": state(("CL", -1.0, 74, 82, 300_000))},
    })
    positions, _ = sweep_positions(c, ["0xa", "0xb"])
    assert sorted(p.coin for p in positions) == ["BTC", "vntl:CL", "vntl:GOLD"]


def test_sweep_falls_back_once_when_the_wildcard_is_rejected():
    """An exchange that does not accept ALL_DEXES must not fail every wallet
    in the universe one at a time."""
    c = FakeClient({"0xa": state(("BTC", 1.0, 99_000, 90_000, 1e6)),
                    "0xb": state(("ETH", 1.0, 3_000, 2_700, 1e6))},
                   reject_dex=True)
    positions, failed = sweep_positions(c, ["0xa", "0xb"])

    assert failed == 0
    assert sorted(p.coin for p in positions) == ["BTC", "ETH"]
    # First wallet tries the wildcard then retries bare; the second must not
    # try the wildcard again.
    assert c.calls == [("0xa", "ALL_DEXES"), ("0xa", ""), ("0xb", "")]


def test_a_failing_wallet_is_skipped_not_fatal():
    class Flaky(FakeClient):
        def clearinghouse_state(self, wallet, dex=""):
            if wallet == "0xbad":
                raise HyperliquidError("500 server error")
            return super().clearinghouse_state(wallet, dex)

    c = Flaky({"0xa": {"native": state(("BTC", 1.0, 99_000, 90_000, 1e6))}})
    positions, failed = sweep_positions(c, ["0xbad", "0xa"])
    assert failed == 1
    assert [p.coin for p in positions] == ["BTC"]


def test_sweep_can_be_pinned_to_the_canonical_dex():
    c = FakeClient({"0xa": state(("BTC", 1.0, 99_000, 90_000, 1e6))})
    sweep_positions(c, ["0xa"], dex="")
    assert c.calls == [("0xa", "")]


# --------------------------------------------------------------------------
# mids across venues
# --------------------------------------------------------------------------

class MidsClient:
    def __init__(self, per_dex, dexes=None, fail=()):
        self.per_dex = per_dex
        self._dexes = dexes if dexes is not None else list(per_dex)
        self.fail = set(fail)

    def all_mids(self, dex=""):
        if dex in self.fail:
            raise HyperliquidError("nope")
        return self.per_dex.get(dex, {})

    def perp_dexs(self):
        return ([{"name": "", "native": True}]
                + [{"name": d, "native": False} for d in self._dexes if d])


def test_mids_are_gathered_from_every_dex_and_namespaced():
    from liqmap.hl import InfoClient
    c = MidsClient({"": {"BTC": 100_000.0},
                    "vntl": {"GOLD": 4_100.0, "CL": 74.0}},
                   dexes=["vntl"])
    mids = InfoClient.all_mids_everywhere(c)
    assert mids == {"BTC": 100_000.0, "vntl:GOLD": 4_100.0, "vntl:CL": 74.0}


def test_one_dead_dex_does_not_lose_the_rest():
    from liqmap.hl import InfoClient
    c = MidsClient({"": {"BTC": 100_000.0}, "vntl": {"GOLD": 4_100.0}},
                   dexes=["vntl", "dead"], fail={"dead"})
    mids = InfoClient.all_mids_everywhere(c)
    assert "vntl:GOLD" in mids and "BTC" in mids


def test_mids_fall_back_to_canonical_when_dex_discovery_fails():
    from liqmap.hl import InfoClient

    class NoDexes(MidsClient):
        def perp_dexs(self):
            raise HyperliquidError("perpDexs unavailable")

    c = NoDexes({"": {"BTC": 100_000.0}})
    assert InfoClient.all_mids_everywhere(c) == {"BTC": 100_000.0}


# --------------------------------------------------------------------------
# perpDexs normalisation
# --------------------------------------------------------------------------

def test_perp_dexs_handles_the_null_native_entry():
    from liqmap.hl import InfoClient

    class C:
        _dexes_cache = None          # perp_dexs caches; give the stub the slot
        DEXES_TTL_S = 600.0

        def post(self, body):
            return [None, {"name": "vntl", "fullName": "Ventuals"},
                    {"name": "para", "fullName": "Paradex"}]

    out = InfoClient.perp_dexs(C())
    assert out[0]["native"] is True and out[0]["name"] == ""
    assert [d["name"] for d in out[1:]] == ["vntl", "para"]
    assert out[1]["full_name"] == "Ventuals"


def test_perp_dexs_handles_a_bare_list_of_names():
    from liqmap.hl import InfoClient

    class C:
        _dexes_cache = None
        DEXES_TTL_S = 600.0

        def post(self, body):
            return ["vntl", "para"]

    assert [d["name"] for d in InfoClient.perp_dexs(C())] == ["vntl", "para"]


# --------------------------------------------------------------------------
# case handling on namespaced symbols
# --------------------------------------------------------------------------

def test_dex_prefix_case_is_preserved_when_normalising():
    """HIP-3 DEX names are lowercase and case-sensitive. Upper-casing the
    whole symbol turns `vntl:GOLD` into `VNTL:GOLD`, which matches nothing on
    the exchange and is indistinguishable from a market that does not exist."""
    import liqmap.web as web

    class FakeHistory:
        def coins_with_data(self):
            return []

    rt = object.__new__(web.Runtime)
    rt.history = FakeHistory()
    rt._markets_cache = None       # resolution also consults the listed universe

    assert web.Runtime.resolve_symbol(rt, "vntl:GOLD")[0] == "vntl:GOLD"
    assert web.Runtime.resolve_symbol(rt, "vntl:gold")[0] == "vntl:GOLD"
    assert web.Runtime.resolve_symbol(rt, "btc")[0] == "BTC"
    assert web.Runtime.resolve_symbol(rt, "")[0] == ""
