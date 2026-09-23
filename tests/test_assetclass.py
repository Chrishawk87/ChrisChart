"""Asset-class tests.

The cost of a wrong label is a market hidden in a group you would never open,
so the rule is: guess only where guessing is safe. An unknown ticker on the
canonical crypto DEX is crypto; an unknown ticker on a builder DEX is not
guessed into a class it might not belong to.
"""

import pytest

from liqmap.assetclass import ORDER, classify, group, label


@pytest.mark.parametrize("symbol,dex,expected", [
    ("BTC", "", "crypto"),
    ("ETH", "", "crypto"),
    ("kPEPE", "", "crypto"),
    ("1000SHIB", "", "crypto"),
    ("@107", "", "crypto"),
    ("HYPE", "", "crypto"),
    ("vntl:GOLD", "vntl", "metals"),
    ("vntl:SILVER", "vntl", "metals"),
    ("PLATINUM", "vntl", "metals"),
    ("COPPER", "vntl", "metals"),
    ("CL", "vntl", "energy"),
    ("BRENTOIL", "vntl", "energy"),
    ("NATGAS", "vntl", "energy"),
    ("EUR", "vntl", "fx"),
    ("GBP", "vntl", "fx"),
    ("JPY", "vntl", "fx"),
    ("JP225", "vntl", "index"),
    ("KR200", "vntl", "index"),
    ("SPX", "vntl", "index"),
    ("SPY", "para", "etf"),
    ("EWZ", "para", "etf"),
    ("KORU", "para", "etf"),
    ("NVDA", "para", "equity"),
    ("AAPL", "para", "equity"),
    ("MSTR", "para", "equity"),
])
def test_known_symbols_land_in_the_right_class(symbol, dex, expected):
    assert classify(symbol, dex) == expected


def test_an_unknown_ticker_on_the_crypto_dex_is_crypto():
    """The canonical DEX is crypto by construction, so this guess is safe."""
    assert classify("ZZZNEWCOIN", "") == "crypto"


def test_an_unknown_long_ticker_on_a_builder_dex_is_not_guessed():
    """A wrong label hides a market in a group nobody opens. 'Other' is the
    honest answer when the symbol says nothing."""
    assert classify("SOMEWEIRDTHING", "para") == "other"


def test_an_fx_pair_is_recognised_from_its_spelling():
    assert classify("EUR/USD", "vntl") == "fx"
    assert classify("GBP-JPY", "vntl") == "fx"


def test_a_crypto_pair_is_recognised_from_its_spelling():
    assert classify("BTC/USDC", "") == "crypto"


def test_leveraged_product_names_read_as_etfs():
    assert classify("TSLABULL3X", "para") == "etf"


def test_empty_and_junk_are_other():
    assert classify("", "") == "other"
    assert classify(":", "x") == "other"


def test_the_dex_prefix_is_stripped_before_matching():
    assert classify("anything:GOLD", "anything") == "metals"


def test_group_buckets_in_display_order():
    rows = [{"symbol": "NVDA", "dex": "para"},
            {"symbol": "BTC", "dex": ""},
            {"symbol": "vntl:GOLD", "dex": "vntl"}]
    g = group(rows)
    assert list(g) == [k for k in ORDER if k in g]
    assert list(g)[0] == "crypto"
    assert "metals" in g and "equity" in g


def test_group_omits_empty_classes():
    g = group([{"symbol": "BTC", "dex": ""}])
    assert list(g) == ["crypto"]


def test_labels_are_human_readable():
    assert label("fx") == "FX"
    assert label("equity") == "Stocks"
    assert label("etf") == "ETFs"
