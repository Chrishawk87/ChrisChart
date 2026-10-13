"""The Ironbeam probe: everything about it that can be tested without a
broker account, which is everything except the socket itself.

IT IS A BROKER API, AND THAT IS THE DIFFERENCE. Every other feed this
project reads can only tell it things. This one could act. So the first
tests here are not about market data at all -- they are about the fact that
an order path is not reachable from this file, by construction rather than
by care.

AND IT DOES NOT KNOW THEIR SCHEMA. Their documentation could not be read
when it was written, so the field detection has to work on shape. These
tests feed it several plausible shapes, including a hostile one, and check
it finds the right fields or says it could not.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load():
    path = ROOT / "scripts" / "ironbeam_probe.py"
    spec = importlib.util.spec_from_file_location("ironbeam_probe", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ironbeam_probe"] = mod
    spec.loader.exec_module(mod)
    return mod


ib = _load()


# ------------------------------------------------ it cannot reach an order

@pytest.mark.parametrize("path", [
    "/order", "/orders/place", "/ORDER/new", "/position/close",
    "/order/cancel", "/trade/execute", "/account/withdraw",
])
def test_every_order_path_is_refused(path):
    """Not discouraged -- refused. A broker client one typo from a live
    order is a different kind of program from the one this is."""
    with pytest.raises(RuntimeError):
        ib._check(path)


def test_a_path_nobody_thought_about_is_refused_by_default():
    """An allowlist, not a blocklist. Anything unrecognised is refused,
    so a new endpoint added by the broker is not automatically reachable."""
    for path in ("/admin", "/funds", "/settings", "/", "/v2/anything"):
        with pytest.raises(RuntimeError):
            ib._check(path)


def test_the_read_only_paths_are_allowed():
    for path in ("/auth", "/stream", "/market/depth", "/info/symbols"):
        assert ib._check(path) == path


def test_no_order_vocabulary_anywhere_in_the_file():
    """The standing constraint, enforced here as it is on the engine.
    Comments and docstrings are stripped first -- this file EXPLAINS the
    rule, and saying so is not breaking it."""
    import ast

    src = (ROOT / "scripts" / "ironbeam_probe.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
            b = node.body
            if (b and isinstance(b[0], ast.Expr)
                    and isinstance(b[0].value, ast.Constant)
                    and isinstance(b[0].value.value, str)):
                b.pop(0)
    code = ast.unparse(tree).lower()
    for word in ("place_order", "submit_order", "market_order",
                 "limit_order", "cancel_order", "new_order"):
        assert word not in code, word


def test_the_forbidden_list_is_checked_before_the_allowlist():
    """'/trade/history' is allowed and '/trade/execute' is not, and they
    share a prefix. Checking the allowlist first would pass the second."""
    assert ib._check("/trade/history")
    with pytest.raises(RuntimeError):
        ib._check("/trade/execute")


# ------------------------------------------------- finding fields by shape

FLAT_TRADE = {"symbol": "MESZ6", "price": 6850.25, "size": 3,
              "side": "B", "timestamp": 1760000000}
NESTED_TRADE = {"d": {"last": {"price": 6850.25, "qty": 3,
                               "aggressor": "SELL"}}, "t": "trade"}
FLAT_BOOK = {"symbol": "MESZ6", "bid": 6850.0, "ask": 6850.25,
             "bidSize": 40, "askSize": 35}
NO_SIDE_TRADE = {"symbol": "MESZ6", "price": 6850.25, "volume": 3}


def test_a_flat_trade_is_recognised_and_its_fields_found():
    p = ib.Probe()
    p.observe(FLAT_TRADE)
    assert p.kinds["trade"] == 1
    assert p.guesses["price"] == "price"
    assert p.guesses["side"] == "side"


def test_a_nested_trade_is_found_two_levels_down():
    """A field buried in the message is as findable as one at the top,
    because nothing here knows how deep their messages are."""
    p = ib.Probe()
    p.observe(NESTED_TRADE)
    assert p.kinds["trade"] == 1
    assert p.guesses["price"].endswith("price")
    assert p.guesses["side"].endswith("aggressor")


def test_a_book_message_is_told_apart_from_a_trade():
    p = ib.Probe()
    p.observe(FLAT_BOOK)
    assert p.kinds["book"] == 1
    assert p.bid == 6850.0 and p.ask == 6850.25


def test_a_trade_with_no_side_field_is_reported_as_such():
    p = ib.Probe()
    p.observe(FLAT_BOOK)
    for _ in range(50):
        p.observe(NO_SIDE_TRADE)
    assert "side" not in p.guesses
    assert p.conv.report().verdict == "no label"


def test_the_shortest_matching_name_wins():
    """`price` beats `priceFormatted`; a longer name is usually a derived
    field, and a derived field is the wrong one to audit against."""
    p = ib.Probe()
    p.observe({"priceFormatted": 9999.0, "price": 6850.25, "size": 1,
               "side": "B"})
    assert p.guesses["price"] == "price"


def test_a_formatted_string_price_is_not_mistaken_for_the_number():
    p = ib.Probe()
    p.observe({"price": "6,850.25", "lastPrice": 6850.25, "size": 1,
               "side": "B"})
    assert p.guesses["price"] == "lastPrice"


def test_a_message_it_cannot_make_sense_of_is_counted_not_dropped():
    p = ib.Probe()
    p.observe({"heartbeat": True, "seq": 12})
    assert p.kinds["other"] == 1


def test_an_empty_or_odd_message_is_not_a_crash():
    p = ib.Probe()
    for msg in ({}, [], None, "", 0):
        p.observe(msg)
    assert sum(p.kinds.values()) == 0


# ------------------------------------- the audit runs on the live stream

def book(bid=6850.0, ask=6850.25):
    return {"bid": bid, "ask": ask, "bidSize": 10, "askSize": 10}


def trade(px, side):
    return {"price": px, "size": 1, "side": side}


def test_a_correctly_labelled_stream_settles_the_convention():
    p = ib.Probe()
    p.observe(book())
    for i in range(80):
        p.observe(trade(6850.25 if i % 2 else 6850.0, "B" if i % 2 else "S"))
    r = p.conv.report()
    assert r.verdict == "b_is_buyer"
    assert r.settled


def test_an_inverted_stream_is_caught():
    p = ib.Probe()
    p.observe(book())
    for i in range(80):
        p.observe(trade(6850.0 if i % 2 else 6850.25, "B" if i % 2 else "S"))
    assert p.conv.report().verdict == "a_is_buyer"


def test_trades_arriving_before_any_book_are_not_counted():
    """There is nothing to compare them against, and assuming a mid would
    invent the answer the whole script exists to measure."""
    p = ib.Probe()
    for i in range(80):
        p.observe(trade(6850.25, "B"))
    assert p.conv.report().quotable == 0


def test_the_book_used_is_the_most_recent_one_before_the_print():
    """Not the one after. A book sampled after the fill already shows the
    level that was just taken, so its mid has drifted towards the
    aggressor and every convention looks correct."""
    p = ib.Probe()
    p.observe(book(6850.0, 6850.25))
    p.observe(trade(6850.25, "B"))
    p.observe(book(6860.0, 6860.25))        # price ran away afterwards
    r = p.conv.report()
    assert r.n == 1 and r.b_above == 1


# --------------------------------------------------------------- the report

def test_the_report_prints_without_a_network_and_names_the_verdict(capsys):
    p = ib.Probe()
    p.observe(book())
    for i in range(80):
        p.observe(trade(6850.25 if i % 2 else 6850.0, "B" if i % 2 else "S"))
    p.print_report("MES")
    out = capsys.readouterr().out
    assert "IRONBEAM PROBE -- MES" in out
    assert "aggressive buyer" in out
    assert "RAW MESSAGES" in out


def test_the_report_says_so_when_nothing_arrived(capsys):
    ib.Probe().print_report("MES")
    out = capsys.readouterr().out
    assert "none" in out


def test_the_report_quantifies_what_inference_would_cost(capsys):
    """With no side field the useful number is how many prints could be
    assigned from the quote at all."""
    p = ib.Probe()
    p.observe(book())
    for i in range(100):
        px = 6850.125 if i < 40 else (6850.25 if i % 2 else 6850.0)
        p.observe({"price": px, "volume": 1})
    p.print_report("MES")
    out = capsys.readouterr().out
    assert "INFERRED" in out
    assert "60%" in out


def test_the_raw_dump_is_capped_so_the_report_stays_pasteable(capsys):
    p = ib.Probe(raw_keep=2)
    for i in range(50):
        p.observe({"price": 6850.0 + i, "size": 1, "side": "B"})
    p.print_report("MES")
    out = capsys.readouterr().out
    assert out.count("{") <= 12


def test_credentials_are_read_from_the_environment_and_never_written():
    import ast

    src = (ROOT / "scripts" / "ironbeam_probe.py").read_text()
    assert "os.environ" in src
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "attr", getattr(node.func, "id", ""))
            assert name not in ("write_text", "dump"), (
                "a credential must not reach a file")


def test_the_listening_window_is_capped():
    assert ib.MAX_SECONDS <= 600
