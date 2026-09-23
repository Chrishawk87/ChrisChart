"""Chart history ingest tests.

An upload parser has two ways to fail badly and only one of them looks like
a failure.

The loud way is refusing a valid file. Annoying, obvious, fixed in minutes.

The quiet way is accepting a broken one: reading milliseconds as seconds so
every bar lands in 1973, capturing "close_time" as the close price, silently
repairing a high that sits below its low. The backtest then runs happily over
nonsense and reports a number. That number gets acted on.

So everything here pins the quiet failures: units, column capture, rejection
rather than repair, and the report telling the truth about what the data can
and cannot train.
"""

import json

import pytest

from liqmap.ingest import CANNOT_TRAIN, merge, parse
from liqmap.structure import Candle

HEADER = "time,open,high,low,close,volume"


def csv_rows(n=10, start=1_700_000_000, step=900, px=100.0):
    lines = [HEADER]
    for i in range(n):
        o = px + i
        lines.append(f"{start + i * step},{o},{o + 1},{o - 1},{o + 0.5},{100 + i}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# the formats people actually upload
# --------------------------------------------------------------------------

def test_a_plain_csv_with_a_header():
    r = parse(csv_rows(20))
    assert r.ok and len(r.candles) == 20
    assert r.interval_s == 900.0
    assert r.interval_name == "15m"


def test_column_order_does_not_matter():
    text = "close,low,time,high,open\n101,99,1700000000,102,100\n" \
           "102,100,1700000900,103,101\n103,101,1700001800,104,102\n" \
           "104,102,1700002700,105,103\n105,103,1700003600,106,104\n" \
           "106,104,1700004500,107,105"
    r = parse(text)
    assert r.ok
    assert r.candles[0].open == 100.0 and r.candles[0].close == 101.0
    assert r.candles[0].high == 102.0 and r.candles[0].low == 99.0


def test_single_letter_headers_work():
    text = "t,o,h,l,c,v\n" + "\n".join(
        f"{1700000000 + i * 60},100,101,99,100.5,10" for i in range(8))
    r = parse(text)
    assert r.ok and r.interval_name == "1m"


def test_semicolons_and_tabs_are_detected():
    for delim in (";", "\t", "|"):
        text = HEADER.replace(",", delim) + "\n" + "\n".join(
            delim.join(str(x) for x in
                       (1700000000 + i * 300, 100, 101, 99, 100.5, 10))
            for i in range(8))
        r = parse(text)
        assert r.ok, f"{delim!r} was not detected"
        assert r.interval_name == "5m"


def test_a_headerless_file_is_read_positionally():
    text = "\n".join(
        f"{1700000000 + i * 3600},100,101,99,100.5,10" for i in range(8))
    r = parse(text)
    assert r.ok and r.interval_name == "1h"


def test_json_list_of_objects():
    rows = [{"time": 1700000000 + i * 900, "open": 100, "high": 101,
             "low": 99, "close": 100.5, "volume": 5} for i in range(8)]
    r = parse(json.dumps(rows))
    assert r.ok and len(r.candles) == 8


def test_json_list_of_lists_in_kline_order():
    rows = [[(1700000000 + i * 900) * 1000, "100", "101", "99", "100.5", "5"]
            for i in range(8)]
    r = parse(json.dumps(rows))
    assert r.ok and len(r.candles) == 8


def test_json_wrapped_in_an_envelope():
    rows = [{"t": 1700000000 + i * 900, "o": 100, "h": 101, "l": 99,
             "c": 100.5} for i in range(8)]
    r = parse(json.dumps({"status": "ok", "data": rows}))
    assert r.ok and len(r.candles) == 8


def test_a_utf8_bom_does_not_break_the_header():
    r = parse(("﻿" + csv_rows(8)).encode("utf-8"))
    assert r.ok


def test_prices_with_commas_and_currency_symbols():
    text = 'time,open,high,low,close\n' + "\n".join(
        f'{1700000000 + i * 900},"$1,200.50","$1,210.00","$1,190.00","$1,205.00"'
        for i in range(8))
    r = parse(text)
    assert r.ok
    assert r.candles[0].open == pytest.approx(1200.50)


# --------------------------------------------------------------------------
# units — the quiet killers
# --------------------------------------------------------------------------

def test_milliseconds_are_not_read_as_seconds():
    """The whole series lands in 1973 if this is wrong, and nothing else
    complains."""
    ms = "\n".join(f"{(1700000000 + i * 900) * 1000},100,101,99,100.5"
                   for i in range(8))
    r = parse(ms)
    assert r.ok
    assert r.candles[0].ts == pytest.approx(1700000000)
    assert r.interval_s == pytest.approx(900)


def test_microseconds_and_nanoseconds_are_scaled_too():
    for scale in (1e6, 1e9):
        text = "\n".join(f"{int((1700000000 + i * 900) * scale)},100,101,99,100.5"
                         for i in range(8))
        r = parse(text)
        assert r.ok and r.candles[0].ts == pytest.approx(1700000000, abs=1)


def test_iso_timestamps_with_and_without_a_zone():
    text = ("time,open,high,low,close\n"
            "2024-03-01T00:00:00Z,100,101,99,100.5\n"
            "2024-03-01T00:15:00Z,100,101,99,100.5\n"
            "2024-03-01T00:30:00,100,101,99,100.5\n"
            "2024-03-01 00:45:00,100,101,99,100.5\n"
            "2024-03-01T01:00:00+00:00,100,101,99,100.5\n"
            "2024-03-01T01:15:00Z,100,101,99,100.5")
    r = parse(text)
    assert r.ok and len(r.candles) == 6
    assert r.interval_s == pytest.approx(900)


def test_close_time_is_never_captured_as_the_close_price():
    """A Binance kline has both. Grabbing the wrong one puts a unix
    timestamp in the close and every bar becomes a vertical line."""
    text = ("open_time,open,high,low,close,volume,close_time\n"
            + "\n".join(
                f"{1700000000 + i * 900},100,101,99,100.5,10,"
                f"{1700000000 + i * 900 + 899}" for i in range(8)))
    r = parse(text)
    assert r.ok
    assert all(c.close == 100.5 for c in r.candles)


def test_open_time_beats_a_bare_time_column_for_the_bar_start():
    text = ("open,high,low,close,opentime\n"
            + "\n".join(f"100,101,99,100.5,{1700000000 + i * 900}"
                        for i in range(8)))
    r = parse(text)
    assert r.ok and r.candles[0].ts == pytest.approx(1700000000)


# --------------------------------------------------------------------------
# rejecting rather than repairing
# --------------------------------------------------------------------------

def test_a_high_below_its_low_is_dropped_not_swapped():
    text = HEADER + "\n" + "\n".join(
        f"{1700000000 + i * 900},100,101,99,100.5,10" for i in range(8))
    text += "\n1700010000,100,90,110,100,10"
    r = parse(text)
    assert len(r.candles) == 8
    assert r.rejected.get("high below low") == 1


def test_a_close_outside_the_bar_range_is_dropped():
    text = HEADER + "\n" + "\n".join(
        f"{1700000000 + i * 900},100,101,99,100.5,10" for i in range(8))
    text += "\n1700010000,100,101,99,150,10"
    r = parse(text)
    assert len(r.candles) == 8
    assert r.rejected.get("open or close outside the range") == 1


def test_zero_and_negative_prices_are_dropped():
    text = HEADER + "\n" + "\n".join(
        f"{1700000000 + i * 900},100,101,99,100.5,10" for i in range(8))
    text += "\n1700010000,0,0,0,0,10\n1700010900,-1,-1,-1,-1,10"
    r = parse(text)
    assert len(r.candles) == 8
    assert r.rejected.get("non-positive price") == 2


def test_duplicate_timestamps_are_kept_once():
    text = HEADER + "\n" + "\n".join(
        f"{1700000000 + i * 900},100,101,99,100.5,10" for i in range(8))
    text += "\n1700000000,100,101,99,100.5,10"
    r = parse(text)
    assert len(r.candles) == 8
    assert r.rejected.get("duplicate timestamp") == 1


def test_rows_are_sorted_even_when_the_file_is_not():
    rows = [f"{1700000000 + i * 900},100,101,99,100.5,10" for i in range(8)]
    r = parse(HEADER + "\n" + "\n".join(reversed(rows)))
    assert r.ok
    assert [c.ts for c in r.candles] == sorted(c.ts for c in r.candles)


def test_a_mostly_broken_file_says_so_loudly():
    good = [f"{1700000000 + i * 900},100,101,99,100.5,10" for i in range(6)]
    bad = [f"{1700100000 + i * 900},100,90,110,100,10" for i in range(20)]
    r = parse(HEADER + "\n" + "\n".join(good + bad))
    assert any("half the rows" in w for w in r.warnings)


def test_a_small_sample_is_flagged_as_unprovable():
    r = parse(csv_rows(30))
    assert any("cannot distinguish it from luck" in w for w in r.warnings)


# --------------------------------------------------------------------------
# failing safely
# --------------------------------------------------------------------------

def test_an_empty_file_returns_a_report_not_an_exception():
    r = parse("")
    assert not r.ok and "empty file" in r.rejected


def test_a_file_with_no_price_columns_explains_what_was_expected():
    r = parse("alpha,beta\n1,2\n3,4")
    assert not r.ok
    assert any("time, open, high, low and close" in w for w in r.warnings)


def test_unreadable_json_does_not_raise():
    r = parse("{not json at all")
    assert not r.ok


def test_arbitrary_binary_does_not_raise():
    r = parse(bytes(range(256)))
    assert isinstance(r.ok, bool)


def test_a_trade_list_uploaded_by_mistake_is_rejected_not_parsed():
    text = "time,side,price,size\n1700000000,buy,100,2\n1700000060,sell,101,3"
    r = parse(text)
    assert not r.ok


# --------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------

def test_the_report_always_states_what_this_data_cannot_train():
    """The single most important sentence this module produces. Uploading
    three months of bars and being told the model is trained on three months
    would be a lie by omission."""
    text = parse(csv_rows(400)).describe()
    assert "seventy percent" in text
    assert "microprice" in text.lower()
    assert "forward" in text


def test_the_dict_form_lists_both_halves_explicitly():
    d = parse(csv_rows(400)).to_dict()
    assert "microprice tilt" in d["cannot_train"]
    assert "absorption" in d["cannot_train"]
    assert any("price action" in t for t in d["trains"])


def test_gaps_are_counted_not_filled():
    rows = [f"{1700000000 + i * 900},100,101,99,100.5,10" for i in range(10)]
    del rows[5]
    r = parse(HEADER + "\n" + "\n".join(rows))
    assert r.gaps() == 1
    assert any("missing" in w or "missing" in r.describe() for w in [""])
    assert len(r.candles) == 9


def test_a_weekend_does_not_become_the_timeframe():
    """Median, not mean: an equity chart with a two-day hole every five days
    must still read as a 15m series."""
    rows = []
    t = 1700000000
    for day in range(4):
        for i in range(30):
            rows.append(f"{t},100,101,99,100.5,10")
            t += 900
        t += 2 * 86400
    r = parse(HEADER + "\n" + "\n".join(rows))
    assert r.interval_name == "15m"


def test_an_empty_report_still_describes_itself():
    assert "No usable bars" in parse("").describe()


# --------------------------------------------------------------------------
# merging
# --------------------------------------------------------------------------

def test_merge_extends_rather_than_duplicating_the_overlap():
    a = parse(csv_rows(10)).candles
    b = parse(csv_rows(10, start=1_700_000_000 + 5 * 900)).candles
    out = merge(a, b)
    assert len({c.ts for c in out}) == len(out)
    assert [c.ts for c in out] == sorted(c.ts for c in out)
    assert len(out) == 15


def test_merge_prefers_the_newly_uploaded_version_of_a_bar():
    a = [Candle(ts=1.0, open=1, high=2, low=0.5, close=1.5)]
    b = [Candle(ts=1.0, open=9, high=10, low=8, close=9.5)]
    assert merge(a, b)[0].open == 9
