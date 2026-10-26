#!/usr/bin/env python3
"""Organic OFI candles from CME Level 1. Self-contained, no time anywhere.

    # against the MBP-10 files already in data/
    python scripts/organic_ofi.py data/ES_c_0_mbp-10_*.dbn.zst --target 4000

    # find a threshold that produces a readable chart first
    python scripts/organic_ofi.py data/ES_c_0_mbp-10_*.dbn.zst --sweep

    # live, if DATABENTO_API_KEY is in the environment
    python scripts/organic_ofi.py --live --symbol ES.c.0 --target 4000

Imports only databento, pandas, numpy and the standard library. Nothing
from liqmap, so this file can be lifted out of the repo whole.

===========================================================================
THE SIGN OF THE ASK SIDE
===========================================================================
The brief for this script defined the ask contributions and then wrote
ΔI = ΔBid - ΔAsk, which negates them. Checked against the engine already
running on the site, over 200,000 random book transitions, that form
agreed 223 times out of 200,000 -- the cases where the ask contributed
nothing at all.

Concretely: the offer ticks up through 300 resting lots and nothing else
moves. Somebody lifted that offer. Cont-Kukanov-Stoikov scores it +300,
buying. The subtracted form scores it -300, selling. Every bar's ask
component would be coloured backwards, and the chart would look
plausible the whole time.

So ASK_SIGN defaults to the published formula, which is also what your
previous brief specified and what liqmap/ofi.py already computes. Pass
--literal-spec to run the subtracted form and see the two side by side.

===========================================================================
WHY THERE IS NO CLOCK IN HERE
===========================================================================
No interval, no poll, no timer, no `time.sleep`. A bar ends when the
flow that built it crosses a line and at no other moment, so every bar
carries the same quantity of imbalance rather than the same quantity of
clock. A quiet hour is one bar; a violent minute is twenty.

The accumulator RESETS AT EVERY CLOSE, and that is not a detail. A
single global accumulator run against a fixed threshold is a latch, not
a filter: cumulative OFI tracks cumulative price, so on any session with
a direction it walks away from zero and never returns. Measured on a
book drifting up at 52% of ticks, that construction printed 99.9% green
and 0.1% everything else over 60,000 updates -- a chart with one word.
Resetting makes the threshold mean "imbalance since the last bar", a
quantity that can go either way and has to be earned again each time.

The ONE concession to time is `flush()`, called once when the stream
ends, so the final partial bar is not silently dropped. That is a
termination condition, not a polling loop.

===========================================================================
MEMORY AND THE HOT PATH
===========================================================================
`process_l1_packet` runs once per book update -- tens of millions of
times over a session file -- so everything it touches is a plain
attribute on a __slots__ object, which CPython resolves through a fixed
array offset rather than an instance dictionary lookup. No dataclass, no
namedtuple, no dict building, no string formatting, and no allocation at
all on the common path: the only object created is the bar dict at a
close, which happens on a small fraction of packets.

Prices stay as the feed's raw integers for as long as possible. A
Databento price is a fixed-point int scaled by 1e-9, and comparing two
ints is exact and fast where comparing two floats invites an equality
test that fails on a value that should have matched. The conversion to a
float price happens once, when a bar is emitted.

Honest note on speed: this is Python, and a claim of sub-microsecond
per-packet processing would be false. CPython's interpreter overhead
alone is a few hundred nanoseconds per bytecode, and a packet costs
dozens. Measured throughput is printed at the end of every run so you
can see the real figure rather than trust a promise. What the structure
above buys is a constant factor, not a change of category.

===========================================================================
This is a charting study. It holds no position, places no order, and
contains no broker or execution code of any kind.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# 1. HARDCODED SYSTEM CONTROLS
# --------------------------------------------------------------------------

CONFIG = {
    # Read from the environment, never written here. A key in a source file
    # is a key in git history.
    "API_KEY": os.environ.get("DATABENTO_API_KEY", ""),
    "DATASET": "GLBX.MDP3",
    "SYMBOL": "ES.c.0",
    "SCHEMA": "mbp-1",          # mbp-10 files work too; only level 0 is read
    "OFI_BLOCK_TARGET": 4000,   # see --sweep: 250 is far below ES's noise floor
    "TICK_RAW": 250_000_000,    # one ES tick in the feed's raw integer units
}

UNDEF_PRICE = 2 ** 63 - 1
PRICE_SCALE = 1e-9

ORGANIC_GREEN = "ORGANIC_GREEN"
ORGANIC_RED = "ORGANIC_RED"
UNCLOSED = "UNCLOSED"

COLUMNS = ("Timestamp", "Open", "High", "Low", "Close",
           "Directional_Color", "Total_OFI", "Packets")


class OrganicOfiCandleEngine:
    """Level 1 in, organically closed candles out.

    __slots__ rather than a dict: every name below is resolved by array
    offset in the hot path. The class is deliberately boring.
    """

    __slots__ = ("target_ofi", "ask_sign", "cumulative_ofi",
                 "prev_bid_price", "prev_bid_size",
                 "prev_ask_price", "prev_ask_size",
                 "c_open", "c_high", "c_low", "c_close", "c_ts_open",
                 "c_packets", "c_ofi_high", "c_ofi_low",
                 "completed_candles", "packets", "skipped")

    def __init__(self, target_ofi: int = 4000, ask_sign: int = +1) -> None:
        if target_ofi < 1:
            raise ValueError("target_ofi must be at least 1 contract")
        if ask_sign not in (+1, -1):
            raise ValueError("ask_sign must be +1 (published) or -1 (literal)")
        self.target_ofi = int(target_ofi)
        # +1 applies the ask contributions as Cont-Kukanov-Stoikov defines
        # them. -1 reproduces the subtracted form from the brief.
        self.ask_sign = int(ask_sign)
        self.cumulative_ofi = 0

        # Level 1 memory registers. None means "no previous packet yet",
        # which is distinct from a price of zero.
        self.prev_bid_price: int | None = None
        self.prev_bid_size = 0
        self.prev_ask_price: int | None = None
        self.prev_ask_size = 0

        # Active candle metrics, in RAW integer price units.
        self.c_open: int | None = None
        self.c_high = -sys.maxsize
        self.c_low = sys.maxsize
        self.c_close: int | None = None
        self.c_ts_open = 0
        self.c_packets = 0
        self.c_ofi_high = 0
        self.c_ofi_low = 0

        self.completed_candles: list[dict] = []
        self.packets = 0
        self.skipped = 0

    # ------------------------------------------------------------------
    def _open_candle(self, mid2: int, ts: int) -> None:
        """Open a bar at twice the mid, kept integer.

        Twice, not the mid itself: the mid of a one-tick spread is a half
        tick, and carrying it as an int avoids a float for the whole
        lifetime of the bar. It is halved once, on emit.
        """
        self.c_open = mid2
        self.c_high = mid2
        self.c_low = mid2
        self.c_close = mid2
        self.c_ts_open = ts
        self.c_packets = 0
        self.c_ofi_high = 0
        self.c_ofi_low = 0
        self.cumulative_ofi = 0

    def _close_candle(self, colour: str, ts: int) -> dict:
        half = 0.5 * PRICE_SCALE
        bar = {
            "Timestamp": ts,
            "Open": self.c_open * half,
            "High": self.c_high * half,
            "Low": self.c_low * half,
            "Close": self.c_close * half,
            "Directional_Color": colour,
            "Total_OFI": self.cumulative_ofi,
            "Packets": self.c_packets,
        }
        self.completed_candles.append(bar)
        # The next bar opens where this one closed. No gap, and no clock
        # involved in deciding when.
        self._open_candle(self.c_close, ts)
        return bar

    # ------------------------------------------------------------------
    def process_l1_packet(self, bid_price: int, bid_size: int,
                          ask_price: int, ask_size: int,
                          ts: int = 0) -> dict | None:
        """One book update. Returns a bar if this packet closed one.

        Raw integers in. The caller does the decoding; this does the
        arithmetic and nothing else.
        """
        # A crossed, locked or absent book is not a state the formula is
        # defined on. Withhold rather than contribute a zero, which would
        # sum like a measurement and quietly drag the accumulator.
        if (bid_price is None or ask_price is None
                or bid_price <= 0 or ask_price <= 0
                or bid_price == UNDEF_PRICE or ask_price == UNDEF_PRICE
                or ask_price <= bid_price):
            self.skipped += 1
            return None

        self.packets += 1
        mid2 = bid_price + ask_price

        prev_bp = self.prev_bid_price
        if prev_bp is None:
            # First packet: seed the registers and open the first bar.
            # No delta exists yet -- there is nothing to difference against.
            self.prev_bid_price = bid_price
            self.prev_bid_size = bid_size
            self.prev_ask_price = ask_price
            self.prev_ask_size = ask_size
            self._open_candle(mid2, ts)
            return None

        prev_bs = self.prev_bid_size
        prev_ap = self.prev_ask_price
        prev_as = self.prev_ask_size

        # ---- ΔBid ----------------------------------------------------
        # A better bid arriving is buying support added at a new level, so
        # the WHOLE new size counts. A bid stepping down is the old level
        # removed, so the whole OLD size counts against. Unchanged, only
        # the difference is new information.
        if bid_price > prev_bp:
            d_bid = bid_size
        elif bid_price < prev_bp:
            d_bid = -prev_bs
        else:
            d_bid = bid_size - prev_bs

        # ---- ΔAsk ----------------------------------------------------
        # Mirrored, and this is the sign discussed at the top of the file.
        # The offer ticking UP means the old offer is gone -- lifted or
        # pulled -- which is pressure in the buyers' favour, so it enters
        # POSITIVE. The offer ticking down is a new seller undercutting,
        # which is negative.
        if ask_price > prev_ap:
            d_ask = prev_as
        elif ask_price < prev_ap:
            d_ask = -ask_size
        else:
            d_ask = prev_as - ask_size

        self.cumulative_ofi += d_bid + self.ask_sign * d_ask

        # ---- roll the registers forward -------------------------------
        self.prev_bid_price = bid_price
        self.prev_bid_size = bid_size
        self.prev_ask_price = ask_price
        self.prev_ask_size = ask_size

        # ---- intrabar state -------------------------------------------
        if self.c_open is None:
            self._open_candle(mid2, ts)
        self.c_packets += 1
        if mid2 > self.c_high:
            self.c_high = mid2
        if mid2 < self.c_low:
            self.c_low = mid2
        self.c_close = mid2
        acc = self.cumulative_ofi
        if acc > self.c_ofi_high:
            self.c_ofi_high = acc
        if acc < self.c_ofi_low:
            self.c_ofi_low = acc

        # ---- the only close condition ---------------------------------
        if acc >= self.target_ofi:
            return self._close_candle(ORGANIC_GREEN, ts)
        if acc <= -self.target_ofi:
            return self._close_candle(ORGANIC_RED, ts)
        return None

    # ------------------------------------------------------------------
    def flush(self, ts: int = 0) -> dict | None:
        """Emit the final partial bar at end of stream.

        A termination condition, not a timer. Without it the last bar of
        every run is dropped, which biases the sample toward whatever
        happens to close quickly.
        """
        if self.c_open is None or self.c_packets == 0:
            return None
        bar = self._close_candle(UNCLOSED, ts or self.c_ts_open)
        self.c_open = None
        return bar

    def emit_candle_dataframe(self) -> pd.DataFrame:
        """Completed bars, ready for the chart."""
        if not self.completed_candles:
            return pd.DataFrame(columns=list(COLUMNS))
        df = pd.DataFrame(self.completed_candles, columns=list(COLUMNS))
        # Nanoseconds since epoch, as the feed gives them.
        df["Time"] = pd.to_datetime(df["Timestamp"], unit="ns", utc=True)
        df["Range_Ticks"] = np.rint(
            (df["High"] - df["Low"]) / (CONFIG["TICK_RAW"] * PRICE_SCALE)
        ).astype(int)
        return df


# --------------------------------------------------------------------------
# 2. FEEDING IT
# --------------------------------------------------------------------------

def _field(rec, name, default=None):
    return getattr(rec, name, default)


def iter_file(path: str):
    """(ts, bid_px, bid_sz, ask_px, ask_sz) from a .dbn or .dbn.zst file.

    Works for mbp-1 and mbp-10 alike -- only level 0 is read, and an
    mbp-10 record carries level 0 in exactly the same place.
    """
    import databento as db

    store = db.DBNStore.from_file(path)
    for rec in store:
        levels = _field(rec, "levels")
        if not levels:
            continue
        l0 = levels[0]
        yield (_field(rec, "ts_event", 0) or 0,
               _field(l0, "bid_px", 0) or 0,
               int(_field(l0, "bid_sz", 0) or 0),
               _field(l0, "ask_px", 0) or 0,
               int(_field(l0, "ask_sz", 0) or 0))


def iter_live(symbol: str, dataset: str, schema: str, key: str):
    """The same tuples, off the live feed.

    Blocking iteration over the client's own callback stream. There is no
    poll interval here: records arrive when the exchange sends them.
    """
    import databento as db

    if not key:
        raise SystemExit("DATABENTO_API_KEY is not set in the environment")
    client = db.Live(key=key)
    client.subscribe(dataset=dataset, schema=schema, symbols=symbol,
                     stype_in="continuous")
    for rec in client:
        levels = _field(rec, "levels")
        if not levels:
            continue
        l0 = levels[0]
        yield (_field(rec, "ts_event", 0) or 0,
               _field(l0, "bid_px", 0) or 0,
               int(_field(l0, "bid_sz", 0) or 0),
               _field(l0, "ask_px", 0) or 0,
               int(_field(l0, "ask_sz", 0) or 0))


# --------------------------------------------------------------------------
# 3. RUNNING IT
# --------------------------------------------------------------------------

# CALIBRATED AGAINST REAL ES, not a synthetic book. An earlier range
# started at 250 and climbed to 64,000; measured on 55M real packets,
# 4,000 produced 119 bars in 102.5 hours -- one every 52 minutes. The
# synthetic feed it was tuned on had far noisier sizes than a real
# book, so the accumulator wandered much further and much faster.
#
# OFI accumulates like a random walk, so first passage to +/-L scales
# as L squared and the bar count goes as 1/L^2. One measured point
# therefore predicts the whole curve: 119 bars at 4,000 means roughly
# 1,900 at 1,000 and 30,000 at 250.
SWEEP = (60, 125, 250, 500, 1000, 2000, 4000)

# A gap longer than this is a session boundary or the join between two
# files, not quiet trading. Counting it as recorded book is how three
# files spanning 102 hours get reported as 414.
SESSION_GAP_NS = 300 * 1_000_000_000


def describe(df: pd.DataFrame, target: int, hours: float) -> str:
    if df.empty:
        return f"  {target:>7,}  no bars closed"
    done = df[df["Directional_Color"] != UNCLOSED]
    if done.empty:
        return f"  {target:>7,}  nothing closed organically"
    g = int((done["Directional_Color"] == ORGANIC_GREEN).sum())
    r = int((done["Directional_Color"] == ORGANIC_RED).sum())
    n = len(done)
    per_hr = n / hours if hours > 0 else 0.0
    return (f"  {target:>7,}{n:>9,}{per_hr:>11.0f}"
            f"{g / n * 100:>8.1f}%{r / n * 100:>8.1f}%"
            f"{done['Range_Ticks'].median():>11.2f}"
            f"{done['Packets'].median():>10.0f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help=".dbn / .dbn.zst (mbp-1 or mbp-10)")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--symbol", default=CONFIG["SYMBOL"])
    ap.add_argument("--dataset", default=CONFIG["DATASET"])
    ap.add_argument("--schema", default=CONFIG["SCHEMA"])
    ap.add_argument("--target", type=int, default=CONFIG["OFI_BLOCK_TARGET"])
    ap.add_argument("--sweep", action="store_true",
                    help="try a range of thresholds and report the shape")
    ap.add_argument("--literal-spec", action="store_true",
                    help="use the subtracted ask form from the brief")
    ap.add_argument("--csv", default="")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N packets (live, or a quick look)")
    a = ap.parse_args()

    ask_sign = -1 if a.literal_spec else +1
    if a.literal_spec:
        print("WARNING: --literal-spec negates the ask side. Verified against "
              "the live engine, that form agrees 223 times in 200,000.\n")

    targets = list(SWEEP) if a.sweep else [a.target]
    engines = {t: OrganicOfiCandleEngine(t, ask_sign) for t in targets}

    if a.live:
        source = iter_live(a.symbol, a.dataset, a.schema, CONFIG["API_KEY"])
        print(f"live {a.symbol} on {a.dataset}/{a.schema}, "
              f"target {a.target:,} contracts. Ctrl-C to stop.")
    else:
        paths = [f for f in a.files if not f.endswith(".ohlcv.dbn.zst")]
        if not paths:
            print("no input files (and --live not given)")
            return 2

        def chained():
            for p in paths:
                print(f"reading {p} ...", flush=True)
                yield from iter_file(p)
        source = chained()

    n = 0
    first_ts = last_ts = 0
    recorded_ns = 0
    started = time.time()
    try:
        for ts, bpx, bsz, apx, asz in source:
            for e in engines.values():
                bar = e.process_l1_packet(bpx, bsz, apx, asz, ts)
                if bar is not None and a.live and len(engines) == 1:
                    print(f"  {bar['Directional_Color']:<14}"
                          f" O {bar['Open']:.2f} H {bar['High']:.2f}"
                          f" L {bar['Low']:.2f} C {bar['Close']:.2f}"
                          f"  OFI {bar['Total_OFI']:+,}"
                          f"  {bar['Packets']} packets", flush=True)
            if first_ts == 0:
                first_ts = ts
            elif ts > last_ts:
                gap = ts - last_ts
                if gap < SESSION_GAP_NS:
                    recorded_ns += gap
            last_ts = ts
            n += 1
            if a.limit and n >= a.limit:
                break
            if n % 2_000_000 == 0:
                print(f"  {n:>12,} packets  "
                      f"{n / (time.time() - started) / 1000:.0f}k/s", flush=True)
    except KeyboardInterrupt:
        print("\nstopped")

    for e in engines.values():
        e.flush(last_ts)

    elapsed = max(1e-9, time.time() - started)
    # RECORDED book, not the calendar span between the first and last
    # record. Three files from different weeks span a fortnight and
    # contain four days of trading.
    hours = recorded_ns / 3.6e12
    span_hours = (last_ts - first_ts) / 3.6e12 if last_ts > first_ts else 0.0
    one = engines[targets[0]]

    print()
    print("=" * 74)
    print("ORGANIC OFI CANDLES")
    print("=" * 74)
    print(f"  {n:,} packets   {hours:.1f} hours of recorded book"
          f"   ({span_hours / 24:.1f} days end to end)")
    print(f"  {one.skipped:,} skipped (crossed, locked or absent)")
    print(f"  {n / elapsed / 1000:.0f}k packets/sec  "
          f"= {elapsed / max(1, n) * 1e6:.2f} microseconds each, measured")
    print(f"  ask sign: {'published / CKS' if ask_sign > 0 else 'LITERAL SPEC'}")
    print()
    print(f"  {'target':>7}{'bars':>9}{'per hour':>11}{'green':>8}"
          f"{'red':>8}{'med ticks':>11}{'packets':>10}")
    for t in targets:
        print(describe(engines[t].emit_candle_dataframe(), t, hours))
    print()
    print("  Pick the row that looks like a chart you would read. A median")
    print("  range of 0 ticks means the threshold sits below this book's")
    print("  noise floor and the bars have no body. A handful an hour means")
    print("  it is above anything that happens.")
    print()
    print("  BARS SCALE AS 1/TARGET^2. The accumulator is a random walk and")
    print("  first passage to +/-L goes as L squared, so halving the target")
    print("  gives roughly four times the bars. One row predicts the rest --")
    print("  no need to re-run the whole file to try another number.")
    print()
    print("  Green and red should sit near even. A long way off means the")
    print("  accumulator is latching or the ask sign is inverted -- both")
    print("  make every bar wrong in the same direction.")

    if a.csv and not a.sweep:
        df = one.emit_candle_dataframe()
        df.to_csv(a.csv, index=False)
        print(f"\n  {len(df):,} bars -> {a.csv}")
        print(f"  columns: {', '.join(df.columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
