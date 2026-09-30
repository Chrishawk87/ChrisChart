#!/usr/bin/env python3
"""Pull ES futures data from Databento, price first.

WHY THIS SCRIPT EXISTS RATHER THAN A ONE-LINER

Databento bills by uncompressed size, and MBO for ES is large -- gigabytes
per session. A mistyped date range or an accidental `mbo` where you meant
`mbp-10` can spend a month's budget in one call. So this never downloads
anything without first asking what it costs and waiting for you to say yes.

    python scripts/fetch_es.py --cost-only          # price it, spend nothing
    python scripts/fetch_es.py                      # price it, then ask

THE KEY NEVER GOES IN A FILE

Read from the environment, so it cannot be committed by accident:

    export DATABENTO_API_KEY='db-...'

Put that line in your shell profile, not in the repo. If you ever paste the
key somewhere public, rotate it at databento.com immediately -- the same
rule as LIQMAP_TOKEN.

WHAT TO BUY FIRST

`mbp-10` (ten levels of aggregated depth) is a small fraction of the size of
`mbo` and covers everything in Phase 1 except absorption and replenishment.
Buy weeks of it, find out whether the read has an edge at all, and only then
spend on narrow `mbo` windows for the order-by-order modules. If mbp-10
shows nothing, the mbo purchase is money saved.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Data lands beside the repo, not inside it -- DBN files are large and have
# no business in git history.
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data"

DATASET = "GLBX.MDP3"

# What each schema is for, so `--help` answers the question you actually have.
SCHEMAS = {
    "mbp-10": "10 levels of aggregated depth. Start here -- small, covers most of Phase 1.",
    "mbo": "Order-by-order with order IDs. Large. Needed for absorption/replenishment.",
    "tbbo": "Trades plus the book at the top. Tiny. Good for a first smoke test.",
    "trades": "Executions only, no book. Tiny.",
    "ohlcv-1m": "One-minute bars. Negligible size. Useful for key levels and context.",
    "ohlcv-1s": "One-second bars.",
}

# A dollar figure that should make you read the request again before saying
# yes. Not a limit -- a speed bump.
LOUD = 25.0


def _usable_end(exc: Exception, asked: str) -> str | None:
    """Pull a usable end timestamp out of a range refusal.

    Databento refuses an over-long window two ways, and both name a date in
    the message:

        data_end_after_available_end   past what the dataset holds
        dataset_unavailable_range      inside the window that also needs a
                                       live subscription

    The first quotes BOTH the limit and the end you asked for -- "has data
    up to '...23:20'. The end in the query ('...00:00') is after the
    available range." Taking the last match picks your own bad value and
    retries it unchanged, which is what the first version of this did. So
    keep only timestamps earlier than what was asked for, and take the
    latest of those.
    """
    found = [t.replace(" ", "T") for t in
             re.findall(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}", str(exc))]
    usable = sorted(t for t in found if t < asked)
    return usable[-1] if usable else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch ES futures data from Databento, cost shown first.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Schemas:\n" + "\n".join(
            f"  {k:<10} {v}" for k, v in SCHEMAS.items()),
    )
    p.add_argument("--schema", default="mbp-10", choices=list(SCHEMAS),
                   help="Data schema (default: mbp-10)")
    p.add_argument("--symbol", default="ES.c.0",
                   help="ES.c.0 is the continuous front month, which is what "
                        "you want for a scalping study -- it rolls for you. "
                        "Pass a raw contract like ESZ6 to pin one expiry.")
    p.add_argument("--start", default=None,
                   help="UTC start, e.g. 2026-09-22T13:30 (default: 5 "
                        "weekdays back at RTH open)")
    p.add_argument("--end", default=None,
                   help="UTC end (default: start + --days)")
    p.add_argument("--days", type=float, default=5.0,
                   help="Days from start when --end is omitted (default: 5)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help=f"Output directory (default: {DEFAULT_OUT})")
    p.add_argument("--cost-only", action="store_true",
                   help="Print the price and record count, then stop.")
    p.add_argument("--yes", action="store_true",
                   help="Skip the confirmation prompt. Think twice.")
    return p.parse_args(argv)


def default_window(days: float) -> tuple[str, str]:
    """Five weekdays back, at the RTH open.

    13:30 UTC is 9:30 Eastern during US daylight time. This is a starting
    point for a first pull, not a session-accurate boundary -- the key-level
    module handles real session edges, including the DST shift.
    """
    now = datetime.now(timezone.utc)
    start = (now - timedelta(days=days + 2)).replace(
        hour=13, minute=30, second=0, microsecond=0)
    while start.weekday() >= 5:               # land on a weekday
        start -= timedelta(days=1)
    return (start.strftime("%Y-%m-%dT%H:%M"),
            (start + timedelta(days=days)).strftime("%Y-%m-%dT%H:%M"))


def human_size(records: int, schema: str) -> str:
    """A rough size, flagged as rough.

    Databento's own cost figure is authoritative; this is only here so the
    record count means something to a human reading it.
    """
    per_record = {"mbo": 56, "mbp-10": 368, "tbbo": 72, "trades": 56,
                  "ohlcv-1m": 56, "ohlcv-1s": 56}.get(schema, 100)
    mb = records * per_record / 1_000_000
    return f"~{mb/1000:.1f} GB" if mb >= 1000 else f"~{mb:.0f} MB"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    key = os.environ.get("DATABENTO_API_KEY", "").strip()
    if not key:
        print("DATABENTO_API_KEY is not set.\n\n"
              "  export DATABENTO_API_KEY='db-...'\n\n"
              "Get the key from databento.com -> Settings -> API keys. Put "
              "the export in your shell profile, never in the repo.",
              file=sys.stderr)
        return 2

    try:
        import databento as db
    except ImportError:
        print("The databento client is not installed:\n\n"
              "  pip install databento\n", file=sys.stderr)
        return 2

    start, end = args.start, args.end
    if start is None:
        start, auto_end = default_window(args.days)
        end = end or auto_end
    elif end is None:
        begun = datetime.fromisoformat(start)
        end = (begun + timedelta(days=args.days)).strftime("%Y-%m-%dT%H:%M")

    # Continuous symbols ("ES.c.0") and raw contracts ("ESZ6") are different
    # symbologies, and passing the wrong stype_in returns an empty result
    # rather than an error -- which looks exactly like "no data for that day".
    stype = "continuous" if "." in args.symbol else "raw_symbol"

    client = db.Historical(key)
    request = dict(dataset=DATASET, symbols=[args.symbol],
                   schema=args.schema, stype_in=stype, start=start, end=end)

    print(f"  dataset   {DATASET}")
    print(f"  symbol    {args.symbol}  ({stype})")
    print(f"  schema    {args.schema}  -- {SCHEMAS[args.schema]}")
    print(f"  window    {start}  ->  {end}  (UTC)")
    print()

    def price(req):
        return (client.metadata.get_cost(**req),
                client.metadata.get_record_count(**req))

    try:
        cost, count = price(request)
    except Exception as exc:                  # noqa: BLE001
        # Two different end-of-range refusals, both of which name a usable
        # timestamp in the message and both of which are tedious to solve
        # by guessing days off the end:
        #
        #   data_end_after_available_end  -- past what the dataset holds
        #   dataset_unavailable_range     -- inside the window that needs a
        #                                    live subscription as well
        #
        # Rather than make you arithmetic your way backwards, take the date
        # the error itself suggests and re-price against it.
        safe = _usable_end(exc, end)
        if safe is None:
            print(f"Could not price the request: {exc}", file=sys.stderr)
            return 1
        print(f"  The end of that window is not available:\n"
              f"    {str(exc).splitlines()[0]}\n")
        print(f"  Re-pricing with end = {safe} instead.\n")
        request["end"] = safe
        end = safe
        try:
            cost, count = price(request)
        except Exception as exc2:             # noqa: BLE001
            print(f"Still refused: {exc2}", file=sys.stderr)
            return 1

    if count == 0:
        print("Zero records for that window.\n\n"
              "Usually one of: a weekend or holiday, a date before the "
              "contract existed, or a raw symbol passed where a continuous "
              "one was meant (ESZ6 vs ES.c.0).", file=sys.stderr)
        return 1

    print(f"  records   {count:,}  ({human_size(count, args.schema)} rough)")
    print(f"  cost      ${cost:,.2f}")
    if cost >= LOUD:
        print(f"\n  ** ${cost:,.2f} is a real amount. Re-read the window "
              f"above before continuing. **")
    print()

    if args.cost_only:
        print("Priced only -- nothing downloaded, nothing spent.")
        return 0

    if not args.yes:
        if input("Download? [y/N] ").strip().lower() not in ("y", "yes"):
            print("Cancelled. Nothing spent.")
            return 0

    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.symbol.replace('.', '_')}_{args.schema}_{start[:10]}"
    path = args.out / f"{stem}.dbn.zst"

    print(f"\nDownloading to {path} ...")

    def fetch(req):
        client.timeseries.get_range(**req).to_file(path)

    try:
        fetch(request)
    except Exception as exc:                  # noqa: BLE001
        # The SAME end-of-range refusal can appear here after pricing
        # succeeded, because `metadata.get_cost` does not enforce the
        # licence boundary and `timeseries.get_range` does. Guarding only
        # the pricing path -- which is what the first version of this fix
        # did -- gets you a clean estimate and then a failed download.
        safe = _usable_end(exc, request["end"])
        if safe is None:
            print(f"Download failed: {exc}", file=sys.stderr)
            return 1
        print(f"  That end is licensed differently:\n"
              f"    {str(exc).splitlines()[0]}\n")
        print(f"  Retrying with end = {safe}\n")
        request["end"] = safe
        try:
            fetch(request)
        except Exception as exc2:             # noqa: BLE001
            print(f"Download failed again: {exc2}", file=sys.stderr)
            return 1

    size = path.stat().st_size / 1_000_000
    print(f"Done. {size:,.1f} MB compressed at {path}")
    print("\nAttach that file in the chat (or a slice of it) so the parser "
          "can be checked against real records rather than my assumptions "
          "about the format.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
