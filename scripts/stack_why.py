#!/usr/bin/env python3
"""Why the stack under your timeframe is blank.

"no read on 15m, 5m, 1m -- the stack is incomplete" is the CONSEQUENCE,
not the cause. The cascade says it when no row reached it, and a row
reaches it only if it was MEASURED -- built from the bar this service
assembled out of fills. A row read from stored candles is present on
the panel and invisible to the stack.

So there are only a few places this can break, and they look identical
on the screen:

    1. the market picker is not on a CME contract this service
       recognises, so there is no tape feed for it at all;
    2. there is a feed, but no builder for that timeframe;
    3. there is a builder, but its bar is not SEEDED -- no fills have
       landed in it yet;
    4. the rows are measured and the cascade is rejecting them for a
       reason of its own.

This prints which one. It reads your running service and invents
nothing.

    BASE=http://127.0.0.1:8000 TOKEN=... python scripts/stack_why.py ES 15m

Nothing here places, sizes or times an order.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request


def get(base: str, token: str, path: str, **params) -> dict:
    url = base.rstrip("/") + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return {"_http": e.code, "_body": e.read().decode()[:400]}
    except Exception as e:                      # noqa: BLE001
        return {"_err": f"{type(e).__name__}: {e}"}


def main() -> int:
    base = os.environ.get("BASE", "http://127.0.0.1:8000")
    token = os.environ.get("TOKEN", os.environ.get("LIQMAP_TOKEN", ""))
    coin = sys.argv[1] if len(sys.argv) > 1 else "ES"
    interval = sys.argv[2] if len(sys.argv) > 2 else "15m"
    tfs = sys.argv[3] if len(sys.argv) > 3 else "15m,5m,1m"

    print(f"\n{'=' * 72}")
    print(f"STACK DIAGNOSIS   {coin} {interval}   {base}")
    print("=" * 72)

    d = get(base, token, "/api/read", coin=coin, interval=interval,
            timeframes=tfs)
    if "_err" in d or "_http" in d:
        print(f"  could not reach the service: "
              f"{d.get('_err') or d.get('_http')} {d.get('_body', '')}")
        print("\n  Set BASE to where it is actually running, and TOKEN to"
              " LIQMAP_TOKEN.")
        return 2

    print(f"  source reported: {d.get('source')!r}"
          f"   (anything but 'cme' means this is NOT the futures tape)")
    for k in ("error", "why", "pressure_error", "feed_interval_error"):
        if d.get(k):
            print(f"  {k}: {d[k]}")

    rows = d.get("timeframes") or []
    if not rows:
        print("\n  NO ROWS AT ALL. The ladder never built one, so the"
              " cascade had nothing")
        print("  to read. That is cause 1 or 2 above -- most often the"
              " market picker is")
        print("  on a symbol this service does not treat as CME.")
    else:
        print(f"\n  {'tf':>6}{'measured':>10}{'source':>9}{'seeded':>8}"
              f"{'forming':>9}{'trades':>8}{'bars':>7}{'age s':>8}")
        for r in rows:
            print(f"  {r.get('timeframe', ''):>6}"
                  f"{str(r.get('measured')):>10}"
                  f"{str(r.get('source')):>9}"
                  f"{str(r.get('seeded')):>8}"
                  f"{str(r.get('forming')):>9}"
                  f"{str(r.get('trades')):>8}"
                  f"{str(r.get('bars_used')):>7}"
                  f"{str(r.get('age_s')):>8}")

    meas = [r["timeframe"] for r in rows if r.get("measured")]
    poll = [r["timeframe"] for r in rows if not r.get("measured")]

    casc = d.get("cascade") or {}
    print(f"\n  cascade state: {casc.get('state')}"
          f"   why: {casc.get('why')}")

    print(f"\n  {'-' * 68}")
    if not rows:
        verdict = ("No rows. Check the market selector first: the OFI"
                   " panel and this one\n  must both be on the same CME"
                   " contract.")
    elif not meas:
        verdict = (f"Rows exist but NONE is measured ({', '.join(poll)}"
                   " came from stored\n  candles). The stack only reads"
                   " measured rows, so it sees nothing.\n  Look at the"
                   " seeded column: all False means no fills have landed"
                   " in\n  those bars, which is cause 3 -- the tape is"
                   " not feeding the builders.")
    elif poll:
        verdict = (f"Measured: {', '.join(meas)}. NOT measured:"
                   f" {', '.join(poll)}.\n  The stack needs every rung,"
                   " so the unmeasured ones are what blanks it.")
    else:
        verdict = ("Every row is measured, so the blank is the cascade's"
                   " own doing --\n  cause 4. Its `why` above is the"
                   " real reason, and it is a market\n  verdict rather"
                   " than a data failure.")
    print(f"  {verdict}")

    # WHICH SYMBOLS THIS SERVICE WILL TREAT AS CME. The ladder asks
    # `cme_feed(coin)` and gets None for anything outside this set, and
    # from there the whole route goes down the crypto path and no row is
    # ever built. A picker holding a symbol one character off this list
    # produces exactly the blank panel, with nothing on screen to say so.
    mk = get(base, token, "/api/markets")
    cme_rows = [r for r in (mk.get("markets") or [])
                if r.get("source") == "cme"]
    print("\n  symbols this service treats as CME:")
    if not cme_rows:
        print("    NONE. The OFI engine is off or misconfigured, so"
              " every symbol goes")
        print("    to the crypto path. That alone produces the blank"
              " stack.")
    else:
        for r in cme_rows:
            print(f"    {r.get('symbol')!r:>12}"
                  f"   active={r.get('active_contract')!r}"
                  f"   front={r.get('front_month')!r}"
                  f"   deferred={r.get('deferred')!r}"
                  f"   has_data={r.get('has_data')}")
        names = {str(r.get("symbol")) for r in cme_rows}
        names |= {str(r.get("front_month")) for r in cme_rows}
        names |= {str(r.get("deferred")) for r in cme_rows if r.get("deferred")}
        if coin.upper() not in {n.upper() for n in names}:
            print(f"\n    >> {coin!r} IS NOT ONE OF THEM. That is the"
                  " blank stack: the route")
            print("       asks for a CME feed by this name, gets None,"
                  " and falls through")
            print("       to the crypto venue. Pass one of the symbols"
                  " above instead, and")
            print("       check the market picker is on it.")

    f = get(base, token, "/api/ofibars", coin=coin, target=2000, bars=5)
    print(f"\n  for comparison, the footprint panel on the same symbol:")
    if f.get("ok"):
        feed = f.get("feed") or {}
        print(f"    ok, feed running={feed.get('running')}"
              f" newest bar {feed.get('last_bar_age_s')}s old")
        print("    -- so the tape IS alive and the problem is between it"
              " and the ladder")
    else:
        print(f"    not ok: {f.get('why') or f.get('_err')}")
        print("    -- the same symbol fails there too, so this is the"
              " feed, not the stack")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
