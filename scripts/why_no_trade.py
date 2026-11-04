#!/usr/bin/env python3
"""Why the trade column is not opening anything.

EVERY GATE, IN THE ORDER THEY ARE CHECKED, with the live value beside
each one. The column prints its reason on the panel, but there are
several places a trade can be stopped and only the first one reached
is shown -- so this walks all of them and says which fired.

    BASE=https://your-app.up.railway.app TOKEN=... \\
        python scripts/why_no_trade.py ES

Reads only. Nothing here places, sizes or times an order.
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
        return {"_http": e.code, "_body": e.read().decode()[:300]}
    except Exception as e:                              # noqa: BLE001
        return {"_err": f"{type(e).__name__}: {e}"}


def tick(ok: bool) -> str:
    return "OK  " if ok else "STOP"


def main() -> int:
    base = os.environ.get("BASE", "http://127.0.0.1:8000")
    token = os.environ.get("TOKEN", os.environ.get("LIQMAP_TOKEN", ""))
    coin = sys.argv[1] if len(sys.argv) > 1 else "ES"
    interval = sys.argv[2] if len(sys.argv) > 2 else "15m"

    print(f"\n{'=' * 72}")
    print(f"WHY NO TRADE   {coin}   {base}")
    print("=" * 72)

    d = get(base, token, "/api/read", coin=coin, interval=interval,
            timeframes="15m,5m,1m")
    if "_err" in d or "_http" in d:
        print(f"  could not reach the service: "
              f"{d.get('_err') or d.get('_http')} {d.get('_body','')}")
        print("\n  Set BASE to where it runs and TOKEN to"
              " LIQMAP_TOKEN.")
        return 2

    j = d.get("journal")
    if j is None:
        print("  THE COLUMN IS NOT ON THIS BUILD. /api/read carries no"
              " `journal` key,")
        print("  which means the deploy predates it. Nothing below can"
              " be read.")
        print(f"  route error: {d.get('error') or d.get('pressure_error')}")
        return 2
    if j.get("error"):
        print(f"  the column itself raised: {j['error']}")
        return 2

    s = j.get("settings") or {}
    w = j.get("window") or {}
    openr = j.get("open")

    # ---- the gates, in order ------------------------------------------
    print(f"\n  {'gate':<34}{'':4}what it says now")
    print(f"  {'-' * 68}")

    auto = bool(s.get("auto"))
    print(f"  {'AUTO switch':<34}{tick(auto)}  "
          f"{'AUTO' if auto else 'MANUAL -- nothing opens by itself'}")

    px = float(j.get("price") or 0.0)
    print(f"  {'live price':<34}{tick(px > 0)}  "
          f"{px if px else 'no print has arrived'}")

    have_win = bool(w)
    print(f"  {'15-minute window exists':<34}{tick(have_win)}  "
          f"{'yes' if have_win else 'none -- the engine has not built one'}")

    if have_win:
        c, o, v = (w.get("mid_dir"), w.get("ofi_dir"), w.get("cvd_dir"))
        allthree = c in ("up", "down") and o == c and v == c
        print(f"  {'all three agree':<34}{tick(allthree)}  "
              f"candle {c} | OFI {o} ({w.get('net_ofi')}) | "
              f"CVD {v} ({w.get('net_cvd')})")
        print(f"  {'   window age':<34}{'':4}"
              f"{round((w.get('ts_last') or 0) - (w.get('start_ts') or 0))}s"
              f" in, {w.get('books')} book updates,"
              f" {w.get('trades')} prints")
        if w.get("unknown"):
            print(f"  {'   prints with no side':<34}{'':4}"
                  f"{w['unknown']} -- these are dropped from CVD"
                  " everywhere else")

    taken = j.get("taken_this_candle") or []
    cap = j.get("max_per_candle", 2)
    room = len(taken) < cap
    print(f"  {'room left in this window':<34}{tick(room)}  "
          f"{'taken: ' + ', '.join(taken) if taken else 'nothing taken yet'}"
          f" (cap {cap} per window, one per direction)")

    print(f"  {'nothing already open':<34}{tick(openr is None)}  "
          f"{'flat' if openr is None else 'a ' + openr['side'] + ' is open'}")

    # ---- the verdict ---------------------------------------------------
    print(f"\n  {'-' * 68}")
    blocked = j.get("blocked") or ""
    idle = j.get("idle") or ""
    agree = j.get("agree") or ""

    if openr is not None:
        print(f"  A TRADE IS OPEN: {openr['side']} from"
              f" {openr['open_px']}, {openr.get('live_ticks', 0)} ticks.")
        print("  Nothing else opens until it closes.")
    elif agree:
        print(f"  THE SIGNAL IS LIVE AND SAYS {agree.upper()}.")
        if idle:
            print(f"  It is not being taken because: {idle}")
        else:
            print("  Nothing is refusing it -- the next poll should"
                  " open it.")
    elif blocked:
        print(f"  NO SIGNAL. {blocked}")
    elif idle:
        print(f"  NO TRADE. {idle}")
    else:
        print("  No signal and no reason given, which is a bug in the"
              " reporting")
        print("  rather than a market state. Send this output on.")

    st = j.get("stats") or {}
    if st.get("trades"):
        v = st.get("verdict") or {}
        print(f"\n  So far: {st['trades']} trades, {st['wins']}W/"
              f"{st['losses']}L, {st['ticks']:+} ticks"
              f" (${st['dollars']:+,.2f}).")
        if v.get("say"):
            print(f"  {v['say']}")
    else:
        print("\n  No trades logged yet on this market.")

    print(f"\n  places_orders: {j.get('places_orders')}"
          "   (this records; it never sends an order)")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
