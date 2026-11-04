#!/usr/bin/env python3
"""Which build is running, and how far behind the chart is.

TWO QUESTIONS, ONE COMMAND:

    1. Is the code serving this page the build you pushed? A cached
       HTML page looks EXACTLY like a deploy that did not land --
       green deploy, numbers moving, and a control that was added in
       the new build simply absent.

    2. Is the tape live, and how old is what you are looking at?

    BASE=https://chrischart-production.up.railway.app TOKEN=... \\
        python scripts/chart_why.py ES

Reads only. Nothing here places, sizes or times an order.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


def fetch(base: str, token: str, path: str, **params):
    """Returns (payload, headers). Payload is a dict or raw text."""
    url = base.rstrip("/") + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    # A diagnostic that gets served a cached answer is worse than no
    # diagnostic, because it reports the cache as if it were the app.
    req.add_header("Cache-Control", "no-cache")
    req.add_header("Pragma", "no-cache")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = r.read().decode("utf-8", "replace")
            hdrs = {k.lower(): v for k, v in r.headers.items()}
            try:
                return json.loads(body), hdrs
            except ValueError:
                return body, hdrs
    except urllib.error.HTTPError as e:
        return {"_http": e.code,
                "_body": e.read().decode("utf-8", "replace")[:200]}, {}
    except Exception as e:                                  # noqa: BLE001
        return {"_err": f"{type(e).__name__}: {e}"}, {}


def tick(ok) -> str:
    return "OK  " if ok else "STOP"


def age(seconds) -> str:
    if seconds is None:
        return "unknown"
    s = float(seconds)
    if s < 90:
        return f"{s:.0f}s"
    if s < 5400:
        return f"{s / 60:.1f} min"
    return f"{s / 3600:.1f} hours"


def main() -> int:
    base = os.environ.get("BASE", "http://127.0.0.1:8000")
    token = os.environ.get("TOKEN", os.environ.get("LIQMAP_TOKEN", ""))
    coin = sys.argv[1] if len(sys.argv) > 1 else "ES"

    print(f"\n{'=' * 72}")
    print(f"CHART WHY   {coin}   {base}")
    print("=" * 72)

    # ---- 1. which build is actually serving ---------------------------
    print("\n  WHICH BUILD IS RUNNING")
    print(f"  {'-' * 68}")

    page, hdrs = fetch(base, token, "/")
    if isinstance(page, dict):
        print(f"  could not load the page: "
              f"{page.get('_err') or page.get('_http')}")
        print("\n  Set BASE to where it runs and TOKEN to LIQMAP_TOKEN.")
        return 2

    has_ticks = "by prints" in page
    print(f"  {'the page has the tick sizes':<34}{tick(has_ticks)}  "
          f"{'yes' if has_ticks else 'NO -- this is the old page'}")

    cc = hdrs.get("cache-control", "")
    cached = ("no-store" not in cc)
    print(f"  {'the page may be cached':<34}{tick(not cached)}  "
          f"{cc or 'no Cache-Control header at all'}")
    for h in ("age", "x-cache", "cf-cache-status"):
        if hdrs.get(h):
            print(f"  {'   ' + h:<34}{'':4}{hdrs[h]}")

    # The ROUTE is the real test. The page can be cached while the
    # code behind it is new; if this answers, the build landed and
    # anything missing on screen is the browser's copy.
    ch, _ = fetch(base, token, "/api/chart", coin=coin, interval="1000t")
    served = isinstance(ch, dict) and ch.get("source") == "tick"
    known = isinstance(ch, dict) and "_http" not in ch and "_err" not in ch
    print(f"  {'the CODE knows tick bars':<34}{tick(known and (served or bool(ch.get('error'))))}  "
          + ("yes -- /api/chart answered as a tick series" if served
             else f"yes, but: {ch.get('error')}" if known and ch.get("error")
             else "NO -- the route does not know '1000t'"))

    if known and (served or ch.get("error")) and not has_ticks:
        print("\n  >> THE BUILD LANDED AND YOUR BROWSER IS SHOWING AN OLD")
        print("     COPY OF THE PAGE. Hard-reload it:")
        print("       Mac Chrome/Edge:  Cmd-Shift-R")
        print("       Mac Safari:       Cmd-Option-R")
        print("     If that does not do it, open it in a private window.")

    # ---- 2. how old is what you are looking at ------------------------
    print("\n  HOW LIVE THE TAPE IS")
    print(f"  {'-' * 68}")

    ob, _ = fetch(base, token, "/api/ofibars", coin=coin)
    if not isinstance(ob, dict) or "_http" in ob or "_err" in ob:
        print(f"  could not read /api/ofibars: "
              f"{ob.get('_err') or ob.get('_http') if isinstance(ob, dict) else ob}")
        return 1

    fd = ob.get("feed") or {}
    run = bool(fd.get("running"))
    print(f"  {'the ingest thread is running':<34}{tick(run)}  "
          f"{'running' if run else 'STOPPED'}")
    if fd.get("error"):
        print(f"  {'   it last said':<34}{'':4}{fd['error']}")
    if fd.get("restarts"):
        print(f"  {'   restarts':<34}{'':4}{fd['restarts']}"
              f"{'  retry in ' + str(fd['retry_in']) if fd.get('retry_in') else ''}")

    bar_age = fd.get("last_bar_age_s")
    fresh = bar_age is not None and float(bar_age) < 120
    print(f"  {'newest bar':<34}{tick(fresh)}  {age(bar_age)} old")

    prints = ob.get("trades")
    print(f"  {'prints seen since start':<34}{'':4}{prints:,}"
          if isinstance(prints, int) else
          f"  {'prints seen since start':<34}{'':4}{prints}")

    # The chart route's own view, which is what the panel draws.
    cm, _ = fetch(base, token, "/api/chart", coin=coin, interval="15m",
                  bars=3)
    if isinstance(cm, dict) and cm.get("bars"):
        last = cm["bars"][-1]
        gap = time.time() - float(last.get("ts") or 0)
        print(f"  {'newest 15m candle on the chart':<34}"
              f"{tick(gap < 1200)}  started {age(gap)} ago"
              f"{'  (forming)' if last.get('live') else '  (CLOSED --'
               ' nothing is building)'}")
    elif isinstance(cm, dict) and cm.get("error"):
        print(f"  {'the 15m chart says':<34}STOP  {cm['error']}")

    # ---- the verdict ---------------------------------------------------
    print(f"\n  {'-' * 68}")
    if not run:
        print("  THE TAPE IS NOT RUNNING. Everything on the chart is the")
        print("  last thing it saw before it stopped. Check the Databento")
        print("  key and the Railway logs; the error above, if any, is")
        print("  what the service itself said.")
    elif bar_age is not None and float(bar_age) > 900:
        print(f"  THE THREAD IS ALIVE BUT HAS PRODUCED NOTHING FOR"
              f" {age(bar_age)}.")
        print("  A running thread that has stopped producing is still a")
        print("  stopped chart. Either the session is closed, or the")
        print("  subscription is not delivering.")
    elif not has_ticks and known:
        print("  THE SERVICE IS FINE AND YOUR PAGE IS STALE. Hard-reload.")
    elif not known:
        print("  THE OLD BUILD IS STILL SERVING. The tick route is not")
        print("  there, so the deploy did not replace the running code.")
        print("  Check that Railway finished, and that it built the")
        print("  commit you pushed.")
    else:
        print("  Build is current and the tape is live.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
