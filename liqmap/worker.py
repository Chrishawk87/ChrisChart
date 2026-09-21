"""Standalone collector, for running the worker as its own Railway service.

Same loop the web service can run on a background thread, without the HTTP
layer. Use this when you want collection isolated from the dashboard -- a
redeploy of one then doesn't interrupt the other, and a hung sweep can't make
the dashboard look down.

If you run this, set LIQMAP_AUTO=false on the web service. Two processes
sweeping the same wallets share one rate-limit budget and will throttle each
other into half-built maps.
"""

from __future__ import annotations

import sys
import time

from .web import Runtime, start_worker, startup_report


def main() -> int:
    rt = Runtime()
    for line in startup_report(rt):
        print(f"[worker] {line}", flush=True)

    cfg = rt.settings()
    if not cfg.auto_run:
        print("[worker] auto_run is false. Set LIQMAP_AUTO=true or flip it in "
              "the dashboard; this process will idle until then.", flush=True)

    start_worker(rt)
    print(f"[worker] collecting {cfg.coins} "
          f"every {cfg.sweep_interval_minutes:.0f} min", flush=True)

    try:
        while True:
            time.sleep(60)
            if rt.last_error:
                print(f"[worker] last error: {rt.last_error}", flush=True)
                rt.last_error = None
    except KeyboardInterrupt:
        print("[worker] stopped.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
