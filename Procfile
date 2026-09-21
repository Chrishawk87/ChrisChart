# Railway reads this to start the service. $PORT is injected by Railway and
# health-checked; binding anything else marks the deploy crashed.
web: uvicorn liqmap.web:app --host 0.0.0.0 --port $PORT

# Optional: run collection as its own Railway service instead of on the web
# service's background thread. Deploy the same repo a second time and set the
# start command to this. Then set LIQMAP_AUTO=false on the web service so the
# two don't both sweep and fight over the rate limit.
worker: python -m liqmap.worker
