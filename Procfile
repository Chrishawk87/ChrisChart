# Railway reads this to start the service. $PORT is injected by Railway and
# health-checked; binding anything else marks the deploy crashed.
web: uvicorn liqmap.web:app --host 0.0.0.0 --port $PORT

# Optional: run collection as its own Railway service instead of on the web
# service's background thread. Deploy the same repo a second time and set the
# start command to this. Then set LIQMAP_AUTO=false on the web service so the
# two don't both sweep and fight over the rate limit.
worker: python -m liqmap.worker

# The EVM arbitrage bot. A THIRD Railway service off this same repo: set the
# root directory to `mev` and the start command to this. It shares nothing
# with the Python package -- different market, different process, its own
# variables. It compiles with tsc at BUILD time and runs plain node, because
# Nixpacks sets NODE_ENV=production and would not install tsx.
# Deploy it only after setting EXECUTOR_ADDRESS, EXECUTION_KEY and
# FLASHBOTS_AUTH_KEY, and leave DRY_RUN unset (it defaults to true) until you
# have watched a week of simulations.
mev: npm start
