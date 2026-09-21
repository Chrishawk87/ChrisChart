# Deploying liqmap to Railway

Goal: a live URL with a working dashboard, and a one-command push to update it.

Two routes below. **CLI** is fastest to something visible. **GitHub** is what
you want for ongoing work, because then `git push` deploys. Do CLI first if you
just want to see it running; switch to GitHub when you start changing things.

---

## 0. Get it into git (both routes need this)

```bash
unzip liqmap.zip && cd liqmap

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest tests/ -q                           # 189 passing
uvicorn liqmap.web:app --reload                      # http://127.0.0.1:8000
```

The dashboard will load and tell you it has no data yet. That's correct — it
has nothing to show until you collect some. Stop it with Ctrl-C.

```bash
git init
git add .
git commit -m "liqmap: liquidation mapping, strength scoring, web service"
```

`.gitignore` already excludes `*.db`, `wallets.json` and `.env`, so your
database and token never enter the repo.

---

## Route A — Railway CLI (fastest to a live URL)

```bash
brew install railway          # macOS
# npm i -g @railway/cli       # anywhere with Node

railway login
railway init                  # name the project, e.g. liqmap
railway up --detach           # builds and deploys
railway domain                # prints your public URL
```

Railway detects Python from `requirements.txt` and reads the start command
from `railway.json`. First build takes a couple of minutes.

Now set the token and the database path:

```bash
railway variables --set "LIQMAP_TOKEN=$(openssl rand -hex 24)"
railway variables --set "LIQMAP_DB=/data/liqmap.db"
railway variables --set "LIQMAP_COINS=BTC,ETH"
railway variables --set "LIQMAP_AUTO=true"

railway variables            # print them; copy LIQMAP_TOKEN, you need it to log in
```

**Add the volume** — do this in the dashboard, it needs a mount path:
open the project → your service → **Variables/Settings → Volumes → New Volume**
→ mount path `/data`.

Or from the terminal:

```bash
railway volume add -m /data
```

Adding a volume restarts the service. Check it came up:

```bash
railway logs
```

You want to see `[liqmap] database: /data/liqmap.db` and **no** ephemeral-disk
warning.

---

## Route B — GitHub (so `git push` deploys)

Create an empty repo on GitHub, then:

```bash
git remote add origin git@github.com:<you>/liqmap.git
git branch -M main
git push -u origin main
```

In Railway: **New Project → Deploy from GitHub repo → liqmap**.

Then set the same variables (**Settings → Variables**), add the volume
(**Settings → Volumes**, mount at `/data`), and generate a URL
(**Settings → Networking → Generate Domain**).

From here every `git push` triggers a deploy:

```bash
git add -A
git commit -m "raise bucket width"
git push
```

Already deployed via CLI and want to switch? Connect the repo in
**Settings → Source** and Railway takes over from GitHub.

---

## 1. First data — the part that makes it visible

Open your URL. Paste your `LIQMAP_TOKEN` into the token box at the top.

The Status panel will say **"No wallet universe yet."** That's the real
starting state — there is no endpoint that lists Hyperliquid positions in
bulk, so you have to build your own list of addresses first.

Click **Harvest wallets** and give it **5 minutes**.

It subscribes to the public trade feed, collects the addresses on both sides of
every fill above your size threshold, and then sweeps them automatically. When
it finishes you'll have a populated map, a positions table, and a cohort
summary.

Then click **Sweep now** once more, a few minutes later.

That second sweep is what makes **DEFENDED** possible. It's defined entirely by
the difference between two sweeps — nothing in a single snapshot can show it.
With `LIQMAP_AUTO=true` the background worker keeps sweeping on its own every
30 minutes, so the change feed fills in over the following hours.

**Five minutes of harvesting is enough to see it work, not enough to trust it.**
Run an hour or more when you're ready for real numbers — the longer it listens,
the more of the large participants it catches, and those are the ones whose
liquidations actually move price.

---

## 2. Changing things

**Settings** — sigma, bucket width, horizon, sweep interval, coins — are
editable in the dashboard and take effect immediately. They live in the
database, so they survive redeploys. No push needed.

**Code changes** need a deploy:

```bash
# GitHub route
git add -A && git commit -m "…" && git push

# CLI route
railway up --detach
```

Useful while iterating:

```bash
railway logs            # runtime output
railway logs --build    # why a build failed
railway status          # what's linked
railway run python -m liqmap.cli demo   # run locally with Railway's env vars
```

---

## If it doesn't come up

**Deploy shows "crashed" but the build succeeded.** Almost always the port.
The app must bind `0.0.0.0:$PORT`; `railway.json` handles this, so check you
didn't override the start command in the service settings.

**Every route returns 503.** `LIQMAP_TOKEN` isn't set. The service starts
locked on purpose — a public URL carrying your position data shouldn't default
to open. Check `railway variables`.

**Data vanished after a deploy.** No volume mounted. This is the expensive one:
sweep history, change events and resolved outcomes are the only things here
that take time to accumulate. `railway logs` prints a loud warning on every
boot when it detects ephemeral disk.

**Harvest finds nothing.** `websocket-client` missing from the build (it's in
`requirements.txt`, so check the build log), or `min_wallet_notional` set too
high for a quiet hour. Try lowering it to 5000 in the dashboard.

**Sweeps are slow.** Expected. The rate limit is 1200 request-weight per minute
and `clearinghouseState` costs 2, so it's 600 wallets a minute. 600 wallets ≈ 1
minute, 5000 ≈ 8 minutes. The limiter blocks rather than getting throttled,
because a half-built map is worse than a slow one.

---

## Cost

Railway's Hobby plan is $5/month and this fits in it comfortably — one small
service, a small volume, no database add-on. Add a second service for the
worker only if you want collection isolated from the dashboard, and if you do,
set `LIQMAP_AUTO=false` on the web service so the two don't both sweep and
throttle each other.
