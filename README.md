# liqmap — liquidation clusters, and whether they mean anything

Builds a forced-selling map from Hyperliquid's published position data, then
runs the test that decides whether the map has any predictive value at all.

The map is the easy part. The test is the point.

---

## Why this exists

In futures you infer stop pools from structure — above the swing high, below
the prior day's low — and you never actually see them. On-chain perps publish
theirs. Hyperliquid's `clearinghouseState` returns, per open position,
`liquidationPx` alongside size, entry, leverage and notional. Bucket those
prices and you have the magnet map, with numbers instead of an educated guess.

And a liquidation is **mechanical**. It fires because margin ran out, not
because the trader was smart or stupid. That's why this is a better use of
leaderboard-style data than copying anyone: you don't need a skill judgment,
only a size and a price.

**The direction convention, which is the easiest thing to get backwards:**

- **Longs liquidate below spot** → forced **selling** → cascade fuel underneath
- **Shorts liquidate above spot** → forced **buying** → squeeze fuel overhead

A cluster isn't just a level. It has a side, and touching it produces flow in a
known direction.

## Why the validation is most of the code

Every liquidation-heatmap product stops at the picture. The picture is always
compelling, because clusters form near price (that's where leverage got put on)
and price moves around near price. So clusters get touched constantly, and that
proves nothing whatsoever.

The only honest question is whether price reaches cluster levels **more often
than an arbitrary level at the same distance would**. That baseline has a
closed form — the first-passage probability of a driftless random walk — and
everything here is built on it.

Three things get measured:

**Touch lift.** Observed touches over baseline-expected touches. 1.0 means the
map adds nothing.

**Does size matter?** Split clusters by notional and compare lift, each band
with its own confidence interval. If big and small clusters behave the same,
the notional weighting — the premise of the whole map — is doing no work.

**What happens after.** Conditional on a touch, does price accelerate through
(cascade) or turn around (absorption)? This is the only one that pays. A level
touched exactly as often as chance predicts, but which reliably produces a
violent move afterwards, is still tradeable.

## See it work first

```bash
pip install -r requirements.txt
python -m liqmap.cli demo
```

No API access needed. It generates two worlds — one where clusters are inert,
one where they genuinely pull price — and prints the full report for each, so
you know what a real null result looks like before you see your own.

Abridged output:

```
WORLD A: clusters are inert
  lift              1.04x  95% CI [0.94, 1.14]
  verdict           indistinguishable from chance
  Lift does not separate across size once the intervals are accounted for.

WORLD B: clusters genuinely pull price
  lift              1.34x  95% CI [1.23, 1.45]
  verdict           clusters ARE touched more than chance
  $720,012   367   0.515   0.227   2.27x  [2.02, 2.55]   <- size separates
```

If your real data reads like World A, the map is decoration. On most assets,
most of the time, it will.

## Position strength — and the inversion that matters

Everything you need is already in `clearinghouseState`: entry price,
liquidation price, unrealised PnL, cumulative funding, margin used, and
account value off `marginSummary`. One call, no extra data source.

`strength.py` scores four things per position:

- **survivability** — how far liquidation sits from spot in *sigma*, softened
  by commitment (margin used / account value), carry (funding annualised) and
  whether margin is isolated or cross
- **commitment** — what share of the account is on this one position
- **carry** — funding paid per year against position value. Someone receiving
  funding can wait forever; someone paying 40% a year is on a clock
- **maturity** — unrealised PnL as a share of position value

**The inversion is the useful part.** The same score means opposite things
depending on the question:

|  | to FOLLOW the trader | as LIQUIDATION FUEL |
|---|---|---|
| far from liquidation | strong, survives | inert, won't fire |
| near liquidation | fragile | **live fuel** |
| deep in profit | move already happened | inert, has buffer |
| deep in loss | under pressure | **live fuel** |
| receiving funding | free to hold | inert |
| paying heavy funding | on a clock | **live fuel** |

Following requires a judgment about skill that this data cannot support. Fuel
requires no judgment at all — a liquidation is mechanical. So the second
column is the defensible use, and `fragility()` is what feeds the map.

Worth stating plainly: **a position showing large unrealised profit is a worse
candidate to copy, not a better one.** The move that made that profit already
happened, and their entry is the part you can't have.

```bash
python -m liqmap.cli positions --coin BTC --sigma 0.0006
```

Prints the scored positions, a cohort summary, then **both maps side by side**
— raw notional and fragility-weighted. Levels that shrink between the two are
held by traders who can sit through the move. They are not fuel.

## Changes beat snapshots

A snapshot says a position exists. The delta between two sweeps says what the
trader *did*, which is the only thing in this data resembling intent.

`history.py` persists every sweep and diffs each new one against the previous,
because **DEFENDED cannot be observed any other way** — it is defined entirely
by a difference between two points in time.

- **DEFENDED** — size unchanged, liquidation moved *away*. They posted more
  margin. The strongest conviction signal available on-chain, and completely
  invisible in a live dashboard.
- **LIQUIDATED** — vanished with spot sitting on its liquidation price. The
  ground truth for whether a cluster ever meant anything.
- **WEAKENED** — liquidation drifted *closer*. Margin pulled, or cross-margin
  pressure from another position.
- **ADDED** / **REDUCED** — size moved. The PnL at the time tells you whether
  ADDED was conviction or averaging into a loser.
- **CLOSED** — gone, and not near liquidation. A decision, not a forced exit.

`conviction_flow()` tallies notional behind commitment (ADDED, DEFENDED)
against retreat (REDUCED, WEAKENED, CLOSED). **LIQUIDATED is deliberately kept
out of that tally** — a forced exit was not a decision, and counting it as one
misreads the tape in exactly the wrong direction.

HELD is counted but not stored. It's the overwhelming majority and carries
nothing; keeping it would bury the signal.

## Running it on Railway

**Step-by-step walkthrough with the exact commands: [DEPLOY.md](DEPLOY.md).**

```bash
python -m liqmap.cli demo        # still works locally, no deploy needed
uvicorn liqmap.web:app --reload  # local dashboard on :8000
```

To deploy: push this repo, create a Railway service from it, and set the
variables from `.env.example`. `railway.json` supplies the start command and
points the health check at `/health`; Nixpacks picks Python up from
`requirements.txt`.

**Two things will bite you if you skip them.**

**Mount a volume.** Railway rebuilds the container filesystem on every deploy.
Without a volume, SQLite goes with it — every sweep, every change event, every
resolved outcome, gone on your next push. Add a Volume in the service
settings, mount it at `/data`, and set `LIQMAP_DB=/data/liqmap.db`. The
service prints a loud warning in the deploy logs when it detects it is running
on ephemeral disk.

**Set `LIQMAP_TOKEN`.** Without it the service starts **locked**: `/health`
answers (Railway needs it) and every other route returns 503. That is
deliberate — a public URL carrying your position data and trading
configuration should not default to open. Generate one with
`openssl rand -hex 24`.

### Making changes without redeploying

Settings live in the database, not in a config file, so you can change sigma,
bucket width, horizon, sweep interval and the coin list from the dashboard and
they take effect immediately and survive redeploys. Bad values are **rejected
rather than stored** — a wrong sigma would silently corrupt every baseline
computed afterwards, and you would not notice until the validation report read
as nonsense.

Secrets stay in Railway variables and are not editable over HTTP.

### Endpoints

| route | what |
|---|---|
| `GET /` | dashboard |
| `GET /health` | open, for Railway's health check |
| `GET /api/status` | settings, history counts, warnings |
| `GET /api/map?coin=BTC` | both maps, raw and fragility-weighted |
| `GET /api/positions?coin=BTC` | scored positions |
| `GET /api/changes?coin=BTC&kind=DEFENDED` | change events + conviction flow |
| `GET /api/report` | the validation report |
| `POST /api/settings` | change settings |
| `POST /api/wallets` | set the wallet universe |
| `POST /api/sweep?coin=BTC` | sweep now |
| `POST /api/resolve` | score elapsed clusters |

Auth is a bearer header or `?token=`, the latter so a browser can open the
dashboard. `GET /api/docs` has the generated OpenAPI page.

### One service or two

The web service can run collection on a background thread — set
`LIQMAP_AUTO=true`. For anything long-running, deploy the repo a second time
with the start command `python -m liqmap.worker` and set `LIQMAP_AUTO=false`
on the web service. A redeploy of one then doesn't interrupt the other.

**Don't run both with auto enabled.** Two processes sweeping the same wallets
share one rate-limit budget and will throttle each other into half-built
maps.

## A note on Hyperdash

Their Terms of Use prohibit automated access — no crawlers, bots, scripts or
"similar or equivalent manual processes" — and restrict use to personal
viewing. There's no public developer API. So this package doesn't touch them.

It costs you nothing: every field listed above comes from Hyperliquid's own
free API. Hyperdash's real value-add is the *leaderboard* — the wallet list —
and `harvest` builds that from the public trade feed instead. If you'd rather
buy a curated list than collect one, HyperTracker publishes a documented API
that covers leaderboards and liquidation data under terms that permit it.

## Running it live

```bash
python -m liqmap.cli check                       # is the API still shaped this way?
python -m liqmap.cli harvest --coins BTC,ETH --minutes 120
python -m liqmap.cli track --coins BTC --minutes 600 &
python -m liqmap.cli snapshot --coin BTC --sigma 0.0006
# ... wait out the horizon ...
python -m liqmap.cli resolve --horizon 240
python -m liqmap.cli report
```

**harvest** builds your wallet universe. There's no bulk endpoint and no
official leaderboard, but the public `trades` WebSocket carries
`users: [buyer, seller]` on every fill — so you collect your own off the tape,
filtered by size. It self-selects toward the participants whose forced exits
actually move price.

**track** must be running before and through every snapshot's horizon. Touches
are resolved from stored prices; no price history means no outcomes.

**snapshot** sweeps the universe and buckets the liquidation prices. At weight
2 per `clearinghouseState` call against a 1200/minute budget, that's **600
wallets a minute** — a 5,000-wallet universe takes about eight minutes. The
rate limiter blocks rather than firing and handling 429s, because a
half-finished map is worse than a slow one.

`--sigma` is required to store a snapshot, and deliberately so. The baseline
must be computed from volatility **as known at snapshot time**. Deriving it
later from the window being tested would leak the answer into the question and
make the map look predictive when it isn't.

## The statistics, and why they're fussy

**Observations are not independent.** Several clusters share one snapshot —
same price, same volatility, same moment — so a quiet window leaves them all
untouched together and a violent one sweeps several at once. Treating them as
independent shrinks the confidence interval to a fiction. Everything here uses
a block bootstrap that resamples whole snapshots. The test suite verifies this
produces a *wider* interval than the naive alternative; if it didn't, the
correction wouldn't be doing anything.

**Placebos are the real control.** Each snapshot also stores levels at the same
distance from spot where no cluster sits. The analytic baseline assumes a
random walk; the placebo assumes nothing. If real and placebo levels get
touched at the same rate, you have your answer regardless of whether the
model's assumptions hold.

**Distance-match everything.** Clusters form near price, so "clusters get
touched" is trivially true. Only the comparison against matched distance means
anything — and note that ±1% are *not* equidistant in log space, which is why
distances are handled in sigma throughout.

**Watch the distance breakdown for noise.** In the demo's null world one band
reads 1.23x purely by chance. Any real effect should be strongest near price
and fade with distance; a "signal" that peaks three sigma out is an artifact.

## Honest limitations

**Coverage is partial.** You see your harvested wallets, on one venue. Most
crypto leverage sits on Binance, Bybit and OKX, which publish nothing. This is
a sample of Hyperliquid's leverage, not the market's.

**`liquidationPx` moves.** It shifts with funding, added collateral, and — for
cross-margin positions — with everything else in that account. Snapshots go
stale in ways that aren't visible.

**People defend.** A trader watching their liquidation approach often adds
margin, and the cluster dissolves without price ever getting there.

**It's reflexive.** Liquidation maps are a popular product. Visible clusters get
front-run and also become self-fulfilling. You can't tell which in advance.

**Crypto only.** This works because on-chain perps publish positions. CME will
never publish a liquidation price, so none of this transfers to gold futures —
the methodology does, the data doesn't.

**The API layer is unverified.** `hl.py` is written to the published
documentation (endpoints, field names and rate-limit weights checked against
Hyperliquid's docs plus two independent mirrors), but it has never run against
the live API — the sandbox it was built in can't reach exchange endpoints.
Everything else is tested. Run `check` first and expect to fix something.

## Layout

```
liqmap/
  firstpassage.py  barrier-hit probability — the baseline clusters must beat
  bucket.py        positions -> liquidation map (raw or fragility-weighted)
  strength.py      survivability, carry, commitment, fragility, change detection
  history.py       sweep persistence + change events over time
  settings.py      runtime config, stored in the DB so it survives redeploys
  web.py           FastAPI dashboard, JSON API, background worker
  worker.py        standalone collector for a second Railway service
  hl.py            Hyperliquid REST + trade-feed harvester  (UNVERIFIED)
  store.py         snapshots, prices, outcomes, placebo generation
  validate.py      lift, size effect, post-touch behaviour, block bootstrap
  cli.py           commands
```

## Tests

```bash
python -m pytest tests/ -q
```

189 tests. The ones that matter:

- the closed-form hit probability is checked against **Monte Carlo simulation**
  across distances, horizons and both sides
- the reflection-principle identity (touch = 2 × terminal at zero drift) is
  asserted exactly
- **the harness is run on data generated with no effect at all**, and must
  report no effect — a validator that finds signal in noise is worse than none
- then on data with an injected effect, which it must find
- the block bootstrap is verified to produce wider intervals than treating
  correlated observations as independent
- the strength inversion is pinned down: strong positions must score as weak
  fuel, and unrealised profit must not quietly inflate survivability
- DEFENDED survives the round trip through storage, which is where a signal
  defined by a difference between two sweeps quietly gets lost
- every data route is verified to reject a missing or wrong token, the locked
  service is verified to refuse everything but `/health`, and `/health` is
  verified to leak no configuration

## Not financial advice

This is a measurement instrument. Its most likely useful output is telling you
that a thing you hoped would work does not.
