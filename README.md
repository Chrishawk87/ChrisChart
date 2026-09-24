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

## The call — a trade to take or leave

`suggest.py` turns the book read into a sentence you can act on or ignore:

```
LONG ETH 84 ticks (+11bps) from 7,736.3 — target 7,744.7,
  invalid below 7,731.8. 1.87R.
Why: leaning to the offer — thinner above; the offer is being eaten;
  57% bid heavy.
Cost: round trip 4.9bps — the target pays 2.2x.
Conviction 64% from 21 book updates, spread 0.26bps. 4m41s left on the 15m.
```

**It suggests. It never places anything.** There is no order-placement code in
this repository and there is not meant to be.

Every number traces to something observed:

| number | where it comes from |
|---|---|
| direction | microprice tilt, replenishment, depletion, absorption |
| target | median high-low range of recent bars on this timeframe, scaled by `sqrt(time left)` and halved |
| invalidation | the largest resting level behind the touch, floored at 15% of a typical bar's range |
| cost | walking the actual book both ways at your size, plus your fee |

**Refusing is the main output.** Most polls return `No trade` naming the gate
that stopped it — flat book, thin conviction, a target that does not clear the
round trip twice over, a stop in the wrong place, one-sided or locked book, a
size bigger than the displayed depth. A tool that finds a trade every time it
is asked is generating sentences, not reading a book.

### Take it or ignore it, and find out which of you is right

Every suggestion is stored, taken or ignored. That second column is the point:

- **The tool's edge** — how all the suggestions did.
- **Your edge** — how the ones you *took* did against the ones you *passed on*.

If the ignored ones win more often, the filter you are applying is costing you,
and there is no way to discover that without writing down the trades you
skipped. The comparison refuses to draw a conclusion under 20 of each, because
twelve trades is an impression and replacing an impression with a measurement
was the whole point.

P&L is reported **net of the round trip**, alongside gross. Gross says whether
the read was right; net says whether the trade made money. Only one of those
pays for anything.

## Book, delta, price — the vote, and the grid that sizes it

`liqmap/vote.py`, `liqmap/sweep.py`

### The decision is three columns added up

```
net = book + delta + price        signed by direction, sized by strength

net > 0  ->  long
net < 0  ->  short
net == 0 ->  nothing
```

That is the whole entry rule. Two agreeing is a trade. One column with two
quiet ones is a trade. Two outvoting one is a trade in the direction of the
two. None of that is decided in code — it falls out of the arithmetic, and
the vote *shape* (`3-0`, `2-1`, `2-0`, `1-1`) is recorded so the scorecard
can tell you afterwards which shapes actually paid.

The only thing producing no trade is all three flat, and that is the absence
of a reading rather than a veto: there is no direction to be long or short
of.

This deliberately removes the gates in `suggest.py` — conviction floors,
breakeven ceilings, cost multiples, agreement age. Each was a guess about
what a good trade looks like, applied before any evidence existed. The
problem is not that they are wrong; it is that a filtered sample cannot
measure its own filter, because the trades it refused have no outcome. The
gated path still exists (`Autopilot(raw=False)`) if you want to compare.

### The grid

A signal does not know where its target is — it is a direction at a price at
a time, identical whether the target is four ticks away or forty. So the
expensive part runs once and scoring a hundred target/stop pairs is a
hundred cheap walks over the same bars.

Set the boxes, hit **Run the grid**. Every take-profit against every stop,
over the candle readings the service already stored, net of cost. Read the
*shape* of the surface rather than the best square:

- a **solid region** of positive cells is a real preference
- a **lone bright square** is the maximum of many noisy numbers, which
  searching alone biases upward — the verdict line says so when the best
  cell's interval straddles zero
- the **share of the grid that is positive** is in the verdict, because 91%
  positive and 9% positive mean very different things about the same best
  cell

Two settlement rules decide whether any of it is true:

- **A bar touching both levels is a stop.** Nothing in OHLC orders the two
  touches. Taking the good one flatters the tightest stops most — exactly
  the cells that would then look best. One-minute bars are used rather than
  the signal's timeframe so the ambiguous window is a minute, not fifteen.
- **Trades that never reach a level are marked out at the close, not
  dropped.** The unresolved ones are disproportionately the ones that went
  nowhere; discarding them is the other classic way to manufacture an edge.

### The live agent runs the same strategy

The levels you set on the agent panel are the ones the grid tests, and in
raw mode the agent exits on those levels only. Invalidation exits are off by
default for one reason: the grid cannot replay an invalidation from bar
data, so leaving them on would make the live book and the backtest two
different strategies and the grid decorative. There is a checkbox to turn
them back on when you want to compare.

There is a test that settles the same bar through both paths and fails if
they disagree.

### One screen at a time

The page had grown to seventeen panels in a single column, which meant the
three you actually watch were separated by a screenful of things you were
not. They are now five tabs, and the tab you were on is remembered:

- **Dashboard** — the call, the agent's own book, the live candle read
- **Testing** — the target/stop grid, the backtest, the agreement table
- **Book & liquidity** — the book call, liquidity, consensus
- **Wallets** — sweeps, positions, the leverage map
- **Settings** — settings and the validation report

One trap worth recording: a pane carrying `.grid{display:grid}` stays on
screen even with the `hidden` attribute set, because an author rule beats
the browser's own `[hidden]{display:none}`. Every tab reported itself
hidden and every panel was still drawn. `.tabpane[hidden]{display:none}`
states it explicitly, and a test pins it.

### A trade never outlives its candle

A five minute trade lasts five minutes. The position was opened on one
candle's book, delta and price; once that candle closes, the reading it
rests on no longer exists and the trade is riding an expired signal.

It is also what keeps the grid honest. Every candle is an independent
test, so a trade allowed to run into the next one is claiming a result the
*next* candle's signal should have had to earn.

Enforced in both places, which is the part that matters:

- the live agent closes at its candle's end, reason `candle_end`
- `sweep.resolve` takes a `deadline` and stops walking bars there

A signal that fires forty seconds before the close gets forty seconds —
the deadline comes from the candle it opened on, not from a fresh bar
measured at entry. The `max_hold_bars` knob is gone; the candle is the
maximum hold and a second answer to that question is just a way to
disagree with itself.

### What the chart shows

### What the chart shows

Markers are the **agent's own trades**, drawn **on the candle** the way
TradingView does it — just clear of the bar's low going long, its high
going short. Not at the entry price: at the price the marker lands on the
body and hides the bar you are trying to verify, and the point of a
per-candle mark is that the candle underneath stays readable.

    L   went long, under the bar
    S   went short, over the bar
    ●   the exit, on ITS own candle, labelled with the net bps

The exit used to be labelled W or L, which collided with L for long. It
carries the number instead, which is less ambiguous and more useful.

Hovering either end gives the whole trade in the corner readout: side, why
it was taken in words, the levels, whether it won, how it ended, and net
after cost.

### The rest of the chart

Built the way a trading chart is built, because that is what it is for:

- **OHLC pinned top-left**, following the crosshair. A floating tooltip
  covers the thing it describes and moves while you read it; in the corner
  it is always in the same place.
- **Last price** as a dashed line carried to a filled label on the scale.
- **Crosshair labels on both scales** — price on the right, date and time
  underneath. A crosshair with a price but no time is half a crosshair.
- **Countdown on the forming bar.** On a chart where every trade dies with
  its candle, the time left is not decoration.
- **Volume in its own pane** under the price pane, coloured by bar, with
  its own scale. This replaced the side-gutter profile, which competed for
  space with the live bar.
- **Drag the price scale** to squash or stretch; `fit` or double-click
  resets.
- **The view holds position** when new bars arrive. Being yanked to the
  right edge every time a candle closes is the single most irritating
  thing a live chart can do.
- **Drawing tools** — horizontal lines and trendlines, kept per market and
  timeframe in your browser. They are a convenience for you; nothing the
  agent decides ever reads them.

One implementation note worth keeping: everything that converts a pixel to
a price reads `chartScale`, which `drawChart` sets. The first cut had
`priceAt()` recomputing the extremes for itself, and two copies of that
arithmetic drift — at which point a line you drew at one price is stored at
another. A test pins it.



Markers are the **agent's own trades**, not the old suggestion table.

They used to come from the suggestions you took or ignored, which after the
agent started keeping its own book meant every marker on the chart read
`pending` forever — nothing decides those rows any more.

Each trade now draws entry to exit: a triangle where it went in, a line to
where it came out (length = how long it held), and a dot coloured by **won
or lost** rather than by which level it was — hitting the target is not the
question, keeping money is. A hollow triangle is still open. Hovering
either end gives the whole trade:

```
SHORT @ 3,469.23    2-1
why: book down, delta down, price up
target 30bps · stop 20bps
WIN — target  +28.0bps net
out at 3,458.82
held 45m   best +30.0bps
```

### The missing mark, and what one bad exit did

Worth recording because the symptom looked nothing like the cause.

`suggest` only sets `entry` on candles it is willing to trade. The agent no
longer cares what `suggest` thinks — it votes — but it was still reading
its price out of that field, so on every refused candle it saw **zero**.

Two things followed:

- It stood aside with `no_read` on 37% of polls while holding a perfectly
  good vote.
- An invalidation exit closed **at zero**, which is a 10,000 basis point
  move. One of those in a book of five put `+2494bps a trade` and an
  `average R of 249.75` on the scorecard.

Fixed in three places, because one was not enough: the payload now always
carries a mark (book mid, else the forming bar's close, else the last
bar's); the agent holds rather than closing when it has no usable price;
and the ledger refuses to write a closed trade at a non-positive price at
all. A bad row that reaches the database hides inside an average where it
is very hard to find again.

There is a **Clear the book** button for exactly this, and the scorecard
now counts trades whose result cannot be true and offers to drop them.

### Tuning, once there are no gates left

The proposal engine was built for the gated agent: tighten a threshold,
walk it forward, ask before adopting. In raw mode that has nothing to say,
because the vote refuses nothing — there is no threshold to tighten.

So it now tunes the thing that is actually left: **where the levels sit**,
using the grid. Same discipline, different parameter. It picks a target and
stop on the older 60% of your stored signals, scores that pair on the newer
40% it has never seen, and proposes only when the out-of-sample interval is
entirely above zero and it beats what is running by at least 1bps a trade.

The pair moves together. A target that measured well beside a 25bps stop is
not evidence for that target beside a 10bps one, so adopting half of a
tested pair is refused — it would produce a setting nobody tested.

The ledger cannot judge levels at all, and this is worth being clear about:
changing a target changes what happens *during* a trade, not which trades
were taken, and an outcome cannot be replayed into a different outcome.
Only the grid can, because it walks real bars.

### Routes

```
GET /api/sweep?coin=ETH&tp_from=5&tp_to=40&tp_step=5&sl_from=5&sl_to=40&sl_step=5
GET /api/signals?coin=ETH&interval=15m     what the columns have been saying
```

## The agent's own book — it decides, logs, and marks itself

`liqmap/autopilot.py`, `liqmap/ledger.py`, `liqmap/score.py`, `liqmap/tuner.py`

The call panel answers a question when asked. This makes the agent commit.
Every poll it says one of five things — long, short, hold, exit, stand aside
— writes it down with the reading behind it, manages what it opened, and
closes it for a stated reason.

**It still does not place orders.** Nothing in these four files can. There
is a test that greps the compiled source (comments and docstrings stripped)
for order-placement vocabulary and fails if any of it appears.

### Why a separate book from yours

`history.py` already records suggestions and what you did with them. That
measures the two of you together, because the sample is filtered by your
judgement — and your judgement is what the agent is being compared against,
not blended into. So it keeps its own. When the two disagree about a market,
that is the interesting row.

### The exit rule

It exits on **invalidation**: the trade was taken because the book was thin
one way and price was going that way, so when that stops being true the
trade is over, stop or no stop. Three guards stop that being trigger-happy:

- **Flat is not against you.** A read going quiet is the absence of a
  signal. Only an *opposite* read counts.
- **It has to persist.** The opposite read must hold continuously for
  `invalidate_s`. One poll does not close a position, and the timer resets
  the moment it stops being against.
- **Quiet is its own exit.** A position whose read went flat and stayed
  flat is not invalidated, it is dead money. Closed as `time`, because "I
  was wrong" and "nothing happened" have different fixes.

Levels are checked against the bar's range, not the last print — a stop
reached between two polls was still reached. A bar touching both levels
counts as a **stop**; nothing in OHLC says which came first, and taking the
good one manufactures a hit rate money will not reproduce.

### The scorecard, and the bug it shipped with

The first version reported a book losing 2bps a trade as "paying for
itself", because its hit rate cleared the rate its entries needed. Both
numbers were right; the inference was not.

Breakeven hit rate assumes every loser loses exactly the planned risk and
every winner makes exactly the target. That holds when the levels are the
only way out — and stops holding the moment an exit rule can close a trade
in between, which is most of this agent's trades.

So **expectancy decides**: net basis points per trade, with a confidence
interval on the mean. Hit-rate-against-breakeven is kept as a diagnostic of
*entry shape*, and when the two disagree the panel says so — "hits 51%
against the 42% its entries needed and still loses 13bps a trade; the
winners are not reaching the target the risk was sized against" is a
specific, fixable finding that neither number gives alone.

Every rate carries a **Wilson interval**. Eleven wins from twenty is 55%,
and the honest reading of eleven from twenty is "somewhere between 32% and
77%", which is compatible with a great strategy and with a coin. Nothing is
described as meaningful under 30 trades, or conclusive under 100.

### Self-improvement, and the three things that keep it honest

- **Walk forward.** A threshold is chosen on the older part of the book and
  scored on the newer part it has never seen.
- **Bounded.** Every knob has a hard range, a step, and a cap of two steps
  per adoption.
- **Ask first.** A proposal is a row with the evidence attached. Nothing
  changes until you adopt it, and adoption is recorded.

Two things it will **not** propose, and the reasons are not cosmetic:

- **Loosening anything.** The book has outcomes for trades it took, not for
  trades it refused. Arguing for a looser threshold from this data would
  rest on rows that do not exist.
- **Exit rules** (`invalidate_s`, `max_hold_bars`, `stale_s`). They change
  what happens *during* a trade; re-running them needs the price path, and
  the ledger stores outcomes. Tune those by hand.

A tightening-only tuner has its own failure mode — a ratchet that filters
until it never trades — so a proposal discarding more than half the book is
refused however good the survivors look.

### Routes

```
POST /api/autopilot?on=true&coin=ETH&interval=15m   start/stop
GET  /api/autopilot                                  state, open position, feed
POST /api/autopilot/step                             one decision now
POST /api/autopilot/close?id=...                     close by hand (logged `manual`)
GET  /api/scorecard                                  the marking, with error bars
GET  /api/ledger                                     every closed paper trade
GET  /api/proposals                                  changes it wants to make
POST /api/proposals/decide?id=...&adopt=true         adopt or reject
POST /api/proposals/revert?param=...                 back to the shipped default
POST /api/proposals/scan                             run the walk-forward now
```

## Uploading chart history

`POST /api/upload-history` takes a CSV or JSON OHLCV export — TradingView, an
exchange, anything with time/open/high/low/close. Columns are matched by name
in any order, delimiters and timestamp units are detected, and malformed rows
are **dropped and counted, never repaired**.

This extends the historical replay well past the few thousand bars the exchange
API serves. What a bar file does **not** contain is the book:

> A bar records four prices and a volume. It does not record what was resting
> at the touch, which side kept replacing size, or whether aggression moved the
> mid. Microprice tilt, replenishment, depletion and absorption are about 70%
> of the weight in the book call, and none of them are in an OHLCV export.

For the book half there is a better source — see below.

## Backfilling the book from Hyperliquid's own archive

Hyperliquid publishes real L2 snapshots, about twice a second, to a public
requester-pays bucket:

```
s3://hyperliquid-archive/market_data/{YYYYMMDD}/{H}/l2Book/{COIN}.lz4
```

That is *higher* resolution than the throttled public WebSocket currently
delivers, and it goes back years. `archive.py` reads it; `backfill.py` replays
those snapshots through the **same `BookReader` the live feed uses** and writes
settled rows into the agreement table, so a question that would take three
weeks of live recording is answered in an afternoon.

Two rules the implementation holds to:

**The two sides stay independent.** The book side comes from the archive. The
price-action side comes from one-minute candles, which the exchange builds from
*fills*. Taking price action from the snapshots' own mid would be the book
agreeing with itself — a superb-looking confirmation rate meaning nothing.
There is a test that parses `backfill.py` and fails if any argument to the
price-action reader is derived from a book.

**It costs real money.** Requester-pays charges every byte to *your* AWS
account. The job estimates before it starts, counts bytes as it runs, reports
them in dollars, and stops at a cap. Set `AWS_ACCESS_KEY_ID` and
`AWS_SECRET_ACCESS_KEY` before using it.

Only `l2Book` lives under `market_data` — no trades, no candles, no spot. The
uploads land roughly monthly with no completeness guarantee, so missing hours
are normal and are counted rather than treated as failures.

Every upload report and every replay result says this. Uploading three months
of bars and being told the model is now trained on three months would be a lie
by omission, and it is exactly the lie that would get acted on.

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
  archive.py       Hyperliquid's S3 book archive — real historical L2
  backfill.py      replays archived books into the agreement table
  confirm.py       book against price action; agreement is a hard gate
  suggest.py       book read -> a trade to take or leave, or a named refusal
  ingest.py        uploaded OHLCV -> candles, with what it can and cannot train
  bookread.py      the order book calls the candle; nothing else consulted
  live.py          websocket feed, live candles, absorption baseline
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
