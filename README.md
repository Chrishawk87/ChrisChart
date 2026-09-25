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

### The rule it runs

Set on the panel, enforced in the agent, and testable on stored history
with the same two filters in the grid:

    ALL THREE AGREE      book, delta and price pointing the same way.
                         3-0 only: not two outvoting one, not one column
                         with two staying quiet.

    PAID FOR OVER 200%   aggressive volume at more than twice what this
                         market normally trades for the span. A direction
                         three columns agree on that nobody is trading
                         through is a reading, not a move.

    SMALL TARGETS        10 by default, in basis points or in the market's
                         own ticks — ten ticks on gold and ten ticks on a
                         cheap perp are different numbers of basis points,
                         so the conversion happens per market at entry
                         from the book's own increment. No tick size
                         available means it reads the number as bps rather
                         than inventing one.

These are gates, and they are back after the others were removed for one
reason: they are your rules with numbers on them, not my guess at what a
good trade looks like. The refusals are still written to the book with the
shape and the volume that was refused, so "what did unanimity cost me" is a
question the ledger can answer later rather than an article of faith.

A candle with no volume reading yet is **excluded**, not counted as quiet.
Absent is not the same as zero, and treating it as zero would refuse every
trade in the first seconds of a bar — and, in the grid, would flatter the
filter by scoring it against candles it never really saw.

Expect this to trade rarely. That is the point of it, but it means the
scorecard will say "too few to read" for a good while; the gate audit under
*What it refused* is the interesting panel in the meantime.

#### How a trade ends

In order of what normally happens:

1. **The target** — 5 or 10, in bps or in the market's own ticks. This is
   the plan and it is what takes most trades out.
2. **The stop**, if price goes the other way first. A bar that touches both
   inside one minute counts as the stop, in the live agent and in the grid
   alike — nothing in OHLC says which came first.
3. **The candle close** is the backstop, not the plan. Whatever has not
   resolved is out `exit_before_s` seconds before the bar prints (5 by
   default). Exiting exactly AT the close assumes a fill at the closing
   print, which is not a price anyone gets, and it leaves the position
   alive into the moment the next candle's reading starts forming.

The grid takes the same buffer, so a result there is a result for the
trade the agent actually takes.

#### What the arithmetic asks of you

Worth knowing before a week of data arrives, because the tool will report
it and it is better not to be surprised by it. With a symmetric target and
stop, breakeven hit rate is `(stop + cost) / (target + stop)`:

    10bps target, 10bps stop, 2bps round trip  ->  needs 60%
    5bps target,  5bps stop,  2bps round trip  ->  needs 70%

A winner nets +8 and a loser nets −12 on the first line. That is not an
argument against small targets — it is the number to watch, and it is the
`needed` column on every row of the scorecard. Two things move it: a lower
round-trip cost, or a target further than the stop. The grid is there to
find out which pairs actually clear it on your own candles.

#### One thing this exposed

Agreement is counted from the three **directions**, which is what the panel
shows and what you check by eye. It used to be counted from the weighted
sum — so a column reporting a direction whose strength rounded to zero
contributed nothing and was treated as flat. The screen would have said all
three agree while the agent stood aside on "only 2 of 3". Strength still
decides the *side* when columns disagree; direction decides who agrees.

### Trading the candles you already have

**Trade every past candle** on the agent panel. Every candle's book, delta
and price reading has been stored since the feed first ran, and the rule
is a function of those readings — so there is no reason to wait a week to
find out what it does.

Each stored candle that meets the rule is settled against real one-minute
bars, with the same one-candle deadline and the same pre-close exit the
live agent uses, and written into the same book marked `backfill`. A
candle already in the book is skipped, so running it twice cannot
double-count anything.

Marked, not blended. A backfilled trade is resolved at one-minute
resolution; a live one is managed poll by poll; a manual one was your
decision rather than the agent's. The scorecard slices by source, because
averaging the three gives a number that describes none of them.

### The alert

**Alert me the moment they agree** gives three channels, because one is
never enough: a banner while you are looking at the page, a two-note sound
while you are not (rising for a long, falling for a short), and a desktop
notification when the tab is behind something else.

It fires on the **edge** — the poll where agreement appears — keyed to the
candle. A rule that stays true for ten polls sounds once. A thing that
beeps every five seconds for a minute is a thing you turn off.

The banner carries the side, the columns, the volume, the price and your
levels, with **I took it** beside it. That records the trade as yours,
`source: manual`, kept out of the agent's own score: a trade you decided
on is not evidence about the agent, and letting it sit inside that score
would make it a measurement of the two of you together — the exact thing
the separate book exists to avoid.

#### A bug worth recording

The first version of this alert never ran at all. The page is one long
script and the candle-read panel already had a `checkAlert`; a second
function declaration with the same name silently **replaces** the first.
So the new one was dead code, the old call site kept firing, and it was
handed a payload with none of the fields it expected — it threw and took
the whole poll down with it. Nothing about that is visible by reading
either function on its own.

Every name in the agreement alert now carries an `agree` prefix, and two
tests fail the build on any duplicate top-level `function` or `const` in
the page.

### The book, as one file

### The book, as one file

**Download the book** on the agent panel gives you a single CSV: a short
summary block — trades closed, winners, losers, hit rate, net bps a trade,
total — then one row per trade with the outcome first and the conditions
that produced it beside it.

```
result | net_bps | exit_reason | side  | shape | book | delta | price | effort_pct
LOSS   |   -12.0 | stop        | short | 3-0   | down | down  | down  |      486.8
WIN    |    +8.0 | target      | short | 3-0   | down | down  | down  |      400.9
LOSS   |    -0.0 | candle_end  | long  | 3-0   | up   | up    | up    |      225.5
```

Sort by `result` and the winners and losers sit in two blocks, with the
effort, the shape, the levels and the hold time beside each one. That is
the file to adjust from.

There is a `source` column — `live`, `backfill` or `manual` — so the three
kinds of evidence can be separated in the spreadsheet too.

Positions still open are included and marked `open` rather than dropped.
Leaving them out of a file you are going to count rows in is how a book
quietly looks better than it is.

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

## The projected candle

`liqmap/project.py` — where the bar still forming is likely to close,
drawn one slot right of the live bar, with a band that narrows as the bar
fills. The `proj` button in the chart toolbar toggles it.

### The reframe

Nothing here predicts a candle. At minute seven of a fifteen minute bar
the open, the high so far, the low so far and the volume so far are all
known exactly — the only unknown is the remaining eight minutes. So the
question is not "what will this bar do", it is "given everything already
printed, where does the close land". Much smaller question, and its
uncertainty collapses as the bar fills.

It also dissolves the horizon problem. Order-book microstructure is worth
something over the next thirty seconds and very little over the next
fifteen minutes. A fifteen minute bar is thirty of those windows, so the
read is not stretched to cover the bar — it is applied to the time
actually left and integrated forward. **Microstructure does not become a
superpower by reaching further. It becomes one by being accumulated.**

### Two projections, always

Every projection is produced twice: once with the drift the signal
implies, once with no drift at all. The second is the null — same
volatility cone, no opinion about direction.

Scored against the bars that follow, the difference between them measures
directly whether the three columns carry any directional information. It
is far faster than trading: every bar closes, so every projection is
graded within minutes, and a few hundred bars arrive in a couple of days
rather than the weeks the same number of trades would take.

If the drifted projection is not better calibrated than the null, the read
carries nothing at that horizon, and no amount of position sizing rescues
it. That is a finding about the signal, not about the projection.

### Getting the answer today

**Test the signal now** on the call panel replays the projection over
every candle already recorded and grades it in one pass. The readings are
in `agreement_states`, the bar-so-far is reconstructable exactly from
one-minute bars, so weeks of labelled outcomes can be produced in minutes
instead of arriving one bar at a time.

Each stored candle is projected at four points through its own bar — 20%,
40%, 60% and 80% of the way in — because the useful question is not
whether it is calibrated at ninety percent of the way through, when
almost nothing is left to be wrong about, but whether it is calibrated
**early**, while there is still a trade in it.

### Two questions, two instruments

**Does it call direction?** A plain binomial on the sign: of the bars
where the read leaned, how many closed that way? Fifty percent is a coin,
and a Wilson interval says whether the gap from fifty is real. This is the
headline, because it is the question.

**Are the bands honest?** The PIT calibration below it.

Keeping these apart matters, and the first version did not. Grading only
the band shape and concluding from it is a mistake of instrument: a real
directional edge barely moves the PIT histogram, because a drift of a few
percent of a standard deviation leaves it almost exactly as flat as it
was. Run against synthetic data with a known ±0.30 sigma tilt, the band
gap could not tell the three cases apart — all three sat within ±0.003 of
each other — while the binomial separated them cleanly:

```
true tilt   direction            band gap
  +0.30     86.7% (82–90%)        +0.001
   0.00     52.3% (47–58%)        -0.001
  -0.30     13.3% (10–18%)        +0.003
```

An inverted signal is reported as inverted rather than as noise, because
predicting with the sign backwards is information and calling it "no
edge" throws it away.

### One bar, one observation

Ten projections on one candle are ten looks at the same outcome. Pooling
them multiplies the apparent sample by ten and shrinks every interval by
root-ten, which would let a few dozen bars masquerade as a finding. The
scoring keeps one row per bar — the earliest look, since the latest is
nearly free — and reports bars separately from rows.

### The drift bug worth recording

The signal's claim is expressed as a fraction of the **remaining
window's** standard deviation, not as a per-second drift:

```
expected move = k * squash(net) * sigma * sqrt(T)
```

A per-second drift accumulates linearly in T while the noise around it
grows only as sqrt(T), so the same coefficient claims more and more as the
horizon lengthens. The first version had `k = 0.15` per second, which on a
fifteen minute bar asserted a **3.4 sigma** move — and it read as "the
signal is actively hurting" on data where the signal was real, because a
wildly over-confident drift is worse than no drift whichever way it
points. Stated as a fraction of the window, `k` is dimensionless and means
the same thing at every horizon.

### How it is scored

Not by counting hits. By the **probability integral transform**: where the
actual close landed inside the predicted distribution. A calibrated model
spreads those evenly over [0, 1]; an over-confident one piles them at both
ends; one whose drift points the wrong way leans to one side.

It grades the *shape* of the forecast, not just its direction — so an
over-confident model is caught even on the bars it happens to get right.
The histogram is reported beside the score because its shape says what is
wrong: a U is over-confidence, a hump is vagueness, a lean is a drift with
the wrong sign.

Nothing is concluded under 20 graded bars, and the comparison against the
null needs both sides ready.

### What it will not do

A well-calibrated projection says "wide" most of the time, and that is the
correct answer. The value is in the minority of bars where the book is
genuinely one-sided, and in knowing which those are. A projection sharp
enough to trade blindly would have had its edge arbitraged away already.

## The Tick Counter PPO

`liqmap/ppo.py` — Chris's Pine indicator, translated and drawn in its own
pane under the volume pane. The `ppo` button in the chart toolbar toggles
it, and the corner readout carries `PPO / sig / h` beside the OHLC.

### What the original actually computes

Two lines of the Pine do less than they look like they do, and a faithful
translation has to match the behaviour rather than the apparent intent.

```
ad_formula  = ((2*close - low - high) / (high - low)) * volume
ad_current  = cum(ad_formula)
ad_previous = cum(ad_formula)[1]
tickCounter = ad_current - ad_previous
```

A running total minus the same running total one bar ago is just the
current bar's contribution, so **`tickCounter` is `ad_formula` for this
bar** — the cumulative cancels out entirely. The momentum half does the
same thing: a cumulative built and then differenced one line later. Both
are implemented as the per-bar value they reduce to, and a test pins the
identity so nobody later "fixes" the indicator by deleting the wrong half.

The rest is as written: each series divided by its own 50-bar population
standard deviation and clamped to ±3, combined 70/30, smoothed, with an
EMA signal line and their difference as the histogram.

### Three honest notes

- **`showArrows` does nothing.** The input is declared in the original and
  never read — there are no arrow plots in the script.
- **It is single-timeframe here.** The original can take A/D and momentum
  from different timeframes. This computes both from the chart's bars. On
  the 15m chart with the original's defaults (A/D blank = chart, momentum
  = 15) those are the same thing; on any other timeframe they are not.
- **The numbers will not match TradingView.** These are our bars, built
  from the venue's own fills with the venue's own volume. The shape should
  agree; the values are not comparable, and expecting them to be is how
  you conclude something is broken when nothing is.

### A property worth knowing

An unvarying series produces **no reading at all**, not a flat line at
zero. The normalisation divides by the window's own standard deviation, so
a market that has not moved makes the ratio undefined rather than infinite.
The pane draws a gap there, because a line pinned at zero would be a claim
about a market that has told us nothing.

### Making it bigger

Two independent ways, because they answer different questions:

- **Drag a pane divider.** Each divider carries a grab handle at its
  centre and the cursor changes to `ns-resize` over it. Dragging up makes
  the pane below it taller, which is the direction every charting tool
  uses. The price pane gives up whatever the lower panes take, down to a
  floor — dragging far enough would otherwise leave the candles a few
  pixels tall, at which point the chart stops being a chart.
- **Drag the chart's bottom-right corner** for a taller chart overall.
  The dividers then split whatever height it has.

Both are remembered per browser. The canvas is redrawn on resize rather
than left to scale its own bitmap, which would make a taller chart a
blurrier one instead of a bigger one.

### Colours

Your blue and orange, nudged into the palette's lightness band:
`#4d8fd1` and `#c17d33`. Validated rather than eyeballed — they hold ΔE 22
under protanopia, which is what two lines crossing in one small pane need.
The line, its signal and the histogram share one symmetric scale, because
they are the same units and separate scales would make a crossover look
like something it is not.

### One panel

Everything the agent does lives on **The call**: the chart with its trades
drawn on the candles, the take profit and stop loss, the rule, the alert,
the buttons, the book and the scorecard. Reading the call and setting what
it does were two jobs in two parts of the page; they are one now.

The separate agent panel is gone, and so are the duplicate candle, size
and fee selectors that came with it — one control each, driving both the
reading and the agent.

Three things went with it, and they were all dead weight rather than
features: the old **take it / ignore it** row (the agent keeps its own
book and the alert records a fill you took yourself, so there is one
record instead of two that disagree), the **you-versus-the-tool
scorecard** it fed, and `paintCompare`, whose two-way row was absorbed
into the three-way row several changes earlier and had had no call site
since.

#### Defined twice, silently

The same failure hit three times in one session, at three levels, and it
is invisible by reading either definition on its own:

- a page `function` — a second `checkAlert` replaced the first, and the
  survivor was handed a payload it could not read
- a **route handler** — a second `/api/calibration` landed on the path the
  candle read already owned; the new one never ran
- a **method on a class** — a second `Runtime.calibration` replaced the
  first, so the caller silently got the other one

None of these is an error in Python or JavaScript. The later definition
simply wins. Five tests now fail the build on any of them: duplicate page
functions, duplicate page consts, duplicate routes by path and by handler
name, duplicate methods in a class, and duplicate module-level functions.

#### The guard that found them

Merging two panels dropped the `aim` selector and a status stamp while the
script still read both. `$('someId')` on an id that is not in the page
returns null, and the next property access throws — which took the whole
poll down, so the chart silently stopped loading.

A test now extracts every `$('id')` literal from the page script and fails
if the id is not in the markup. It found four more dead references the
moment it was written, one of which had been there for several rounds.
Together with the duplicate-name tests, the page's three worst failure
modes — a dead reference, a shadowed function, a redeclared const — are
now build failures rather than things you find by watching a panel not
update.

### One screen at a time

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
- **Countdown on the forming bar**, on the price scale directly under the
  last-price label, where TradingView puts it. It ticks every second —
  a countdown that only moves when new data arrives is a clock that lies
  between polls — and it stops when the tab is hidden or the chart is on
  another tab. Under thirty seconds it turns red, because on a chart
  where every trade is out before the close, the last half minute is the
  part you act on.
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
