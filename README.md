# Crypto Regime Detection: A Validation Study

**Research question.** Can unsupervised regime detection on cryptocurrency markets
produce a trading strategy with genuine out-of-sample value — and what do the data
limitations tell us about whether that question is even answerable at this scale?

**Answer.** The regimes are real. The edge is not.

Unsupervised clustering on price/volume/technical features recovers market regimes that
are economically meaningful and validate against events the model never saw. Converting
that description into out-of-sample tradeable edge fails every significance test applied
to it. The apparent Sharpe advantage is deleveraging, not skill.

Those are two different claims, and keeping them separate is the point of this project.

**This README is in three parts:**

1. **[What this project was before](#part-i--what-this-project-was-before)** — the
   54.35% return, the 1.437 Sharpe, and the specific reasons all of it was wrong.
2. **[What I changed, and why](#part-ii--what-i-changed-and-why)** — the rebuild, and
   the methodology that replaced the original approach.
3. **[What we actually found](#part-iii--what-we-actually-found)** — the honest results,
   positive and negative.

---

# Part I — What this project was before

## The headline number

For most of its life, this repository was a trading project whose README opened with a
performance table. The final version of that table read:

| metric | reported value |
|---|---|
| **Total return** | **54.35%** |
| **Sharpe ratio** | **1.437** |
| Max drawdown | 13.27% |
| Win rate | 53.3% |
| Trades | 137 |
| Period | "2024-01-01 to 2025-01-01" |

A 1.437 Sharpe on a long-only crypto strategy with a 13% max drawdown, in a market whose
buy-and-hold drawdown over the same era exceeded 70%, is not a good result. It is an
implausible one. The correct reaction to seeing it was suspicion. My reaction at the time
was to write it in bold at the top of the README and start building a C++ paper-trading
system to run it live.

**Every one of those numbers is invalid.** They are preserved here — in git history, in
`backtest_system/`, and in this section — deliberately. They are the control condition.
The distance between that table and the honest result *is* the finding of this project.

## The number was never stable, and that should have been the first clue

The headline block did not survive contact with its own commit history. It moved every
time the strategy was touched:

| commit | date | total return | Sharpe | max DD | win rate | trades |
|---|---|---|---|---|---|---|
| `6eb112a` initial commit | 2025-08-19 | **23.45%** | **1.87** | 8.2% | 84% | 156 |
| `39a54f2` "risk-free rate 3.95%" | 2025-08-27 | 23.71% | 1.85 | 8.21% | 84.2% | 155 |
| `45946e1` "profitable research-based strategy" | 2025-08-27 | 13.63% | 0.345 | 11.26% | 53.0% | 132 |
| `aa742bd` "further optimize based on performance analysis" | 2025-08-27 | 21.12% | 0.780 | 8.80% | 54.1% | 122 |
| `2188844` "Sharpe optimization +28.5%" | 2025-08-27 | 28.90% | 0.798 | 15.12% | 53.2% | 106 |
| `9c1a205` "update system configuration" | 2025-08-29 | **54.35%** | **1.437** | 13.27% | 53.3% | 137 |

Two things are visible in that table that were not visible to me while I was producing it.

**First, a Sharpe of 1.87 with an 84% win rate became a Sharpe of 0.345 with a 53% win
rate** — after a commit whose message describes making the strategy *more* research-based.
That is a ~5x change in the headline metric from an edit that was supposed to improve it.
A number that can move that far under a refactor is not measuring the strategy.

**Second, and worse: the 28.90% → 54.35% jump changed no strategy code at all.**
`git show --stat 9c1a205` touches `README.md`, `CHANGELOG.md`, `.gitignore`, the analysis
notebook and the paper-trading files. `backtest_trading_strategy.py`,
`sharpe_optimized_strategy.py` and `backtest_engine.py` were last modified two days
earlier; `backtest_data_manager.py` had not been touched since the initial commit. The
return nearly doubled with **identical Python parameters**. What actually changed was
which strategy class the notebook instantiated, and the fact that the data cache had been
rebuilt one day prior — over a different window (see below).

I was reading those moves as progress. They were noise, and part of the time they were
not even noise about the strategy.

## The four defects

The audit in `Overfitting_Diagnosis.md` was a read-only pass over the code and data
files — no code modified, no pipeline re-run, every count computed directly from the
pickles and CSVs. It found four independent defects. **Any one of them alone invalidates
the result.**

### 1. Full-sample statistics leaked into every trading decision

This is the fatal one.

`generate_historical_regimes()` computed `persistence`, `avg_duration` and
`frequency_percentage` for each regime **over the entire dataset**, then broadcast each
value as a constant onto every row belonging to that regime. Verified from the pickle —
each regime has exactly one distinct value:

| regime | persistence | avg_duration | frequency_% |
|---|---|---|---|
| 0 | 1.000 | 29.00 | 8.66 |
| 1 | 0.895 | 8.29 | 17.31 |
| 2 | 0.870 | 7.15 | 27.76 |
| 3 | 0.769 | 4.00 | 11.94 |
| 4 | 1.000 | 42.00 | 12.54 |
| 5 | 0.962 | 17.67 | 15.82 |
| 6 | 1.000 | 20.00 | 5.97 |

Those three columns are not diagnostics. They feed `should_trade`, the coin-score base,
the position-sizing persistence bonus, the stop-loss persistence adjuster and the
take-profit duration bonus. **On day 1, the strategy already knew how persistent, how
long-lived and how frequent each regime would turn out to be over the whole year.**

The same class of problem exists one level up: the `StandardScaler`, the 25-component PCA
and the 7-cluster k-means were all fitted **once, over the entire backtest window**, then
pickled and replayed timestamp by timestamp. There is no rolling or expanding refit
anywhere in the backtest path. Scaler means and PCA rotations are themselves whole-sample
statistics — that is in-sample scaling leakage even with zero forward returns.

Worth being precise about what *was* clean, because the honest version of this section
has to include it: **no forward return, P&L, or backtest metric ever entered PCA or
k-means.** The features were genuinely backward-looking. The regime detection was not the
leak. Everything downstream of it was.

### 2. The backtest did not run on the dates it reported

The cache file is named `crypto_data_20240101_20250101.pkl`. The engine prints
`Period: 2024-01-01 to 2025-01-01`. The actual contents:

| | reported | actual |
|---|---|---|
| Price data range | 2024-01-01 → 2025-01-01 | **2024-08-29 → 2025-08-28** |
| Regime index range | — | **2024-09-28 → 2025-08-28** |
| Symbols | 15 configured | **4 present** (BTC, ETH, ADA, SOL) |
| Observations | — | **335** |

The cause is one line. `fetch_coin_data_coingecko()` sends
`days=min((end-start).days, 365)`, and CoinGecko's `days` parameter returns a trailing
window anchored to the **request date**. So the evaluated window was fixed by *when the
cache happened to be built* (2025-08-28), not by `start_date` or `end_date` — those two
arguments only affected the filename and the printed banner. Separately, the
no-API-key branch restricted collection to 4 coins while the strategy iterated all 15, so
11 symbols silently had no prices at all.

Two consequences worth stating plainly:

- The reported date range in the README was **fiction**, and the notebook's own stored
  output printed "Processing 4 cryptocurrencies … 335 observations" directly beneath
  "Period: 2024-01-01 to 2025-01-01" without anything flagging the contradiction.
- The result is reproducible only while those pickles survive. Both loaders
  short-circuit on `force_refresh=False`, so deleting them — or passing
  `force_refresh=True` once — silently re-anchors the whole study to a new trailing
  365-day window. No error, no warning, a different dataset under the same filename.

### 3. The sample was far too small, and smaller than it looked

335 daily observations. Roughly 11 months. Seven regimes fitted on it.

But observations are the wrong unit. Within one contiguous regime spell, consecutive days
are near-perfectly autocorrelated; the quantity that behaves like sample size is
**independent regime episodes**. From `n / avg_duration`:

| regime | observations | avg duration | **episodes** |
|---|---|---|---|
| 0 STABLE_GROWTH | 29 | 29.0 | **1** |
| 1 MODERATE_MOMENTUM | 58 | 8.29 | 7 |
| 2 BASELINE_MARKET | 93 | 7.15 | 13 |
| 3 EXTREME_OUTLIER | 40 | 4.00 | 10 |
| 4 DEFENSIVE_STABLE | 42 | 42.0 | **1** |
| 5 BREAKOUT_MOMENTUM | 53 | 17.67 | 3 |
| 6 EXTREME_VOLATILITY | 20 | 20.0 | **1** |

**Three of seven regimes occurred exactly once in the entire dataset.** A per-regime
allocation, entry threshold, stop-loss and win rate were fitted to each of them anyway.
Fitting seven parameters to one observation is not estimation; it is transcription.

And the tradeable subset was smaller still — regimes 3 and 6 never traded (see below), so
the strategy's actual evidence base was **275 observations across 25 episodes**.

### 4. The most-favoured parameters were dead code

Regimes 3 and 6 both mapped to `strategy = "WAIT_AND_SEE"`, and three separate gates
blocked them: `should_trade_regime()` returned `False`, `calculate_position_size()`
forced `base_percent = 0.0`, and the `should_trade` column was `False` for every one of
their rows.

Regime 3 was the regime awarded **the largest allocation in the entire ladder (0.40)**
and **the lowest entry threshold (0.02)**, on the strength of being ranked "1st (BEST)"
at "2.230R expected value." It never took a single trade. Neither did regime 6, at the
bottom of the same ladder. Whatever those two numbers were fitted to, they contributed
nothing to the 54.35%.

The dead code went further than the allocation ladder:

- **The stop-loss and risk/reward tables were keyed on strings the regime mapper never
  emitted.** Emitted: `STABLE_GROWTH`, `MOMENTUM`, `BALANCED`, `WAIT_AND_SEE`,
  `CONSERVATIVE`. Keyed: `CRISIS`, `BASELINE`, `STABLE_GROWTH`, `MOMENTUM`, `BREAKOUT`,
  `DEFENSIVE`, `EXTREME_VOLATILITY`. Only two keys were reachable. The largest tradeable
  regime (regime 2, 93 observations) and regime 4 both fell through to the defaults. The
  advertised "BASELINE: 4% stop, 3.0 R:R, for our best regime" **never executed once.**
- **`USE_EV_FILTER` did not exist.** The README displayed it in a code block captioned
  `# backtest_engine.py`. `grep` finds it in exactly one file: `README.md`. The real EV
  filter was inert anyway — its guard required `ev_initialized`, set only inside
  `initialize_ev_system()`, which nothing ever called.
- **The `process_regime_change()` rebalancing body sat after a `return`.** Unreachable.
- **`getMinimumTradeThreshold()` in the C++ live path had no return statement at all.**
- **Tether (`USDTUSD`) was scored as a tradeable asset** and given explicit score bonuses
  in the defensive branches. It is a dollar peg.

### And a fifth, upstream of every number: the OHLC bars were synthetic

`high = price * 1.001`, `low = price * 0.999`, `volume = 1000000` as a constant. Fills,
stops and take-profits were then all evaluated on the close only. A position that pierced
its stop intrabar and recovered by the close was recorded as never having stopped out.
The stop-loss system — several hundred lines of per-regime tables, volatility adjusters,
persistence adjusters and trailing logic — was being evaluated against data that could
not, in principle, tell it whether a stop had been hit.

## Where I was wrong

The defects above are code. This part is judgment, and it is the part I would want a
reader to take seriously.

**I optimized against a number I had already contaminated.** The seven-value allocation
ladder, the seven-value entry-threshold ladder and the seven per-regime EV figures are
ordered strictly monotonically with each other. That ordering came from realized backtest
P&L on the same 335 observations the strategy was then evaluated on. I was not testing a
hypothesis; I was fitting a curve to my test set and reading the fit back as evidence.

**I treated "no forward returns in the clustering" as proof there was no leakage.** It
was true, and I checked it, and I stopped there. The leak was not in the features. It was
in three summary statistics computed downstream, and in the scaler, and in the PCA
rotation — all of which are look-ahead in exactly the same way and none of which look
like a forward return.

**I read a rising number as a working strategy.** 23% → 13% → 21% → 28% → 54% across
eleven days, each move accompanied by a commit message describing an improvement. I never
asked what the *variance* of that metric was under changes that should not have mattered.
Had I checked, I would have found that the largest single jump came from a commit that
modified no strategy code.

**I never ran the strategy on data it had not been built on.** Not once. There was no
train/test split, no walk-forward, no holdout of any kind. The word "backtest" was doing
work that the design did not support: a backtest over a window the model was fitted on
is an in-sample fit report, and I was presenting it as evidence about the future.

**I mistook complexity for rigor.** The strategy had per-regime stop losses, volatility
adjusters, persistence adjusters, stress adjusters, a three-factor PC sizing cascade,
correlation-group exposure caps and trailing stops. Large parts of it were unreachable,
and I did not know, because the sophistication of the code was itself the reassurance.
A simpler strategy would have made its own failure obvious.

**I was building the live system before validating the research.** A C++ paper-trading
implementation existed, with a *third* independent copy of the parameters that already
disagreed with the other two. The engineering was ahead of the evidence by a wide margin,
and effort spent there is effort not spent asking whether the result was real.

The generalizable version: **the failure was not a bug I missed. It was that I had built
no mechanism that could have told me.** Nothing in the repository was capable of
producing the sentence "this doesn't work." That is what the rebuild changed.

---

# Part II — What I changed, and why

The rebuild ran in phases, each reviewed before the next began, and each documented:
`PHASE1_DATA_REPORT.md` (data), `PHASE2_REPORT.md` (leak removal),
`PHASE3_REPORT.md` (modeling), `PHASE4_REPORT.md` (validation).

**The governing rule was set before any of it ran: a negative or inconclusive result is
an acceptable outcome and will not be massaged.** That commitment has to be made in
advance, because after the fact every result can be improved by one more idea.

| the original approach | what replaced it | why |
|---|---|---|
| CoinGecko `days=365`, trailing window from the request date | Binance daily klines via `ccxt`, paginated to full history, cached as parquet | The window is now determined by the data, not by the day the cache was built |
| 4 of 15 symbols, silently | 6 assets on their explicit common window, 2018-05-04 → 2026-08-26 | The universe is a stated choice with a stated reason |
| 335 observations, ~11 months | 2977 modeling observations, **2250 out-of-sample days** | The evidence base is out-of-sample days, not sample days |
| Scaler / PCA / k-means fitted on the full sample | Fitted per fold, on training data only, applied frozen | This is the leak, removed at the source |
| Full-sample `persistence` / `avg_duration` / `frequency` | Expanding-window regime statistics, causal by construction | Same leak, one level down |
| Synthetic OHLC, stops evaluated on close | No stops fitted at all | Daily bars cannot adjudicate intraday stops; the honest move is not to claim them |
| Per-regime parameters read off backtest P&L | Per-regime weights fitted in-fold, gated by a significance test | Parameters no longer see the data they are scored on |
| A single in-sample "backtest" | 9-fold expanding walk-forward + three independent significance tests | The design can now return a negative answer |
| Correctness asserted in comments | Two mechanical guard tests that fail the build | Intentions do not survive refactors; tests do |
| `.ffill().bfill().fillna(0)` | Backward-fill removed; the original path is quarantined | `bfill` pulls future values backwards. It is a leak in one method call |

## Data

Binance daily klines via `ccxt`, paginated to full history. 6 assets — BTC, ETH, BNB,
LTC, ADA, XRP — on their common window **2018-05-04 → 2026-08-26**.

The 6-asset universe was chosen over larger ones deliberately, and the reasoning is a
direct consequence of the sample-size lesson in Part I. For a *regime* model the quantity
that behaves like sample size is **distinct market regimes visited**, not observation
count. Adding a 2020-listed altcoin adds breadth to days already in the sample; extending
the start date backward adds days unlike any other. Only the 6-asset tier contains the
**2018 bear market**. A 14-asset universe would have started 2020-09-22 and held exactly
one full bull-bear transition.

| count | meaning |
|---|---|
| 3037 | raw common-history observations |
| 2977 | after 60-day rolling-feature warmup — the modeling sample |
| **2250** | **stitched out-of-sample test days — the actual evidence base** |

## Features

78 features in three groups, all causal:

- **54 per-asset** (9 each across 6 assets): 1d/7d/30d returns, RSI-14, RSI-21,
  price/SMA-14 ratio, 20d volatility, 20d VaR-95, 14d momentum
- **3 market-wide**: 14d momentum, positive breadth, BTC/ETH correlation
- **21 relational**: BTC volume-share dominance and relative strength, alt-vs-BTC
  spreads per asset, aggregate rotation signals and breadth, cross-sectional return
  dispersion, 30d/60d average pairwise correlation

The relational group is why six assets rather than one: per-asset technicals cannot
distinguish "BTC leads, alts bleed" from "alts outrun BTC." Both look like ordinary
momentum asset by asset.

Raw moving-average *levels* are excluded. Under fold-scoped scaling they are not a
nuisance but a correctness failure: a scaler fit on 2018-2020 sees BTC's SMA near $8k,
the 2021 test fold sees $50k, and every test point falls outside the training
distribution. The original pipeline included them and never noticed, because it fitted
the scaler on everything at once — the leak was concealing the problem it created.

## Models and validation

`StandardScaler` → `PCA` (19 components at a pre-registered 0.897 variance target) →
k-means and Gaussian Mixture (full covariance), k=4.

**Validation.** 9-fold expanding-window walk-forward, t0=500, 250-day test blocks,
**60-day purge** at every boundary — set by the longest rolling feature
(`avg_pairwise_corr_60d`), not the 30-day one a first pass assumed. Scaler, PCA, cluster
centroids and every trading rule are fitted on training data only and applied frozen;
test labels come from `.predict()` on the frozen pipeline.

## Why 4 regimes?

This is the first question the project draws, and the honest answer has three parts.
All three are required — the short version sounds like a failure; the full version is
the methodological point.

**1. k=4 is exogenous.** It comes from Two Sigma's factor-regime work
(Crisis / Steady State / Inflation / Walking-on-Ice), chosen for comparability. It was
never fitted to this data, so using it downstream is not selection on the test window.
Contrast the original k=7, which was picked where two of six disagreeing criteria
happened to agree — and which the research notebook that ostensibly justified it
contradicted, hardcoding `final_k = 5`.

**2. No selection criterion has a genuine interior optimum at 4.**

| criterion | optimum |
|---|---|
| Gap statistic (Tibshirani SE rule) | fires at k=4 **and** k=5 |
| Gap statistic (argmax) | k=12 — the edge of the search range |
| Silhouette | k=2 |
| GMM full-covariance BIC | k=5 |
| GMM diagonal-covariance BIC/AIC | k=12 — edge of range |

The gap statistic is *consistent with* 4; it does not select it. The SE rule fires at 4
because of a local flattening — `gap(4) − gap(5) = 1.6e-04` against `s_k(5) = 4.5e-03`,
**3.6% of the standard error** — not because 4 is a maximum. And that weak consistency
does not survive restriction to the first 1500 observations, where the rule fires at 11.

**3. Here is why the criteria are unreliable.** Silhouette, BIC, AIC and the gap
statistic all assume iid observations. Market regimes are contiguous time blocks — mean
spell length 5.3 days, up to 42 — so nearby days are near-duplicates and every criterion
counts correlated samples as independent evidence. That is why they disagree by a factor
of six, why BIC/AIC run to the edge of the range instead of finding an interior optimum,
and why picking k by argmax would be wrong.

The same reasoning about serial dependence governs the significance testing in Part III.
The project treats autocorrelation consistently throughout.

## The guards

Two tests enforce the rebuild's discipline mechanically rather than by intention. Both
are part of the method, not incidental tooling. They exist because of the Part I lesson
that a correct-looking comment is not a control.

### `test_causality.py` — proves features cannot see the future

Rebuilds every feature on truncated prefixes of the data and requires the result to
equal the full-series features on those same rows. Any full-window statistic — a mean, a
std, a correlation, a scaler fit on the whole sample — changes when the tail is removed.
A trailing rolling window does not. This test would have caught defect 1 immediately.

Result: differences are **exactly 0.000e+00** across 5 cut points, for all 78 features
and for the causal regime statistics. This cannot be satisfied by inspection or by a
reassuring comment.

### `test_quarantine.py` — proves the leak cannot walk back in

`backtest_system/` is **retained deliberately** so the naive result stays reproducible.
It still leaks by design: full-sample `StandardScaler`/PCA/KMeans, and
`.ffill().bfill().fillna(0)`. The highest-probability way this rebuild fails silently is
one convenience import.

So the quarantine is enforced: the test AST-parses every pipeline module and fails on any
import from `backtest_system/`, catching direct imports, `importlib.import_module`, and
`sys.path` insertion. Verified by planting a deliberate breach — it caught all three
routes and exited non-zero.

### An honest note on a third guard

Regime identity across folds is anchored to the first training fold's raw-feature
profiles via Hungarian matching, replacing a rank-based ordering that was not fold-stable.
Tested across 9 expanding folds, the two methods produced **identical label maps every
time**. The fix therefore corrected no observed error — it converts a property that
happened to hold into one that is guaranteed. Reported this way because a guard that
caught nothing is worth more stated plainly than implied to have caught something.

---

# Part III — What we actually found

## Result 1 — the regimes are real (positive finding)

k-means recovers four regimes, labelled from **feature profiles only**:

| regime | share | label | defining profile |
|---|---|---|---|
| 0 | 16.2% | **Crisis / Capitulation** | deep negative VaR-95 across all assets (z ≈ −1.4), negative momentum, peak pairwise correlation |
| 1 | 44.1% | **Steady State** | mildly negative RSI, low dispersion |
| 2 | 32.6% | **Broad Rally** | RSI elevated across every asset (+0.83 to +0.90), near-zero dispersion |
| 3 | 7.0% | **Alt-Season / Rotation** | dispersion +2.43, alt-vs-BTC spread +1.74, BTC dominance −1.05 |

**They validate against events the model never saw.** No returns, no dates, no labels
entered the fitting:

| event | date | k-means regime |
|---|---|---|
| COVID crash | 2020-03-08 | **Crisis** |
| China mining ban | 2021-05-19 | **Crisis** |
| Terra/LUNA collapse | 2022-05-11 | **Crisis** |
| FTX collapse | 2022-11-09 | **Crisis** |
| Alt-season | 2021-01-15 | **Alt-Season** |
| Alt-season peak | 2021-04-10 | **Alt-Season** |

Crisis captures **all four** major crashes. Its longest spells are the Nov–Dec 2018
capitulation, March 2020, May 2021 and May/June 2022. Alt-Season's longest spells are
Jan–Mar 2021 and Nov–Dec 2024 — the actual historical alt-seasons.

This is a genuine positive finding, and it is worth contrasting with Part I: the original
project's seven regimes included two that occurred once and were named
`EXTREME_OUTLIER` and `EXTREME_VOLATILITY` — labels describing 1- and 2-observation
pockets in one fit that held 40 and 20 observations in another, while inheriting the same
names and the same allocations. These four regimes are named for what their feature
profiles say, and they land on the right dates.

**The relational features earned their place — partly.** Alt-Season is unambiguously a
capital-rotation state and would not be separable without them. But for the other three
regimes the best relational feature ranks **#43–49 of 78** discriminators; those regimes
are driven by per-asset RSI and volatility. The six-asset universe bought **one**
genuinely new regime, not a rotation-structured partition throughout.

## Result 2 — the edge does not survive (negative finding)

Out-of-sample, 2020-01-14 → 2026-03-12, 2250 stitched test days:

| strategy | total return | ann. vol | **Sharpe** | max DD |
|---|---|---|---|---|
| buy & hold (EW basket) | **+1505%** | 61.6% | 0.528 | −75.7% |
| no-regime vol-target | +913% | 45.3% | 0.566 | −61.1% |
| **k-means regime strategy** | +518% | 31.5% | **0.594** | −44.8% |
| block-shuffled regimes (500 draws, mean) | — | — | 0.519 | — |
| equal-weight-all-regimes | +161% | 18.8% | 0.393 | −30.8% |
| **GMM regime strategy** | **−17%** | 21.8% | **−0.278** | −51.2% |

The regime strategy has the highest Sharpe. **That is deleveraging, not skill** — and
the table shows it: the Sharpe ordering is *monotone in decreasing volatility*
(0.594 at 31.5% > 0.566 at 45.3% > 0.528 at 61.6%). A vol-target with **no regime model
at all** reaches within 0.03 Sharpe of the regime strategy. It also earns nearly double
the total return while doing so.

Note what happened to the headline number in the process. 54.35% in-sample on 11 months
became a 6-year out-of-sample result that **underperforms buy-and-hold by a factor of
three** on total return, and whose Sharpe advantage is explained by holding less.

### Three independent tests, all negative

**1. Block bootstrap** (2000 draws, 60-day blocks so regime spells survive resampling —
an iid bootstrap would assume away the very autocorrelation this project argues
invalidates the k-criteria):

| comparison | ΔSharpe | 95% CI |
|---|---|---|
| regime − buy & hold | +0.065 | [−0.699, +0.740] |
| regime − no-regime vol-target | +0.028 | [−0.774, +0.645] |

Both cross zero.

**2. Placebo test — the decisive one.** Regime labels were block-shuffled: the series is
split into contiguous spells and the **spells permuted with their labels attached**, so a
35-day Alt-Season spell stays a 35-day Alt-Season spell and merely moves in time.
Spell-length distribution and regime frequencies are exactly preserved; only the
correspondence between regime and market state is destroyed.

Over 500 draws, the real strategy's 0.594 **beats 311 — empirical p = 0.378** — sitting
just 0.24 standard deviations above a shuffled mean of 0.519. Random labels with the
same temporal structure match or beat it more than a third of the time.

**3. Concentration.** 7 of 9 folds are positive, but one carries it:

| | total return | Sharpe |
|---|---|---|
| full OOS | +518% | 0.594 |
| excluding fold 2 (2020-09→2021-05) | +125% | **0.274** |

One 8-month window is 54% of gross return. Removing it halves the Sharpe.

### It is not a four-regime strategy

The gate rejected two of four regimes in **every single fold**:

| regime | folds gated out | OOS contribution | days invested |
|---|---|---|---|
| Crisis | 3 / 9 | **−23.1%** | 302 |
| Steady State | **9 / 9** | never invested | 0 |
| Broad Rally | **9 / 9** | never invested | 0 |
| Alt-Season | 0 / 9 | **+703.6%** | 574 |

Plainly: **this is a one-regime rotation-timing rule that goes long during alt-season**,
not a four-regime conditional strategy. This is the same failure mode as Part I's dead
allocation ladder — most of the machinery does nothing — with the difference that the
walk-forward reports it rather than concealing it.

### On k-means vs GMM — neither won

The OOS gap is large (0.594 vs −0.278), but the k-means variant **fails the same
significance tests**. The honest reading is that both are noise around zero, not that
k-means is the better method. Adjusted Rand index between the two partitions is 0.299.

### The nuance that cuts the other way

`equal-weight-all-regimes` (0.393) is **worse** than both conditioning on regimes and not
conditioning at all. Averaging the fitted weights across regimes destroys something the
conditioning preserves — so the conditioning captures real structure. It simply is not
large or stable enough to beat a temporal-structure-matched null at 2977 autocorrelated
observations. The claim is not "regimes are useless"; it is narrower and truer than that.

## Result 3 — the most instructive finding: leak-free ≠ reliable

The Crisis regime passed the significance gate in folds 4–9 on genuinely strong training
evidence, and lost money out of sample:

| fold | train n | train t | out-of-sample |
|---|---|---|---|
| 4 | 457 | +2.54 | +10.89% |
| 7 | 696 | +3.16 | −0.42% |
| 8 | 166 | +2.68 | −3.04% |
| **9** | **828** | **+3.67** | **−22.64%** |

A spell-adjusted **t of +3.67 on 828 training observations reversed out of sample.** This
was checked for a sign or alignment error; there is none.

**Walk-forward machinery removes the leak. It does not, by itself, make per-regime
parameter estimates reliable at this sample size.** Those are separate problems, and
solving the first does not solve the second. This is the single most transferable result
in the project — and the reason the rebuild alone would not have been enough without the
three significance tests on top of it.

---

## Data limitations

**The evidence base is 2250 out-of-sample days on 6 assets.** Everything below follows
from that being small.

**Scale, versus the literature.** Regime models of the kind this replicates are fitted on
decades of data across many asset classes and factors. Two Sigma's factor-regime work
spans multiple decades of cross-asset factor returns. This study has under 6 years of
out-of-sample daily crypto on six correlated assets whose average pairwise correlation
is **0.739**. Six assets at that correlation are closer to one-and-a-half independent
series than to six.

**Autocorrelation shrinks it further.** Mean regime spell is 5.3 days. The effective
number of independent observations is nearer the ~560 regime spells than the 2977 days.

**Regime identity drifts.** Two measurements, different setups:
- Phase 3 (PCA fitted once, rank ordering): drift 0.245 → 0.675 as folds lengthen.
- Phase 4 (PCA refit per fold, as the walk-forward actually runs): **0.000 → 3.051 by
  fold 3, staying ≈2.25.**

The Phase 4 figures are the ones that bear on the validation. "Alt-Season" in fold 9 is
matched to, but materially different from, fold 1's. A per-regime weight carried across
folds is applied to a moving target — so even the surviving Alt-Season effect is not a
clean estimate of a stable object.

**Daily bars cannot adjudicate intraday stops.** A daily candle cannot say whether a stop
or a target was hit first. Stops were therefore not fitted at all. The original pipeline
concealed this by synthesizing highs and lows as `close*1.001` / `close*0.999` and
evaluating exits on the close only.

**A synthetic splice exists in the data cache.** Polygon's MATIC→POL migration was
stitched (Binance closed MATIC/USDT 2024-09-10, opened POL/USDT 2024-09-13; observed seam
ratio 1.083 across the gap, treated as a 1:1 rebrand with no rescaling). A spliced series
invites the question of whether the seam created an artificial regime boundary.
**It cannot have done so here: the 6-asset analysis universe excludes MATIC entirely.**
The splice exists in the cache and in no result in this repository.

**Survivorship.** The universe is six assets that still trade in 2026. Assets that failed
between 2018 and now are absent, which flatters any long-biased result.

## What proper validation would require

The binding constraint is not observations. It is **independent regime episodes** — the
same unit that made 335 observations across 25 episodes hopeless in Part I, applied to a
sample forty times larger and still not large enough.

The Crisis result gives the arithmetic: t = +3.67 on 828 training observations reversed
out of sample, because those 828 days were roughly 90 episodes, and which side of a crash
a given window lands on is close to arbitrary at that count.

At the observed spell lengths, for the rarest regime (Alt-Season, ~7% of days):

| episodes per regime | implied history at 5.3d spells | at 10d spells |
|---|---|---|
| 30 | ~6 years | ~12 years |
| 50 | ~10 years | ~20 years |
| 100 | ~21 years | ~39 years |

Crypto's liquid history begins around 2017. **The data required to settle this question
for a 7%-frequency regime does not yet exist**, regardless of how clean the pipeline is.
That is a statement about the setting, not about this implementation.

What would move the answer:

1. **More independent episodes** — via a longer history (unavailable), or cross-sectional
   replication across many uncorrelated markets so episodes accumulate in parallel.
2. **Fewer regimes, or regimes defined to be more frequent.** A 7% regime carrying all
   the performance is the core statistical problem here.
3. **Intraday data**, if stops are to be part of the strategy at all.
4. **A stronger gate than a t-statistic.** The Crisis result shows t ≥ 1.0 on ~800
   observations is not sufficient protection. Nested cross-validation inside each
   training fold, or a substantially higher threshold, would be the next step.
5. **Testing the regime model on assets outside the fitting universe**, which would
   separate "these regimes describe crypto" from "these regimes describe these six coins."

---

## Repository structure

```
crypto_cluster_pca/
  data_pipeline/            # authoritative — all Phase 2+ work lives here
    binance_data.py         #   paginated Binance daily klines
    features.py             #   78 causal features; no standardization (deferred to folds)
    regime_stats.py         #   expanding-window regime statistics
    regime_models.py        #   k-means / GMM, model selection, fold-stable identity
    walkforward.py          #   9-fold expanding walk-forward + baselines
    test_causality.py       #   GUARD: truncation invariance
    test_quarantine.py      #   GUARD: import isolation
  data_cache/               # parquet OHLCV + result artifacts
  research/                 # exploratory notebooks (original analysis)
  src/                      # original standalone regime analysis (pre-rebuild)
  backtest_system/          # LEGACY — retained deliberately, leaks by design,
                            #   quarantined by test_quarantine.py. Kept so the naive
                            #   54.35% result stays reproducible as the control.

Overfitting_Diagnosis.md    # the audit that started the rebuild
PHASE1..4_REPORT.md         # data, leak removal, modeling, validation
```

## Reproducing

```bash
pip install -r requirements.txt
python crypto_cluster_pca/data_pipeline/binance_data.py      # fetch (cached)
python crypto_cluster_pca/data_pipeline/test_causality.py    # guard
python crypto_cluster_pca/data_pipeline/test_quarantine.py   # guard
python crypto_cluster_pca/data_pipeline/run_phase3.py        # regime models
python crypto_cluster_pca/data_pipeline/run_phase4.py        # walk-forward
```

## What this project is

A validation study that reaches a negative result, with the reasoning that gets there
documented at each step. The deliverable is the rigor, not a return figure — the original
return figure was the problem.

## Disclaimer

Research code. Not investment advice. The strategy studied here is explicitly shown *not*
to have demonstrable out-of-sample value.

## License

MIT — see `LICENSE`.
