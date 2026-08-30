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

---

## The arc

This repository began as a trading project reporting **54.35% return and a 1.437 Sharpe
ratio**. That number is invalid, and the work of establishing why — then rebuilding the
study so the question could be answered honestly — is what the repository now contains.

| stage | what happened | artifact |
|---|---|---|
| 1. Naive result | 54.35% return, 1.437 Sharpe reported on 335 observations | preserved in git history, `backtest_system/` |
| 2. Audit | Read-only diagnosis found leaks, an unreproducible window, and dead code | `Overfitting_Diagnosis.md` |
| 3. Rebuild | Real data, causal features, mechanical guards | `PHASE1_DATA_REPORT.md`, `PHASE2_REPORT.md` |
| 4. Principled modeling | k-means and GMM, model selection reported honestly | `PHASE3_REPORT.md` |
| 5. Honest validation | 9-fold walk-forward, three independent significance tests | `PHASE4_REPORT.md` |

**The naive 54.35% is kept visible on purpose.** It is the control condition. The
distance between it and the honest result *is* the finding.

### What was wrong with the original result

Four independent defects, any one of which invalidates it:

1. **Full-sample statistics leak.** `persistence`, `avg_duration` and
   `frequency_percentage` were computed over the entire dataset and broadcast onto every
   row, then used in `should_trade`, position sizing and stops. On day 1 the strategy
   already knew how persistent each regime would prove over the whole year.
2. **Unreproducible window.** The CoinGecko collector sent `days=365`, which returns a
   trailing window anchored to the *request date*. The backtest printed "Period:
   2024-01-01 to 2025-01-01" while running on 2024-09-28 → 2025-08-28, on 4 of 15
   configured symbols.
3. **Sample size.** 335 daily observations, with 3 of 7 regimes occurring exactly once.
4. **Dead code.** The highest-allocation regime (40% of capital, "highest EV") mapped to
   `WAIT_AND_SEE` and never traded. Entire stop-loss tables were keyed on strings the
   regime mapper never emitted.

---

## Method

**Data.** Binance daily klines via `ccxt`, paginated to full history.
6 assets — BTC, ETH, BNB, LTC, ADA, XRP — on their common window **2018-05-04 → 2026-08-26**.

The 6-asset universe was chosen over larger ones deliberately. For a *regime* model the
quantity that behaves like sample size is **distinct market regimes visited**, not
observation count. Adding a 2020-listed altcoin adds breadth to days already in the
sample; extending the start date backward adds days unlike any other. Only the 6-asset
tier contains the **2018 bear market**. A 14-asset universe would have started
2020-09-22 and held exactly one full bull-bear transition.

| count | meaning |
|---|---|
| 3037 | raw common-history observations |
| 2977 | after 60-day rolling-feature warmup — the modeling sample |
| **2250** | **stitched out-of-sample test days — the actual evidence base** |

**Features.** 78 features in three groups, all causal:
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
distribution.

**Models.** `StandardScaler` → `PCA` (19 components at a pre-registered 0.897 variance
target) → k-means and Gaussian Mixture (full covariance), k=4.

**Validation.** 9-fold expanding-window walk-forward, t0=500, 250-day test blocks,
**60-day purge** at every boundary (set by the longest rolling feature, not the
30-day one that a first pass assumed). Scaler, PCA, cluster centroids and every trading
rule are fitted on training data only and applied frozen; test labels come from
`.predict()` on the frozen pipeline.

---

## Why 4 regimes?

This is the first question the project draws, and the honest answer has three parts.
All three are required — the short version sounds like a failure; the full version is
the methodological point.

**1. k=4 is exogenous.** It comes from Two Sigma's factor-regime work
(Crisis / Steady State / Inflation / Walking-on-Ice), chosen for comparability. It was
never fitted to this data, so using it downstream is not selection on the test window.

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

The same reasoning about serial dependence governs the significance testing below. The
project treats autocorrelation consistently throughout.

---

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

**The relational features earned their place — partly.** Alt-Season is unambiguously a
capital-rotation state and would not be separable without them. But for the other three
regimes the best relational feature ranks **#43–49 of 78** discriminators; those regimes
are driven by per-asset RSI and volatility. The six-asset universe bought **one**
genuinely new regime, not a rotation-structured partition throughout.

---

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
not a four-regime conditional strategy.

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

---

## The most instructive finding: leak-free ≠ reliable

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
in the project.

---

## Methodology: the guards

Two tests enforce the rebuild's discipline mechanically rather than by intention. Both
are part of the method, not incidental tooling.

### `test_causality.py` — proves features cannot see the future

Rebuilds every feature on truncated prefixes of the data and requires the result to
equal the full-series features on those same rows. Any full-window statistic — a mean, a
std, a correlation, a scaler fit on the whole sample — changes when the tail is removed.
A trailing rolling window does not.

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

---

## What proper validation would require

The binding constraint is not observations. It is **independent regime episodes**.

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
