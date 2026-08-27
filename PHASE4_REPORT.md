# Phase 4 — Walk-Forward Validation

**Status: complete. Stopping here for review before Phase 5.**

**Headline: the regime strategy does not demonstrate out-of-sample value.** Its Sharpe is
statistically indistinguishable from buy-and-hold *and* from a naive volatility target,
it fails a block-shuffle placebo test, and more than half its gross return comes from a
single fold. Details and the reasoning below; nothing here has been massaged.

## 1. Setup

9 expanding folds (not the 7 estimated in Phase 1 — usable observations are 2977, more
than that estimate assumed), t0=500, 250-day test, **60-day purge**. OOS window
**2020-01-14 → 2026-03-12, 2250 stitched test days**.

Per fold: `StandardScaler` → `PCA` (16–19 components at the pre-registered 0.897 target)
→ clusterer, all fitted on training data only. Test labels come from `.predict()` on the
frozen pipeline; the clusterer is never refit on test data. Purge is asserted at runtime
(`train.index.max() + 60d <= test.index.min()`) and the fitted weight from day *t*'s
regime is applied to day *t+1*'s return.

GMM uses **full** covariance as confirmed. Regime identity is anchored to fold 1's
raw-feature profiles via Hungarian matching.

### What was fitted per fold — and what was dropped, with the arithmetic

The brief listed "allocation, thresholds, sizing, stops." Two families were fitted:

- **allocation / sizing**: `w_r = min(1.0, 0.50 / annualized_vol_r)`, from training data
- **threshold (gate)**: a regime is tradeable only if it has ≥5 completed spells and a
  training t-stat ≥ 1.0

The t-statistic uses **spell count, not observation count**, for degrees of freedom.
Days inside one spell are near-duplicates; using n would overstate significance by
roughly √(spell length).

**Stops were not fitted, deliberately.** Measured before building the harness, per-fold
per-regime training data goes as low as **20 observations in 1 spell** (Crisis, fold 2)
and 30 observations in 3 spells (Alt-Season, fold 3). Fitting per-regime stop
percentages and R-multiples *on top of* allocations at that sample size is fitting
noise — the failure the audit documented. Stops are also not supportable by daily bars
at all: a daily candle cannot adjudicate whether the stop or the target was hit first
intraday. The legacy pipeline concealed this by synthesizing highs/lows as
`close*1.001`/`close*0.999` and evaluating exits on the close (diagnosis finding 5.5).

The four constants (5 spells, t≥1.0, 50% vol target, no leverage) were fixed a priori
and tuned on nothing.

## 2. Results — out of sample

| strategy | total return | ann. return | ann. vol | **Sharpe** | max DD | days |
|---|---|---|---|---|---|---|
| buy & hold (EW basket) | **+1505%** | 36.5% | 61.6% | 0.528 | −75.7% | 2250 |
| no-regime vol-target | +913% | 29.6% | 45.3% | 0.566 | −61.1% | 2250 |
| **k-means regime strategy** | +518% | 22.6% | 31.5% | **0.594** | −44.8% | 2250 |
| block-shuffled regimes (mean of 500) | — | — | — | 0.519 | — | 2250 |
| equal-weight-all-regimes | +161% | 11.3% | 18.8% | 0.393 | −30.8% | 2250 |
| **GMM regime strategy** | **−17%** | −2.1% | 21.8% | **−0.278** | −51.2% | 2250 |

The k-means strategy has the highest Sharpe. That is the entire case for it, and it does
not survive testing.

### It fails every test of that edge

**Against buy-and-hold and against a naive vol target.** 2000-draw **block** bootstrap
with 60-day blocks, matching the purge and the longest feature window, so regime spells
and serial dependence survive resampling. An iid bootstrap would assume away exactly the
autocorrelation this report elsewhere argues invalidates the k-selection criteria:

| comparison | ΔSharpe | 95% CI (block) | verdict |
|---|---|---|---|
| regime − buy & hold | +0.065 | [−0.699, +0.740] | **not distinguishable from 0** |
| regime − no-regime vol-target | +0.028 | [−0.774, +0.645] | **not distinguishable from 0** |

(The iid bootstrap gave [−0.770, +0.808] and [−0.734, +0.812] — materially the same, so
the width is a genuine property of the data, not a resampling artifact.)

**Against the placebo — the decisive test.** Regime labels were block-shuffled and the
identical pipeline refit. The null preserves structure and breaks only alignment: the
series is split into contiguous spells and the **spells are permuted with their labels
attached**, so a 35-day Alt-Season spell stays a 35-day Alt-Season spell and simply
moves in time. Spell-length distribution and regime frequencies are therefore exactly
preserved; what is destroyed is the correspondence between regime and market state.
(This is a stronger null than shuffling labels within a fixed spell skeleton.)

Over **500 draws**:

| | Sharpe |
|---|---|
| real strategy | **0.594** |
| shuffled mean | 0.519 |
| shuffled sd | 0.316 |
| shuffled range | −0.174 to 1.535 |
| shuffled 50th / 90th / 95th pct | 0.486 / 0.928 / 1.085 |

The real strategy **beats 311 of 500 shuffles — empirical p = 0.378**, and sits just
0.24 standard deviations above the shuffled mean. Random labels with the same temporal
structure match or beat it more than a third of the time.

**Concentration.** 7 of 9 folds are positive, but one fold carries it:

| | total return | Sharpe |
|---|---|---|
| full OOS | +518% | 0.594 |
| **excluding fold 2** (2020-09→2021-05, +175%) | +125% | **0.274** |
| buy & hold, same excluded window | +81% | 0.067 |

Removing one 8-month window — the 2020-21 bull run — cuts the Sharpe by more than half.
Fold 2 is **54% of gross return**. The strategy's record is one very good bull market.

The lower drawdown (−44.8% vs −75.7%) and lower volatility are real, but they are what
being in cash 61% of the time buys. The `no-regime vol-target` baseline gets nearly the
same Sharpe by deleveraging alone, with no regime model at all.

*One reporting note: the 21.4% headline hit rate is a cash artifact — flat days count as
misses. On days actually invested the hit rate is **54.9%**, versus 54.2% for
buy-and-hold. That is a difference of 0.7 percentage points.*

## 3. It is not a four-regime strategy

The gate rejected two of four regimes in **every single fold**:

| regime | folds gated out | contribution to OOS return | days invested |
|---|---|---|---|
| Crisis | 3 / 9 | **−23.1%** (see §3b) | 302 |
| SteadyState | **9 / 9** | never invested | 0 |
| BroadRally | **9 / 9** | never invested | 0 |
| **AltSeason** | 0 / 9 | **+703.6%** | 574 |

SteadyState and BroadRally never cleared t ≥ 1.0 in any training fold. Crisis was
allocated in 6 folds and **lost money**. Essentially all performance comes from
Alt-Season — the one regime the relational features made separable.

So the honest description is **"a rotation-timing rule that goes long the basket during
alt-season,"** not a four-regime conditional strategy. That is a narrower and more
fragile claim, and it should be stated that way everywhere.

GMM is worse and differently broken: it gated out Crisis in 9/9 folds, took no position
at all in folds 1–2, and ended negative.

### 3b. The gate is insufficient — Crisis is the demonstration

Crisis passed the gate in folds 4–9 on genuinely strong training evidence and then lost
money out of sample. Train vs test, per fold:

| fold | train n | train mean/day | **train t** | w | test n | **test total** |
|---|---|---|---|---|---|---|
| 1 | 51 | +1.60% | +0.97 | 0.00 (gated out) | 0 | — |
| 2 | 20 | −0.43% | −0.04 | 0.00 (gated out) | 0 | — |
| 3 | 30 | +2.36% | +0.67 | 0.00 (gated out) | 0 | — |
| 4 | 457 | +1.07% | **+2.54** | 0.94 | 71 | **+10.89%** |
| 5 | 529 | +1.09% | **+2.84** | 0.96 | 0 | — |
| 6 | 615 | +1.07% | **+3.01** | 0.98 | 3 | −7.11% |
| 7 | 696 | +1.01% | **+3.16** | 1.00 | 86 | −0.42% |
| 8 | 166 | +2.56% | **+2.68** | 0.58 | 68 | −3.04% |
| 9 | 828 | +1.01% | **+3.67** | 1.00 | 74 | **−22.64%** |

I checked this for a sign or alignment error; there is none (`W.shift(1)` is applied
correctly, and fold 4 is positive). It is a real result, and an instructive one.

**A t-statistic of +3.67 on 828 training observations did not survive out of sample.**
The training signal is not weak or marginal — it is the strongest gate evidence anywhere
in the run — and it reversed. The likely mechanism is that Crisis's in-sample edge comes
from sharp post-crash rebounds, and which side of a crash a given window lands on is
close to arbitrary at this sample size.

The finding: **a spell-adjusted t ≥ 1.0 gate fitted on 500–800 training observations is
not sufficient protection against exactly the overfitting this rebuild set out to
eliminate.** Proper walk-forward machinery removed the leak; it did not, on its own,
make per-regime parameter estimates reliable.

## 4. Regime identity is not stable across folds

Profile drift from fold 1's reference, in reference units:

| fold | train n | drift |
|---|---|---|
| 1 | 500 | 0.000 |
| 3 | 1000 | **3.051** |
| 5 | 1500 | 2.265 |
| 7 | 2000 | 2.237 |
| 9 | 2500 | 2.246 |

Drift jumps sharply by fold 3 and stays elevated. This is worse than the smooth
0.245 → 0.675 progression Phase 3 measured on rank-ordering, because here PCA is also
refit per fold, rotating the space the clusters live in.

**This bears directly on what the walk-forward measured.** "Alt-Season" in fold 9 is
matched to but materially different from "Alt-Season" in fold 1. A per-regime weight
carried across folds is being applied to a moving target, which means even the
Alt-Season result is not a clean estimate of a stable effect.

## 5. What this phase concludes

**The research question — can unsupervised regime detection on crypto produce a strategy
with genuine out-of-sample value — is answered "not demonstrably, on this data."**

Specifically, and each independently sufficient:

1. The Sharpe edge over buy-and-hold (+0.065) and over a naive vol target (+0.028) is
   inside the bootstrap noise band in both cases.
2. It beats block-shuffled regime labels 311 times in 500 (**p = 0.378**), sitting 0.24
   sd above the placebo mean. A null with identical temporal structure does about as
   well more than a third of the time.
3. Half the gross return is one 8-month fold; excluding it halves the Sharpe.
4. Two of four regimes never trade, so it is a one-regime rule with a two-regime
   comparator, not a four-regime strategy.
5. Regime identity drifts materially across folds, so even the surviving effect is not
   measured against a stable object.
6. The GMM comparator is outright negative (−0.278).
7. The Sharpe ordering across all strategies is **monotone in decreasing volatility** —
   regime 0.594 at 31.5% vol > vol-target 0.566 at 45.3% > buy-and-hold 0.528 at 61.6%.
   That is the signature of deleveraging, not of skill. It is the single cleanest
   explanation for the entire apparent "edge."

One nuance worth keeping: `equal-weight-all-regimes` (Sharpe 0.393) is **worse** than
both conditioning on regimes and not conditioning at all. Averaging the fitted weights
across regimes destroys something the conditioning preserves — so the regime
conditioning is doing *something* real. It just is not something that survives the
placebo test.

This is a negative result reported as-is. It is also the correct outcome of the rebuild:
the naive pipeline reported 54.35% return and 1.437 Sharpe on 335 leak-contaminated
observations. With the leaks removed, 9× the data, and honest out-of-sample testing,
the effect does not survive.

**What is *not* claimed:** that regime detection cannot work on crypto. Phase 3 showed
the regimes are economically real — Crisis captured all four major crashes, Alt-Season
peaked in Feb 2021 and Nov 2024. The regimes describe the market accurately. What fails
is converting that description into out-of-sample tradeable edge at this sample size.
Those are different claims, and Phase 6 should keep them separate.

## 6. Carried into Phase 5

- ARI 0.299 between models was the Phase 3 caution; the OOS gap is in fact large
  (0.594 vs −0.278) — but given points 1–3 above, this is better read as *both* being
  noise around zero than as k-means winning.
- Time in market 38.9%, mean weight when invested 0.80.
- Profile drift table above.
- Fold-level returns are in `data_cache/phase4_oos_kmeans.csv`; summary in
  `phase4_summary.csv`.
