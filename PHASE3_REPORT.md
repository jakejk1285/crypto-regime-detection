# Phase 3 — Principled Regime Modeling: k-means vs GMM

**Status: complete. Stopping here for review before Phase 4.**

Setup: 6-asset tier, **2977 obs × 78 features** (price levels now excluded),
2018-07-03 → 2026-08-26. StandardScaler → PCA → cluster. **19 PCA components** at the
pre-registered 0.897 variance target (achieved 0.8986).

## 0. Decisions applied and guardrails installed

- `include_price_levels = False` — 90 → **78 features**, zero `sma_10`/`sma_20` remain.
- `src/regime_scheduler.py` **deleted** (647 lines).
- **Import quarantine is now mechanical.** `test_quarantine.py` AST-parses every
  pipeline module and fails on any import from `backtest_system/`. It catches three
  evasion routes — direct import, `importlib.import_module`, and `sys.path` insertion.
  Verified by planting a deliberate breach: the guard flagged all three and exited 1.
  Currently: **6 modules, quarantine intact.**

### Why PCA is load-bearing, not a carryover
A full-covariance GMM on 78 raw features needs 3159 parameters per component — ~12,600
for k=4 against 2977 observations. Covariances would be singular and the BIC penalty
would swamp the likelihood, making the curve monotone and uninformative. At 19
components it is 839 parameters (full) or 155 (diag). The 0.897 target was fixed in
advance from the original project's `pca_analysis_summary.json` and **not adjusted after
seeing any clustering result**.

## 1. The headline finding: forced 4 vs. what the criteria picked

**k = 4 is exogenous** — it comes from the Two Sigma comparison, not from this data. The
curves below are *diagnostic reporting*, not selection: nothing downstream consumes the
criterion-optimal k. That is what keeps this from being selection on the test window.

| criterion | model | optimum | vs forced k=4 |
|---|---|---|---|
| Gap statistic (SE rule) | k-means | k = 4 **or 5** | **consistent with 4, does not select it** |
| Gap statistic (argmax) | k-means | k = 12 (range edge) | forced 4 far lower |
| Silhouette | k-means | k = 2 | forced 4 is higher |
| BIC | GMM (full cov) | k = 5 | forced 4 is close |
| BIC | GMM (diag cov) | k = 12 (range edge) | forced 4 far lower |
| AIC | GMM (both) | k = 12 (range edge) | forced 4 far lower |

**The gap statistic is *consistent with* k=4. It does not independently select it**, and
the distinction matters because this is the load-bearing claim in any defense of k=4.
The detail:

- `satisfies_se_rule` is True at **both k=4 and k=5**. Tibshirani's rule takes the
  smallest such k, which is why the code returns 4 — but "smallest k passing a threshold
  that 5 also passes" is not an optimum.
- The gap **rises essentially monotonically through k=12** (2.3232 → 2.4240). Its argmax
  is k=12, the edge of the search range.
- The SE rule fires at 4 because of a local flattening, not a maximum:
  `gap(4) − gap(5) = 1.6e-04` against `s_k(5) = 4.5e-03`. The flattening is **3.6% of the
  standard error** — an order of magnitude below the noise the statistic itself
  measures.

So the honest statement is that **no criterion tested has a genuine interior optimum at
4**; the gap statistic merely fails to rule it out. The train-only k=11 result below is
not a separate anomaly — it is the same fact from another angle.

### Why the criteria disagree by a factor of six — and why that is expected

The old analysis reported silhouette 2, CH 10, DB 12, BIC 7, AIC 11 and never accounted
for the spread. The explanation is **autocorrelation**. Silhouette, BIC, AIC and gap all
assume iid observations. Market regimes are contiguous time blocks — mean spell length
here is 5.3 days, longest 35 — so any two nearby days are near-duplicates. More clusters
keep looking better because the criteria are effectively counting correlated samples.
This is why BIC/AIC run to the edge of the search range rather than finding an interior
optimum, and it is the honest reason not to pick k by argmax.

Note that **GMM with full covariance does find an interior BIC optimum at k=5**, while
diagonal covariance does not. The heavier parameterization penalizes additional
components enough to stop the runaway.

### Train-only cross-check — an instability worth stating

Criteria recomputed on the first 1500 observations (~50%):

| criterion | full sample | train-only | stable? |
|---|---|---|---|
| silhouette (k-means) | 2 | 2 | yes |
| **gap SE-rule (k-means)** | **4** | **11** | **no** |
| BIC (GMM diag) | 12 | 12 | yes |
| AIC (GMM diag) | 12 | 12 | yes |

The gap statistic's consistency with k=4 **does not survive** restriction to the first
half of the sample, where the SE rule fires at 11. Given that the gap curve has no
interior optimum on the full sample either, the two results are one finding: **the SE
rule is picking out shallow flattenings in a rising curve, and where those fall is not
stable.** The claim available is "k=4 is not contradicted by the full sample," not "the
data chose 4." Reported as-is rather than quoting only the convenient half.

## 2. Regime interpretability — k-means wins clearly

### Regime identity across refits — fixed before Phase 4 needs it

The original ordering ranked clusters by within-sample mean of a raw feature. That is
**not fold-stable**: the sort key is a sample statistic, so when the fold window changes
a regime can move rank without changing character, and nothing pins the cluster
boundaries. Phase 4 refits per fold, and per-fold parameters keyed on regime id would be
meaningless — the exact instability the diagnosis documented, where
`regime_strategy_mapping` was hardcoded across two fits whose clusters disagreed (2
observations in cluster 3 for one fit, 40 for the other).

Replaced with reference anchoring: regime identity is established once from the **first
training fold's raw-feature profiles**, and every later fold is matched to it by optimal
(Hungarian) assignment, guaranteeing a one-to-one mapping. Matching uses raw features
rather than PCA centroids because PCA is refit per fold and its axes — sign included,
which sklearn does not fix — rotate between folds.

**Honest result of testing it:** across 9 expanding folds (n = 1700 → 2900), rank-based
ordering and reference-matching produced **identical label maps every time**. So on this
data the fix corrects no observed error; it converts a property that happened to hold
into one that is guaranteed. That is worth having before Phase 4 rather than after.

The same test surfaced something that *is* a finding: **profile drift rises
monotonically** with fold length, 0.245 (n=1700) → 0.675 (n=2900) in reference units.
The matched regimes are progressively less like their reference selves. Phase 5 should
report this rather than assume regime identity held across the walk-forward.

### k-means regimes, labelled from feature profiles only

| regime | n (share) | economic label | defining feature profile |
|---|---|---|---|
| 0 | 483 (16.2%) | **Crisis / Capitulation** | deep negative VaR-95 across all assets (z ≈ −1.4), negative market momentum, highest pairwise correlation (+0.63) |
| 1 | 1314 (44.1%) | **Steady State** | mildly negative RSI, low dispersion — the unremarkable majority |
| 2 | 971 (32.6%) | **Broad Rally** | RSI elevated across every asset (+0.83 to +0.90), dispersion near zero — everything rising together |
| 3 | 209 (7.0%) | **Alt-Season / Rotation** | dispersion +2.43, ADA/XRP 30d returns +2.11, alt-vs-BTC spread +1.74, BTC dominance −1.05, BTC relative strength −1.67 |

### The labels validate against events the model never saw

Regime assignment on known crypto events — the model saw no returns and no dates:

| event | date | k-means | GMM |
|---|---|---|---|
| COVID crash | 2020-03-08 | **0 (Crisis)** | 1 |
| China mining ban crash | 2021-05-19 | **0 (Crisis)** | 3 |
| Terra/LUNA collapse | 2022-05-11 | **0 (Crisis)** | 1 |
| FTX collapse | 2022-11-09 | **0 (Crisis)** | 3 |
| Alt-season | 2021-01-15 | **3 (Alt-Season)** | 3 |
| Alt-season peak | 2021-04-10 | **3 (Alt-Season)** | 3 |

k-means regime 0 captures **all four** major crashes. GMM scatters them across two
different regimes. k-means regime 0's longest spells are the Nov–Dec 2018 capitulation
(28d), March 2020 COVID (25d), May 2021 China ban (25d), and May/June 2022 Terra (23d
each). Regime 3's longest spells are Jan–Mar 2021 (35d), Nov–Dec 2024 (31d), and April
2021 (20d) — the actual alt-season windows.

### Did the relational features earn their place? Partly — and the negative half matters

**Yes for the rotation regime.** Regime 3 is unambiguously a capital-rotation state:
alt-spread +1.74, BTC dominance −1.05, BTC relative strength −1.67, and it peaks in 2021
(33% of days) and Nov 2024 — exactly the historical alt-seasons. Without the relational
features this regime would not be separable. That is the six-asset universe doing the
work it was chosen for.

**No for the other three.** For regimes 0, 1 and 2 the best-ranked relational feature
sits **#43–#49 of 78** discriminators; those regimes are driven by per-asset RSI, VaR and
volatility. So the honest summary is: the relational features bought **one** genuinely
new regime, not a rotation-structured partition throughout.

### Epoch check — the price-levels fix worked
Regime shares by calendar year confirm k-means regimes are **not** time blocks: regime 1
appears in 19–60% of days in every year from 2018 to 2026. GMM is weaker here — its
regime 0 drifts monotonically from 25% (2018) to 76% (2026), which is epoch-like drift
rather than a recurring state.

## 3. k-means vs GMM

| axis | k-means | GMM (diag) |
|---|---|---|
| flip rate | **0.187** | 0.225 |
| spells | **557** | 672 |
| mean spell length | **5.3 d** | 4.4 d |
| median spell length | 2 d | 2 d |
| crash events captured in one regime | **4 of 4** | 0 of 4 (split across two) |
| epoch drift | none material | regime 0 drifts 25% → 76% |
| interior BIC optimum | n/a | only with full covariance |

**Raw hard-label agreement is 39.4%**, but that number is partly an artifact of label
permutation — two models can produce different ids for similar partitions. Measured
permutation-invariantly:

| measure | value |
|---|---|
| raw label agreement | 39.4% |
| agreement under **optimal** label matching | **64.8%** |
| **adjusted Rand index** | **0.299** |
| adjusted mutual information | 0.319 |

The finding survives but in weaker form. ARI = 0.299 (0 = chance, 1 = identical) means
the two methods find **substantially but not entirely different** partitions — the
agreement is real and well above chance, and considerably better than the raw 39.4%
suggested. With this much autocorrelation and this few effective observations, the
partition is only moderately robust to algorithm choice.

### GMM soft probabilities
Mean max-probability **0.877**; **8.9%** of days fall below 0.60, i.e. genuinely
ambiguous between regimes. A day at 0.51/0.49 is materially different from one at 0.99,
and hard assignment discards that — Phase 4 can use the probabilities to size positions
by confidence.

One counter-intuitive result worth flagging: on days where the two models **disagree**,
GMM's mean confidence is **0.885** — slightly *higher* than its overall average. The
disagreement is not concentrated in GMM's uncertain days. Both models are confidently
assigning the same days to different regimes, which is a stronger caution than if the
disagreement had been confined to borderline cases.

### Post-hoc return description (never an input to fitting or selection)

| k-means regime | label | mean fwd return | volatility | hit rate |
|---|---|---|---|---|
| 0 | Crisis | +0.238% | 5.26% | 54.9% |
| 1 | Steady State | +0.004% | 3.20% | 52.1% |
| 2 | Broad Rally | +0.213% | 3.29% | 53.9% |
| 3 | Alt-Season | **+0.616%** | 5.56% | 58.4% |

Kept in a separate function from the feature profiles so the boundary is structural.
**These numbers must not be read as strategy performance** — they are in-sample
descriptions of a full-sample fit. Phase 4 is what determines whether any of this
survives out of sample, and the spread here is well within what 2977 autocorrelated
observations can produce by chance.

## 4. NaN policy — decided, not deferred

**Clustering is unaffected.** It consumes the feature matrix, which has no NaNs after
warmup. The NaN `avg_duration` values arise in `regime_stats.py` and affect the *trading
rules* in Phase 4 only.

Prevalence:

| model | undefined obs | share | last undefined | contiguous from start? |
|---|---|---|---|---|
| k-means | 38 of 2977 | 1.3% | 2018-09-25 | **yes** |
| GMM | 27 of 2977 | 0.9% | 2018-09-24 | **yes** |

**Decision: exclude, never impute.** The undefined block is contiguous at the very start
of the sample and ends by late September 2018 — before any plausible first test fold
begins (t0=500 puts fold 1 in late 2019). So the exclusion costs **zero test
observations**; it only trims the head of the first training fold.

`fillna(0)` is explicitly rejected: 0 means "never persists," a real and meaningful
value for these statistics, not "unknown." Imputing it would silently distort the
persistence-based sizing and stop-loss rules.

## 5. Recommendation for Phase 4

Carry **k-means** as the primary model: better stability (flip rate 0.187 vs 0.225),
longer spells, no epoch drift, and regimes that map onto real events. Carry **GMM as the
comparator** as specified, using its soft probabilities.

Two cautions to carry into Phase 4 and Phase 5:
1. **ARI = 0.299** between the two models means a meaningful part of any performance
   difference will be partition noise, not method quality. Phase 5 should expect to say
   the difference is within noise rather than declare a winner.
2. **No criterion has a genuine interior optimum at k=4.** "Why 4 regimes?" rests on
   Two Sigma comparability, with the data merely not contradicting it on the full
   sample — and contradicting it (k=11) on the first half. That full statement, not the
   short version, should appear wherever k=4 is justified.
3. **GMM covariance type must be settled.** `fit_gmm` defaults to `diag`, but only
   `full` produces an interior BIC optimum (k=5). Phase 4 should carry **`full`** for
   the GMM comparator — 839 parameters against 2977 observations is tight but tractable,
   and diag's monotone BIC makes it the less informative model. Flagging rather than
   silently switching, since it changes the comparator.
4. **Profile drift grows with fold length** (0.245 → 0.675). Report it in Phase 5.
