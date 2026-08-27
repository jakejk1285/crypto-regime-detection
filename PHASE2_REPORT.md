# Phase 2 — Leak Removal, Dead-Code Removal, Relational Features

**Status: complete. Stopping here for review before Phase 3.**

Tier in use: **6 assets** — BTC, ETH, BNB, LTC, ADA, XRP. Common window
2018-05-04 → 2026-08-26, 3037 raw observations, **2977 after rolling-window warmup**.

> **Correction to Phase 1.** That report stated the longest rolling feature window was
> 30 days (`return_30d`) and sized the fold-capacity table on it. The true binding
> window is **60** — `avg_pairwise_corr_60d`. The arithmetic confirms it: 3037 raw −
> 2977 in the feature matrix = a head loss of exactly 60. This matters because Phase 4's
> purge buffer must equal the longest window; at 30 the last 30 days of each training
> fold would share input data with the first rows of the test fold, which is the bleed
> the purge exists to stop. Corrected fold capacity for the approved 6-asset tier
> (usable 2977, purge 60d): **7 folds** at t0=500d/250d test, **10 folds** at
> t0=500d/180d test. The tier ranking is unchanged and 6 assets remains the right call.

## 1. What was removed

### C++ execution system — deleted
| path | size | note |
|---|---|---|
| `paper_trading/` | 2.7 MB, 72 files | full C++ trading system, Alpaca client, CMake build |
| `shared_regime_data/` | 4 KB | existed only as the Python→C++ handoff (`regime_for_cpp.json`) |

`setup.py` had two references (a `shared_regime_data/regime_output` mkdir and a
"Build C++ trading system" step); both replaced with the data-fetch and causality-test
steps. No stale references remain.

### Dead / inert code — deleted from `backtest_trading_strategy.py`
1462 → 1326 lines.

| removed | why it was dead |
|---|---|
| `expected_value_analyzer.py` (entire 619-line module) | its only consumer was the EV filter below |
| `initialize_ev_system()`, `get_ev_summary()`, `update_ev_settings()`, `use_ev_filter`, `ev_analyzer`, `ev_initialized` | `initialize_ev_system()` was **never called**, so `ev_initialized` was permanently `False` and the `should_take_trade()` branch was unreachable regardless of the flag |
| `build_rejection_reasoning()` | only caller was the unreachable EV branch |
| allocation-ladder entries for regimes **3 (0.40)** and **6 (0.01)** | both map to `WAIT_AND_SEE`, forced to `base_percent = 0.0` and blocked by `should_trade_regime()` |
| threshold-ladder entries for regimes **3 (0.02)** and **6 (0.85)** | same |
| 11 lines of unreachable code after the `return` in `get_regime_strategy_name()` | included a `coin_scores < 0.4` rebalancing rule that never ran |
| `"CRISIS"` branches in the coin scorer and `get_max_positions()` | `CRISIS` is never emitted by the regime mapping |

**Key-mismatch bug fixed rather than deleted.** The `base_stop_loss` and
`risk_reward_ratios` tables were keyed on `BASELINE / BREAKOUT / DEFENSIVE / CRISIS /
EXTREME_VOLATILITY`, none of which the regime mapping ever emits. Every regime except
`STABLE_GROWTH` and `MOMENTUM` silently fell through to the defaults — so regime 2
(`BALANCED`, the largest tradeable regime) and regime 4 (`CONSERVATIVE`) never received
their intended stops. The tables are now keyed on the strings actually produced.
Verified: `BALANCED` now resolves to a 4% stop at 3:1 R/R, `CONSERVATIVE` to 8% at 1:1.

*Note for later: `Overfitting_Diagnosis.md` cites file:line inside
`expected_value_analyzer.py`. Those citations now point at git history rather than the
working tree. The diagnosis is a record of what was found, so this is expected, but it
should be stated in the Phase 6 write-up rather than discovered later.*

### One judgment call, not made unilaterally
`crypto_cluster_pca/src/regime_scheduler.py` (647 lines) is the 15-minute production
refit loop. It is an **execution** concern in a now research-only project, and the
diagnosis found its refit-every-15-minutes behavior interacts badly with the hardcoded
cluster-id→name mapping. It is not C++, so it fell outside "remove the C++ system."
**Recommend deleting it in Phase 6; left in place pending your confirmation.**

## 2. How each statistic was made causal

New module: `crypto_cluster_pca/data_pipeline/regime_stats.py`.

The original (`backtest_data_manager.py:432-477`) computed each statistic over the
**entire sample** and broadcast one value per regime onto every row of that regime.
Those values drove `should_trade`, the coin-score base, the position-sizing persistence
bonus, the stop-loss persistence adjuster, and the take-profit duration bonus.

| statistic | was | now |
|---|---|---|
| `persistence` | P(stay) over all transitions in the full sample | P(stay) over transitions observed in `[0, t]` only |
| `avg_duration` | mean spell length over the whole sample, **including the trailing unfinished spell** | mean length of spells **completed strictly before t**; the current spell's length is unknowable until it ends |
| `frequency_percentage` | regime's share of the whole sample | regime's share of `[0, t]` |

The `avg_duration` fix matters most: the old code appended the in-progress spell
(`:455-456`), which is how regime 4 reported `avg_duration = 42` from a single possibly
unfinished episode.

**Explicit early-sample policy.** With fewer than 2 completed spells, `avg_duration` is
emitted as **NaN**, not as a number derived from one observation. The first walk-forward
folds will inherit those NaNs and Phase 4 must handle them deliberately. Emitting a
confident value from a single spell is the exact failure this rebuild corrects.

The original leaking computation is preserved as `full_sample_regime_stats()` — never
used for a trading decision, retained only to quantify the leak for Phase 6.

### Magnitude of the leak that was removed
Measured as `|full-sample − causal|` per observation:

| statistic | mean gap | max gap |
|---|---|---|
| `persistence` | 0.0126 | **0.559** |
| `avg_duration` | 0.773 | **10.04** days |
| `frequency_percentage` | 2.44 pp | **98.85 pp** |

The maxima are concentrated early in the sample — exactly where a walk-forward's first
folds live — which is why the leak inflated early-period results most.

## 3. Relational feature set (new)

New module: `crypto_cluster_pca/data_pipeline/features.py`. **90 features / 2977 obs.**

| group | count | contents |
|---|---|---|
| per-asset | 66 | returns (1/7/30d), RSI (14/21), SMA (10/20), price/SMA14 ratio, 20d volatility, 20d VaR-95, 14d momentum — carried over unchanged |
| market-wide | 3 | 14d market momentum, positive breadth, BTC/ETH 20d correlation — carried over unchanged |
| **relational** | **21** | **new — see below** |

### The 21 relational features

**BTC dominance proxy (4).** Binance klines carry no circulating supply, so true
market-cap dominance is **not computable from this data**. Substituted: BTC's share of
universe **quote** volume (`close × volume`; ccxt returns base-asset volume at index 5,
so the multiply is required). Volume verified non-degenerate — 3297 distinct values,
~$1.2B/day median for BTC — unlike the old CoinGecko path which hardcoded
`volume = 1000000` (diagnosis finding 5.5).
`btc_volume_dominance`, `..._ma30`, `..._chg30` (rotation direction),
`btc_relative_strength_30d` (price-based companion).

**Alt-vs-BTC return spreads (13).** `return(alt) − return(BTC)` per alt plus a 14d mean
(10 features), then `alt_spread_mean`, `alt_spread_mean_ma30`, and `alt_breadth_vs_btc`
(how many alts are beating BTC — breadth of the rotation, not just its size).

**Cross-asset return dispersion (2).** `return_dispersion` (same-day cross-sectional
stdev) and its 20d mean. High = idiosyncratic/rotational; low = everything moving
together.

**Average pairwise correlation (2).** Mean of the 15 off-diagonal pairwise correlations
on trailing 30d and 60d windows.

### These features measure what they claim

| check | result |
|---|---|
| avg pairwise corr level | median **0.739** — inside the expected 0.4–0.8 band |
| crisis signature | COVID crash (2020-03-12..20) mean **0.970** vs calm July 2019 **0.727**; the **top 5 highest-correlation days in 8 years are all March 2020** |
| dominance tracks known cycles | yearly mean 0.60+ in 2018/19/20/22/23 bear-and-BTC years, dropping to **0.421 in 2021** and **0.441 in 2025** — the alt-rotation years |
| rotation signal | peak alt-season month = **Feb 2021**, the actual historic alt-season peak; most BTC-dominant = July 2019 |

## 4. Causality verification — mechanical, not by inspection

New: `crypto_cluster_pca/data_pipeline/test_causality.py`. Written **before** the
features, as a constraint rather than a postscript.

**The test.** Rebuild every feature on truncated prefixes of the data and require the
result to equal the full-series features on those same rows, exactly. A feature using
any full-window statistic — mean, std, correlation, a scaler fit on the whole sample —
changes when the tail is removed. A trailing rolling window does not. This is precisely
the "same leak wearing a different hat" check you flagged.

```
TEST 1  Feature truncation-invariance         (90 features, 5 cut points)
  cut=  500  rows compared=  180  max|diff|=0.000e+00  PASS
  cut= 1000  rows compared=  680  max|diff|=0.000e+00  PASS
  cut= 1500  rows compared= 1180  max|diff|=0.000e+00  PASS
  cut= 2000  rows compared= 1680  max|diff|=0.000e+00  PASS
  cut= 2500  rows compared= 2180  max|diff|=0.000e+00  PASS

TEST 2  Regime-statistic truncation-invariance
  all 5 cuts: max|diff|=0.000e+00, nan-pattern-match=True  PASS

TEST 3  No forward-return leakage into features
  highest |corr| with NEXT-day return: 0.0792 (eth_return_1d)
  PASS: below the 0.20 leak threshold
```

Differences are **exactly zero**, not merely small.

**No standardization anywhere in `features.py`.** The module emits raw features only.
All scaling is deferred to the fold-scoped pipeline in Phase 4, so the walk-forward
scaler is the only place normalization happens — the structural fix for diagnosis item
2(i), where one `StandardScaler` was fit across the whole backtest window.

Warmup NaNs are dropped from the head only. **No backfill anywhere** — `bfill` moves
future values backward in time and is the leak this module exists to prevent (the old
pipeline used `.ffill().bfill().fillna(0)` at `backtest_data_manager.py:394`).

## 5. Forward-return leakage in regime detection — confirmed absent

Three independent lines of evidence:

1. **Structurally (Test 1) — this is the actual proof.** All 90 features are trailing
   rolling windows or same-day cross-sections, established by exact truncation
   invariance.
2. **By construction.** `forward_returns()` is a separate function never merged into the
   feature matrix. No feature references a negative shift.
3. **Empirically (Test 3) — a cheap cross-check, not co-equal evidence.** Max
   |correlation| with the next-day universe return is 0.0792. This test is a weak net:
   a slow-moving leak like the old full-sample `avg_duration` would not show up as
   correlation with a *next-day* return at all. It catches only crude forward-shift
   errors. Test 1 is what carries the claim.

## 5b. Quarantine: the old pipeline is still in the tree and still leaks

`backtest_system/backtest_data_manager.py` is **untouched and still importable**. It
fits `StandardScaler`/PCA/KMeans on the full sample (`:404-417`) and does
`.ffill().bfill().fillna(0)` (`:394`) — the very bfill this report cites as avoided.
It is retained because the naive 54.35% result must stay reproducible for the Phase 6
narrative.

**`crypto_cluster_pca/data_pipeline/` is authoritative from here on. Phase 3 onward must
not import anything from `backtest_system/`.** One convenience import would walk the
leak straight back in.

Related: the notebook's `backtester.trading_strategy.use_ev_filter = False` now sets an
attribute that no longer exists on the class. It no longer errors — it silently creates
an unused attribute — so that line is stale and should go in Phase 6.

## 6. One flaw found, kept, and flagged — this one needs a decision, not just a note

`sma_10` and `sma_20` are raw **price levels**. This is not a leak — it is a design
flaw — and your brief was to fix leaks, not redesign the feature set, so I kept them
behind a flag (`include_price_levels`, default `True`) rather than changing them
unilaterally. But I under-stated the severity in framing it as an interpretability
concern:

**Under fold-scoped scaling this breaks Phase 4 mechanically.** A training fold covering
2018-2020 sees BTC's SMA near \$8k and fits the scaler to that range. The 2021 test fold
sees \$50k. Every scaled test observation then lands far outside the training
distribution, so out-of-sample points get assigned to whichever cluster sits at the
extreme — regardless of actual market state. The walk-forward would produce numbers, and
they would be meaningless.

The stationary `price_sma14_ratio` already carries the same information scale-free, so
nothing is lost by dropping the levels.

**This needs to be `False` before Phase 4, and should be `False` for Phase 3 so the two
phases are consistent.** It is a one-flag change; I have left the default at `True`
pending your word because it is a feature-set decision you reserved.

## 7. Carry-forwards recorded for Phase 6

1. **MATIC→POL splice.** The 6-asset tier **excludes MATIC**, so the splice exists in
   the cache but not in the analysis. This is a *stronger* caveat than originally
   anticipated, not a weaker one: the Data Limitations section should state that a
   synthetic splice was created and then say the analysis universe does not touch it.
2. **CHANGELOG.md rewrite.** Currently celebrates EV-based allocation and the 7-regime
   design as features and still lists the C++ system. To be replaced with a v2.0.0
   entry covering: removed leak-contaminated EV allocation, removed C++ execution,
   replaced unprincipled 7-regime count with principled selection, rebuilt around
   walk-forward validation.
3. **`Overfitting_Diagnosis.md` line citations** into `expected_value_analyzer.py` now
   resolve against git history rather than the tree.
