# Overfitting Diagnosis — Read-Only Findings

Diagnostic pass only. No code was modified and no pipeline was executed (running
`backtest_engine.py` or `crypto_regime_analysis.py` would overwrite the cache
pickles and `crypto_cluster_pca/data/*.csv` that this report cites). All counts and
date ranges below were computed directly from the data files with pandas.

**Provenance legend used in the tables**

- **(a)** hardcoded fixed default, chosen independently of backtest results
- **(b)** derived from backtest-observed performance (realized P&L / win rate / EV) —
  whether computed at runtime or transcribed by hand into a literal
- **(c)** unclear from the code alone

A note on how (b) was assigned without relying on comments or the README: a value is
called (b) only when the *code structure itself* carries the evidence — the seven
per-regime literals are ordered strictly monotonically by the seven per-regime EV
figures, and those exact EV figures appear as return values of
`get_regime_ev_estimate()` / `get_regime_win_rate()`, which are code, not comments.
Where only a comment asserts backtest derivation, the row is (c) and the comment is
quoted as context.

---

## Preliminary: there are three parameter sets, not one

Any provenance answer has to name which implementation it is about.

| Implementation | File | On the reported-results path? |
|---|---|---|
| `RegimeBasedTradingStrategy` | `crypto_cluster_pca/backtest_system/backtest_trading_strategy.py` | **Yes** — this is what produced 54.35% / 1.437 |
| `SharpeOptimizedStrategy` (subclass, overrides sizing, thresholds, SL/TP, regime filter) | `crypto_cluster_pca/backtest_system/sharpe_optimized_strategy.py` | No — not instantiated anywhere on the current results path |
| `ResearchBasedTradingStrategy` (live paper trading) | `paper_trading/src/trading_strategy.cpp` | No — separate C++ implementation with a *third*, different parameter set |

The notebook that produces the current reported numbers
(`backtest_system/backtest_analysis.ipynb`, cell 2) constructs
`RegimeBasedBacktester(...)`, which instantiates the **base**
`RegimeBasedTradingStrategy` (`backtest_engine.py:35`). `SharpeOptimizedStrategy` is
only instantiated inside its own `test_sharpe_optimized_strategy()` function
(`sharpe_optimized_strategy.py:363`), which nothing on the reported path calls.

### The `USE_EV_FILTER = False` line does not exist in the code

`grep` over all `.py`, `.cpp`, `.h`, `.ipynb` finds `USE_EV_FILTER` **only in
`README.md:241`**, inside a block captioned `# backtest_engine.py`. There is no such
constant in `backtest_engine.py` or anywhere else. What actually exists:

- `backtest_trading_strategy.py:52` — constructor default `use_ev_filter: bool = True`
- `backtest_engine.py:35` — constructs the strategy **without** passing the flag, so it defaults to `True`
- `backtest_analysis.ipynb` cell 2 — sets `backtester.trading_strategy.use_ev_filter = False` after construction

So the EV filter is off on the reported path, but by a line in a notebook, not by the
config constant the README shows. It is in fact inert regardless of that line: the
guard at `backtest_trading_strategy.py:675` is `if self.use_ev_filter and
self.ev_initialized`, and `ev_initialized` is set `True` only inside
`initialize_ev_system()` (`:435`) — which `grep` finds is **never called**, from any
`.py` file or from the notebook. So `ev_initialized` is `False` for the entire run and
the `expected_value_analyzer.should_take_trade()` branch is unreachable whatever
`use_ev_filter` is set to. The README's "entry thresholds adjusted per regime based on
EV data" therefore does *not* describe a live EV filter — it describes the hardcoded
ladders in item 1, which are a different mechanism.

---

## 1. Parameter inventory with provenance

### 1a. Reported-results path — `backtest_trading_strategy.py`

| Parameter | Current value | Prov. | Code location |
|---|---|---|---|
| Regime base allocation (inline, used for sizing) | R3 0.40, R5 0.30, R0 0.25, R2 0.15, R1 0.08, R4 0.03, R6 0.01 | **(b)** | `backtest_trading_strategy.py:158-174` |
| Regime base allocation (duplicate table, used for logging) | same seven values | **(b)** | `backtest_trading_strategy.py:841-849` |
| Regime entry threshold on coin score | R3 0.02, R5 0.06, R0 0.08, R2 0.25, R1 0.45, R4 0.65, R6 0.85 | **(b)** | `backtest_trading_strategy.py:382-398` |
| Per-regime EV in R (returned as data) | 3:2.230, 5:0.761, 0:0.494, 2:0.166, 1:0.024, 4:-0.206, 6:-0.901 | **(b)** | `backtest_trading_strategy.py:866-869` |
| Per-regime win rate % (returned as data) | 3:75, 5:60, 0:54, 2:58, 1:44, 4:40, 6:0 | **(b)** | `backtest_trading_strategy.py:876-878` |
| Per-regime EV rank string | 3 "1st (BEST)" … 6 "7th (WORST)" | **(b)** | `backtest_trading_strategy.py:856-859` |
| Base stop-loss % by strategy | CRISIS .06, BASELINE .04, STABLE_GROWTH .04, MOMENTUM .05, BREAKOUT .06, DEFENSIVE .08, EXTREME_VOLATILITY .10; default .05 | **(c)** | `backtest_trading_strategy.py:318-328` |
| Risk/reward ratios by strategy | BASELINE 3.0, BREAKOUT 2.5, STABLE_GROWTH 2.0, MOMENTUM 2.0, CRISIS 1.5, DEFENSIVE 1.0, EXTREME_VOLATILITY 1.0; default 2.0 | **(c)** | `backtest_trading_strategy.py:344-354` |
| SL volatility / persistence / stress adjusters | `max(0.7,min(1.5,1+(pc2-1)*0.2))`, `max(0.8,min(1.2,2-persist))`, `max(1.0,min(1.3,1+stress*0.3))` | **(c)** | `backtest_trading_strategy.py:331-337` |
| SL clamp | 2%–12% | (a) | `backtest_trading_strategy.py:341` |
| Take-profit duration bonus | ×1.2 when `avg_duration > 10` | **(c)** | `backtest_trading_strategy.py:358-359` |
| PC1 sizing multiplier | >2.0 → `min(1.4, 1+(s-2.0)*0.2)`; >1.5 → `min(1.2, 1+(s-1.5)*0.4)`; else `max(0.7, 1-(1.5-s)*0.2)` | **(c)** | `backtest_trading_strategy.py:186-193` |
| PC2 sizing adjustment | >2.5 → `max(0.6, 1-(s-2.5)*0.15)`; <0.8 → `min(1.3, 1+(0.8-s)*0.4)`; else 1.0 | **(c)** | `backtest_trading_strategy.py:196-203` |
| PC3 sizing bonus | >1.5 → `min(1.15, 1+(s-1.5)*0.3)`; else 1.0 | **(c)** | `backtest_trading_strategy.py:206-210` |
| Combined PC multiplier clamp | 0.5–1.6 | (a) | `backtest_trading_strategy.py:214` |
| Persistence bonus | >0.8 → 1.3; >0.6 → 1.1; <0.3 → 0.7; else 1.0 | **(c)** | `backtest_trading_strategy.py:218-225`, dup at `:885-892` |
| Concentration limit | `max(0.7, 0.95 - exposure)` | **(c)** | `backtest_trading_strategy.py:232` |
| Final position-size clamp | 1%–45% | **(c)** | `backtest_trading_strategy.py:240` |
| Coin-score base weights | `persistence*0.35 + freq_pct*0.25`, `+0.4` floor | **(c)** | `backtest_trading_strategy.py:112-114` |
| Coin-score PC bonuses by symbol group | 0.1 / 0.05 / 0.15 / 0.25 / −0.10 | **(c)** | `backtest_trading_strategy.py:119-136` |
| Volatility penalty on coin score | `abs(pc2)*0.02` when `abs(pc2) > 2.0` | **(c)** | `backtest_trading_strategy.py:139` |
| Threshold stress / persistence penalties | +0.10 if stress>0.7; +0.15 if persist<0.25; +0.1 if stress>0.8 | **(c)** | `backtest_trading_strategy.py:401-410` |
| Max positions by strategy | CRISIS 3, WAIT_AND_SEE 2, MOMENTUM/BREAKOUT 8, else 6 | **(c)** | `backtest_trading_strategy.py:473-480` |
| Trailing stop activation / distance | activate at +10% profit, trail 5% below high | **(c)** | `backtest_trading_strategy.py:997-1004` |
| Min trade value | $100 | (a) | `backtest_trading_strategy.py:82` |
| Max position count | 5 | (a) | `backtest_trading_strategy.py:83` |
| Max portfolio risk | 0.15 | **(c)** | `backtest_trading_strategy.py:84` |
| Max single-position risk | 0.05 | **(c)** | `backtest_trading_strategy.py:85` |
| Max correlation exposure | 0.50 | **(c)** | `backtest_trading_strategy.py:86` |
| Correlation groups | major/defi/alt/legacy/exchange | (a) | `backtest_trading_strategy.py:276-282` |
| `min_ev_threshold` / `max_risk_per_trade` | 0.1 / 0.02 | **(c)** | `backtest_trading_strategy.py:60` |
| `should_trade_regime` gates | persistence>0.4, stress<0.9; hard block R1 when stress>0.85 | **(c)** | `backtest_trading_strategy.py:525-533` |
| Regime→strategy map + `risk_mult` | R0 1.3, R1 1.1, R2 1.0, R3 0.3, R4 0.8, R5 1.2, R6 0.2 | **(c)** | `backtest_data_manager.py:480-488` |
| Global `should_trade` gates | persistence>0.5, stress<0.7, avg_duration>3.0 | **(c)** | `backtest_data_manager.py:523-528` |
| Risk-free rate | 0.0395 | (a) — sourced to 10Y Treasury | `backtest_engine.py:27` |
| `random_state` for PCA and k-means | 42 | (a) | `backtest_data_manager.py:408, 416`; `crypto_regime_analysis.py:908` |
| `rebalance_frequency` | `'D'` (D/W/M enumerated) | **(c)** | `backtest_engine.py:80-95`; passed as `'D'` in `backtest_analysis.ipynb` cell 2 |

Note the (c) rows are (c) for a specific reason, not as a hedge: many of them carry
comments like "increased from 0.3", "lowered from 0.3", "relaxed", "increased for
better diversification" (e.g. `:86`, `:114`, `:405-410`, `:474-480`). Those comments
record that the value was *changed*, but the code contains nothing that says what the
change was optimized against. Compare with the item-4 commit history, which shows a
headline backtest number moving with each such revision — that is suggestive, but it
is evidence about commits, not about the code, so those rows stay (c).

### Why the seven-value ladders are (b) and not (c)

The chain is fully visible in code:

1. `expected_value_analyzer.py:207-318` `_simulate_trades_from_cache()` reads
   `backtest_data_cache/crypto_data_*.pkl` and `historical_regimes_*.pkl` — the same
   files the backtest itself consumes — and generates trades by entering at each
   timestamp and exiting a fixed 5 bars later (`exit_idx = min(idx + 5, ...)`,
   line 252).
2. `expected_value_analyzer.py:116-119` groups those into per-regime `EVMetrics`;
   `:389` computes `expected_value_r = win_rate*avg_win_r - loss_rate*|avg_loss_r|`.
3. Those per-regime EV and win-rate figures appear verbatim as literals in
   `backtest_trading_strategy.py:866-878`.
4. The allocation ladder (`:158-174`) and threshold ladder (`:382-398`) are ordered
   strictly monotonically in exactly that EV ranking — allocation descending, entry
   threshold ascending, across all seven regimes with no exception.

So the parameters that decide *how much* to trade and *when* to trade were set from
per-regime performance measured by the same pipeline, over effectively the same
trailing window, that the backtest then scored. ("Effectively" rather than "identical":
`backtest_trading_strategy.py` was last modified in `2188844` on 2025-08-27, while the
cache pickles the reported run consumed are dated 2025-08-28, so the constants were
transcribed from a cache built a day earlier than the one finally scored. Same source,
same construction, one day apart.) That is in-sample fitting, established from code
alone.

Two accuracy points about those numbers:

- The EV figures come from the analyzer's **synthetic** fixed-5-day-hold trades, not
  from the strategy's own realized trades. The README's own per-regime breakdown
  (regime 0: 80.0% WR, regime 2: 57.7%, regime 1: 27.3%) does not match the hardcoded
  table (0:54, 2:58, 1:44). The hardcoded win rates are therefore not the realized
  backtest win rates; they are the simulator's.
- `get_regime_win_rate()` and `get_regime_ev_estimate()` are consumed only by the
  decision-logging paths (`:551-553`, `:656-657`, `:772-774`). They shape reporting,
  not execution. The *execution*-affecting (b) parameters are the allocation and
  threshold ladders.

### 1b. `sharpe_optimized_strategy.py` (off the reported path)

Every value here is written as a delta from the base strategy's value — a second
tuning pass over the same single backtest window.

| Parameter | Value | Prov. | Location |
|---|---|---|---|
| Regime base allocation | R3 0.35, R5 0.25, R0 0.20, R2 0.08, R1 0.05, R4 0.02, R6 0.005 | **(b)** | `sharpe_optimized_strategy.py:60-75` |
| Regime entry threshold | R3 0.08, R5 0.12, R0 0.15, R2 0.40, R1 0.55, R4 0.70, R6 0.85 | **(b)** | `sharpe_optimized_strategy.py:182-197` |
| Risk caps | single 0.04, portfolio 0.12, correlation 0.40 | **(c)** | `sharpe_optimized_strategy.py:31-33` |
| Position clamp | 0.008–0.35 | **(c)** | `sharpe_optimized_strategy.py:39-40, 163` |
| Base stop-loss / RR tables | .05/.035/.035/.045/.055/.07/.09; RR 2.8/2.3/1.9/1.9/1.5/1.2/1.0 | **(c)** | `sharpe_optimized_strategy.py:269-302` |
| Regime hard blocks | pc2>4.0 block; R4/R6 blocked when stress>0.6; R6 blocked when pc2>2.5 | **(b)** — gates are keyed on the negative-EV regime ids | `sharpe_optimized_strategy.py:324-341` |
| Diversification bonus | ×1.1 at ≥3 positions, ×1.15 at ≥5 | **(c)** | `sharpe_optimized_strategy.py:151-156` |

### 1c. `paper_trading/src/trading_strategy.cpp` (live path)

| Parameter | Value | Prov. | Location |
|---|---|---|---|
| Regime allocation, keyed by **strategy name** not regime id | STABLE_GROWTH 0.20, MOMENTUM 0.15, BALANCED 0.15, BREAKOUT 0.15, DEFENSIVE 0.10, CONSERVATIVE 0.10, EXTREME_VOLATILITY 0.05, CRISIS 0.0, WAIT_AND_SEE 0.0 | **(c)** | `trading_strategy.cpp:103-117` |
| Volatility adjustment | `max(0.5, 1-(pc2-1)*0.1)` when pc2>1 | **(c)** | `trading_strategy.cpp:128` |
| Persistence bonus | `1 + (persistence-0.5)*0.4` | **(c)** | `trading_strategy.cpp:131` |
| Base stop-loss / RR tables | same numbers as Python base, plus a `CONSERVATIVE` entry Python lacks | **(c)** | `trading_strategy.cpp:545-556, 576-586` |
| Regime filter | persistence>0.4; R1 blocked when stress>0.85 | **(c)** | `trading_strategy.cpp:498-511` |

The C++ allocation scheme is structurally different from Python's: nine strategy-name
buckets spanning 0.0–0.20, versus seven regime-id buckets spanning 0.01–0.40. They are
not the same strategy despite the "exact replication" docstring at
`backtest_trading_strategy.py:48`. Separately, the README's "C++ Configuration" block
(`README.md:246-252`) lists four `constexpr` constants — `MIN_TRADE_VALUE`,
`MAX_POSITIONS`, `MAX_PORTFOLIO_RISK`, `MAX_SINGLE_POSITION_RISK` — that do not exist
anywhere in `paper_trading/`.

---

## 2. Is regime detection clean?

**Free of forward-return leakage: confirmed.** No forward return, P&L, or backtest
metric enters PCA or k-means anywhere. Features are returns, RSI, SMAs, rolling
volatility, VaR, momentum, breadth, correlation — all backward-looking
(`backtest_data_manager.py:270-336`; `crypto_regime_analysis.py` feature stage). The
clustering call sites (`backtest_data_manager.py:404-417`,
`crypto_regime_analysis.py:826-831, 908-909`) receive only the scaled feature matrix.

**But "clean" is not the same as "usable out-of-sample."** Three separate issues, all
established from code:

**(i) Whole-sample fitting, no refit inside the backtest.** `generate_historical_regimes()`
fits the `StandardScaler` (`:404`), the 25-component PCA (`:409`), and the 7-cluster
k-means (`:417`) **once, over the entire backtest window**, then pickles the result.
`backtest_engine.py:60` loads that pickle and `run_backtest()` iterates it timestamp by
timestamp. Every regime label used on day 1 was produced by a model that saw the whole
year. There is no rolling/expanding refit anywhere in the backtest path. The scaler
means and PCA rotation are themselves whole-sample statistics — in-sample scaling
leakage even with zero forward returns.

**(ii) Per-regime statistics are computed over the full sample and then used as live inputs.**
`persistence`, `avg_duration`, and `frequency_percentage` are computed across all 335
observations (`backtest_data_manager.py:432-477`) and broadcast as a **constant** onto
every row of that regime (`:472-477`). Verified from the pickle — each regime has
exactly one distinct value for each:

| regime | persistence | avg_duration | frequency_% |
|---|---|---|---|
| 0 | 1.000 | 29.00 | 8.66 |
| 1 | 0.895 | 8.29 | 17.31 |
| 2 | 0.870 | 7.15 | 27.76 |
| 3 | 0.769 | 4.00 | 11.94 |
| 4 | 1.000 | 42.00 | 12.54 |
| 5 | 0.962 | 17.67 | 15.82 |
| 6 | 1.000 | 20.00 | 5.97 |

These feed `should_trade` (`:523-528`), the coin-score base (`:112`), the persistence
sizing bonus (`:218-225`), the SL persistence adjuster (`:334`), and the take-profit
duration bonus (`:358`). This is direct look-ahead: on day 1 the strategy already knows
how persistent and how frequent each regime will turn out to be over the full year.
This is downstream of clustering, but it is inside the regime-data pipeline, not the
trading rules.

**(iii) Hyperparameter provenance is unsupervised, but inconsistent.**
`crypto_cluster_pca/data/clustering_analysis_metadata.json` records the k selection
explicitly:

```
recommended_clusters: 7
optimal_clusters_by_method: silhouette 2, calinski_harabasz 10,
                            davies_bouldin 12, bic 7, aic 11, elbow 7
```

k=7 was chosen where BIC and elbow agreed. Every criterion listed is unsupervised —
none is a backtest metric. Likewise `pca_analysis_summary.json` records
`optimal_components: 25` at `variance_captured: 0.8968` from 117 cleaned features, a
variance-threshold choice. So the answer to "was k or the component count tuned by
checking backtest results" is **no, as recorded**. It was, however, a pick among six
criteria that disagreed by a factor of six (2 to 12), and
`research/04_clustering_analysis.ipynb` cell 4 still hardcodes `final_k = 5`.

**Fit once or refit — the answer differs by path:**

| Path | Behavior |
|---|---|
| Backtest (`backtest_data_manager.py:368-545`) | Fit **once** over the whole window, pickled, replayed |
| Research (`research/*.ipynb`, `data/*.csv`) | Fit once, exported to CSV, 2025-07-09/10 |
| Live production (`crypto_regime_analysis.py:1519-1524` via `regime_scheduler.py`) | **Refit from scratch every 15 minutes** on the full then-available history |

**A consequence of the refit that is worth stating plainly:** k-means cluster ids are
arbitrary and unstable across fits, but `regime_strategy_mapping` — which assigns
regime 3 → `WAIT_AND_SEE` and regime 6 → `EXTREME_VOLATILITY` — is hardcoded
identically in `backtest_data_manager.py:480-488` and `crypto_regime_analysis.py:1178-1185`.
The two fits already disagree. From the data:

| regime id | research fit (280 obs) | backtest fit (335 obs) |
|---|---|---|
| 0 | 56 | 29 |
| 1 | 57 | 58 |
| 2 | 102 | 93 |
| 3 | **2** | **40** |
| 4 | 20 | 42 |
| 5 | 42 | 53 |
| 6 | **1** | **20** |

Worth being precise about the mechanism: within a fixed dataset the labels are stable
only because `random_state=42` is hardcoded (`backtest_data_manager.py:416`,
`crypto_regime_analysis.py:908`). That seed pins cluster identity to nothing more than
an initialization; change the seed, or fit on a different fold or window, and the same
integer ids attach to different clusters — silently, since nothing validates the
mapping. This is the coupling that makes the id→name→allocation table fragile under
any refit.

In the research fit, clusters 3 and 6 are 2-observation and 1-observation outlier
pockets — which is what "EXTREME_OUTLIER" and "EXTREME_VOLATILITY" describe. In the
backtest's independent refit they hold 40 and 20 observations and mean something else
entirely, yet inherit the same names, the same `risk_mult`, and the same 0.40 / 0.01
allocations. Nothing in the code checks that a cluster id still denotes the same thing
after a refit.

---

## 3. Data available for a walk-forward redesign

### The backtest does not run on the dates it reports

The cache is named `crypto_data_20240101_20250101.pkl` and the engine prints
`Period: 2024-01-01 to 2025-01-01`. The actual contents:

| | value |
|---|---|
| Price data range | **2024-08-29 → 2025-08-28** |
| Rows per symbol | 366 |
| Symbols present | **BTCUSD, ETHUSD, ADAUSD, SOLUSD — 4 of 15** |
| Regime observations after feature NaN-drop | **335** |
| Regime index range | **2024-09-28 → 2025-08-28** |
| Features built | 47 |

Cause is visible in code: `fetch_coin_data_coingecko()` sends only
`days=min((end-start).days, 365)` (`backtest_data_manager.py:81-89`) — CoinGecko
returns the trailing 365 days from the request date, so the window is fixed by *when
the cache was built* (2025-08-28), not by `start_date`/`end_date`. Those two arguments
affect only the filename and the printed banner. Separately, the no-API-key branch
(`:164-167`) restricts collection to 4 coins, while the strategy iterates all 15
(`backtest_trading_strategy.py:68-74`), so 11 symbols silently have no prices.

The notebook's own stored output confirms the run used this cache: "Processing 4
cryptocurrencies", "Created 47 features, 335 observations", "Total trading
timestamps: 335" — printed directly beneath "Period: 2024-01-01 to 2025-01-01".

A consequence: the 54.35% baseline is reproducible only for as long as those two
pickle files survive. Both loaders short-circuit on `force_refresh=False`
(`backtest_data_manager.py:141-144, 374-377`), so deleting them, or passing
`force_refresh=True` once, silently re-anchors the evaluation window to the new
trailing 365 days — no error, no warning, a different dataset under the same filename.

### Regime observation counts — the backtest fit (what the strategy actually saw)

335 daily observations, 2024-09-28 → 2025-08-28.

| id | name | strategy | n | % of sample | avg_duration | episodes implied |
|---|---|---|---|---|---|---|
| 0 | STABLE_GROWTH | STABLE_GROWTH | 29 | 8.66% | 29.0 | 1 |
| 1 | MODERATE_MOMENTUM | MOMENTUM | 58 | 17.31% | 8.29 | 7 |
| 2 | BASELINE_MARKET | BALANCED | 93 | 27.76% | 7.15 | 13 |
| 3 | EXTREME_OUTLIER | WAIT_AND_SEE | 40 | 11.94% | 4.00 | 10 |
| 4 | DEFENSIVE_STABLE | CONSERVATIVE | 42 | 12.54% | 42.0 | 1 |
| 5 | BREAKOUT_MOMENTUM | MOMENTUM | 53 | 15.82% | 17.67 | 3 |
| 6 | EXTREME_VOLATILITY | WAIT_AND_SEE | **20** | 5.97% | 20.0 | **1** |

### Regime observation counts — the research fit (`data/historical_regime_classifications.csv`)

280 observations, 2024-10-01 → 2025-07-02.

| id | name | n | % |
|---|---|---|---|
| 0 | STABLE_GROWTH | 56 | 20.00% |
| 1 | MODERATE_MOMENTUM | 57 | 20.36% |
| 2 | BASELINE_MARKET | 102 | 36.43% |
| 3 | EXTREME_OUTLIER | **2** | 0.71% |
| 4 | DEFENSIVE_STABLE | 20 | 7.14% |
| 5 | BREAKOUT_MOMENTUM | 42 | 15.00% |
| 6 | EXTREME_VOLATILITY | **1** | 0.36% |

### The number that matters for per-fold fitting

The relevant unit is not observations but **independent regime episodes**, since
observations within one contiguous spell are near-perfectly autocorrelated. In the
backtest fit, from `n / avg_duration`:

- Regime 0: 29 observations in **1 episode**
- Regime 4: 42 observations in **1 episode**
- Regime 6 (EXTREME_VOLATILITY): 20 observations in **1 episode**
- Regime 5: 53 observations in 3 episodes
- Regime 1: 7 episodes, Regime 3: 10 episodes, Regime 2: 13 episodes

Three of the seven regimes occur exactly once in the entire dataset. The full sample is
335 daily observations, or roughly 11 months.

### The `WAIT_AND_SEE` regimes are never traded at all

Regimes 3 and 6 both map to `strategy = "WAIT_AND_SEE"`
(`backtest_data_manager.py:484, 487`). Verified from the pickle, `should_trade` is
`False` for every row of regimes 3 and 6 and `True` for all others (275 True / 60 False).
Three separate gates then block them: `should_trade_regime()` returns `False` on
`WAIT_AND_SEE` (`backtest_trading_strategy.py:521-522`); `calculate_position_size()`
forces `base_percent = 0.0` (`:177-178`); and `should_trade` is `False`.

Consequence: **regime 3 — the regime given the largest allocation (0.40) and the lowest
entry threshold (0.02) on the strength of being "highest EV, 2.230R" — never trades.**
Nor does regime 6. The README's own per-regime breakdown (`README.md:224-230`) lists
trades for regimes 0, 5, 2, 4, 1 only, which corroborates this. Whatever the 0.40 and
0.01 allocations were fitted to, they contribute nothing to the reported 54.35%.

So the tradeable universe is regimes 0, 1, 2, 4, 5: **275 observations across 25
episodes**, and the two largest-allocation entries in the fitted ladder are dead.

---

## 4. The README self-contradiction

Both figures are real outputs of this repo. Neither pair is from a "different date
range" in the sense of a deliberately chosen window — every run in this history used
whatever trailing-365-day window CoinGecko returned on the day its cache was built.

Full history of the README's headline block:

| commit | date | Total Return | Sharpe | Max DD | Win Rate | Trades |
|---|---|---|---|---|---|---|
| `6eb112a` initial commit | 2025-08-19 | **23.45%** | **1.87** | 8.2% | 84% | 156 |
| `39a54f2` "risk-free rate 3.95%" | 2025-08-27 | 23.71% | 1.85 | 8.21% | 84.2% | 155 |
| `45946e1` "profitable research-based strategy" | 2025-08-27 | 13.63% | 0.345 | 11.26% | 53.0% | 132 |
| `aa742bd` "further optimize based on performance analysis" | 2025-08-27 | 21.12% | 0.780 | 8.80% | 54.1% | 122 |
| `2188844` "Sharpe optimization +28.5%" | 2025-08-27 | 28.90% | 0.798 | 15.12% | 53.2% | 106 |
| `cd8a290`, `26e4f70` | 2025-08-27 | 28.90% | 0.798 | 15.12% | 53.2% | 106 |
| `9c1a205` "update system configuration" | 2025-08-29 | **54.35%** | **1.437** | 13.27% | 53.3% | 137 |
| `0b1d22c` (HEAD) | 2025-08-29 | 54.35% | 1.437 | 13.27% | 53.3% | 137 |

**Answer to the question asked:**

- **23.45% / 1.87 is stale.** It is the initial-commit figure from 2025-08-19,
  superseded within the repo eight days later. The GitHub "About" sidebar is stored on
  GitHub, not in the repository, so it was never updated when the README moved on. Its
  companion stats — 84% win rate over 156 trades — appear nowhere in the current code's
  output and correspond to a strategy version that no longer exists in the tree.
- **54.35% / 1.437 reflects the current code**, with two caveats. It is reproduced in
  the committed notebook outputs (`backtest_analysis.ipynb` cells 3 and 4), produced by
  the **base** `RegimeBasedTradingStrategy` with `use_ev_filter = False`, over the
  335-observation 2024-09-28 → 2025-08-28 cache, on 4 of the 15 configured symbols.
  It is *labeled* "2024-2025, 1-Year" and "Period: 2024-01-01 to 2025-01-01"; that
  labeling is wrong per item 3.

**On the 28.90% → 54.35% jump:** `git show --stat 9c1a205` shows it touched
`README.md`, `CHANGELOG.md`, `.gitignore`, `backtest_analysis.ipynb`,
`paper_trading/*`, and deleted three scripts — **no Python strategy file changed**.
`backtest_trading_strategy.py`, `sharpe_optimized_strategy.py`, and
`backtest_engine.py` were all last modified in `2188844` (2025-08-27) and
`backtest_data_manager.py` / `expected_value_analyzer.py` back at the initial commit.
So the near-doubling came with identical Python parameters. Two differences are
visible: the reported run switched from `SharpeOptimizedStrategy` (which owns the
28.90% / 0.798 figure — `sharpe_optimized_strategy.py:363`) back to the base strategy
via the notebook, and the cache pickles are dated 2025-08-28, one day before the
commit, i.e. rebuilt over a later trailing window. Which of the two dominates cannot be
determined from the repo without re-running, which this pass does not do.

**One further discrepancy in the same block:** `README.md:189` states Sharpe 1.437
while the notebook output for the same run prints `Risk-Free Rate: 0.04%` against a
configured 3.95%. That is a display-formatting mismatch in the notebook's summary cell,
not necessarily an error in the Sharpe figure itself, which
`calculate_sharpe_ratio()` computes with the passed 0.0395
(`backtest_trading_strategy.py:1051-1096`).

---

## 5. Other backtest-fitted or backtest-contingent findings not asked about

**5.1 — Whole per-regime SL/TP tables are unreachable dead code.** The strategy strings
actually emitted by `regime_strategy_mapping` are `STABLE_GROWTH`, `MOMENTUM`,
`BALANCED`, `WAIT_AND_SEE`, `CONSERVATIVE`. The `base_stop_loss` and
`risk_reward_ratios` dicts are keyed on `CRISIS`, `BASELINE`, `STABLE_GROWTH`,
`MOMENTUM`, `BREAKOUT`, `DEFENSIVE`, `EXTREME_VOLATILITY`. Verified against the pickle:

- Reachable keys: `STABLE_GROWTH`, `MOMENTUM` only
- Unreachable: `BASELINE`, `BREAKOUT`, `CRISIS`, `DEFENSIVE`, `EXTREME_VOLATILITY`

So regime 2 (`BALANCED`, 93 obs — the largest tradeable regime) and regime 4
(`CONSERVATIVE`, 42 obs) both fall through to the defaults `base_stop = 0.05` and
`risk_reward = 2.0`. The advertised `BASELINE: 0.04 / 3.0 R:R` "for best regime" never
executes. Same key mismatch in the coin-scorer (`:126-136`, `CRISIS` branch
unreachable) and `get_max_positions()` (`:473`, `CRISIS` unreachable). The C++ file
uses `BALANCED` and adds `CONSERVATIVE` (`trading_strategy.cpp:548, 553, 578, 585`), so
it does not have this bug — a further divergence from the "exact replication" claim.

**5.2 — The EV simulator's exits look forward.** `_simulate_trades_from_cache()`
(`expected_value_analyzer.py:207-318`) enters at bar `idx` and exits at `idx + 5`
unconditionally, then classifies the outcome. This is fine as a way to characterize a
regime post hoc, but the resulting per-regime EV was transcribed into live trading
parameters (item 1), which turns a hindsight statistic into a forward-looking rule.
Its stop-loss approximations (5% / 3% / 4%, `:261-266`) also do not match the stop
losses the strategy actually uses, so the R multiples are not the strategy's R
multiples.

**5.3 — The EV analyzer's cache can silently reuse stale metrics.** `analyze_historical_trades()`
returns cached `ev_metrics_cache.pkl` before looking at the trades it was passed
(`expected_value_analyzer.py:88-95`), and `load_from_backtest_cache()` accepts any
cache with `total_trades > 10` as valid (`:150-151`). No file currently exists in the
tree, so this did not affect the reported run, but the cache is keyed on nothing —
not the date range, not the parameter set.

**5.4 — Duplicated parameter tables that can drift.** The allocation ladder exists twice
(`backtest_trading_strategy.py:158-174` for execution, `:841-849` for logging), the
persistence bonus twice (`:218-225`, `:885-892`), and the whole PC1/PC2/PC3 multiplier
cascade twice (`:186-210` for execution, `:750-763` inside `build_entry_reasoning`).
They currently agree; nothing enforces it, and `SharpeOptimizedStrategy` overrides only
the execution copy — so under that strategy the logged reasoning reports the base
class's multipliers, not the ones actually applied.

**5.5 — Synthetic OHLC.** `high = price * 1.001` and `low = price * 0.999`, with
`volume = 1000000` as a constant placeholder (`backtest_data_manager.py:113-116`).
Fills, stops, and take-profits are all evaluated on `close` only
(`backtest_trading_strategy.py:1009-1018`), so intrabar stop and target hits are never
detected — a position that pierced its stop and recovered by the close is recorded as
never having stopped out. This affects the realized return and the win rate directly.
It is a backtest-realism issue rather than a fitted parameter, but it is upstream of
every performance number in item 4.

**5.6 — `USDTUSD` is scored and tradeable.** Tether appears in `crypto_coins`
(`:72`) and receives explicit score bonuses in the `STABLE_GROWTH`/`DEFENSIVE` and
`CRISIS` branches (`:128-134`). It is a dollar peg. It is absent from the 4-symbol
cache so it did not trade in the reported run, but it would if the API-key path were
used.

**5.6b — `getMinimumTradeThreshold()` has no return statement.**
`trading_strategy.cpp:153-156` declares `double getMinimumTradeThreshold(...)`, casts
its argument to void, and closes on a comment ("Minimum trade value of $100") with no
`return`. This reinforces item 1c: the README's "C++ Configuration" `constexpr` block
describes constants that do not exist, and the function that would have carried one of
them returns nothing.

**5.7 — Duplicate/unreachable code after `return`.** `process_regime_change()`'s actual
rebalancing body sits after the `return` in `get_regime_strategy_name()`
(`backtest_trading_strategy.py:583-594`) — the `coin_scores < 0.4` position-closing
logic, including that 0.4 literal, is unreachable. Whether that threshold was ever
active in an earlier version cannot be determined from the current code.

**5.8 — Three different values of k in the tree.** `k = 7` in both production paths
(`backtest_data_manager.py:416`, `crypto_regime_analysis.py:1524`), `k = 5` hardcoded
in `research/04_clustering_analysis.ipynb` cell 4, and `recommended_clusters: 7` with
silhouette preferring 2 in `clustering_analysis_metadata.json`. The research notebook
that ostensibly justifies the choice does not implement it.

---

## What could not be determined from the code

- Whether the (c) rows — the PC1/PC2/PC3 multiplier breakpoints, the SL/RR tables, the
  risk caps, the coin-score weights — were arrived at by observing backtest results.
  Many carry comments recording that they were *adjusted* ("increased from", "lowered
  from", "relaxed"), and the item-4 commit history shows the headline metric moving
  alongside such edits, but the code contains no artifact tying any specific value to a
  specific measured outcome. They are (c), not (a) and not (b).
- Whether the 28.90% → 54.35% jump came from the strategy-class switch, the rebuilt
  data cache, or both. Both changed; neither can be isolated without re-running, which
  this pass deliberately did not do.
- Whether `SharpeOptimizedStrategy` was ever the reported path at any commit. It is not
  instantiated on the current one.
