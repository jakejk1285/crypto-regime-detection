# Phase 1 — Data Acquisition Report

**Status: complete. Stopping here for review before Phase 2, per instruction.**

## What changed

The CoinGecko collector (`backtest_system/backtest_data_manager.py:55-214`) is replaced
by `crypto_cluster_pca/data_pipeline/binance_data.py`, which pulls daily klines from
Binance via `ccxt`.

Two bugs from `Overfitting_Diagnosis.md` are structurally prevented in the new loader:

1. **Trailing-window bug (diagnosis item 3).** The old request sent
   `days=min(n, 365)`, which returns the trailing 365 days from the *request date* and
   ignores `start_date`/`end_date` entirely. Binance caps klines at 1000 per call, so a
   naive `fetch_ohlcv` would reproduce the identical failure. The new loader paginates
   on `since` until the exchange stops advancing (`fetch_full_daily`, lines 74-108) and
   is verified below by a rows-vs-calendar-span assertion.
2. **Filename asserting a range the contents lack.** The old cache was named
   `crypto_data_20240101_20250101.pkl` while holding 2024-08-29 → 2025-08-28. The new
   cache derives every date from the returned payload and writes a `manifest.json`
   recording the fetch timestamp, source host, and per-symbol first/last dates.

The current UTC day is dropped from every series — it is a partial candle and would
inject a fractional-day return into every feature.

**Network note:** `api.binance.com` returns HTTP 451 from this location. The public
market-data mirror `data-api.binance.vision` serves the identical global-Binance klines
without the geo restriction, so all REST hosts are routed there. `binanceus` was
rejected as a fallback: its BTC history starts 2019-09-23 rather than 2017-08-17, which
would have discarded two years for no reason.

`backtest_data_cache/` (the two pickles behind the naive 54.35% result) is untouched, so
the "before" half of the Phase 6 narrative remains reproducible.

## Per-symbol daily history

All 14 symbols, longest history first. Every series ends **2026-08-26**.

| pair | asset | rows | first | last |
|---|---|---|---|---|
| BTC/USDT | bitcoin | 3297 | 2017-08-17 | 2026-08-26 |
| ETH/USDT | ethereum | 3297 | 2017-08-17 | 2026-08-26 |
| BNB/USDT | binancecoin | 3216 | 2017-11-06 | 2026-08-26 |
| LTC/USDT | litecoin | 3179 | 2017-12-13 | 2026-08-26 |
| ADA/USDT | cardano | 3054 | 2018-04-17 | 2026-08-26 |
| XRP/USDT | ripple | 3037 | 2018-05-04 | 2026-08-26 |
| LINK/USDT | chainlink | 2780 | 2019-01-16 | 2026-08-26 |
| MATIC/USDT (→POL) | matic-network | 2678 | 2019-04-26 | 2026-08-26 |
| ATOM/USDT | cosmos | 2677 | 2019-04-29 | 2026-08-26 |
| ALGO/USDT | algorand | 2623 | 2019-06-22 | 2026-08-26 |
| DOGE/USDT | dogecoin | 2610 | 2019-07-05 | 2026-08-26 |
| SOL/USDT | solana | 2207 | 2020-08-11 | 2026-08-26 |
| DOT/USDT | polkadot | 2200 | 2020-08-18 | 2026-08-26 |
| AVAX/USDT | avalanche-2 | 2165 | 2020-09-22 | 2026-08-26 |

### Continuity assertion (the check that proves pagination worked)

For each symbol, `rows` vs `(last - first).days + 1`:

**13 of 14 symbols have exactly zero missing days.** The single exception is
MATIC/USDT with 2 missing days, which is the rebrand gap described below. Had the
loader returned a trailing window, every symbol would show a first date near
2023-12 (1000 days back) rather than its true listing date.

## Universe substitutions

| original | resolution |
|---|---|
| `tether` (USDT) | **Dropped.** USDT is Binance's quote currency — no USDT/USDT pair exists. It is also a dollar peg: near-zero return variance would dominate `StandardScaler` and load onto a PCA component as pure noise. This also resolves diagnosis finding 5.6, where USDT was scored and tradeable in the old strategy. |
| `matic-network` | **Stitched MATIC→POL.** Polygon rebranded; Binance closed MATIC/USDT on 2024-09-10 and opened POL/USDT on 2024-09-13. The swap was 1:1 — observed seam ratio POL_first/MATIC_last = **1.083** across a 3-day gap, i.e. ordinary price drift, not a split — so the series are concatenated with no rescaling. The 2-day gap is left missing rather than filled. |

All other 13 assets have continuous `/USDT` spot pairs. No symbol was unavailable.

One interaction to flag before Phase 2 touches feature code: MATIC's 2 missing days
(2024-09-11/12) will propagate NaN through `pct_change(30)` and the rolling windows for
roughly 30 days afterward, and the pipeline's cross-symbol `dropna()` would then drop
those rows from *every* symbol. Small (~30 observations) but it needs a deliberate
decision rather than a silent drop — and it disappears entirely if the 6-asset tier is
chosen, since MATIC is not in it.

The universe chosen is the 15-asset set in `backtest_data_manager.py:31-47`, the
broadest of the four universes in the repo, minus tether → **14 assets**.

## The number that sizes Phase 4: common-history intersection

The feature pipeline `dropna()`s across all symbols, so **the binding constraint is the
intersection, not the per-symbol history**. Adding a late-listing asset truncates
everything. This is the tradeoff to decide before Phase 4 is built:

| symbols | added at this tier | common start | common obs | ≈ years |
|---|---|---|---|---|
| 2 | BTC, ETH | 2017-08-17 | **3297** | 9.0 |
| 6 | + BNB, LTC, ADA, XRP | 2018-05-04 | **3037** | 8.3 |
| 10 | + LINK, MATIC, ATOM, ALGO | 2019-06-22 | **2621** | 7.2 |
| 11 | + DOGE | 2019-07-05 | **2608** | 7.1 |
| 14 | + SOL, DOT, AVAX | 2020-09-22 | **2163** | 5.9 |

Note the shape: going from 6 to 14 assets costs **874 observations (2.4 years)** and
buys 8 more assets. Going from 6 to 10 costs only 416 observations. The three 2020
listings (SOL, DOT, AVAX) are what force the 2020-09-22 start.

## Total available for regime modeling, and resulting fold capacity

Longest rolling window in the existing feature set is `return_30d`, so **30 days** is
both the warmup loss and the purge buffer Phase 4 requires at each fold boundary.

Fold count is **not** a property of the data alone — it depends on two choices I have
not made yet: the initial training block and the test-fold length. Both are shown as
axes below rather than baked into a single number, each test fold preceded by a 30-day
purge:

| symbols | usable obs (after 30d warmup) | t0=500, 250d test | t0=500, 180d test | t0=750, 250d test | t0=750, 180d test |
|---|---|---|---|---|---|
| 2 | 3267 | 9 | 13 | 8 | 11 |
| 6 | 3007 | 8 | 11 | 8 | 10 |
| 10 | 2591 | 7 | 9 | 6 | 8 |
| 11 | 2578 | 7 | 9 | 6 | 8 |
| 14 | **2133** | **5** | **7** | **4** | **6** |

*(t0 = initial training block in days. A 500-day block is roughly two years — enough for
a k=4 model to have seen each regime several times; 750 is more conservative.)*

Every tier above is a **subset of what is already cached on disk** — all 14 symbols were
fetched. Choosing any tier requires no refetch.

## Comparison to the dataset behind the naive result

| | old (CoinGecko) | new (Binance, 14 assets) | new (Binance, 6 assets) |
|---|---|---|---|
| observations | 335 | **2163** (6.5×) | **3037** (9.1×) |
| symbols with data | 4 of 15 | 14 of 14 | 6 of 6 |
| range | 2024-09-28 → 2025-08-28 | 2020-09-22 → 2026-08-26 | 2018-05-04 → 2026-08-26 |
| reproducible window | no — trailing from fetch date | yes — fixed listing dates | yes |
| market cycles spanned | ~1 partial | 2020 bull, 2022 bear, 2023-26 | + 2018 bear, 2021 cycle |

## The open question for Phase 4 sizing

Observation count is the obvious axis, but for a **regime** model it is the weaker one.
The quantity that behaves like sample size is **distinct market regimes actually
visited**, and the diagnosis established exactly how that bites: three of seven regimes
occurred once in the old dataset, and a regime seen once cannot be fit and validated.
Adding a 2020-listed altcoin adds cross-sectional breadth to a day that is already in
the sample; extending the start date backward adds *days that are unlike any other day
in the sample*. Those are not equivalent purchases.

| tier | obs | folds (t0=500, 250d) | market cycles spanned |
|---|---|---|---|
| 6 assets | 3037 | 8 | 2018 bear, 2019 recovery, 2021 bull, 2022 bear, 2023-26 |
| 10 assets | 2621 | 7 | 2019 recovery, 2021 bull, 2022 bear, 2023-26 |
| 14 assets | 2163 | 5 | 2021 bull, 2022 bear, 2023-26 |

The 2018 bear market is a genuine crisis regime — a drawdown deeper and longer than
anything after it — and only the 6-asset tier contains it. The 14-asset tier starts in
September 2020 and therefore contains **exactly one** full bull-bear transition, which
is close to the failure mode the audit found.

**Recommendation: the 6-asset tier (BTC, ETH, BNB, LTC, ADA, XRP).** I initially
leaned to 10 for cross-sectional breadth, but the cycle-coverage argument outweighs it:
6 assets buys 8 folds instead of 5 and the only true crisis regime in the data, at the
cost of altcoins that are largely redundant with BTC/ETH beta anyway. If cross-sectional
breadth turns out to matter for the feature set in Phase 2, 10 assets is the fallback —
it costs the 2018 bear but keeps 7 folds.

This is the decision you asked to sanity-check. Nothing downstream is built until you
pick a tier.
