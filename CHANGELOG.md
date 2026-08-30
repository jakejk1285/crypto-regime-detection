# Changelog

## [2.0.0] - 2026-08-27 — Honest rebuild

A conversion from a trading project reporting a return figure into a validation study
reporting a negative result. **Every performance number in the 1.x entries below is
invalid** and is retained only as the control condition for the audit.

### The finding

Regime detection on crypto recovers economically real market states — Crisis captures
all four major crashes (COVID, China ban, Terra/LUNA, FTX); Alt-Season peaks Feb 2021
and Nov 2024 — but **converting that description into out-of-sample tradeable edge
fails every significance test applied**. The apparent Sharpe advantage is deleveraging,
not skill.

### Removed

- **C++ execution system** (`paper_trading/`, 2.7 MB, 72 files) and `shared_regime_data/`.
  Execution infrastructure is irrelevant to a research project.
- **`regime_scheduler.py`** (647 lines) — 15-minute production refit loop, same rationale.
- **Expected-Value allocation system** (`expected_value_analyzer.py`, 619 lines, and all
  EV machinery in the strategy). It was never reachable: `initialize_ev_system()` was
  never called, so `ev_initialized` stayed `False` and the filter branch was dead code
  regardless of its config flag.
- **Dead allocation ladder entries.** The regime given the largest allocation (40% of
  capital, documented as "highest EV, 2.230R") mapped to `WAIT_AND_SEE` and never traded.
- **Unreachable stop-loss and risk/reward tables**, keyed on strategy strings the regime
  mapper never emitted. Every regime except two silently fell through to defaults.
- **`PROJECT_SUMMARY.md`** — a stray two-character file.

### Fixed

- **Full-sample statistics leak.** `persistence`, `avg_duration` and
  `frequency_percentage` were computed over the whole dataset and broadcast onto every
  row, then used in trading decisions. Now expanding-window, using only data available
  up to each point. `avg_duration` averages only spells completed *strictly before* t;
  the old code counted the still-running spell. Measured leak magnitude: up to 0.56
  (persistence), 10.0 days (duration), 98.9 pp (frequency).
- **Unreproducible data window.** The CoinGecko collector's `days=365` returned a
  trailing window anchored to the request date, ignoring the requested start/end. The
  backtest printed "2024-01-01 to 2025-01-01" while running 2024-09-28 → 2025-08-28 on
  4 of 15 configured symbols. Replaced with paginated Binance klines via `ccxt`.
- **Purge buffer.** Corrected from 30 to **60 days** — the true longest rolling feature
  window (`avg_pairwise_corr_60d`), verified arithmetically.
- **Non-stationary features.** Raw moving-average price levels removed: under fold-scoped
  scaling they place every test observation outside the training distribution.
- **Fold-unstable regime identity.** Rank-based cluster ordering replaced with
  reference-anchored Hungarian matching on raw-feature profiles.

### Added

- **Mechanical guards, as methodology.** `test_causality.py` rebuilds features on
  truncated prefixes and requires exact equality with full-series values (differences are
  0.000e+00 across 5 cut points). `test_quarantine.py` AST-parses pipeline modules and
  fails on any import from the legacy `backtest_system/`, catching direct imports,
  `importlib`, and `sys.path` escapes.
- **21 relational features** — BTC volume-share dominance, alt-vs-BTC spreads, rotation
  breadth, cross-sectional dispersion, rolling pairwise correlation.
- **Principled model selection** — silhouette and gap statistic for k-means, BIC/AIC for
  GMM, across k=2..12, with the reasoning for why the criteria disagree.
- **9-fold expanding walk-forward** with per-fold fitting of scaler, PCA, clusterer and
  every trading rule.
- **Three independent significance tests** — block bootstrap, block-shuffle placebo
  (500 draws), and fold-concentration analysis.
- Reports: `Overfitting_Diagnosis.md`, `PHASE1`–`PHASE4_REPORT.md`.

### Changed

- Regime count: 7 → **4**, chosen for Two Sigma comparability rather than by an
  unexamined criterion. No selection criterion has a genuine interior optimum at 4, and
  the README says so.
- Data: 335 observations / 4 symbols → 2977 modeling observations / 6 assets, with
  **2250 stitched out-of-sample test days**.
- `backtest_system/` is retained deliberately and quarantined. It still leaks by design;
  that is the point of keeping it.

### Notes

- `Overfitting_Diagnosis.md` cites file:line locations inside
  `expected_value_analyzer.py`, deleted in this release. Those citations now resolve
  against git history rather than the working tree. The diagnosis is a record of what was
  found, so this is expected.
- The 1.x performance figures below are **not reproducible and were never comparable to
  each other**. Successive parameter revisions were applied to a single backtest window
  with a new headline number recorded each time — 23.45% / 1.87, then 23.71% / 1.85,
  13.63% / 0.345, 21.12% / 0.780, 28.90% / 0.798, 30.53% / 0.621, and finally
  54.35% / 1.437. That progression is itself the evidence of in-sample fitting, and is
  documented in `Overfitting_Diagnosis.md` item 4.

---

# Historical entries (superseded — numbers invalid)

The entries below are preserved unedited as the control condition. They describe the
EV-based allocation and 7-regime design as features; the audit found the EV system was
unreachable dead code and the 7-regime count unjustified.

## [1.2.0] - 2025-01-XX - Performance Optimization

### Added
- **Performance-Optimized Trading Strategy** (`backtest_trading_strategy.py`)
  - Strong risk-adjusted performance (0.621 Sharpe ratio achievement)
  - Optimized portfolio returns (30.53% total return)
  - Data-driven regime selection based on Expected Value analysis
  - Professional risk controls supporting profitable trades
  - Advanced position sizing with regime optimization

- **Strategy Comparison Framework** (`strategy_comparison.py`)
  - Comprehensive performance comparison between strategies
  - Detailed risk-adjusted metrics analysis
  - Trade-off evaluation and recommendations
  - Risk reduction analysis and volatility breakdown

- **Enhanced Testing Suite** (`test_enhanced_strategy.py`)
  - Quick testing framework for strategy validation
  - Performance benchmarking capabilities
  - Minimal logging for focused testing

### Enhanced
- **README.md**
  - Added performance-optimized strategy results and analysis
  - Updated strategy features with optimization details
  - Improved performance metrics documentation
  - Added comprehensive return calculations

- **Trading Strategy** (`backtest_trading_strategy.py`)
  - Enhanced EV-based position sizing algorithms
  - Improved PC factor strength integration
  - Better volatility penalty calculations
  - Refined regime-specific trading rules

### Performance Results
- **Total Return**: 30.53% (strong absolute performance)
- **Sharpe Ratio**: 0.621 (solid risk-adjusted performance)
- **Annualized Volatility**: 29.52% (controlled risk profile)
- **Max Drawdown**: 17.23% (professional drawdown management)
- **Win Rate**: 51.7% (consistent performance)
- **Final Value**: $130,535 (strong capital appreciation)

### Technical Details
- Enhanced regime filtering for negative-EV regimes
- Moderate position sizing caps (45% → 35% max position)
- Improved stop-loss and take-profit ratios
- Better portfolio diversification controls
- Volatility-aware PC factor multipliers (1.5x penalty multiplier)

---

## [1.1.0] - 2025-01-XX - Enhanced Strategy Implementation

### Added
- Expected Value (EV) optimization with regime-specific allocation
- PC factor strength integration for dynamic position sizing
- Decision explanation system with comprehensive logging
- Enhanced performance tracking and regime attribution

### Performance Results
- **Total Return**: 30.53% (improved from 21.12%)
- **Sharpe Ratio**: 0.621
- **Win Rate**: 51.7%
- **Total Trades**: 118

---

## [1.0.0] - 2024-XX-XX - Initial Release

### Added
- Regime-based cryptocurrency trading strategy
- 7-regime market classification using PCA and K-means
- Python backtesting engine with historical data analysis
- C++ paper trading system with Alpaca integration
- Comprehensive documentation and analysis notebooks