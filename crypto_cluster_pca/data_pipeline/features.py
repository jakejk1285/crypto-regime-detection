#!/usr/bin/env python3
"""
Causal feature engineering for regime detection.

Every feature here is a function of data at or before its own timestamp. The module
emits RAW features only — no standardization, no z-scoring, no centering. All scaling
is deferred to the fold-scoped pipeline in Phase 4, so the walk-forward scaler is the
only place normalization happens. This is the structural fix for Overfitting_Diagnosis.md
item 2(i), where a single StandardScaler was fit across the whole backtest window.

Causality is enforced mechanically, not by inspection: `test_causality.py` rebuilds the
features on truncated prefixes of the data and requires the result to match the
full-series features row for row. Any full-window statistic — a mean, a std, a
correlation over the whole sample — changes when the tail is removed and fails that test.

Feature groups
--------------
per-asset   : returns, RSI, moving averages, volatility, VaR, momentum
              (carried over unchanged from backtest_data_manager.create_historical_regime_features)
market-wide : momentum, breadth, BTC/ETH correlation (carried over unchanged)
relational  : BTC volume-share dominance, alt-vs-BTC return spreads, cross-sectional
              return dispersion, average pairwise correlation  (NEW in Phase 2)

The relational group is the reason for using six assets rather than one: it encodes
capital-rotation structure (BTC-dominance rising vs alt-season) that per-asset
technicals cannot express.
"""

from itertools import combinations

import numpy as np
import pandas as pd

# Phase 1 approved tier: common history from 2018-05-04, 3037 observations.
TIER_6 = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "LTC/USDT", "ADA/USDT", "XRP/USDT"]

BASE = "BTC/USDT"          # rotation reference asset

# Longest rolling lookback in the feature set, and therefore the purge buffer Phase 4
# must place at every train/test fold boundary. This is 60, set by
# avg_pairwise_corr_60d — NOT 30 (return_30d). Verified arithmetically: the 6-asset
# intersection is 3037 raw observations and the feature matrix is 2977, a head loss of
# exactly 60. A purge of 30 would leave the last 30 days of each training fold sharing
# input data with the first rows of the test fold — the exact bleed the purge prevents.
LONGEST_WINDOW = 60

# Raw moving-average *levels* (sma_10, sma_20) are EXCLUDED.
#
# They are non-stationary, and under the fold-scoped scaling that Phase 4 requires that
# is not a nuisance but a correctness failure: a training fold covering 2018-2020 fits
# the scaler to BTC's SMA near $8k, the 2021 test fold sees $50k, and every scaled test
# observation then falls outside the training distribution — so out-of-sample points get
# assigned to whichever cluster sits at the extreme regardless of market state. The
# walk-forward would emit numbers that mean nothing.
#
# The stationary price_sma14_ratio carries the same information scale-free, so nothing
# is lost. Set include_price_levels=True only to reproduce the pre-Phase-3 feature set.
INCLUDE_PRICE_LEVELS_DEFAULT = False


def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI. Rolling means only — no forward reference."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    return 100 - (100 / (1 + gain / loss))


def build_panel(data: dict, symbols: list = None) -> tuple:
    """
    Align the universe onto its common-history intersection.

    Returns (close, quote_volume) frames indexed by date, columns = symbols. The
    intersection is taken over the symbols requested, so the window is a property of
    the chosen tier and nothing else.
    """
    symbols = symbols or TIER_6
    idx = None
    for sym in symbols:
        i = data[sym].index
        idx = i if idx is None else idx.intersection(i)
    idx = idx.sort_values()

    close = pd.DataFrame({s: data[s].loc[idx, "close"] for s in symbols}, index=idx)
    # ccxt returns BASE-asset volume; quote (USDT) volume is close * volume.
    qvol = pd.DataFrame(
        {s: data[s].loc[idx, "close"] * data[s].loc[idx, "volume"] for s in symbols},
        index=idx,
    )
    return close, qvol


def per_asset_features(close: pd.DataFrame, include_price_levels: bool) -> dict:
    """Per-asset technicals, carried over from the original pipeline unchanged."""
    out = {}
    for sym in close.columns:
        p = close[sym]
        tag = sym.split("/")[0].lower()

        out[f"{tag}_return_1d"] = p.pct_change()
        out[f"{tag}_return_7d"] = p.pct_change(7)
        out[f"{tag}_return_30d"] = p.pct_change(30)

        out[f"{tag}_rsi_14"] = rsi(p, 14)
        out[f"{tag}_rsi_21"] = rsi(p, 21)

        if include_price_levels:
            out[f"{tag}_sma_10"] = p.rolling(10).mean()
            out[f"{tag}_sma_20"] = p.rolling(20).mean()
        out[f"{tag}_price_sma14_ratio"] = p / p.rolling(14).mean()

        r = p.pct_change()
        out[f"{tag}_volatility_20d"] = r.rolling(20).std()
        out[f"{tag}_var_95_20d"] = r.rolling(20).quantile(0.05)
        out[f"{tag}_momentum_14d"] = p / p.shift(14) - 1

    return out


def market_features(close: pd.DataFrame) -> dict:
    """Market-wide aggregates, carried over from the original pipeline unchanged."""
    rets = close.pct_change()
    btc, eth = rets[BASE], rets["ETH/USDT"]
    return {
        "market_momentum_14d": (btc.rolling(14).mean() + eth.rolling(14).mean()) / 2,
        "breadth_positive": (rets > 0).sum(axis=1) / rets.shape[1],
        "btc_eth_correlation": btc.rolling(20).corr(eth),
    }


def relational_features(close: pd.DataFrame, qvol: pd.DataFrame) -> dict:
    """
    Cross-sectional structure: what the six-asset universe is *for*.

    Per-asset technicals cannot distinguish "BTC leads, alts bleed" (rising dominance)
    from "alts outrun BTC" (alt-season) — both look like ordinary momentum asset by
    asset. These features make the distinction explicit.

    Every one is causal: dominance and dispersion are same-day cross-sections, and
    every smoothing or correlation term is a trailing rolling window.
    """
    out = {}
    rets = close.pct_change()

    # --- BTC dominance proxy -------------------------------------------------
    # Binance klines carry no circulating supply, so true market-cap dominance is not
    # computable from this data. Substitute BTC's share of universe QUOTE volume, which
    # is a same-day cross-section and therefore causal by construction.
    share = qvol[BASE] / qvol.sum(axis=1)
    out["btc_volume_dominance"] = share
    out["btc_volume_dominance_ma30"] = share.rolling(30).mean()
    # Direction of rotation: dominance rising = capital moving into BTC.
    out["btc_volume_dominance_chg30"] = share - share.shift(30)

    # A price-based companion: BTC's share of total universe price momentum. Captures
    # rotation even when volume share is flat.
    cum = (1 + rets).rolling(30).apply(np.prod, raw=True)
    out["btc_relative_strength_30d"] = cum[BASE] / cum.mean(axis=1)

    # --- alt-vs-BTC return spreads ------------------------------------------
    alts = [s for s in close.columns if s != BASE]
    spreads = {}
    for sym in alts:
        tag = sym.split("/")[0].lower()
        sp = rets[sym] - rets[BASE]
        spreads[tag] = sp
        out[f"{tag}_spread_vs_btc"] = sp
        out[f"{tag}_spread_vs_btc_ma14"] = sp.rolling(14).mean()

    # Aggregate rotation signal: positive = alt-season, negative = BTC-dominance.
    spread_df = pd.DataFrame(spreads, index=close.index)
    out["alt_spread_mean"] = spread_df.mean(axis=1)
    out["alt_spread_mean_ma30"] = spread_df.mean(axis=1).rolling(30).mean()
    # Breadth of the rotation: how many alts are actually beating BTC.
    out["alt_breadth_vs_btc"] = (spread_df > 0).sum(axis=1) / spread_df.shape[1]

    # --- cross-asset return dispersion --------------------------------------
    # Same-day cross-sectional stdev. High = idiosyncratic/rotational market,
    # low = everything moving together (typical of crisis or strong trend).
    disp = rets.std(axis=1, ddof=1)
    out["return_dispersion"] = disp
    out["return_dispersion_ma20"] = disp.rolling(20).mean()

    # --- average pairwise correlation ---------------------------------------
    # Mean of the 15 off-diagonal pairwise correlations, on trailing windows.
    # Rises toward 1 in crises as everything sells off together.
    for win in (30, 60):
        pair_corrs = [
            rets[a].rolling(win).corr(rets[b]) for a, b in combinations(close.columns, 2)
        ]
        out[f"avg_pairwise_corr_{win}d"] = pd.concat(pair_corrs, axis=1).mean(axis=1)

    return out


def build_features(
    data: dict,
    symbols: list = None,
    include_price_levels: bool = INCLUDE_PRICE_LEVELS_DEFAULT,
) -> pd.DataFrame:
    """
    Full causal feature matrix for the universe.

    Rows with NaN from rolling-window warmup are dropped from the head only; no
    interpolation, no backfill. Backfilling would move future values backward in time,
    which is the leak this module exists to prevent.
    """
    symbols = symbols or TIER_6
    close, qvol = build_panel(data, symbols)

    feats = {}
    feats.update(per_asset_features(close, include_price_levels))
    feats.update(market_features(close))
    feats.update(relational_features(close, qvol))

    df = pd.DataFrame(feats, index=close.index)
    df = df.replace([np.inf, -np.inf], np.nan)

    # Drop only the leading warmup block; an interior NaN is a data problem that should
    # surface rather than be silently filled.
    first_valid = df.dropna().index.min()
    return df.loc[first_valid:].dropna()


def forward_returns(data: dict, symbols: list = None, horizon: int = 1) -> pd.Series:
    """
    Equal-weight forward return of the universe.

    Kept in a SEPARATE function, never merged into the feature matrix, so it cannot
    reach the regime model by accident. For evaluation only.
    """
    symbols = symbols or TIER_6
    close, _ = build_panel(data, symbols)
    return close.pct_change(horizon).shift(-horizon).mean(axis=1)
