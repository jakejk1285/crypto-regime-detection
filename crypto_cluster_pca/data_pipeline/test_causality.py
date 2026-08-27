#!/usr/bin/env python3
"""
Mechanical causality check for the Phase 2 pipeline.

The test: rebuild every feature on a truncated PREFIX of the data and require the
result to equal the full-series features on those same rows, exactly.

A feature computed with any full-window statistic — a mean, a std, a correlation, a
scaler fit over the whole sample — changes value when the tail is removed, and fails.
A trailing rolling window does not. This covers the relational features (dominance,
spreads, dispersion, pairwise correlation) and the causal regime statistics in one
pass, and it cannot be satisfied by inspection or by a reassuring comment.

Run: python crypto_cluster_pca/data_pipeline/test_causality.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from binance_data import load_stitched
from features import TIER_6, build_features, build_panel, forward_returns
from regime_stats import causal_regime_stats, full_sample_regime_stats

CUTS = [500, 1000, 1500, 2000, 2500]


def _truncate(data: dict, k: int) -> dict:
    return {s: d.iloc[:k] for s, d in data.items()}


def test_feature_causality(data: dict) -> bool:
    full = build_features(data, TIER_6)
    print(f"  full feature matrix: {full.shape[0]} obs x {full.shape[1]} features")
    print(f"  window: {full.index[0].date()} -> {full.index[-1].date()}\n")

    ok = True
    for k in CUTS:
        trunc = build_features(_truncate(data, k), TIER_6)
        if trunc.empty:
            print(f"  cut={k:5d}  (no rows after warmup, skipped)")
            continue

        common = full.index.intersection(trunc.index)
        a = full.loc[common]
        b = trunc.loc[common, a.columns]

        diff = (a - b).abs()
        worst = diff.max().max()
        bad = diff.columns[(diff > 1e-10).any()].tolist()

        status = "PASS" if not bad else "FAIL"
        print(f"  cut={k:5d}  rows compared={len(common):5d}  max|diff|={worst:.3e}  {status}")
        if bad:
            ok = False
            print(f"        leaking features: {bad[:8]}")
    return ok


def test_regime_stats_causality(data: dict) -> bool:
    """Same truncation test on the expanding-window regime statistics."""
    rng = np.random.default_rng(0)
    close, _ = build_panel(data, TIER_6)
    # Synthetic persistent labels: the test is about the statistic, not the clusterer.
    lab = pd.Series(
        rng.choice(4, size=len(close), p=[0.4, 0.3, 0.2, 0.1]).repeat(1)[: len(close)],
        index=close.index,
    )
    lab = lab.rolling(15, min_periods=1).median().round().astype(int)  # add persistence

    full = causal_regime_stats(lab)
    ok = True
    for k in CUTS:
        trunc = causal_regime_stats(lab.iloc[:k])
        a = full.iloc[:k]
        b = trunc
        diff = (a - b).abs()
        worst = np.nanmax(diff.to_numpy())
        # NaN pattern must match too, else a value appeared that shouldn't exist yet.
        nan_match = (a.isna().to_numpy() == b.isna().to_numpy()).all()
        good = (worst < 1e-10 or np.isnan(worst)) and nan_match
        ok &= good
        print(f"  cut={k:5d}  max|diff|={worst:.3e}  nan-pattern-match={nan_match}  "
              f"{'PASS' if good else 'FAIL'}")
    return ok


def test_no_forward_return_in_features(data: dict) -> bool:
    """
    Regime detection must not see forward returns.

    Correlate every feature against the 1-day FORWARD universe return. A feature that
    accidentally contains future information shows a correlation far above what a
    genuine predictor of next-day crypto returns could plausibly reach.
    """
    feats = build_features(data, TIER_6)
    fwd = forward_returns(data, TIER_6, horizon=1).reindex(feats.index)
    valid = fwd.notna()

    corrs = feats[valid].corrwith(fwd[valid]).abs().sort_values(ascending=False)
    print(f"  features tested: {len(corrs)}")
    print("  highest |corr| with NEXT-day return:")
    for name, c in corrs.head(5).items():
        print(f"    {c:.4f}  {name}")

    suspicious = corrs[corrs > 0.20]
    if len(suspicious):
        print(f"  FAIL: {len(suspicious)} feature(s) above 0.20")
        return False
    print(f"  PASS: max |corr| = {corrs.iloc[0]:.4f}, below the 0.20 leak threshold")
    return True


def test_leak_magnitude(data: dict) -> None:
    """Quantify how far the OLD full-sample statistics sat from the causal ones."""
    rng = np.random.default_rng(0)
    close, _ = build_panel(data, TIER_6)
    lab = pd.Series(rng.choice(4, size=len(close)), index=close.index)
    lab = lab.rolling(15, min_periods=1).median().round().astype(int)

    causal = causal_regime_stats(lab)
    leaked = full_sample_regime_stats(lab)
    for col in ["persistence", "avg_duration", "frequency_percentage"]:
        d = (leaked[col] - causal[col]).abs()
        print(f"  {col:22s} mean|full-sample - causal| = {d.mean():.4f}  max = {d.max():.4f}")


if __name__ == "__main__":
    print("Loading Binance daily data...")
    data = load_stitched()

    print("\n" + "=" * 72)
    print("TEST 1  Feature truncation-invariance")
    print("=" * 72)
    t1 = test_feature_causality(data)

    print("\n" + "=" * 72)
    print("TEST 2  Regime-statistic truncation-invariance")
    print("=" * 72)
    t2 = test_regime_stats_causality(data)

    print("\n" + "=" * 72)
    print("TEST 3  No forward-return leakage into features")
    print("=" * 72)
    t3 = test_no_forward_return_in_features(data)

    print("\n" + "=" * 72)
    print("REFERENCE  Magnitude of the leak that was removed")
    print("=" * 72)
    test_leak_magnitude(data)

    print("\n" + "=" * 72)
    ok = t1 and t2 and t3
    print("ALL CAUSALITY TESTS PASS" if ok else "CAUSALITY TESTS FAILED")
    print("=" * 72)
    sys.exit(0 if ok else 1)
