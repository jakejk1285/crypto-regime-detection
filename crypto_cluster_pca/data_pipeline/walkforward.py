#!/usr/bin/env python3
"""
Expanding-window walk-forward validation.

Everything that touches a test fold is fitted on that fold's training data only:
the StandardScaler, the PCA rotation, the cluster centroids, and every trading-rule
parameter. Test-fold regime labels come from `.predict()` on the frozen training
pipeline — the clusterer is never refit on test data.

Strategy specification, and what was deliberately NOT fitted
------------------------------------------------------------
The brief listed "allocation, thresholds, sizing, stops." Only the first three are
fitted, and the arithmetic is why: fold 1 has 500 training observations spread over 4
regimes, and the per-fold spell counts (measured before building this harness) go as low
as **20 observations in 1 spell** for Crisis in fold 2 and 30 observations in 3 spells
for Alt-Season in fold 3. Fitting per-regime stop-loss percentages, R-multiples,
coin-score thresholds AND allocations on that is fitting noise — the exact failure the
audit documented.

Stops are additionally not supportable by the data at all: daily bars cannot adjudicate
whether a stop or a target was hit first *within* a day. The legacy pipeline hid this by
synthesizing highs and lows as close*1.001 / close*0.999 (Overfitting_Diagnosis.md
finding 5.5) and evaluating exits on the close only.

So the strategy is a regime-conditional allocation to the equal-weight universe basket:

    per regime r, fitted on the TRAINING fold only:
        mu_r    mean daily basket return while in regime r
        sd_r    daily volatility while in regime r
        s_r     completed spells (the honest degrees of freedom, not observation count)
        t_r     mu_r / (sd_r / sqrt(s_r))

    gate  (threshold):  regime is tradeable iff s_r >= MIN_SPELLS and t_r >= MIN_T
    size  (sizing)   :  w_r = min(MAX_W, TARGET_ANN_VOL / (sd_r * sqrt(252)))
                        w_r = 0 if the gate fails

The four constants below are fixed a priori and are NOT tuned on any result — not on
the training folds and certainly not on the out-of-sample series. The *fitted* quantities
are mu_r, sd_r and s_r, re-estimated inside every training fold.

The t-statistic uses **spell count**, not observation count, for degrees of freedom.
Days inside one regime spell are near-duplicates; using n would overstate significance
by roughly sqrt(spell length).
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import regime_models as rm

# --- a priori constants, never tuned -------------------------------------
MIN_SPELLS = 5          # a regime seen fewer than 5 times is not estimated
MIN_T = 1.0             # minimum training-fold t-stat (spell-based dof)
TARGET_ANN_VOL = 0.50   # annualized volatility target; crypto-appropriate, fixed
MAX_W = 1.0             # no leverage
N_COMPONENTS_TARGET = rm.PCA_VARIANCE_TARGET
K = rm.PRIMARY_K
PURGE = 60              # = features.LONGEST_WINDOW (avg_pairwise_corr_60d)
SEED = 42

REGIME_NAMES = {0: "Crisis", 1: "SteadyState", 2: "BroadRally", 3: "AltSeason"}


def make_folds(n: int, t0: int = 500, test: int = 250, purge: int = PURGE):
    """
    Expanding-window folds as (train_start, train_end, test_start, test_end).

    The purge sits between train_end and test_start, so the training fold ENDS `purge`
    days before the test fold begins. Both indices are half-open.
    """
    out, start = [], t0
    while start + purge + test <= n:
        out.append((0, start, start + purge, start + purge + test))
        start += test
    return out


def spell_count(mask: np.ndarray) -> int:
    """Number of contiguous runs where mask is True."""
    return int((np.diff(np.concatenate([[0], mask.astype(int)])) == 1).sum())


def fit_regime_rules(labels: np.ndarray, basket_ret: np.ndarray) -> dict:
    """
    Fit per-regime allocation on TRAINING data only.

    `basket_ret[i]` is the equal-weight basket's return realized on day i, so pairing it
    with `labels[i]` asks: what did the basket do while we were in this regime? The
    resulting weight is applied to the NEXT day in `apply_rules`, never the same day.
    """
    rules = {}
    for r in range(K):
        m = labels == r
        n, s = int(m.sum()), spell_count(m)
        if n < 2:
            rules[r] = {"w": 0.0, "mu": np.nan, "sd": np.nan, "n": n, "spells": s,
                        "t": np.nan, "gated": False, "reason": "n<2"}
            continue
        mu, sd = float(basket_ret[m].mean()), float(basket_ret[m].std(ddof=1))
        t = mu / (sd / np.sqrt(max(s, 1))) if sd > 0 else 0.0

        if s < MIN_SPELLS:
            w, gated, reason = 0.0, False, f"spells {s} < {MIN_SPELLS}"
        elif t < MIN_T:
            w, gated, reason = 0.0, False, f"t {t:.2f} < {MIN_T}"
        else:
            ann = sd * np.sqrt(252)
            w = float(min(MAX_W, TARGET_ANN_VOL / ann)) if ann > 0 else 0.0
            gated, reason = True, "tradeable"

        rules[r] = {"w": w, "mu": mu, "sd": sd, "n": n, "spells": s, "t": float(t),
                    "gated": gated, "reason": reason}
    return rules


def apply_rules(labels: np.ndarray, rules: dict) -> np.ndarray:
    """Weight for each day from its regime. No forward reference."""
    return np.array([rules.get(int(l), {"w": 0.0})["w"] for l in labels])


def fit_fold(train_feats: pd.DataFrame, model: str = "kmeans", seed: int = SEED):
    """
    Fit scaler -> PCA -> clusterer on training data. Returns a frozen predictor.

    Nothing here sees the test fold. The returned closure only transforms and predicts.
    """
    scaler = StandardScaler().fit(train_feats)
    Xs = scaler.transform(train_feats)
    probe = PCA(random_state=seed).fit(Xs)
    ncomp = int(np.searchsorted(np.cumsum(probe.explained_variance_ratio_),
                                N_COMPONENTS_TARGET) + 1)
    pca = PCA(n_components=ncomp, random_state=seed).fit(Xs)
    Xp = pca.transform(Xs)

    if model == "kmeans":
        est = KMeans(n_clusters=K, random_state=seed, n_init=10).fit(Xp)
    elif model == "gmm":
        # full covariance: the only variant with an interior BIC optimum (Phase 3)
        est = GaussianMixture(n_components=K, covariance_type="full",
                              random_state=seed, n_init=5, max_iter=500).fit(Xp)
    else:
        raise ValueError(model)

    def predict(feats: pd.DataFrame) -> np.ndarray:
        return est.predict(pca.transform(scaler.transform(feats)))

    return predict, est.predict(Xp), ncomp


def run_walkforward(feats: pd.DataFrame, basket: pd.Series, model: str = "kmeans",
                    t0: int = 500, test: int = 250, seed: int = SEED) -> dict:
    """
    Full expanding-window walk-forward. Returns stitched OOS series plus per-fold detail.

    Regime identity is anchored to the FIRST fold's raw-feature profiles and every later
    fold is matched to it by Hungarian assignment, so a per-regime weight means the same
    thing in fold 9 as in fold 1.
    """
    folds = make_folds(len(feats), t0, test, PURGE)
    reference = None
    oos_w, oos_lab, oos_idx, detail = [], [], [], []

    for i, (a, b, c, d) in enumerate(folds, 1):
        tr, te = feats.iloc[a:b], feats.iloc[c:d]

        # purge assertion: training must END at least PURGE days before test begins
        gap = (te.index.min() - tr.index.max()).days
        assert gap >= PURGE, f"fold {i}: purge gap {gap}d < {PURGE}d"

        predict, tr_lab_raw, ncomp = fit_fold(tr, model=model, seed=seed)

        if reference is None:
            tr_lab = rm.canonical_order(tr_lab_raw, tr)
            reference = rm.fit_reference_profiles(tr_lab_raw, tr)
        else:
            tr_lab = rm.match_to_reference(tr_lab_raw, tr, reference)

        te_lab_raw = predict(te)
        te_lab = rm.match_to_reference(te_lab_raw, te, reference)

        rules = fit_regime_rules(tr_lab, basket.loc[tr.index].to_numpy())
        w = apply_rules(te_lab, rules)

        oos_w.append(pd.Series(w, index=te.index))
        oos_lab.append(pd.Series(te_lab, index=te.index))
        oos_idx.append(te.index)
        detail.append({"fold": i, "train_n": b - a, "n_components": ncomp,
                       "test_start": te.index.min(), "test_end": te.index.max(),
                       "rules": rules,
                       "tradeable": [r for r in range(K) if rules[r]["gated"]]})

    W = pd.concat(oos_w).sort_index()
    L = pd.concat(oos_lab).sort_index()
    # weight chosen from day t's regime is applied to day t+1's return: no same-day peek
    ret = W.shift(1).fillna(0.0) * basket.reindex(W.index)
    return {"weights": W, "labels": L, "returns": ret.dropna(), "folds": detail}


def block_shuffle(labels: pd.Series, seed: int = SEED) -> pd.Series:
    """
    Shuffle regime labels preserving spell structure.

    Splits the series into contiguous spells and permutes the spells, so spell-length
    distribution and total regime frequencies are preserved but the labels no longer
    align with market state. If the real regimes do not beat this out of sample, the
    regime detection contributes nothing.
    """
    rng = np.random.default_rng(seed)
    v = labels.to_numpy()
    edges = np.flatnonzero(np.diff(v)) + 1
    blocks = np.split(v, edges)
    rng.shuffle(blocks)
    return pd.Series(np.concatenate(blocks)[: len(v)], index=labels.index)


def performance(ret: pd.Series, rf: float = 0.0395) -> dict:
    """Standard annualized metrics on a daily return series."""
    r = ret.dropna()
    if len(r) < 2 or r.std() == 0:
        return {k: 0.0 for k in ("total_return", "ann_return", "ann_vol",
                                 "sharpe", "max_drawdown", "hit_rate")}
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    ann = eq.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(252)
    return {
        "total_return": float(eq.iloc[-1] - 1),
        "ann_return": float(ann),
        "ann_vol": float(vol),
        "sharpe": float((ann - rf) / vol) if vol > 0 else 0.0,
        "max_drawdown": float((eq / eq.cummax() - 1).min()),
        "hit_rate": float((r > 0).mean()),
        "n_days": int(len(r)),
    }
