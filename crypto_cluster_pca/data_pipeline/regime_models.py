#!/usr/bin/env python3
"""
Regime detection: k-means and Gaussian Mixture, with principled model selection.

Design decisions, all fixed BEFORE looking at any clustering result
-------------------------------------------------------------------
1. PCA is load-bearing, not a stylistic carryover. A full-covariance GMM on the 78 raw
   features would need 78 + 78*79/2 = 3159 parameters per component — ~12,600 for k=4
   against 2977 observations. Covariances would be singular and the BIC penalty would
   swamp the likelihood, making the curve monotone in k and therefore uninformative.

2. The PCA component count comes from a variance target fixed in advance: 0.897, the
   value recorded in the original project's `pca_analysis_summary.json`. On the current
   78-feature set that yields 19 components. This target is NOT adjusted after seeing
   clustering results — tuning a hyperparameter on the outcome is the failure the audit
   documented.

3. k=4 is EXOGENOUS. It comes from the Two Sigma factor-regime work (Crisis / Steady
   State / Inflation / Walking-on-Ice), not from this data. The selection-criterion
   curves below are therefore *diagnostic reporting*, not selection: nothing downstream
   consumes the criterion-optimal k. This is what keeps "choose k on the full sample,
   then walk-forward test with it" from being selection on the test window.

4. Regime identity across refits is anchored to the FIRST training fold's raw-feature
   profiles, matched by optimal assignment (`fit_reference_profiles` /
   `match_to_reference`). Rank-based ordering within each fold is NOT fold-stable — the
   sort key is a within-sample mean, so a regime can change rank without changing
   character. Matching uses raw features rather than PCA centroids because PCA is refit
   per fold and its axes (sign included) rotate between folds.

NaN policy
----------
Clustering consumes the FEATURE matrix, which has no NaNs after warmup. The NaN
`avg_duration` values from regime_stats.py affect the trading rules in Phase 4, not the
clustering here. This module reports their prevalence so Phase 4's fold-exclusion
decision is made on numbers. Imputation is never used: 0 means "never persists," a real
value, not "unknown."
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

PCA_VARIANCE_TARGET = 0.897   # pre-registered, from the original pca_analysis_summary.json
PRIMARY_K = 4                 # exogenous: Two Sigma comparability
K_RANGE = range(2, 13)
ORDER_BY = "avg_pairwise_corr_30d"   # raw feature; descending => regime 0 = most crisis-like
SEED = 42


def build_design_matrix(features: pd.DataFrame, variance_target: float = PCA_VARIANCE_TARGET):
    """Standardize then PCA. Returns (scores, pca, scaler, n_components)."""
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    probe = PCA(random_state=SEED).fit(X)
    n = int(np.searchsorted(np.cumsum(probe.explained_variance_ratio_), variance_target) + 1)
    pca = PCA(n_components=n, random_state=SEED).fit(X)
    return pca.transform(X), pca, scaler, n


def canonical_order(labels: np.ndarray, features: pd.DataFrame, order_by: str = ORDER_BY) -> np.ndarray:
    """
    Rank-based ordering by descending mean of a raw feature. USE ONLY FOR A SINGLE FIT.

    NOT fold-stable, and deliberately not used by Phase 4: the sort key is a
    within-sample mean, so when the sample changes across folds a regime can move rank
    without changing character, and nothing pins the cluster boundaries. Use
    `fit_reference_profiles` + `match_to_reference` for anything that refits.
    """
    key = features[order_by].to_numpy()
    means = {c: key[labels == c].mean() for c in np.unique(labels)}
    ranked = sorted(means, key=lambda c: -means[c])
    remap = {old: new for new, old in enumerate(ranked)}
    return np.array([remap[l] for l in labels])


# --------------------------------------------------------------------------
# Fold-stable regime identity
# --------------------------------------------------------------------------
#
# Phase 4 refits per fold. Without a rule anchoring identity across refits, "regime 2"
# in fold 1 and fold 3 are different clusters and any per-fold parameter keyed on regime
# id is meaningless. This is exactly the instability Overfitting_Diagnosis.md documented:
# `regime_strategy_mapping` was hardcoded across two fits whose clusters disagreed
# (research fit had 2 observations in cluster 3; the backtest refit had 40).
#
# Matching is done on RAW-FEATURE z-profiles, not on PCA centroids, because PCA is refit
# per fold and its axes (including sign, which sklearn does not fix) rotate between
# folds. Raw-feature profiles live in a space that is the same for every fold.

PROFILE_FEATURES = [
    "avg_pairwise_corr_30d", "return_dispersion_ma20", "btc_volume_dominance",
    "alt_spread_mean_ma30", "btc_relative_strength_30d", "market_momentum_14d",
    "btc_rsi_14", "btc_volatility_20d", "btc_var_95_20d",
]


def fit_reference_profiles(labels: np.ndarray, features: pd.DataFrame,
                           order_by: str = ORDER_BY) -> pd.DataFrame:
    """
    Establish canonical regime identity from the FIRST training fold.

    Returns a (k x len(PROFILE_FEATURES)) frame of standardized per-regime means, with
    rows already in canonical order (descending `order_by`). Every later fold matches
    against this, so regime 0 means the same kind of market for the whole walk-forward.

    Standardization uses the reference fold's own mean/std, which are returned alongside
    so later folds are scored in the reference fold's units rather than their own — a
    later fold's mean is not available at the time the reference is set.
    """
    ordered = canonical_order(labels, features, order_by)
    mu, sd = features[PROFILE_FEATURES].mean(), features[PROFILE_FEATURES].std()
    z = (features[PROFILE_FEATURES] - mu) / sd
    prof = z.groupby(ordered).mean()
    prof.attrs["mu"], prof.attrs["sd"] = mu, sd
    return prof


def match_to_reference(labels: np.ndarray, features: pd.DataFrame,
                       reference: pd.DataFrame) -> np.ndarray:
    """
    Relabel a later fold's clusters to the reference fold's regime identities.

    Uses optimal (Hungarian) assignment on Euclidean distance between raw-feature
    z-profiles, so the mapping is one-to-one: two of this fold's clusters cannot both
    claim to be regime 0. Greedy nearest-match would allow that.
    """
    from scipy.optimize import linear_sum_assignment

    mu, sd = reference.attrs["mu"], reference.attrs["sd"]
    z = (features[PROFILE_FEATURES] - mu) / sd          # reference fold's units
    prof = z.groupby(labels).mean()

    cost = np.linalg.norm(
        prof.to_numpy()[:, None, :] - reference.to_numpy()[None, :, :], axis=2
    )
    rows, cols = linear_sum_assignment(cost)
    remap = {int(prof.index[r]): int(reference.index[c]) for r, c in zip(rows, cols)}
    return np.array([remap[l] for l in labels])


def ordering_drift(labels_a: np.ndarray, features_a: pd.DataFrame,
                   labels_b: np.ndarray, features_b: pd.DataFrame) -> float:
    """
    Diagnostic: how far a later fold's matched profiles sit from the reference.

    Large drift means the matched regime is only nominally the same state, and Phase 5
    should report it rather than assume identity held.
    """
    ref = fit_reference_profiles(labels_a, features_a)
    mu, sd = ref.attrs["mu"], ref.attrs["sd"]
    matched = match_to_reference(labels_b, features_b, ref)
    z = (features_b[PROFILE_FEATURES] - mu) / sd
    prof_b = z.groupby(matched).mean()
    common = ref.index.intersection(prof_b.index)
    return float(np.linalg.norm(ref.loc[common].to_numpy() - prof_b.loc[common].to_numpy(), axis=1).mean())


# --------------------------------------------------------------------------
# Model selection criteria
# --------------------------------------------------------------------------

def gap_statistic(X: np.ndarray, k_range=K_RANGE, B: int = 20, seed: int = SEED):
    """
    Tibshirani gap statistic with a PCA-ALIGNED reference box.

    The reference distribution is uniform over the bounding box of the data *rotated
    into its principal axes*, then rotated back — the axis-aligned box overstates the
    reference volume for correlated data and biases the gap upward.

    Returns a DataFrame with gap, s_k, and the standard-error rule flag: the smallest k
    with gap(k) >= gap(k+1) - s(k+1). Reported instead of the raw argmax, which is
    noisy.
    """
    rng = np.random.default_rng(seed)
    Xc = X - X.mean(0)
    U, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    Xp = Xc @ Vt.T                                  # principal-axis coordinates
    lo, hi = Xp.min(0), Xp.max(0)

    def inertia(D, k):
        return KMeans(n_clusters=k, random_state=seed, n_init=10).fit(D).inertia_

    rows = []
    for k in k_range:
        wk = np.log(inertia(X, k))
        refs = []
        for _ in range(B):
            Z = rng.uniform(lo, hi, size=Xp.shape) @ Vt + X.mean(0)
            refs.append(np.log(inertia(Z, k)))
        refs = np.array(refs)
        gap = refs.mean() - wk
        sk = refs.std(ddof=1) * np.sqrt(1 + 1 / B)
        rows.append({"k": k, "gap": gap, "s_k": sk})

    df = pd.DataFrame(rows)
    df["gap_next_minus_se"] = df["gap"].shift(-1) - df["s_k"].shift(-1)
    df["satisfies_se_rule"] = df["gap"] >= df["gap_next_minus_se"]
    return df


def kmeans_selection(X: np.ndarray, k_range=K_RANGE, seed: int = SEED) -> pd.DataFrame:
    """Silhouette + inertia across k."""
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X)
        rows.append({
            "k": k,
            "silhouette": silhouette_score(X, km.labels_),
            "inertia": km.inertia_,
        })
    return pd.DataFrame(rows)


def gmm_selection(X: np.ndarray, k_range=K_RANGE, covariance_type: str = "diag",
                  seed: int = SEED) -> pd.DataFrame:
    """BIC + AIC across k for a given covariance type."""
    rows = []
    for k in k_range:
        gm = GaussianMixture(n_components=k, covariance_type=covariance_type,
                             random_state=seed, n_init=5, max_iter=500).fit(X)
        rows.append({
            "k": k,
            "bic": gm.bic(X),
            "aic": gm.aic(X),
            "n_params": int(gm._n_parameters()),
            "converged": bool(gm.converged_),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------

def fit_kmeans(X, features, k=PRIMARY_K, seed=SEED):
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X)
    return canonical_order(km.labels_, features), km


def fit_gmm(X, features, k=PRIMARY_K, covariance_type="diag", seed=SEED):
    """
    Returns (hard_labels, soft_probabilities, model).

    Soft probabilities are retained deliberately: a day sitting 0.51/0.49 between two
    regimes is materially different from one at 0.99, and a hard assignment discards
    that. Columns of the probability frame are in canonical order.
    """
    gm = GaussianMixture(n_components=k, covariance_type=covariance_type,
                         random_state=seed, n_init=5, max_iter=500).fit(X)
    raw = gm.predict(X)
    labels = canonical_order(raw, features)

    # map probability columns through the same permutation
    key = features[ORDER_BY].to_numpy()
    means = {c: key[raw == c].mean() for c in np.unique(raw)}
    ranked = sorted(means, key=lambda c: -means[c])
    proba = pd.DataFrame(gm.predict_proba(X)[:, ranked],
                         index=features.index,
                         columns=[f"p_regime_{i}" for i in range(k)])
    return labels, proba, gm


# --------------------------------------------------------------------------
# Description and stability
# --------------------------------------------------------------------------

def regime_profiles(features: pd.DataFrame, labels: np.ndarray, top: int = 6) -> dict:
    """
    Standardized per-regime feature means, and the top discriminators by |z|.

    Uses FEATURES ONLY. Realized returns are reported separately in
    `regime_return_description` — describing a regime by its realized returns after the
    fact is legitimate; using returns to define or select regimes would be the leak.
    """
    z = (features - features.mean()) / features.std()
    out = {}
    for r in sorted(np.unique(labels)):
        m = z[labels == r].mean().sort_values(key=np.abs, ascending=False)
        out[r] = {
            "n_obs": int((labels == r).sum()),
            "share": float((labels == r).mean()),
            "top_features": m.head(top).round(3).to_dict(),
        }
    return out


def regime_return_description(labels: np.ndarray, fwd_returns: pd.Series) -> pd.DataFrame:
    """
    POST HOC description only — never an input to fitting or selection.

    Kept in a separate function from regime_profiles() so the boundary is structural.
    """
    df = pd.DataFrame({"regime": labels, "fwd": fwd_returns.to_numpy()})
    g = df.groupby("regime")["fwd"]
    return pd.DataFrame({
        "n": g.size(),
        "mean_fwd_return": g.mean(),
        "volatility": g.std(),
        "hit_rate": g.apply(lambda s: (s > 0).mean()),
    })


def stability(labels: np.ndarray) -> dict:
    """Label flip rate and mean spell length — Phase 5 needs both."""
    flips = (labels[1:] != labels[:-1]).sum()
    spells, cur = [], 1
    for i in range(1, len(labels)):
        if labels[i] == labels[i - 1]:
            cur += 1
        else:
            spells.append(cur)
            cur = 1
    spells.append(cur)
    return {
        "flip_rate": float(flips / (len(labels) - 1)),
        "n_spells": len(spells),
        "mean_spell_length": float(np.mean(spells)),
        "median_spell_length": float(np.median(spells)),
    }
