#!/usr/bin/env python3
"""Phase 3 analysis: k-means vs GMM, model selection, regime labelling."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from binance_data import load_stitched
from features import build_features, forward_returns, TIER_6
from regime_stats import causal_regime_stats
import regime_models as rm

TRAIN_PREFIX = 1500   # ~50% of sample, for the train-only criterion cross-check
out = {}

print("Loading data and building features...")
data = load_stitched()
feats = build_features(data, TIER_6)
X, pca, scaler, ncomp = rm.build_design_matrix(feats)
print(f"features: {feats.shape}  PCA components at {rm.PCA_VARIANCE_TARGET:.3f} variance: {ncomp}")
print(f"explained: {pca.explained_variance_ratio_.sum():.4f}\n")
out['n_obs'], out['n_features'], out['n_components'] = len(feats), feats.shape[1], ncomp

print("="*74); print("MODEL SELECTION — k-means"); print("="*74)
km_sel = rm.kmeans_selection(X)
gap = rm.gap_statistic(X)
km_sel = km_sel.merge(gap, on="k")
print(km_sel[['k','silhouette','gap','s_k','satisfies_se_rule']].round(4).to_string(index=False))
sil_opt = int(km_sel.loc[km_sel.silhouette.idxmax(),'k'])
gap_opt = int(km_sel[km_sel.satisfies_se_rule].k.min()) if km_sel.satisfies_se_rule.any() else None
print(f"\n  silhouette optimum: k={sil_opt}")
print(f"  gap statistic (SE rule): k={gap_opt}")

print("\n"+"="*74); print("MODEL SELECTION — GMM"); print("="*74)
gmm_res={}
for cov in ("diag","full"):
    g = rm.gmm_selection(X, covariance_type=cov)
    gmm_res[cov]=g
    b,a = int(g.loc[g.bic.idxmin(),'k']), int(g.loc[g.aic.idxmin(),'k'])
    print(f"\n  covariance_type='{cov}'  (params at k=4: {int(g.loc[g.k==4,'n_params'].iloc[0])}, all converged: {g.converged.all()})")
    print(g[['k','bic','aic','n_params']].round(1).to_string(index=False))
    print(f"    BIC optimum: k={b}   AIC optimum: k={a}")
    gmm_res[cov+'_opt']=(b,a)

print("\n"+"="*74); print(f"TRAIN-ONLY CROSS-CHECK (first {TRAIN_PREFIX} obs)"); print("="*74)
Xtr,_,_,ntr = rm.build_design_matrix(feats.iloc[:TRAIN_PREFIX])
km_tr = rm.kmeans_selection(Xtr); gap_tr = rm.gap_statistic(Xtr)
gm_tr = rm.gmm_selection(Xtr, covariance_type="diag")
km_tr = km_tr.merge(gap_tr,on='k')
sil_tr=int(km_tr.loc[km_tr.silhouette.idxmax(),'k'])
gap_tr_opt=int(km_tr[km_tr.satisfies_se_rule].k.min()) if km_tr.satisfies_se_rule.any() else None
print(f"  components: {ntr} (vs {ncomp} full)")
print(f"  silhouette: k={sil_tr} (full: {sil_opt})   gap SE-rule: k={gap_tr_opt} (full: {gap_opt})")
print(f"  GMM diag BIC: k={int(gm_tr.loc[gm_tr.bic.idxmin(),'k'])} (full: {gmm_res['diag_opt'][0]})   AIC: k={int(gm_tr.loc[gm_tr.aic.idxmin(),'k'])} (full: {gmm_res['diag_opt'][1]})")

print("\n"+"="*74); print(f"PRIMARY FIT — k={rm.PRIMARY_K}"); print("="*74)
km_lab, _ = rm.fit_kmeans(X, feats)
gm_lab, gm_proba, _ = rm.fit_gmm(X, feats, covariance_type="diag")

for name, lab in (("k-means",km_lab),("GMM",gm_lab)):
    st = rm.stability(lab)
    print(f"\n  {name}: flip_rate={st['flip_rate']:.3f}  spells={st['n_spells']}  "
          f"mean_spell={st['mean_spell_length']:.1f}d  median={st['median_spell_length']:.0f}d")
    print("    sizes:", {int(r):int((lab==r).sum()) for r in sorted(np.unique(lab))})

agree=(km_lab==gm_lab).mean()
print(f"\n  hard-label agreement k-means vs GMM: {agree:.1%}")
conf = gm_proba.max(axis=1)
print(f"  GMM mean max-probability: {conf.mean():.3f}   share of days below 0.60: {(conf<0.60).mean():.1%}")
print(f"  on days where models DISAGREE, GMM mean confidence: {conf[km_lab!=gm_lab].mean():.3f}")

print("\n"+"="*74); print("REGIME PROFILES (features only)"); print("="*74)
for name,lab in (("K-MEANS",km_lab),("GMM",gm_lab)):
    print(f"\n--- {name} ---")
    prof = rm.regime_profiles(feats, lab)
    for r,p in prof.items():
        print(f"  regime {r}: n={p['n_obs']:4d} ({p['share']:.1%})")
        for k_,v in p['top_features'].items():
            print(f"      {v:+.2f}  {k_}")

print("\n"+"="*74); print("POST-HOC RETURN DESCRIPTION (never an input)"); print("="*74)
fwd = forward_returns(data, TIER_6, 1).reindex(feats.index)
valid=fwd.notna()
for name,lab in (("k-means",km_lab),("GMM",gm_lab)):
    print(f"\n  {name}:")
    print(rm.regime_return_description(lab[valid.to_numpy()], fwd[valid]).round(5).to_string())

print("\n"+"="*74); print("NaN PREVALENCE for Phase 4 (avg_duration)"); print("="*74)
for name,lab in (("k-means",km_lab),("GMM",gm_lab)):
    cs = causal_regime_stats(pd.Series(lab, index=feats.index))
    nan = cs.avg_duration.isna()
    last = feats.index[nan][-1] if nan.any() else None
    print(f"  {name}: {nan.sum()} of {len(nan)} obs undefined ({nan.mean():.1%}); last undefined = {last.date() if last is not None else 'n/a'}")
    print(f"      contiguous from start? {bool(nan.iloc[:nan.to_numpy().argmin()].all()) if not nan.all() else True}")

pd.DataFrame({'date':feats.index,'kmeans':km_lab,'gmm':gm_lab}).to_csv('crypto_cluster_pca/data_cache/phase3_labels.csv',index=False)
km_sel.to_csv('crypto_cluster_pca/data_cache/phase3_kmeans_selection.csv',index=False)
gmm_res['diag'].to_csv('crypto_cluster_pca/data_cache/phase3_gmm_diag_selection.csv',index=False)
gmm_res['full'].to_csv('crypto_cluster_pca/data_cache/phase3_gmm_full_selection.csv',index=False)
print("\nartifacts written to crypto_cluster_pca/data_cache/")
