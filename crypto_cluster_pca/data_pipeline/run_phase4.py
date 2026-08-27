#!/usr/bin/env python3
"""Phase 4: walk-forward validation vs baselines."""
import sys, json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from binance_data import load_stitched
from features import build_features, build_panel, TIER_6
import walkforward as wf, regime_models as rm

data=load_stitched(); feats=build_features(data,TIER_6)
close,_=build_panel(data,TIER_6)
basket=close.pct_change().mean(axis=1).reindex(feats.index)   # equal-weight universe
print(f"features {feats.shape}  basket days {basket.notna().sum()}")

F=wf.make_folds(len(feats)); print(f"folds: {len(F)}  (t0=500, test=250, purge={wf.PURGE})\n")

res={}
for model in ("kmeans","gmm"):
    print("="*78); print(f"WALK-FORWARD — {model.upper()}"); print("="*78)
    r=wf.run_walkforward(feats,basket,model=model)
    res[model]=r
    print(f"{'fold':>4s} {'test window':>24s} {'ncomp':>5s} | tradeable regimes (w) | fold return")
    for f in r['folds']:
        tr=", ".join(f"{wf.REGIME_NAMES[x][:5]}={f['rules'][x]['w']:.2f}" for x in f['tradeable']) or "NONE"
        seg=r['returns'].loc[f['test_start']:f['test_end']]
        fr=(1+seg).prod()-1
        print(f"{f['fold']:4d} {str(f['test_start'].date())+'->'+str(f['test_end'].date()):>24s} {f['n_components']:5d} | {tr:34s} | {fr:+7.2%}")
    print("\n  gate rejections by regime (why a regime got w=0):")
    for k in range(4):
        rs=[(f['fold'],f['rules'][k]['reason']) for f in r['folds'] if not f['rules'][k]['gated']]
        print(f"    {wf.REGIME_NAMES[k]:12s} blocked in {len(rs)}/{len(r['folds'])} folds" + (f"  e.g. fold {rs[0][0]}: {rs[0][1]}" if rs else ""))

print("\n"+"="*78); print("BASELINES"); print("="*78)
km=res['kmeans']; idx=km['returns'].index
bh=basket.reindex(idx)

# equal-weight-all-regimes: regime model used, but same weight everywhere (mean of fitted w)
eqw=[]
for f in km['folds']:
    ws=[f['rules'][k]['w'] for k in range(4)]
    m=np.mean(ws)
    seg=km['labels'].loc[f['test_start']:f['test_end']]
    eqw.append(pd.Series(m,index=seg.index))
eqw=pd.concat(eqw).sort_index()
eq_ret=(eqw.shift(1).fillna(0)*basket.reindex(eqw.index)).dropna()

# no-regime-conditioning: one vol-targeted weight fit on whole training fold
nrc=[]
for f,(a,b,c,d) in zip(km['folds'],wf.make_folds(len(feats))):
    tr=basket.iloc[a:b].dropna()
    ann=tr.std()*np.sqrt(252); w=min(wf.MAX_W, wf.TARGET_ANN_VOL/ann) if ann>0 else 0
    nrc.append(pd.Series(w,index=feats.index[c:d]))
nrc=pd.concat(nrc).sort_index()
nrc_ret=(nrc.shift(1).fillna(0)*basket.reindex(nrc.index)).dropna()

# block-shuffled regime labels
sh_rets=[]
for s in range(10):
    sw=[]
    for f in km['folds']:
        seg=km['labels'].loc[f['test_start']:f['test_end']]
        shuf=wf.block_shuffle(seg,seed=100+s)
        sw.append(pd.Series(wf.apply_rules(shuf.to_numpy(),f['rules']),index=seg.index))
    sw=pd.concat(sw).sort_index()
    sh_rets.append((sw.shift(1).fillna(0)*basket.reindex(sw.index)).dropna())

rows={}
rows['k-means regime strategy']=wf.performance(km['returns'])
rows['GMM regime strategy']=wf.performance(res['gmm']['returns'])
rows['equal-weight-all-regimes']=wf.performance(eq_ret)
rows['no-regime-conditioning']=wf.performance(nrc_ret)
rows['buy & hold (EW basket)']=wf.performance(bh)
sh_perf=[wf.performance(x) for x in sh_rets]
rows['block-shuffled regimes (mean of 10)']={k:float(np.mean([p[k] for p in sh_perf])) for k in sh_perf[0]}

df=pd.DataFrame(rows).T[['total_return','ann_return','ann_vol','sharpe','max_drawdown','hit_rate','n_days']]
print(f"\nOut-of-sample window: {idx.min().date()} -> {idx.max().date()}  ({len(idx)} days)\n")
print(df.round(4).to_string())

sh_sharpes=sorted(p['sharpe'] for p in sh_perf)
print(f"\n  block-shuffled Sharpe range over 10 draws: {sh_sharpes[0]:.3f} to {sh_sharpes[-1]:.3f}")
print(f"  k-means Sharpe {rows['k-means regime strategy']['sharpe']:.3f} — beats {sum(rows['k-means regime strategy']['sharpe']>s for s in sh_sharpes)}/10 shuffles")

print("\n=== FOLD-LEVEL BREAKDOWN (aggregate hides concentration) ===")
for name,r in (('k-means',km),('GMM',res['gmm'])):
    fr=[]
    for f in r['folds']:
        seg=r['returns'].loc[f['test_start']:f['test_end']]
        fr.append((1+seg).prod()-1)
    pos=sum(x>0 for x in fr)
    print(f"  {name}: {pos}/{len(fr)} folds positive | best {max(fr):+.2%} worst {min(fr):+.2%} | "
          f"top fold = {max(fr)/sum(abs(x) for x in fr):.0%} of gross")

print("\n=== profile drift across folds (regime identity stability) ===")
ref_tr=feats.iloc[:500]
Xr,_,_,_=rm.build_design_matrix(ref_tr); lr,_=rm.fit_kmeans(Xr,ref_tr)
for f,(a,b,c,d) in zip(km['folds'],wf.make_folds(len(feats))):
    if f['fold'] in (1,3,5,7,9):
        tr=feats.iloc[a:b]; X,_,_,_=rm.build_design_matrix(tr); _,kmm=rm.fit_kmeans(X,tr)
        print(f"  fold {f['fold']}: train_n={b-a:4d}  drift={rm.ordering_drift(lr,ref_tr,kmm.labels_,tr):.3f}")

km['returns'].to_csv('crypto_cluster_pca/data_cache/phase4_oos_kmeans.csv')
df.to_csv('crypto_cluster_pca/data_cache/phase4_summary.csv')
print("\nartifacts written")
