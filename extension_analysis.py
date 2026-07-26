#!/usr/bin/env python3
# Extension analyses for the ecancer revision:
#  (1) HER2 asymmetry sensitivity (exclude 24 HER2+ from training)
#  (2) Post-hoc power simulation under the permutation framework
#  (3) Permutation B (random-gene-set) bumped to 2000 iterations
#  (4) ER-status baseline computed DIRECTLY (binary predictor) with bootstrap CI
import sys, os, warnings, json
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'data')   # place processed GEO CSVs here (see README)
OUT = HERE
SEED = 42
rng = np.random.default_rng(SEED)
PROBES = ['201887_at', '213872_at']   # IL13RA1, C6orf62 (pre-specified)

def load_expr(path):
    return pd.read_csv(path, index_col=0)

xt = load_expr(os.path.join(DATA_DIR, 'X_GSE20194.csv'))
xv = load_expr(os.path.join(DATA_DIR, 'X_GSE25066.csv'))
if 'ID_REF' in xt.columns:
    xt = xt.drop(columns=['ID_REF'])
common = xt.columns.intersection(xv.columns)
xt = xt[common]; xv = xv[common]

# ---- Training clinical (GSE20194) ----
ct = pd.read_csv(os.path.join(DATA_DIR, 'GSE20194_clinical_labels.csv')).set_index('Sample_geo_accession')
ct = ct.loc[ct.index.intersection(xt.index)]
xt = xt.loc[ct.index]
er   = ct['characteristics: ER_status'].values
her2 = ct['HER2 Status'].values
pcr  = (ct['characteristics: pCR_vs_RD'].values == 'pCR').astype(int)
mask_er  = er == 'P'
mask_er_her2neg = (er == 'P') & (her2 == 'N')
ytr_all = pcr[mask_er]
ytr_ens = pcr[mask_er_her2neg]
print(f"[Train ER+] n={len(ytr_all)} pCR={ytr_all.sum()} ({ytr_all.mean():.1%})")
print(f"[Train ER+/HER2-] n={len(ytr_ens)} pCR={ytr_ens.sum()} ({ytr_ens.mean():.1%})")
# 24 HER2+ subgroup pCR rate (for Table 1)
her2pos_erpos = (er=='P') & (her2=='P')
print(f"[Train ER+/HER2+] n={her2pos_erpos.sum()} pCR={pcr[her2pos_erpos].sum()} ({pcr[her2pos_erpos].mean():.1%})")

# ---- Validation (GSE25066 ER+/HER2-) ----
cv = pd.read_csv(os.path.join(DATA_DIR, 'clinical_GSE25066.csv')).set_index(
    pd.read_csv(os.path.join(DATA_DIR, 'clinical_GSE25066.csv')).columns[0])
cv = cv.loc[cv.index.intersection(xv.index)]
xv = xv.loc[cv.index]
erv  = cv['ER'].values; her2v = cv['HER2'].values; pcrv = cv['pCR'].values.astype(int)
mask_v = (erv == '+') & (her2v == '-')
Xv_full = xv[mask_v].values.astype(float); yv = pcrv[mask_v]
print(f"[Val ER+/HER2-] n={len(yv)} pCR={yv.sum()} ({yv.mean():.1%})")

# variance-top200 candidate pool (original training: all ER+)
Xtr_full = xt[mask_er].values.astype(float)
variances = np.var(Xtr_full, axis=0)
top200_idx = np.argsort(variances)[::-1][:200]
cand_cols = np.array(common)[top200_idx].tolist()
Xtr_c = Xtr_full[:, top200_idx]; Xv_c = Xv_full[:, top200_idx]

# variance-top200 candidate pool for HER2-restricted training (ER+/HER2-)
Xtr_full_ens = xt[mask_er_her2neg].values.astype(float)
variances_ens = np.var(Xtr_full_ens, axis=0)
top200_idx_ens = np.argsort(variances_ens)[::-1][:200]
Xtr_c_ens = Xtr_full_ens[:, top200_idx_ens]; Xv_c_ens = Xv_full[:, top200_idx_ens]

def fit_pred(Xtr, ytr, Xv, C=1e10, penalty='l2'):
    sc = StandardScaler(); Xtrs = sc.fit_transform(Xtr); Xvs = sc.transform(Xv)
    m = LogisticRegression(C=C, penalty=penalty, solver='liblinear',
                            class_weight='balanced' if penalty=='l1' else None,
                            max_iter=5000, random_state=42)
    m.fit(Xtrs, ytr)
    return m.predict_proba(Xvs)[:,1]

def auc_with_ci(y, pred, n_boot=2000, seed=7):
    auc = roc_auc_score(y, pred)
    r = np.random.default_rng(seed)
    bs = []
    y = np.asarray(y); pred = np.asarray(pred)
    for _ in range(n_boot):
        idx = r.choice(len(y), len(y), replace=True)
        if len(np.unique(y[idx])) < 2: continue
        bs.append(roc_auc_score(y[idx], pred[idx]))
    bs = np.array(bs)
    return auc, np.percentile(bs, 2.5), np.percentile(bs, 97.5)

# ============ (1) HER2 sensitivity: train ER+/HER2- only ============
Xtr_ens = xt.loc[xt.index[mask_er_her2neg], PROBES].values.astype(float)
Xv2     = xv.loc[xv.index[mask_v], PROBES].values.astype(float)
pred_ens = fit_pred(Xtr_ens, ytr_ens, Xv2)
auc_ens, lo, hi = auc_with_ci(yv, pred_ens)
print(f"\n[HER2 SENS] ER+/HER2- training -> val AUC = {auc_ens:.4f} (95% CI {lo:.3f}-{hi:.3f})")

# permutation B (random-gene-set) with the HER2-restricted training, 2000 iters
N_PERM = 2000
permB_ens = np.zeros(N_PERM); r2 = np.random.default_rng(SEED)
for p in range(N_PERM):
    pick = r2.choice(len(cand_cols), size=2, replace=False)
    Xb = StandardScaler().fit_transform(Xtr_c_ens[:, pick]); Xvb = StandardScaler().fit_transform(Xv_c_ens[:, pick])
    m = LogisticRegression(C=0.05, penalty='l1', solver='liblinear', class_weight='balanced', max_iter=5000).fit(Xb, ytr_ens)
    permB_ens[p] = 0.5 if np.abs(m.coef_[0]).sum()==0 else roc_auc_score(yv, m.predict_proba(Xvb)[:,1])
pB_ens = np.mean(permB_ens >= auc_ens)
print(f"[HER2 SENS permB] mean={permB_ens.mean():.4f} 95th={np.percentile(permB_ens,95):.4f} p={pB_ens:.4f}")

# ============ (3) Permutation B on ORIGINAL training, 2000 iters ============
Xtr2 = xt.loc[xt.index[mask_er], PROBES].values.astype(float)
pred_orig = fit_pred(Xtr2, ytr_all, Xv2)
auc_orig, lo_o, hi_o = auc_with_ci(yv, pred_orig)
print(f"\n[ORIG] val AUC = {auc_orig:.4f} (95% CI {lo_o:.3f}-{hi_o:.3f})")
permB = np.zeros(N_PERM); r3 = np.random.default_rng(SEED)
for p in range(N_PERM):
    pick = r3.choice(len(cand_cols), size=2, replace=False)
    Xb = StandardScaler().fit_transform(Xtr_c[:, pick]); Xvb = StandardScaler().fit_transform(Xv_c[:, pick])
    m = LogisticRegression(C=0.05, penalty='l1', solver='liblinear', class_weight='balanced', max_iter=5000).fit(Xb, ytr_all)
    permB[p] = 0.5 if np.abs(m.coef_[0]).sum()==0 else roc_auc_score(yv, m.predict_proba(Xvb)[:,1])
pB = np.mean(permB >= auc_orig)
null_95 = np.percentile(permB, 95)
print(f"[ORIG permB] mean={permB.mean():.4f} SD={permB.std():.4f} 95th={null_95:.4f} max={permB.max():.4f} p={pB:.4f}")

# ============ (4) ER baseline DIRECT (binary ER-status predictor) ============
er_binary = (erv == '-').astype(int)   # 1 if ER-, 0 if ER+
auc_er_direct, er_lo, er_hi = auc_with_ci(pcrv, er_binary)
print(f"\n[ER BASELINE direct] AUC(ERstatus, pCR) full GSE25066 = {auc_er_direct:.4f} (95% CI {er_lo:.3f}-{er_hi:.3f})")
# sanity: univariate logistic AUC
Xer = StandardScaler().fit_transform(er_binary.reshape(-1,1))
auc_er_logit = roc_auc_score(pcrv, LogisticRegression(C=1e10,solver='liblinear').fit(Xer,pcrv).predict_proba(Xer)[:,1])
print(f"[ER BASELINE univariate-logit] AUC = {auc_er_logit:.4f}")

# ============ (2) Power simulation ============
# Power = P(observed signature AUC > null 95th) under a true signal of strength A.
# Simulate validation scores for a true AUC=A with fixed event rate r=11/289.
from scipy.stats import norm
r_event = yv.mean()
def simulate_observed_auc(A, S=2000, seed=123):
    d = np.sqrt(2) * norm.ppf(A)          # separation for two equal-var normals
    mu1, mu0 = d/2, -d/2
    r = np.random.default_rng(seed)
    aucs = []
    for _ in range(S):
        scores = np.where(yv==1, r.normal(mu1,1,len(yv)), r.normal(mu0,1,len(yv)))
        aucs.append(roc_auc_score(yv, scores))
    return np.array(aucs)
grid = np.round(np.arange(0.55, 0.86, 0.01), 2)
rows = []
for A in grid:
    sim = simulate_observed_auc(A)
    power = np.mean(sim > null_95)
    rows.append((A, power))
    print(f"  true AUC={A:.2f} -> power={power:.3f}")
pow_df = pd.DataFrame(rows, columns=['true_AUC','power'])
# interpolate AUC for 80% power
if pow_df['power'].min() < 0.80 < pow_df['power'].max():
    a = np.interp(0.80, pow_df['power'].values, pow_df['true_AUC'].values)
    print(f"\n[AUC for 80% power] ~ {a:.3f}")
else:
    a = float('nan')
    print("\n80% power not bracketed in grid")

# ============ SAVE ============
summary = {
 'train_ERpos_n':[len(ytr_all)], 'train_ERpos_pCR':[int(ytr_all.sum())],
 'train_ERposHER2neg_n':[len(ytr_ens)], 'train_ERposHER2neg_pCR':[int(ytr_ens.sum())],
 'train_ERposHER2pos_n':[int(her2pos_erpos.sum())], 'train_ERposHER2pos_pCR':[int(pcr[her2pos_erpos].sum())],
 'val_n':[len(yv)], 'val_pCR':[int(yv.sum())],
 'HER2sens_AUC':[auc_ens], 'HER2sens_AUC_lo':[lo], 'HER2sens_AUC_hi':[hi], 'HER2sens_permB_p':[pB_ens],
 'orig_AUC':[auc_orig], 'orig_AUC_lo':[lo_o], 'orig_AUC_hi':[hi_o],
 'permB2000_mean':[permB.mean()], 'permB2000_SD':[permB.std()], 'permB2000_95th':[null_95], 'permB2000_p':[pB],
 'ERbaseline_direct_AUC':[auc_er_direct], 'ERbaseline_direct_lo':[er_lo], 'ERbaseline_direct_hi':[er_hi],
 'ERbaseline_logit_AUC':[auc_er_logit],
 'power_AUC_for_80pct':[a],
}
pd.DataFrame(summary).T.to_csv(os.path.join(OUT,'extension_summary.csv'), header=['value'])
pow_df.to_csv(os.path.join(OUT,'power_curve.csv'), index=False)
pd.DataFrame({'perm_auc_orig': permB}).to_csv(os.path.join(OUT,'permutation_randomgene_null_2000.csv'), index=False)
pd.DataFrame({'perm_auc_her2sens': permB_ens}).to_csv(os.path.join(OUT,'permutation_her2sens_null.csv'), index=False)
print("\nSAVED extension_summary.csv, power_curve.csv, permutation_*_2000.csv")
print("DONE.")
