#!/usr/bin/env python3
# Rerun of the JENCI ER+/HER2- pCR 2-probe analysis on REAL local data.
import sys, os, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'data')   # place processed GEO CSVs here (see README)
OUT = HERE
os.makedirs(OUT, exist_ok=True)
SEED = 42
np.random.seed(SEED)
PROBES = ['201887_at', '213872_at']   # IL13RA1, C6orf62

def load_expr(path):
    return pd.read_csv(path, index_col=0)

# ---------- Load & align probes ----------
xt = load_expr(os.path.join(DATA_DIR, 'X_GSE20194.csv'))
xv = load_expr(os.path.join(DATA_DIR, 'X_GSE25066.csv'))
if 'ID_REF' in xt.columns:
    xt = xt.drop(columns=['ID_REF'])          # stray column in GSE20194
common = xt.columns.intersection(xv.columns)
xt = xt[common]; xv = xv[common]
print(f"Common probes aligned: {len(common)}")

# ---------- Training clinical (GSE20194) ----------
ct = pd.read_csv(os.path.join(DATA_DIR, 'GSE20194_clinical_labels.csv')).set_index('Sample_geo_accession')
ct = ct.loc[ct.index.intersection(xt.index)]
xt = xt.loc[ct.index]
er   = ct['characteristics: ER_status'].values
pcr  = (ct['characteristics: pCR_vs_RD'].values == 'pCR').astype(int)
mask_tr = er == 'P'
Xtr_full = xt[mask_tr].values.astype(float); ytr = pcr[mask_tr]
print(f"[Train ER+] n={len(ytr)} pCR={ytr.sum()} ({ytr.mean():.1%})")

# ---------- Validation clinical (GSE25066 ER+/HER2-) ----------
cv_raw = pd.read_csv(os.path.join(DATA_DIR, 'clinical_GSE25066.csv'))
cv = cv_raw.set_index(cv_raw.columns[0])
cv = cv.loc[cv.index.intersection(xv.index)]
xv = xv.loc[cv.index]
erv  = cv['ER'].values; her2v = cv['HER2'].values; pcrv = cv['pCR'].values.astype(int)
mask_v = (erv == '+') & (her2v == '-')
Xv_full = xv[mask_v].values.astype(float); yv = pcrv[mask_v]
print(f"[Val ER+/HER2-] n={len(yv)} pCR={yv.sum()} ({yv.mean():.1%})")
print(f"[Full GSE25066] n={len(pcrv)} ER+ pCR={pcrv[erv=='+'].sum()}/{ (erv=='+').sum() } "
      f"ER- pCR={pcrv[erv=='-'].sum()}/{ (erv=='-').sum() }")

# ---------- 2-probe observed model ----------
Xtr2 = xt.loc[xt.index[mask_tr], PROBES].values.astype(float)
Xv2  = xv.loc[xv.index[mask_v], PROBES].values.astype(float)
sc = StandardScaler(); Xtr2s = sc.fit_transform(Xtr2); Xv2s = sc.transform(Xv2)
model = LogisticRegression(C=1e10, solver='liblinear', random_state=42)
model.fit(Xtr2s, ytr)
pred = model.predict_proba(Xv2s)[:,1]
auc_obs = roc_auc_score(yv, pred)
fpr,tpr,thr = roc_curve(yv, pred); yi = np.argmax(tpr-fpr); t = thr[yi]
pc = (pred >= t).astype(int); tn,fp,fn,tp = confusion_matrix(yv, pc).ravel()
print(f"\n[OBSERVED] 2-probe AUC = {auc_obs:.4f}")
print(f"  Sens={tp/(tp+fn):.0%} Spec={tn/(tn+fp):.0%} PPV={tp/(tp+fp):.1%} NPV={tn/(tn+fn):.1%}")

# ---------- Bootstrap stability (regularized LASSO, stability selection) ----------
variances = np.var(Xtr_full, axis=0)
top200_idx = np.argsort(variances)[::-1][:200]
cand_cols = np.array(common)[top200_idx].tolist()
Xtr_c = Xtr_full[:, top200_idx]; Xv_c = Xv_full[:, top200_idx]
N_BOOT = 500
sel = np.zeros(Xtr_c.shape[1]); coefs = np.zeros((N_BOOT, Xtr_c.shape[1]))
for b in range(N_BOOT):
    idx = np.random.choice(len(ytr), len(ytr), replace=True)
    if ytr[idx].sum() < 2 or (len(ytr)-ytr[idx].sum()) < 2: continue
    m = LogisticRegression(C=0.05, penalty='l1', solver='liblinear',
                           class_weight='balanced', max_iter=5000, random_state=b)
    m.fit(StandardScaler().fit_transform(Xtr_c[idx]), ytr[idx])
    sel += (np.abs(m.coef_[0]) > 1e-6).astype(int); coefs[b] = m.coef_[0]
freq = sel / N_BOOT * 100
gene_map = {'201887_at':'IL13RA1','213872_at':'C6orf62'}
bs = pd.DataFrame({'probe': cand_cols,
                   'gene': [gene_map.get(p,'') for p in cand_cols],
                   'sel_freq_pct': freq, 'coef_mean': coefs.mean(0), 'coef_sd': coefs.std(0)})
bs_sorted = bs.sort_values('sel_freq_pct', ascending=False)
print("\n[BOOTSTRAP] top stable (>=30%):")
print(bs_sorted[bs_sorted['sel_freq_pct']>=30][['probe','gene','sel_freq_pct']].to_string(index=False))
bs.to_csv(os.path.join(OUT,'bootstrap_stability_all200.csv'), index=False)

# ---------- Permutation A: label-permutation on training ----------
N_PERM = 500; permA = np.zeros(N_PERM); rng = np.random.default_rng(SEED)
for p in range(N_PERM):
    yp = rng.permutation(ytr)
    m = LogisticRegression(C=1e10, solver='liblinear').fit(Xtr2s, yp)
    permA[p] = roc_auc_score(yv, m.predict_proba(Xv2s)[:,1])
pA = np.mean(permA >= auc_obs)
print(f"\n[PERM A label-shuffle] mean={permA.mean():.4f} SD={permA.std():.4f} "
      f"95th={np.percentile(permA,95):.4f} max={permA.max():.4f} p={pA:.4f}")

# ---------- Permutation B: random-gene-set (manuscript-described) ----------
permB = np.zeros(N_PERM); rng2 = np.random.default_rng(SEED)
for p in range(N_PERM):
    pick = rng2.choice(len(cand_cols), size=2, replace=False)
    Xb = StandardScaler().fit_transform(Xtr_c[:, pick]); Xvb = StandardScaler().fit_transform(Xv_c[:, pick])
    m = LogisticRegression(C=0.05, penalty='l1', solver='liblinear', class_weight='balanced', max_iter=5000).fit(Xb, ytr)
    permB[p] = 0.5 if np.abs(m.coef_[0]).sum()==0 else roc_auc_score(yv, m.predict_proba(Xvb)[:,1])
pB = np.mean(permB >= auc_obs)
print(f"[PERM B random-gene]  mean={permB.mean():.4f} SD={permB.std():.4f} "
      f"95th={np.percentile(permB,95):.4f} max={permB.max():.4f} p={pB:.4f}")

# ---------- DCA ----------
print("\n[DCA]")
ths = np.arange(0,0.51,0.01); n=len(yv); erate=yv.mean(); nb_model=[]; nb_ta=[]
for t in ths:
    pct = (pred>=t).astype(int)
    nb = max(0, (pct@yv)/n - (pct@(1-yv))/n * t/(1-t))
    ta = erate - (1-erate)*t/(1-t)
    nb_model.append(nb); nb_ta.append(ta)
    if round(t,2) in (0.05,0.10,0.20):
        print(f"  t={t:.2f}: NB_model={nb:.4f} NB_treatAll={ta:.4f}")

# ---------- ER baseline (full cohort) ----------
er_score = np.where(erv=='-', 0.85, 0.3)
auc_er_coding = roc_auc_score(pcrv, er_score)
Xer = StandardScaler().fit_transform((erv=='-').astype(int).reshape(-1,1))
auc_er_logit = roc_auc_score(pcrv, LogisticRegression(C=1e10,solver='liblinear').fit(Xer,pcrv).predict_proba(Xer)[:,1])
print(f"\n[ER BASELINE] coding AUC={auc_er_coding:.4f} univariate-logit AUC={auc_er_logit:.4f}")

# ---------- Diagnostic: AUC of top-2 stable probes from THIS bootstrap ----------
top2 = bs_sorted.iloc[0:2]['probe'].tolist()
Xtr_t2 = xt.loc[xt.index[mask_tr], top2].values.astype(float)
Xv_t2  = xv.loc[xv.index[mask_v], top2].values.astype(float)
s2 = StandardScaler(); Xtr_t2s = s2.fit_transform(Xtr_t2); Xv_t2s = s2.transform(Xv_t2)
m_t2 = LogisticRegression(C=1e10, solver='liblinear').fit(Xtr_t2s, ytr)
auc_top2 = roc_auc_score(yv, m_t2.predict_proba(Xv_t2s)[:,1])
print(f"\n[DIAGNOSTIC] AUC of top-2 stable probes ({top2}) = {auc_top2:.4f}")

# ---------- Save ----------
pd.DataFrame({'perm_auc': permA}).to_csv(os.path.join(OUT,'permutation_label_null.csv'), index=False)
pd.DataFrame({'perm_auc': permB}).to_csv(os.path.join(OUT,'permutation_randomgene_null.csv'), index=False)
pd.DataFrame({'threshold': ths,'model_NB': nb_model,'treat_all_NB': nb_ta}).to_csv(os.path.join(OUT,'dca_results.csv'), index=False)
def fv(p):
    v = bs.loc[bs.probe==p,'sel_freq_pct'].values
    return float(v[0]) if len(v) else float('nan')
summary = {'observed_AUC_IL13RA1_C6orf62':[auc_obs],
 'observed_AUC_top2_stable':[auc_top2],'top2_stable_probes':[','.join(top2)],
 'train_n':[len(ytr)],'train_pCR':[int(ytr.sum())],
 'val_n':[len(yv)],'val_pCR':[int(yv.sum())],
 'IL13RA1_freq_pct':[fv('201887_at')],'C6orf62_freq_pct':[fv('213872_at')],
 'permA_mean':[permA.mean()],'permA_SD':[permA.std()],'permA_95th':[np.percentile(permA,95)],'permA_p':[pA],
 'permB_mean':[permB.mean()],'permB_SD':[permB.std()],'permB_95th':[np.percentile(permB,95)],'permB_p':[pB],
 'ER_baseline_AUC':[auc_er_coding]}
pd.DataFrame(summary).T.to_csv(os.path.join(OUT,'rerun_summary.csv'), header=['value'])
print(f"\nALL OUTPUTS SAVED TO {OUT}\nDONE.")
