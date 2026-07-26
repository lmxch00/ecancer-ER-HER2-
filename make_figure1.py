# -*- coding: utf-8 -*-
"""Regenerate Figure 1 for the ecancer manuscript from the verified 2000-permutation null.

Panel A: AUC (with 95% CI) of three models in the ER+/HER2- validation set (n=289).
Panel B: histogram of 2000 random-gene-set permutation AUCs; observed signature AUC,
         null mean, and null 95th percentile marked.
Writes Figure_1.png and updates figure1_numbers.json with the verified values.
"""
import sys, os, json
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
NULL_CSV = os.path.join(HERE, 'permutation_randomgene_null_2000.csv')
FIG_OUT  = os.path.join(HERE, 'Figure_1.png')
JSON_OUT = os.path.join(HERE, 'figure1_numbers.json')

# --- verified numbers (from extension_summary.csv / manuscript draft) ---
OBSERVED = 0.569
FIXED_CI = (0.333, 0.789)     # pre-specified 2-gene signature
TOP2_CI  = (0.589, 0.879)     # data-mined bootstrap top-2 probes
ER_AUC   = 0.717
ER_CI    = (0.658, 0.773)     # mixed-cohort ER-status baseline (reference only)

# --- load 2000-perm null ---
null = np.loadtxt(NULL_CSV, skiprows=1)
null = null[~np.isnan(null)]
mean = float(np.mean(null))
sd = float(np.std(null, ddof=1))
p95 = float(np.percentile(null, 95))
pval = float(np.mean(null >= OBSERVED))   # one-sided permutation p

print(f"null n={len(null)} mean={mean:.4f} SD={sd:.4f} 95th={p95:.4f} p={pval:.4f}")

# --- update json ---
nums = {
    "fixed": [OBSERVED, FIXED_CI[0], FIXED_CI[1]],
    "top2":  [0.751, TOP2_CI[0], TOP2_CI[1]],
    "er":    [ER_AUC, ER_CI[0], ER_CI[1]],
    "perm_mean": mean,
    "perm95": p95,
    "perm_sd": sd,
    "p": pval,
    "n_perm": int(len(null)),
}
with open(JSON_OUT, 'w', encoding='utf-8') as f:
    json.dump(nums, f, indent=2)
print("wrote", JSON_OUT)

# --- figure ---
fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.2, 4.6))

# Panel A: three models
labels = ['Pre-specified\nsignature', 'Data-mined\ntop-2 probes', 'ER status\n(mixed-cohort)']
aucs   = [OBSERVED, 0.751, ER_AUC]
lo     = [FIXED_CI[0], TOP2_CI[0], ER_CI[0]]
hi     = [FIXED_CI[1], TOP2_CI[1], ER_CI[1]]
err_lo = [a - l for a, l in zip(aucs, lo)]
err_hi = [h - a for a, h in zip(aucs, hi)]
ypos = np.arange(len(labels))[::-1]
colors = ['#c0392b', '#7f8c8d', '#2980b9']
axA.barh(ypos, aucs, color=colors, height=0.55, zorder=2)
axA.errorbar(aucs, ypos,
             xerr=[err_lo, err_hi], fmt='none', ecolor='black',
             capsize=4, elinewidth=1.2, zorder=3)
axA.axvline(0.5, color='gray', ls='--', lw=1, zorder=1)
axA.set_yticks(ypos); axA.set_yticklabels(labels, fontsize=9)
axA.set_xlim(0.2, 0.95)
axA.set_xlabel('AUC (95% CI)', fontsize=10)
axA.set_title('A. Model performance in ER+/HER2- validation set (n=289)', fontsize=10, weight='bold')
for y, a in zip(ypos, aucs):
    axA.text(a + 0.01, y, f'{a:.3f}', va='center', fontsize=8.5)

# Panel B: null histogram
axB.hist(null, bins=30, color='#d6eaf8', edgecolor='#2e86c1', alpha=0.9)
axB.axvline(OBSERVED, color='#c0392b', lw=2.2, label=f'Observed signature AUC = {OBSERVED:.3f}')
axB.axvline(mean, color='black', lw=1.6, label=f'Null mean = {mean:.3f}')
axB.axvline(p95, color='#e67e22', lw=1.8, ls='--', label=f'Null 95th pct = {p95:.3f}')
axB.set_xlabel('Permutation AUC (random gene sets)', fontsize=10)
axB.set_ylabel('Frequency', fontsize=10)
axB.set_title(f'B. 2000 random-gene-set permutation null (p = {pval:.2f})', fontsize=10, weight='bold')
axB.legend(fontsize=8, loc='upper left', framealpha=0.95)

plt.tight_layout()
plt.savefig(FIG_OUT, dpi=300, bbox_inches='tight')
print("wrote", FIG_OUT)
