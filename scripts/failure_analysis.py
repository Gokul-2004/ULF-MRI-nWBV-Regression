"""
Failure Analysis: Where Does the Model Break?
=============================================
Generates Fig 9: a 4-panel failure analysis figure showing:
  (A) Residual plot — prediction error vs true nWBV (OASIS)
  (B) Error by nWBV tertile — where systematic bias lives
  (C) Real 64mT: error vs age scatter with worst-case labels
  (D) CI width vs actual error — calibration check

Usage: python3 scripts/failure_analysis.py
Output: paper_figures/fig9_failure_analysis.{pdf,png}
        experiments/paper_statistics/failure_analysis.json
"""

import sys, json
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

FIGURES_DIR = project_root / "paper_figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = project_root / "experiments" / "paper_statistics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────

with open(project_root / "experiments" / "oasis_finetune" / "finetune_results.json") as f:
    d = json.load(f)
ps = d['per_subject_test']
oasis_trues = np.array([s['true_nwbv'] for s in ps])
oasis_preds = np.array([s['pred_nwbv'] for s in ps])
oasis_cdrs  = [s['cdr'] for s in ps]
oasis_errors = oasis_preds - oasis_trues
oasis_abs    = np.abs(oasis_errors)

with open(project_root / "experiments" / "real64mt_eval" / "mc_dropout_ci.json") as f:
    mc = json.load(f)

ages = {}
with open(project_root / "data" / "ds006557_data" / "participants.tsv") as f:
    for line in f.readlines()[1:]:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            try:
                ages['sub-HYPE' + parts[0].replace('HYPE','').zfill(2)] = float(parts[1])
            except: pass

real_ages, real_errs, real_abs, real_ci, real_names = [], [], [], [], []
for s in mc['per_subject']:
    subj = s['subject']
    age = ages.get(subj, None)
    if age is None: continue
    err = s['mean'] - s['gt']
    real_ages.append(age); real_errs.append(err)
    real_abs.append(abs(err)); real_ci.append(s['ci_width'])
    real_names.append(subj.replace('sub-HYPE', 'S'))

real_ages = np.array(real_ages)
real_errs = np.array(real_errs)
real_abs  = np.array(real_abs)
real_ci   = np.array(real_ci)

# ── Figure ─────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle('Failure Analysis: Where Does the Model Break?\n'
             'OASIS-1 Test Set (n=38) + Real 64\u202fmT (n=23)',
             fontsize=12, fontweight='bold', color='#1B4F72')

# ── Panel A: Residual plot (pred error vs true nWBV) ─────────────────────
ax = axes[0, 0]

# Colour by CDR
cdr_colors = {0.0: '#1A7A8A', 0.5: '#E67E22', 1.0: '#C0392B', None: '#95A5A6'}
for i, (t, e, c) in enumerate(zip(oasis_trues, oasis_errors, oasis_cdrs)):
    col = cdr_colors.get(c, '#95A5A6')
    ax.scatter(t, e, c=col, s=45, alpha=0.85, edgecolors='white', lw=0.5, zorder=3)

# Annotate worst cases
worst5 = np.argsort(oasis_abs)[::-1][:5]
for i in worst5:
    ax.annotate(f'err={oasis_errors[i]:+.3f}',
                (oasis_trues[i], oasis_errors[i]),
                textcoords='offset points', xytext=(6, 3),
                fontsize=7, color='#C0392B', fontweight='bold')

ax.axhline(0, color='black', lw=1.2, ls='--', alpha=0.5, label='Zero error')
ax.axhline(np.mean(oasis_errors), color='#D35400', lw=1.5, ls='-',
           alpha=0.7, label=f'Mean bias = {np.mean(oasis_errors):+.3f}')

# Shade atrophy zone (low nWBV where model fails)
ax.axvspan(oasis_trues.min(), 0.76, alpha=0.08, color='#C0392B',
           label='Atrophy zone (nWBV < 0.76)')

# CDR legend
patches = [mpatches.Patch(color=cdr_colors[0.0], label='CDR 0.0'),
           mpatches.Patch(color=cdr_colors[0.5], label='CDR 0.5'),
           mpatches.Patch(color=cdr_colors[1.0], label='CDR 1.0'),
           mpatches.Patch(color=cdr_colors[None], label='CDR unknown')]
ax.legend(handles=patches, fontsize=7.5, loc='upper left')
ax.set_xlabel('True nWBV (FreeSurfer)', fontsize=10)
ax.set_ylabel('Prediction Error (pred − true)', fontsize=10)
ax.set_title('(A) Residual Plot — OASIS-1 Test Set\n'
             'Model over-predicts for atrophied brains', fontsize=10, fontweight='bold')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.yaxis.grid(True, alpha=0.3)

# ── Panel B: Error by nWBV tertile ────────────────────────────────────────
ax2 = axes[0, 1]

pcts = [np.percentile(oasis_trues, 33), np.percentile(oasis_trues, 67)]
masks = [
    oasis_trues < pcts[0],
    (oasis_trues >= pcts[0]) & (oasis_trues < pcts[1]),
    oasis_trues >= pcts[1]
]
labels = [f'Low nWBV\n(<{pcts[0]:.3f})', f'Mid nWBV\n({pcts[0]:.3f}–{pcts[1]:.3f})',
          f'High nWBV\n(>{pcts[1]:.3f})']
maes   = [np.mean(oasis_abs[m]) for m in masks]
biases = [np.mean(oasis_errors[m]) for m in masks]
ns     = [m.sum() for m in masks]
colors_b = ['#C0392B', '#E67E22', '#1A7A8A']

bars = ax2.bar(labels, maes, color=colors_b, alpha=0.85, width=0.5,
               edgecolor='white', linewidth=1.2)
for bar, mae, bias, n in zip(bars, maes, biases, ns):
    ax2.text(bar.get_x() + bar.get_width()/2, mae + 0.002,
             f'MAE={mae:.3f}\nbias={bias:+.3f}\nn={n}',
             ha='center', fontsize=8, fontweight='bold')

ax2.axhline(0.020, color='#27AE60', ls='--', lw=1.5,
            label='Clinical threshold (0.020)')
ax2.set_ylabel('MAE (nWBV units)', fontsize=10)
ax2.set_title('(B) Error by True nWBV Tertile — OASIS\n'
              'Failure concentrated in atrophied subjects',
              fontsize=10, fontweight='bold')
ax2.legend(fontsize=8.5)
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
ax2.yaxis.grid(True, alpha=0.3)

# ── Panel C: Real 64mT — error vs age ─────────────────────────────────────
ax3 = axes[1, 0]

scatter = ax3.scatter(real_ages, real_abs, c=real_abs,
                      cmap='RdYlGn_r', s=55, alpha=0.9,
                      edgecolors='white', lw=0.5, zorder=3,
                      vmin=0, vmax=0.08)

# Label worst 4
worst4_idx = np.argsort(real_abs)[::-1][:4]
for i in worst4_idx:
    ax3.annotate(f'{real_names[i]}\n(age {real_ages[i]:.0f})',
                 (real_ages[i], real_abs[i]),
                 textcoords='offset points', xytext=(6, 3),
                 fontsize=7, color='#C0392B', fontweight='bold')

# Regression line
m, b = np.polyfit(real_ages, real_abs, 1)
xs = np.linspace(real_ages.min(), real_ages.max(), 100)
r_age, p_age = stats.pearsonr(real_ages, real_abs)
ax3.plot(xs, m*xs + b, color='#C0392B', lw=2, alpha=0.7,
         label=f'Trend: r={r_age:.3f}, p={p_age:.3f}')

ax3.axhline(0.020, color='#27AE60', ls='--', lw=1.5,
            label='Clinical threshold (0.020)', alpha=0.8)

plt.colorbar(scatter, ax=ax3, label='Absolute error', shrink=0.8)
ax3.set_xlabel('Subject Age (years)', fontsize=10)
ax3.set_ylabel('Absolute Error (nWBV units)', fontsize=10)
ax3.set_title(f'(C) Error vs Age — Real 64\u202fmT\n'
              f'Older subjects harder (r={r_age:.3f}, p={p_age:.3f})',
              fontsize=10, fontweight='bold')
ax3.legend(fontsize=8.5, loc='upper left')
ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)
ax3.yaxis.grid(True, alpha=0.3)

# ── Panel D: CI width vs actual error (calibration) ───────────────────────
ax4 = axes[1, 1]

r_ci, p_ci = stats.pearsonr(real_ci, real_abs)
ax4.scatter(real_ci, real_abs, c='#5D6D7E', s=55, alpha=0.85,
            edgecolors='white', lw=0.5, zorder=3)

# Label worst cases
for i in worst4_idx:
    ax4.annotate(real_names[i],
                 (real_ci[i], real_abs[i]),
                 textcoords='offset points', xytext=(4, 3),
                 fontsize=7, color='#C0392B')

# Ideal calibration: CI width should correlate with error
xs_ci = np.linspace(real_ci.min(), real_ci.max(), 100)
m_ci, b_ci = np.polyfit(real_ci, real_abs, 1)
ax4.plot(xs_ci, m_ci*xs_ci + b_ci, color='#5D6D7E', lw=1.5, ls='--',
         alpha=0.6, label=f'Fit: r={r_ci:.3f} (p={p_ci:.3f})')

ax4.axhline(0.020, color='#27AE60', ls='--', lw=1.5,
            label='Clinical threshold (0.020)', alpha=0.8)

# Box showing miscalibration zone
ax4.fill_between(xs_ci, 0.020, real_abs.max()+0.005,
                 alpha=0.05, color='#C0392B',
                 label='Above clinical threshold')

ax4.set_xlabel('MC Dropout 95% CI Width', fontsize=10)
ax4.set_ylabel('Actual Absolute Error (nWBV)', fontsize=10)
ax4.set_title(f'(D) CI Width vs Actual Error — Real 64\u202fmT\n'
              f'Poor calibration: r={r_ci:.3f} (p={p_ci:.3f})',
              fontsize=10, fontweight='bold')
ax4.legend(fontsize=8)
ax4.spines['top'].set_visible(False); ax4.spines['right'].set_visible(False)
ax4.yaxis.grid(True, alpha=0.3)

fig.text(0.5, -0.01,
         'Key failure mode: model predicts near-constant ≈0.84 nWBV; '
         'fails for atrophied brains (nWBV < 0.76) and older subjects. '
         'MC-Dropout CIs do not predict where errors occur (miscalibrated).',
         ha='center', fontsize=8.5, color='#5D6D7E',
         style='italic', wrap=True)

plt.tight_layout(pad=1.8)
for ext in ['pdf', 'png']:
    fig.savefig(FIGURES_DIR / f"fig9_failure_analysis.{ext}",
                dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("Fig 9 saved.")

# ── Save JSON summary ──────────────────────────────────────────────────────
summary = {
    'oasis_test': {
        'n': len(oasis_trues),
        'mae_overall': round(float(np.mean(oasis_abs)), 4),
        'bias_overall': round(float(np.mean(oasis_errors)), 4),
        'max_abs_error': round(float(np.max(oasis_abs)), 4),
        'mae_by_tertile': {
            'low_nwbv': {'thresh': round(float(pcts[0]),3),
                          'mae': round(float(maes[0]),4),
                          'bias': round(float(biases[0]),4), 'n': int(ns[0])},
            'mid_nwbv': {'mae': round(float(maes[1]),4),
                          'bias': round(float(biases[1]),4), 'n': int(ns[1])},
            'high_nwbv': {'thresh': round(float(pcts[1]),3),
                           'mae': round(float(maes[2]),4),
                           'bias': round(float(biases[2]),4), 'n': int(ns[2])},
        },
        'worst_cases': [
            {'idx': int(i), 'true': round(float(oasis_trues[i]),4),
             'pred': round(float(oasis_preds[i]),4),
             'error': round(float(oasis_errors[i]),4), 'cdr': oasis_cdrs[i]}
            for i in worst5
        ]
    },
    'real_64mt': {
        'n': len(real_ages),
        'age_vs_error_pearson_r': round(float(r_age), 4),
        'age_vs_error_p': round(float(p_age), 4),
        'ci_vs_error_pearson_r': round(float(r_ci), 4),
        'ci_vs_error_p': round(float(p_ci), 4),
        'mae_by_age_group': {
            'under35': round(float(np.mean(real_abs[real_ages < 35])), 4),
            '35_55':   round(float(np.mean(real_abs[(real_ages >= 35) & (real_ages < 55)])), 4),
            'over55':  round(float(np.mean(real_abs[real_ages >= 55])), 4),
        }
    },
    'key_finding': (
        'Model predicts near-constant ~0.84 nWBV; '
        'fails for atrophied brains (low nWBV). '
        'MC-Dropout CIs not calibrated to actual errors.'
    )
}
with open(OUT_DIR / "failure_analysis.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Summary saved.")
print("\nKey findings:")
print(f"  Low nWBV MAE:  {maes[0]:.4f}  (atrophy zone — model fails here)")
print(f"  High nWBV MAE: {maes[2]:.4f}  (normal range — model works)")
print(f"  Age vs error:  r={r_age:.3f}, p={p_age:.3f}")
print(f"  CI calibration: r={r_ci:.3f}, p={p_ci:.3f} (poor)")
