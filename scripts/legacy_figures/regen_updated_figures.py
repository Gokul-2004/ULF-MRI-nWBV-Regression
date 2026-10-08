"""
Regenerate only the three updated figures into updated_figures/ subfolder,
then remove the originals from paper_figures/.
Figures: fig3_cdr_boxplot, fig1_architecture, fig_loocv_icc
"""
import json, csv, os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

ROOT     = Path(__file__).parent.parent
OUT_DIR  = ROOT / "standalone_paper" / "paper_figures"
UPD_DIR  = OUT_DIR / "updated_figures"
EXP      = ROOT / "experiments"

OASIS_FT = EXP / "oasis_finetune"  / "finetune_results.json"
LOOCV    = EXP / "loocv_cross_session" / "results.json"
PARTS    = ROOT / "data" / "ds006557_data" / "participants.tsv"

UPD_DIR.mkdir(parents=True, exist_ok=True)

with open(OASIS_FT) as f: oasis_ft = json.load(f)
with open(LOOCV)    as f: loocv    = json.load(f)

ages = {}
with open(PARTS) as f:
    for row in csv.DictReader(f, delimiter='\t'):
        ages[f"sub-{row['participant_id']}"] = float(row['age'])

plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'font.size':        11,
    'axes.titlesize':   12,
    'axes.labelsize':   11,
    'xtick.labelsize':  10,
    'ytick.labelsize':  10,
    'legend.fontsize':  10,
    'figure.dpi':       150,
    'savefig.dpi':      300,
    'savefig.bbox':     'tight',
    'savefig.pad_inches': 0.08,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'lines.linewidth':  1.5,
})

BLUE   = '#1f77b4'
ORANGE = '#ff7f0e'
GREEN  = '#2ca02c'
RED    = '#d62728'
PURPLE = '#9467bd'
GRAY   = '#7f7f7f'
THRESH = 0.020

def savefig_upd(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(str(UPD_DIR / f"{name}.{ext}"), format=ext)
    plt.close(fig)
    print(f"  Saved updated: {name}.png / .pdf  →  {UPD_DIR}")

# ── FIG 1: Architecture ───────────────────────────────────────────────────────
def fig1_architecture():
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis('off')

    boxes = [
        (0.3, 1.2, 1.4, "IXI\nPre-training\n(n=156)",       BLUE,   "white"),
        (2.2, 1.2, 1.4, "OASIS-1\nFine-tuning\n(n=375)",    ORANGE, "white"),
        (4.1, 1.2, 1.4, "LOOCV\nAdaptation\n(n=23)",        GREEN,  "white"),
        (6.2, 0.6, 1.4, "ViT3D\n4.23M params\n64³ → nWBV",  PURPLE, "white"),
        (8.1, 1.2, 1.4, "Inference\n47 ms\nno FreeSurfer",   RED,    "white"),
    ]
    for x, y, w, label, color, tc in boxes:
        rect = plt.Rectangle((x, y), w, 0.9, linewidth=1.5,
                               edgecolor=color, facecolor=color, alpha=0.85)
        ax.add_patch(rect)
        ax.text(x + w/2, y + 0.45, label, ha='center', va='center',
                fontsize=9, color=tc, fontweight='bold', linespacing=1.4)

    arrows = [(1.7, 2.0), (3.6, 2.0), (5.5, 2.0), (7.6, 2.0)]
    for x, y in arrows:
        ax.annotate('', xy=(x+0.4, y), xytext=(x, y),
                    arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5))

    labels_y = [
        (0.55, "Physics 64 mT\nsimulation"),
        (2.45, "nWBV labels\n(FreeSurfer)"),
        (4.35, "Cross-session\nLOOCV"),
        (6.55, "FastSurfer GT\n(3T paired)"),
    ]
    for x, txt in labels_y:
        ax.text(x + 0.7, 0.35, txt, ha='center', va='center',
                fontsize=8, color=GRAY, style='italic', linespacing=1.3)

    ax.set_title("Three-Stage Training Pipeline", fontsize=13,
                 fontweight='bold', pad=14)
    fig.subplots_adjust(top=0.88)
    savefig_upd(fig, "fig1_architecture")

# ── FIG 3: CDR Boxplot ────────────────────────────────────────────────────────
def fig3_cdr_boxplot():
    subjects = oasis_ft['per_subject_test']
    cdr_groups = {0.0: [], 0.5: [], 1.0: []}
    for s in subjects:
        if s['cdr'] in cdr_groups:
            cdr_groups[s['cdr']].append(s['true_nwbv'])

    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    labels     = ['CDR 0.0\n($n=8$)', 'CDR 0.5\n($n=8$)', 'CDR 1.0\n($n=2$)']
    data       = [cdr_groups[0.0], cdr_groups[0.5], cdr_groups[1.0]]
    BOX_COLORS = ['#4da6ff', '#1a6ed8', '#0a3d8f']

    bp = ax.boxplot(data, patch_artist=True, widths=0.48,
                    medianprops=dict(color='white', lw=2.5),
                    whiskerprops=dict(lw=1.4, color='#444444'),
                    capprops=dict(lw=1.4, color='#444444'),
                    flierprops=dict(marker='o', markersize=5, markerfacecolor='#888'))
    for patch, color in zip(bp['boxes'], BOX_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.88)
        patch.set_edgecolor('#333333')
        patch.set_linewidth(1.4)

    means = [np.mean(cdr_groups[k]) for k in [0.0, 0.5, 1.0]]
    box_tops = [max(cdr_groups[k]) for k in [0.0, 0.5, 1.0]]
    for i, (m, top, c) in enumerate(zip(means, box_tops, BOX_COLORS), 1):
        # Place label above the highest data point / whisker cap
        label_y = top + 0.022
        ax.text(i, label_y, f'μ={m:.3f}', ha='center', va='bottom',
                fontsize=9.5, color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=c,
                          edgecolor='white', linewidth=1.2, alpha=0.95))

    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('True nWBV (FastSurfer)', fontsize=11)
    ax.set_title('nWBV by CDR Stage\nOASIS-1 Test Set ($n=18$)',
                 fontsize=11, fontweight='bold')
    ax.set_ylim(0.58, 1.08)

    bracket_y = 1.00
    tick_len  = 0.006
    ax.plot([1, 1, 3, 3],
            [bracket_y - tick_len, bracket_y, bracket_y, bracket_y - tick_len],
            color='#333333', lw=1.4)
    ax.text(2, bracket_y + 0.006, r'$p = 0.022$   CDR 0 vs CDR 1',
            ha='center', va='bottom', fontsize=9, color='#222222')

    ax.text(0.5, -0.16,
            'Literature: 5–10% nWBV reduction per CDR stage',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=8, style='italic', color='#555555',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f4ff',
                      edgecolor='#aabbdd', alpha=0.95))

    fig.tight_layout()
    fig.subplots_adjust(top=0.90, bottom=0.20)
    savefig_upd(fig, "fig3_cdr_boxplot")

# ── FIG loocv_icc ─────────────────────────────────────────────────────────────
def fig_loocv_icc():
    subjects  = loocv['per_subject']
    trues     = np.array([s['true_nwbv'] for s in subjects])
    preds_hfe = np.array([s['pred_hfe']  for s in subjects])
    preds_hfc = np.array([s['pred_hfc']  for s in subjects])
    abs_errs  = np.abs(preds_hfe - trues)
    subj_ids  = [s['subject'].replace('sub-HYPE', 'HY') for s in subjects]
    sort_idx  = np.argsort(abs_errs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # Panel A: per-subject error bar
    colors_e = [RED if e > THRESH else BLUE for e in abs_errs[sort_idx]]
    axes[0].barh([subj_ids[i] for i in sort_idx],
                  abs_errs[sort_idx], color=colors_e, alpha=0.8)
    axes[0].axvline(THRESH, color=RED, lw=2, linestyle='--',
                    label=f'Reference threshold ({THRESH})')
    axes[0].axvline(np.mean(abs_errs), color=BLUE, lw=2, linestyle='-',
                    label=f'MAE $= {np.mean(abs_errs):.4f}$')
    axes[0].set_xlabel('|Error| (nWBV)', fontsize=11)
    axes[0].set_title(f'(A) Per-Subject Absolute Error\n'
                       f'LOOCV $n=23$: {int(np.sum(abs_errs < THRESH))}/23 below threshold',
                       fontsize=11, fontweight='bold')
    axes[0].legend(fontsize=9)

    # Panel B: ICC scatter HFC vs HFE predictions
    axes[1].scatter(preds_hfc, preds_hfe, c=PURPLE, s=60, alpha=0.85,
                    edgecolors='white', lw=0.5, zorder=3)
    lims = [min(preds_hfc.min(), preds_hfe.min()) - 0.005,
            max(preds_hfc.max(), preds_hfe.max()) + 0.005]
    axes[1].plot(lims, lims, 'k--', lw=1, alpha=0.5, label='Identity')
    axes[1].set_xlim(lims); axes[1].set_ylim(lims)
    axes[1].set_aspect('equal')

    for i, sid in enumerate(subj_ids):
        if abs(preds_hfe[i] - preds_hfc[i]) > 0.012:
            axes[1].annotate(sid, (preds_hfc[i], preds_hfe[i]),
                             textcoords='offset points', xytext=(4, 3),
                             fontsize=7.5, color=GRAY)

    icc_val = loocv['icc_31_hfc_hfe']
    ci      = loocv['icc_31_bootstrap_ci_95']
    axes[1].text(0.05, 0.93,
                 f'ICC(3,1) $= {icc_val:.3f}$\n95% CI [{ci[0]:.3f}–{ci[1]:.3f}]',
                 transform=axes[1].transAxes, fontsize=10,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))
    axes[1].set_xlabel('HFC Session Prediction (nWBV)', fontsize=11)
    axes[1].set_ylabel('HFE Session Prediction (nWBV)', fontsize=11)
    axes[1].set_title('(B) Inter-Session Reliability\n'
                       'Paired HFC vs HFE Predictions ($n=23$)',
                       fontsize=11, fontweight='bold')
    axes[1].legend(fontsize=9)

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    savefig_upd(fig, "fig_loocv_icc")


if __name__ == "__main__":
    print("Regenerating 3 updated figures → updated_figures/\n")
    fig1_architecture()
    fig3_cdr_boxplot()
    fig_loocv_icc()

    # Remove originals from paper_figures/
    for name in ("fig1_architecture", "fig3_cdr_boxplot", "fig_loocv_icc"):
        for ext in ("png", "pdf"):
            p = OUT_DIR / f"{name}.{ext}"
            if p.exists():
                p.unlink()
                print(f"  Deleted original: {p.name}")

    print("\nDone. Updated figures are in updated_figures/")
    for f in sorted(UPD_DIR.iterdir()):
        print(f"  {f.name}  ({f.stat().st_size:,} bytes)")
