"""
Generate Figures 1, 6, 7 for IEEE paper:
  Fig 1 — Architecture diagram
  Fig 6 — 3-way baseline comparison
  Fig 7 — Scan comparison: 3T vs sim 64mT vs real 64mT
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import nibabel as nib

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
FIGURES_DIR = project_root / "experiments" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DS_DIR = project_root / "data" / "ds006557_data"

C_BLUE   = "#1B4F72"
C_TEAL   = "#1A7A8A"
C_GREEN  = "#1E8449"
C_ORANGE = "#D35400"
C_GREY   = "#5D6D7E"

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def box(ax, x, y, w, h, label, sub=None, color=C_BLUE, fs=8.5):
    ax.add_patch(FancyBboxPatch((x,y), w, h,
        boxstyle="round,pad=0.025", facecolor=color,
        edgecolor='white', linewidth=1.5, alpha=0.93, zorder=2))
    cy = y + h*(0.65 if sub else 0.5)
    ax.text(x+w/2, cy, label, ha='center', va='center',
            color='white', fontsize=fs, fontweight='bold', zorder=3)
    if sub:
        ax.text(x+w/2, y+h*0.27, sub, ha='center', va='center',
                color='#D6EAF8', fontsize=fs-1.5, zorder=3)

def arrow(ax, x1, y1, x2, y2, color='#2C3E50'):
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
        arrowprops=dict(arrowstyle='->', color=color, lw=1.6),
        zorder=3)

def save(fig, name):
    for ext in ['pdf','png']:
        fig.savefig(FIGURES_DIR/f"{name}.{ext}", dpi=200,
                    bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {name}.pdf  {name}.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 1 — ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────

def fig1():
    fig, ax = plt.subplots(figsize=(15, 5.8))
    ax.set_xlim(0, 15); ax.set_ylim(0, 5.8); ax.axis('off')

    ax.text(7.5, 5.5, 'Physics-Constrained ViT3D: Two-Stage Pipeline for nWBV Prediction from 64 mT MRI',
            ha='center', va='center', fontsize=11, fontweight='bold', color=C_BLUE)

    # ── Stage labels ──
    for xc, lbl, col in [(1.2,'STAGE 1\nPre-training',C_TEAL),
                          (6.2,'STAGE 2\nFine-tuning',C_ORANGE),
                          (11.2,'DEPLOYMENT',C_GREEN)]:
        ax.text(xc, 5.15, lbl, ha='center', fontsize=8, color=col, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=col, lw=1.2))

    # ── STAGE 1 ──
    box(ax, 0.1, 3.4, 1.4, 0.85, 'IXI Dataset', 'n=156  3T T1w', C_GREY)
    box(ax, 0.1, 1.95, 1.4, 1.15, 'Physics Sim', 'T1/T2 @ 64mT\nRician noise\nB0 inhom.', C_TEAL)
    box(ax, 0.1, 0.7, 1.4, 0.95, 'Sim. 64mT', '64×64×64\nvoxels', C_TEAL)
    arrow(ax, 0.8, 3.4, 0.8, 3.1)
    arrow(ax, 0.8, 1.95, 0.8, 1.65)

    box(ax, 1.75, 1.55, 1.6, 1.85, 'ViT3D\nEncoder', 'patch=16³\ndim=256\n4 layers\n8 heads', C_BLUE)
    arrow(ax, 1.5, 1.17, 1.75, 2.1)

    box(ax, 3.55, 1.95, 1.15, 0.9, '4-Output\nHead', 'BTF TCR\nVBR MCI', C_BLUE)
    arrow(ax, 3.35, 2.47, 3.55, 2.4)
    ax.text(4.12, 1.8, 'MSE↓', ha='center', fontsize=7.5, color=C_TEAL, style='italic')

    ax.plot([0.05,0.05,4.8,4.8],[0.55,0.4,0.4,0.55], color=C_TEAL, lw=1.2)
    ax.text(2.4, 0.2, 'Stage 1: Physics-informed pre-training on IXI (n=156)',
            ha='center', fontsize=7.5, color=C_TEAL)

    # Stage 1 → 2 arrow (weight transfer)
    arrow(ax, 4.82, 2.47, 5.1, 2.47, color=C_ORANGE)
    ax.text(4.96, 2.65, 'weights\ntransfer', ha='center', fontsize=7, color=C_ORANGE)

    # ── STAGE 2 ──
    box(ax, 5.1, 3.4, 1.4, 0.85, 'OASIS-1', 'n=375  CDR 0–1', C_GREY)
    box(ax, 5.1, 1.95, 1.4, 1.15, 'Physics Sim', '64mT from 3T\nOASIS scans', C_ORANGE)
    box(ax, 5.1, 0.7, 1.4, 0.95, 'Sim. 64mT\n(OASIS)', '300/37/38\ntrain/val/test', C_ORANGE)
    arrow(ax, 5.8, 3.4, 5.8, 3.1)
    arrow(ax, 5.8, 1.95, 5.8, 1.65)

    box(ax, 6.75, 1.55, 1.6, 1.85, 'ViT3D\n(fine-tune)', '← pretrained\nfull fine-tune\nLR=5×10⁻⁵\n50 epochs', C_BLUE)
    arrow(ax, 6.5, 1.17, 6.75, 2.1)

    box(ax, 8.55, 1.95, 1.15, 0.9, '1-Output\nHead', 'nWBV\nregression', C_ORANGE)
    arrow(ax, 8.35, 2.47, 8.55, 2.4)
    ax.text(9.12, 1.8, 'MSE↓', ha='center', fontsize=7.5, color=C_ORANGE, style='italic')

    ax.plot([5.05,5.05,9.8,9.8],[0.55,0.4,0.4,0.55], color=C_ORANGE, lw=1.2)
    ax.text(7.4, 0.2, 'Stage 2: Clinical fine-tuning on OASIS-1  →  r=0.892 [CI: 0.801–0.943]',
            ha='center', fontsize=7.5, color=C_ORANGE)

    arrow(ax, 9.82, 2.47, 10.1, 2.47, color=C_GREEN)

    # ── DEPLOYMENT ──
    box(ax, 10.1, 3.4, 1.4, 0.85, 'Real 64mT', 'Hyperfine  n=23', C_GREY)
    box(ax, 10.1, 1.95, 1.4, 1.15, 'Pre-process', 'Resize 64³\nNorm [0,1]', C_GREEN)
    box(ax, 10.1, 0.7, 1.4, 0.95, 'Domain\nAdapt.', 'SynthSeg+\npseudo-labels', C_GREEN)
    arrow(ax, 10.8, 3.4, 10.8, 3.1)
    arrow(ax, 10.8, 1.95, 10.8, 1.65)

    box(ax, 11.75, 1.6, 1.55, 1.5, 'ViT3D\n(deployed)', '130 ms\nno FreeSurfer\nCPU/GPU', C_GREEN)
    arrow(ax, 11.5, 1.17, 11.75, 2.0)

    box(ax, 13.5, 1.95, 1.4, 0.9, 'nWBV\nOutput', 'MAE=0.010\nAge ρ=−0.597', C_GREEN)
    arrow(ax, 13.3, 2.4, 13.5, 2.4)

    ax.plot([10.05,10.05,14.95,14.95],[0.55,0.4,0.4,0.55], color=C_GREEN, lw=1.2)
    ax.text(12.5, 0.2, 'Deployment: 130 ms inference, no FreeSurfer dependency',
            ha='center', fontsize=7.5, color=C_GREEN)

    ax.axvline(5.0, color='#BDC3C7', lw=1, ls='--', alpha=0.5)
    ax.axvline(10.0, color='#BDC3C7', lw=1, ls='--', alpha=0.5)

    plt.tight_layout(pad=0.3)
    save(fig, 'fig1_architecture')


# ─────────────────────────────────────────────────────────────────────────────
# FIG 6 — BASELINE COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def fig6():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    fig.suptitle('Method Comparison on Real 64 mT Hyperfine Data (n=23)',
                 fontsize=12, fontweight='bold', color=C_BLUE)

    methods = ['SynthSeg+\nMRR\n(Vása 2025)', 'SynthSeg+\nSingle Axial\n(Baseline B)',
               'CNN3D\n(Ours)', 'ViT3D+Adapt\n(Ours)']
    colors  = [C_GREY, C_GREY, C_ORANGE, C_GREEN]
    hatch   = ['///', '///', '', '']
    x       = np.arange(len(methods))

    def bar_panel(ax, vals, ylabel, title, ylim, threshold=None, thresh_label=None, fmt='.3f'):
        bars = ax.bar(x, vals, color=colors, width=0.55, edgecolor='white', lw=1.2, zorder=2)
        for bar, h in zip(bars, hatch):
            bar.set_hatch(h)
        for bar, v in zip(bars, vals):
            if v is not None:
                ax.text(bar.get_x()+bar.get_width()/2,
                        v + ylim*0.015,
                        f'{v:{fmt}}', ha='center', va='bottom',
                        fontsize=8.5, fontweight='bold')
        if threshold:
            ax.axhline(threshold, color='red', lw=1.2, ls='--', alpha=0.6,
                       label=thresh_label, zorder=3)
            ax.legend(fontsize=7.5, loc='lower right')
        ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=7.5)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_ylim(0, ylim)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, alpha=0.3, zorder=0)

    bar_panel(axes[0], [0.91, 0.918, 0.514, 0.247],
              'Pearson r vs FreeSurfer GT', '(A) Correlation', 1.1,
              threshold=0.70, thresh_label='r = 0.70 threshold')

    bar_panel(axes[1], [0.005, 0.005, 0.076, 0.010],
              'MAE (nWBV units)', '(B) Absolute Error', 0.10,
              threshold=0.020, thresh_label='Clinical threshold (0.02)')

    bar_panel(axes[2], [300, 150, 0.095, 0.047],
              'Inference Time (seconds)', '(C) Speed', 360, fmt='.3f')
    for xi, lbl in enumerate(['~5 min','~2.5 min','95 ms','47 ms']):
        axes[2].text(xi, [300,150,0.13,0.13][xi]+6, lbl,
                     ha='center', va='bottom', fontsize=8, fontweight='bold')

    legend_patches = [
        mpatches.Patch(facecolor=C_GREY, hatch='///', label='Prior work / SynthSeg+ baselines'),
        mpatches.Patch(facecolor=C_ORANGE, label='CNN3D baseline (ours)'),
        mpatches.Patch(facecolor=C_GREEN, label='ViT3D + domain adaptation (ours)'),
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=3,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.06), frameon=True)

    plt.tight_layout(pad=1.5)
    save(fig, 'fig6_baseline_comparison')


# ─────────────────────────────────────────────────────────────────────────────
# FIG 7 — SCAN COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def load_slice(path, pct=(2,98)):
    img = nib.load(str(path))
    vol = img.get_fdata(dtype=np.float32)
    if vol.ndim == 4: vol = vol[...,0]
    mid = vol.shape[1]//2
    sl = vol[:,mid,:]
    lo,hi = np.percentile(sl, pct[0]), np.percentile(sl, pct[1])
    if hi>lo: sl = np.clip(sl,lo,hi); sl=(sl-lo)/(hi-lo)
    return sl

def fig7():
    subjects = ['sub-HYPE00','sub-HYPE09','sub-HYPE17']

    try:
        from utils.field_conversion import FieldConverter
        conv = FieldConverter({})
        have_sim = True
    except:
        have_sim = False

    fig, axes = plt.subplots(3, 3, figsize=(10, 9.5))
    fig.patch.set_facecolor('black')

    titles = ['3T MPRAGE (High-Field Reference)', 'Physics-Simulated 64 mT', 'Real Hyperfine 64 mT']
    tcols  = ['#85C1E9','#F0B27A','#82E0AA']

    for ci,(t,c) in enumerate(zip(titles,tcols)):
        axes[0,ci].set_title(t, color=c, fontsize=10, fontweight='bold', pad=8)

    for ri, subj in enumerate(subjects):
        t1_path = DS_DIR/subj/"ses-GE"/"anat"/f"{subj}_ses-GE_acq-MPRAGE_T1w.nii.gz"
        hfc_path= DS_DIR/subj/"ses-HFC"/"anat"/f"{subj}_ses-HFC_acq-axi_T2w.nii.gz"

        # Col 0: 3T
        if t1_path.exists():
            axes[ri,0].imshow(load_slice(t1_path).T, cmap='gray', origin='lower', aspect='auto')
        else:
            axes[ri,0].text(0.5,0.5,'N/A',color='white',ha='center',transform=axes[ri,0].transAxes)

        # Col 1: physics sim
        if t1_path.exists() and have_sim:
            try:
                v = nib.load(str(t1_path)).get_fdata(dtype=np.float32)
                if v.ndim==4: v=v[...,0]
                lo,hi=np.percentile(v,1),np.percentile(v,99)
                if hi>lo: v=np.clip(v,lo,hi); v=(v-lo)/(hi-lo)
                sim = conv.convert(v, method='hyperfine')
                mid = sim.shape[1]//2
                sl = sim[:,mid,:]
                lo2,hi2=np.percentile(sl,2),np.percentile(sl,98)
                if hi2>lo2: sl=np.clip(sl,lo2,hi2); sl=(sl-lo2)/(hi2-lo2)
                axes[ri,1].imshow(sl.T, cmap='gray', origin='lower', aspect='auto')
            except:
                axes[ri,1].text(0.5,0.5,'Sim\nerror',color='white',ha='center',
                                transform=axes[ri,1].transAxes,fontsize=8)
        else:
            axes[ri,1].text(0.5,0.5,'N/A',color='white',ha='center',transform=axes[ri,1].transAxes)

        # Col 2: real HFC
        if hfc_path.exists():
            axes[ri,2].imshow(load_slice(hfc_path, pct=(1,99)).T,
                              cmap='gray', origin='lower', aspect='auto')
        else:
            axes[ri,2].text(0.5,0.5,'N/A',color='white',ha='center',transform=axes[ri,2].transAxes)

        axes[ri,0].set_ylabel(f'Subject {ri+1}', color='white', fontsize=9)

    for ax in axes.flat:
        ax.set_facecolor('black'); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_edgecolor('#555')

    fig.suptitle('MRI Quality Across Field Strengths — Same Subjects',
                 color='white', fontsize=11, fontweight='bold')
    fig.text(0.5,-0.01,
             'SNR: 3T ~200  |  Physics-simulated 64mT ~32  |  Real Hyperfine 64mT ~309 (ETL-80 averaging)',
             ha='center', color='#AAB7B8', fontsize=8, style='italic')

    plt.tight_layout(pad=0.8)
    save(fig, 'fig7_scan_comparison')


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Generating Fig 1, 6, 7...\n")
    print("[1/3] Architecture diagram...")
    fig1()
    print("[2/3] Baseline comparison...")
    fig6()
    print("[3/3] Scan comparison...")
    fig7()
    print(f"\nDone. Saved to {FIGURES_DIR}/")
