#!/bin/bash
# =============================================================================
# SynthSeg+ on single axial T2w 64mT scans (ds006557, n=23)
# Baseline B: matched single-scan comparison vs our ViT3D
#
# Uses FreeSurfer 7.4.1 mri_synthseg (contrast/resolution agnostic)
# Input:  ses-HFC axial T2w (same input as our model)
# Output: nWBV computed from segmentation volumes
#
# Usage: bash scripts/run_synthseg_ds006557.sh
# Runtime: ~2-3 min/subject × 23 = ~1 hr sequential
# =============================================================================

set -e

DS_DIR="/home/gk-krishnan/Desktop/VIT Paper/data/ds006557_data"
OUT_DIR="/home/gk-krishnan/Desktop/VIT Paper/experiments/synthseg_output"
LICENSE="/home/gk-krishnan/freesurfer_license/license.txt"
FS_IMG="freesurfer/freesurfer:7.4.1"
PARALLEL_JOBS=4

mkdir -p "$OUT_DIR"

echo "=== SynthSeg+ batch run: $(date) ==="
echo "Input: single axial T2w (ses-HFC)"
echo "Parallel: $PARALLEL_JOBS jobs"
echo ""

SUBJECTS=$(ls "$DS_DIR" | grep "^sub-HYPE")
TOTAL=$(echo "$SUBJECTS" | wc -l)
echo "Found $TOTAL subjects"
echo ""

run_subject() {
    local SUBJ=$1
    local T2W="$DS_DIR/$SUBJ/ses-HFC/anat/${SUBJ}_ses-HFC_acq-axi_T2w.nii.gz"
    local SUBJ_OUT="$OUT_DIR/$SUBJ"
    local SEG_OUT="$SUBJ_OUT/${SUBJ}_synthseg.mgz"
    local STATS_OUT="$SUBJ_OUT/${SUBJ}_synthseg.stats"
    local VOL_OUT="$SUBJ_OUT/${SUBJ}_volumes.csv"

    if [ ! -f "$T2W" ]; then
        echo "  [SKIP] $SUBJ — T2w not found"
        return
    fi

    if [ -f "$VOL_OUT" ]; then
        echo "  [DONE] $SUBJ — already complete, skipping"
        return
    fi

    echo "  [RUN ] $SUBJ — started $(date +%H:%M:%S)"
    mkdir -p "$SUBJ_OUT"

    docker run --rm \
        --user root \
        -v "$DS_DIR/$SUBJ/ses-HFC/anat:/input:ro" \
        -v "$SUBJ_OUT:/output" \
        -v "$LICENSE:/usr/local/freesurfer/license.txt:ro" \
        -e FS_LICENSE=/usr/local/freesurfer/license.txt \
        "$FS_IMG" \
        mri_synthseg \
        --i "/input/${SUBJ}_ses-HFC_acq-axi_T2w.nii.gz" \
        --o "/output/${SUBJ}_synthseg.mgz" \
        --vol "/output/${SUBJ}_volumes.csv" \
        --robust \
        --threads 3 \
        > "$SUBJ_OUT/synthseg.log" 2>&1

    if [ -f "$VOL_OUT" ]; then
        echo "  [OK  ] $SUBJ — complete $(date +%H:%M:%S)"
    else
        echo "  [FAIL] $SUBJ — check $SUBJ_OUT/synthseg.log"
    fi
}

export -f run_subject
export DS_DIR OUT_DIR LICENSE FS_IMG

echo "=== Starting SynthSeg+ segmentation ==="
RUNNING=0
for SUBJ in $SUBJECTS; do
    run_subject "$SUBJ" &
    RUNNING=$((RUNNING + 1))
    if [ "$RUNNING" -ge "$PARALLEL_JOBS" ]; then
        wait -n 2>/dev/null || wait
        RUNNING=$((RUNNING - 1))
    fi
done
wait

echo ""
echo "=== All segmentations complete: $(date) ==="
echo ""
echo "=== Extracting nWBV from SynthSeg+ volumes ==="

python3 - <<'PYEOF'
import os, glob
import pandas as pd
import numpy as np
from scipy import stats

OUT_DIR = "/home/gk-krishnan/Desktop/VIT Paper/experiments/synthseg_output"
GT_CSV  = "/home/gk-krishnan/Desktop/VIT Paper/experiments/fastsurfer_output/nwbv_ground_truth.csv"

# SynthSeg volume CSV columns:
# subject, Left-Cerebral-White-Matter, Right-Cerebral-White-Matter,
# Left-Cerebral-Cortex, Right-Cerebral-Cortex, ..., Total intracranial

results = []

for subj in sorted(os.listdir(OUT_DIR)):
    vol_csv = os.path.join(OUT_DIR, subj, f"{subj}_volumes.csv")
    if not os.path.isfile(vol_csv):
        print(f"  MISSING: {subj}")
        continue

    try:
        df = pd.read_csv(vol_csv)
        # Column names vary — find GM, WM, TIV
        cols = [c.lower() for c in df.columns]

        # Total intracranial volume
        tiv = None
        for col in df.columns:
            if 'intracranial' in col.lower() or 'total intracranial' in col.lower():
                tiv = float(df[col].iloc[0])
                break

        # White matter (left + right cerebral WM)
        wm = 0
        for col in df.columns:
            if 'white-matter' in col.lower() or 'white matter' in col.lower():
                if 'cerebral' in col.lower() or 'left' in col.lower() or 'right' in col.lower():
                    wm += float(df[col].iloc[0])

        # Gray matter / cortex (left + right cerebral cortex + subcortical)
        gm = 0
        for col in df.columns:
            if ('cortex' in col.lower() or 'gray' in col.lower() or
                'caudate' in col.lower() or 'putamen' in col.lower() or
                'thalamus' in col.lower() or 'hippocampus' in col.lower() or
                'amygdala' in col.lower()):
                try: gm += float(df[col].iloc[0])
                except: pass

        if tiv and tiv > 0 and wm > 0 and gm > 0:
            nwbv = (gm + wm) / tiv
            results.append({
                'subject': subj,
                'gray_matter_vol': round(gm, 1),
                'white_matter_vol': round(wm, 1),
                'tiv': round(tiv, 1),
                'nWBV_synthseg': round(nwbv, 4)
            })
            print(f"  {subj}: nWBV={nwbv:.4f}  GM={gm:.0f}  WM={wm:.0f}  TIV={tiv:.0f}")
        else:
            print(f"  {subj}: Could not extract (GM={gm}, WM={wm}, TIV={tiv})")
            print(f"    Columns: {list(df.columns)}")

    except Exception as e:
        print(f"  {subj}: ERROR — {e}")

if not results:
    print("No results extracted.")
else:
    df_res = pd.DataFrame(results)
    out_csv = os.path.join(OUT_DIR, "nwbv_synthseg.csv")
    df_res.to_csv(out_csv, index=False)
    print(f"\nSaved {len(results)} subjects to {out_csv}")

    # Compare against FreeSurfer ground truth
    if os.path.isfile(GT_CSV):
        gt = pd.read_csv(GT_CSV)
        merged = df_res.merge(gt[['subject','nWBV_freesurfer']], on='subject')
        if len(merged) >= 5:
            r, p = stats.pearsonr(merged['nWBV_synthseg'], merged['nWBV_freesurfer'])
            rho, p_sp = stats.spearmanr(merged['nWBV_synthseg'], merged['nWBV_freesurfer'])
            mae = float(np.mean(np.abs(merged['nWBV_synthseg'] - merged['nWBV_freesurfer'])))
            print(f"\n=== SynthSeg+ vs FreeSurfer Ground Truth (n={len(merged)}) ===")
            print(f"Pearson r  = {r:.4f}  (p={p:.4f})")
            print(f"Spearman ρ = {rho:.4f}  (p={p_sp:.4f})")
            print(f"MAE        = {mae:.4f}")

            # Load participants for age correlation
            tsv = "/home/gk-krishnan/Desktop/VIT Paper/data/ds006557_data/participants.tsv"
            if os.path.isfile(tsv):
                ages = {}
                with open(tsv) as f:
                    for line in f.readlines()[1:]:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            try: ages['sub-' + parts[0]] = float(parts[1])
                            except: pass
                age_pairs = [(ages[s], n) for s, n in
                             zip(df_res['subject'], df_res['nWBV_synthseg'])
                             if s in ages]
                if len(age_pairs) >= 5:
                    a_arr = np.array([x[0] for x in age_pairs])
                    n_arr = np.array([x[1] for x in age_pairs])
                    ar, _ = stats.pearsonr(a_arr, n_arr)
                    arho, ap = stats.spearmanr(a_arr, n_arr)
                    print(f"\nAge-nWBV correlation (SynthSeg+):")
                    print(f"Pearson r  = {ar:.4f}")
                    print(f"Spearman ρ = {arho:.4f}  (p={ap:.4f})")

            # Save comparison
            import json
            comp = {
                'method': 'SynthSeg+ single axial T2w (64mT)',
                'n': len(merged),
                'vs_freesurfer_gt': {
                    'pearson_r': round(r, 4), 'pearson_p': round(p, 4),
                    'spearman_r': round(rho, 4), 'spearman_p': round(p_sp, 4),
                    'mae': round(mae, 4)
                }
            }
            with open(os.path.join(OUT_DIR, "synthseg_comparison.json"), 'w') as f:
                json.dump(comp, f, indent=2)
            print(f"\nComparison saved to {OUT_DIR}/synthseg_comparison.json")
PYEOF
