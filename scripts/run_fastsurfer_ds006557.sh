#!/bin/bash
# =============================================================================
# FastSurfer segmentation for all 23 ds006557 subjects (GE 3T T1w)
# Produces aseg.stats → nWBV ground truth for real 64mT validation
#
# Usage: bash scripts/run_fastsurfer_ds006557.sh
# Runtime: ~3 min/subject × 23 = ~1.5 hrs sequential
#          Run 4 in parallel → ~25 min total
# =============================================================================

set -e

DS_DIR="/home/gk-krishnan/Desktop/VIT Paper/data/ds006557_data"
OUT_DIR="/home/gk-krishnan/Desktop/VIT Paper/experiments/fastsurfer_output"
LICENSE="/home/gk-krishnan/freesurfer_license/license.txt"
FASTSURFER_IMG="deepmi/fastsurfer:cpu-v2.3.3"
PARALLEL_JOBS=4   # max simultaneous subjects (adjust to CPU cores / 3)

mkdir -p "$OUT_DIR"

# Verify license exists
if [ ! -f "$LICENSE" ]; then
    echo "ERROR: FreeSurfer license not found at $LICENSE"
    echo "Save the license.txt from your email to that path and re-run."
    exit 1
fi

echo "=== FastSurfer batch run: $(date) ==="
echo "Dataset : $DS_DIR"
echo "Output  : $OUT_DIR"
echo "Parallel: $PARALLEL_JOBS jobs"
echo ""

SUBJECTS=$(ls "$DS_DIR" | grep "^sub-HYPE")
TOTAL=$(echo "$SUBJECTS" | wc -l)
echo "Found $TOTAL subjects"

run_subject() {
    local SUBJ=$1
    local T1W="$DS_DIR/$SUBJ/ses-GE/anat/${SUBJ}_ses-GE_acq-MPRAGE_T1w.nii.gz"
    local SUBJ_OUT="$OUT_DIR/$SUBJ"

    if [ ! -f "$T1W" ]; then
        echo "  [SKIP] $SUBJ — T1w not found: $T1W"
        return
    fi

    if [ -f "$SUBJ_OUT/stats/aseg.stats" ]; then
        echo "  [DONE] $SUBJ — already complete, skipping"
        return
    fi

    echo "  [RUN ] $SUBJ — started $(date +%H:%M:%S)"
    mkdir -p "$SUBJ_OUT"

    docker run --rm \
        --user root \
        -v "$DS_DIR/$SUBJ/ses-GE/anat:/input:ro" \
        -v "$SUBJ_OUT:/output" \
        -v "$LICENSE:/opt/freesurfer/license.txt:ro" \
        "$FASTSURFER_IMG" \
        --t1 "/input/${SUBJ}_ses-GE_acq-MPRAGE_T1w.nii.gz" \
        --sid "$SUBJ" \
        --sd /output \
        --seg_only \
        --no_cereb \
        --parallel \
        --threads 3 \
        --allow_root \
        > "$SUBJ_OUT/fastsurfer.log" 2>&1

    if [ -f "$SUBJ_OUT/$SUBJ/stats/aseg.stats" ]; then
        echo "  [OK  ] $SUBJ — complete $(date +%H:%M:%S)"
    else
        echo "  [FAIL] $SUBJ — check $SUBJ_OUT/fastsurfer.log"
    fi
}

export -f run_subject
export DS_DIR OUT_DIR LICENSE FASTSURFER_IMG

# Run in parallel using background jobs
echo ""
echo "=== Starting segmentation ==="
RUNNING=0
for SUBJ in $SUBJECTS; do
    run_subject "$SUBJ" &
    RUNNING=$((RUNNING + 1))
    if [ "$RUNNING" -ge "$PARALLEL_JOBS" ]; then
        wait -n 2>/dev/null || wait   # wait for any one job to finish
        RUNNING=$((RUNNING - 1))
    fi
done
wait  # wait for all remaining jobs

echo ""
echo "=== All segmentations complete: $(date) ==="
echo ""
echo "=== Extracting nWBV values ==="
python3 - <<'PYEOF'
import os, re
import pandas as pd

OUT_DIR = "/home/gk-krishnan/Desktop/VIT Paper/experiments/fastsurfer_output"
results = []

for subj in sorted(os.listdir(OUT_DIR)):
    stats_file = os.path.join(OUT_DIR, subj, subj, "stats", "aseg.stats")
    if not os.path.isfile(stats_file):
        print(f"  MISSING: {subj}")
        continue

    with open(stats_file) as f:
        content = f.read()

    # Extract eTIV (estimated total intracranial volume)
    etiv_match = re.search(r'EstimatedTotalIntraCranialVol,\s*[\d.]+,\s*([\d.]+)', content)
    # Extract Gray matter and White matter volumes
    gm_match   = re.search(r'Total gray matter volume,\s*([\d.]+)', content)
    wm_match   = re.search(r'Total cerebral white matter volume,\s*([\d.]+)', content)

    # Also try parsing from table
    gm_vol = wm_vol = etiv = None
    for line in content.split('\n'):
        if 'CortexVol' in line or 'TotalGrayVol' in line:
            parts = line.split()
            if len(parts) >= 4:
                try: gm_vol = float(parts[3])
                except: pass
        if 'CerebralWhiteMatterVol' in line:
            parts = line.split()
            if len(parts) >= 4:
                try: wm_vol = float(parts[3])
                except: pass

    if etiv_match: etiv = float(etiv_match.group(1))

    # FreeSurfer aseg.stats header format for eTIV and nWBV
    for line in content.split('\n'):
        if 'EstimatedTotalIntraCranialVol' in line:
            parts = line.split(',')
            if len(parts) >= 4:
                try: etiv = float(parts[3].strip())
                except: pass
        if 'TotalGrayVol' in line and not line.startswith('#'):
            parts = line.split()
            if len(parts) >= 4:
                try: gm_vol = float(parts[3])
                except: pass
        if 'CerebralWhiteMatterVol' in line and not line.startswith('#'):
            parts = line.split()
            if len(parts) >= 4:
                try: wm_vol = float(parts[3])
                except: pass

    if gm_vol and wm_vol and etiv and etiv > 0:
        nwbv = (gm_vol + wm_vol) / etiv
        results.append({
            "subject": subj,
            "gray_matter_vol": round(gm_vol, 1),
            "white_matter_vol": round(wm_vol, 1),
            "etiv": round(etiv, 1),
            "nWBV_freesurfer": round(nwbv, 4)
        })
        print(f"  {subj}: nWBV = {nwbv:.4f}  (GM={gm_vol:.0f}, WM={wm_vol:.0f}, eTIV={etiv:.0f})")
    else:
        print(f"  {subj}: Could not extract volumes (GM={gm_vol}, WM={wm_vol}, eTIV={etiv})")

if results:
    df = pd.DataFrame(results)
    out_csv = os.path.join(OUT_DIR, "nwbv_ground_truth.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {len(results)} subjects to {out_csv}")
    print(f"nWBV range: {df['nWBV_freesurfer'].min():.4f} -- {df['nWBV_freesurfer'].max():.4f}")
    print(f"nWBV mean:  {df['nWBV_freesurfer'].mean():.4f} ± {df['nWBV_freesurfer'].std():.4f}")
else:
    print("No results extracted — check segmentation logs")
PYEOF
# NOTE: sub-HYPE00 rerun added — crashed during stats consolidation on first run
