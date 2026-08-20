#!/bin/bash
# =============================================================================
# FastSurfer segmentation for the van den Broek Zenodo cohort (3T highres T1w)
# Produces aseg.stats -> nWBV ground truth for EXTERNAL validation (Reviewer R1.3)
#
# Same pipeline (--seg_only) and same nWBV extraction as the ds006557 run, so the
# external nWBV labels are apples-to-apples with the paper's OASIS/ds006557 labels.
#
# Usage: bash scripts/run_fastsurfer_zenodo.sh
# Runtime: ~1-1.5h/subject on CPU x 11 subjects, 2 parallel -> ~6-9h
# =============================================================================

set -e

BASE="/home/gk-krishnan/Desktop/VIT Paper/Paper 2"
ZEN="$BASE/Website_dat/extracted/Paired 64mT and 3T Brain MRI Scans of Healthy Subjects for Neuroimaging Research v2/Data/3T data"
OUT_DIR="$BASE/experiments/fastsurfer_zenodo"
LICENSE="/home/gk-krishnan/freesurfer_license/license.txt"
FASTSURFER_IMG="deepmi/fastsurfer:cpu-v2.3.3"
PARALLEL_JOBS=1   # sequential: 2-parallel triggered OOM (only ~10GB free RAM)

mkdir -p "$OUT_DIR"

if [ ! -f "$LICENSE" ]; then
    echo "ERROR: FreeSurfer license not found at $LICENSE"; exit 1
fi

echo "=== FastSurfer Zenodo run: $(date) ==="
echo "Data   : $ZEN"
echo "Output : $OUT_DIR"
echo "Parallel: $PARALLEL_JOBS jobs"
echo ""

SUBJECTS=$(ls "$ZEN" | grep "^sub-")
TOTAL=$(echo "$SUBJECTS" | wc -l)
echo "Found $TOTAL subjects: $(echo $SUBJECTS | tr '\n' ' ')"

run_subject() {
    local SUBJ=$1
    local ANAT="$ZEN/$SUBJ/anat"
    local T1W_NAME="${SUBJ}_acq-highres_T1w.nii.gz"
    local SUBJ_OUT="$OUT_DIR/$SUBJ"

    if [ ! -f "$ANAT/$T1W_NAME" ]; then
        echo "  [SKIP] $SUBJ - T1w not found: $ANAT/$T1W_NAME"; return
    fi
    if [ -f "$SUBJ_OUT/$SUBJ/stats/aseg+DKT.stats" ]; then
        echo "  [DONE] $SUBJ - already complete, skipping"; return
    fi

    echo "  [RUN ] $SUBJ - started $(date +%H:%M:%S)"
    mkdir -p "$SUBJ_OUT"

    docker run --rm \
        --user root \
        -v "$ANAT:/input:ro" \
        -v "$SUBJ_OUT:/output" \
        -v "$LICENSE:/opt/freesurfer/license.txt:ro" \
        "$FASTSURFER_IMG" \
        --t1 "/input/$T1W_NAME" \
        --sid "$SUBJ" \
        --sd /output \
        --seg_only \
        --no_cereb \
        --no_hypothal \
        --parallel \
        --threads 3 \
        --allow_root \
        > "$SUBJ_OUT/fastsurfer.log" 2>&1

    if [ -f "$SUBJ_OUT/$SUBJ/stats/aseg+DKT.stats" ]; then
        echo "  [OK  ] $SUBJ - complete $(date +%H:%M:%S)"
    else
        echo "  [FAIL] $SUBJ - check $SUBJ_OUT/fastsurfer.log"
    fi
}
export -f run_subject
export ZEN OUT_DIR LICENSE FASTSURFER_IMG

echo ""
echo "=== Starting segmentation ==="
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
echo "=== Extracting nWBV values ==="
python3 - "$OUT_DIR" <<'PYEOF'
import os, re, sys
import pandas as pd
OUT_DIR = sys.argv[1]

def measure(content, name):
    m = re.search(rf'# Measure {name},\s*\w+,\s*[^,]+,\s*([\d.]+)', content)
    return float(m.group(1)) if m else None

results = []
for subj in sorted(os.listdir(OUT_DIR)):
    # nWBV uses the main segmentation stats (aseg+DKT.stats), which --seg_only writes
    # BEFORE the (optional, sometimes-crashing) hypothalamus step.
    stats_file = os.path.join(OUT_DIR, subj, subj, "stats", "aseg+DKT.stats")
    if not os.path.isfile(stats_file):
        print(f"  MISSING: {subj}"); continue
    content = open(stats_file).read()
    # Definition matching the paper's OASIS/ds006557 labels:
    #   eTIV  = MaskVol
    #   brain = BrainSegVol  (= GM + WM)
    #   WM    = CerebralWhiteMatterVol ;  GM = BrainSeg - WM
    #   nWBV  = BrainSegVol / MaskVol
    mask = measure(content, "Mask")
    brainseg = measure(content, "BrainSeg")
    wm = measure(content, "CerebralWhiteMatter")
    if mask and brainseg and mask > 0:
        gm = (brainseg - wm) if wm else None
        nwbv = brainseg / mask
        results.append({"subject": subj,
                        "gray_matter_vol": round(gm, 1) if gm else None,
                        "white_matter_vol": round(wm, 1) if wm else None,
                        "brainseg_vol": round(brainseg, 1),
                        "etiv_mask": round(mask, 1),
                        "nWBV_freesurfer": round(nwbv, 4)})
        print(f"  {subj}: nWBV = {nwbv:.4f}")
    else:
        print(f"  {subj}: could not extract (Mask={mask}, BrainSeg={brainseg})")
if results:
    df = pd.DataFrame(results)
    out_csv = os.path.join(OUT_DIR, "nwbv_ground_truth_zenodo.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {len(results)} subjects -> {out_csv}")
    print(f"nWBV range: {df['nWBV_freesurfer'].min():.4f} -- {df['nWBV_freesurfer'].max():.4f}")
    print(f"nWBV mean:  {df['nWBV_freesurfer'].mean():.4f} +- {df['nWBV_freesurfer'].std():.4f}")
else:
    print("No results extracted - check logs")
PYEOF
echo "=== DONE: $(date) ==="
