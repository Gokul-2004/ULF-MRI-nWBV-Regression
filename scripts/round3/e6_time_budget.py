"""
E6 — End-to-end time budget (R2.12)
==================================
Separates the 4.53 ms model forward pass from everything else a user would
actually wait for. Every row is labelled MEASURED (this machine, with source)
or CITED/LOG-DERIVED.

  acquisition          AcquisitionDuration from the ds006557 acq-axi_T2w sidecars
  load+preprocess      NIfTI load, 1-99 percentile normalisation, resize to 64^3
                       (headline loocv_cross_session functions), 46 real scans
  forward pass         experiments/inference_latency (cold, CPU) + end-to-end
                       timing here on real scans (load -> prediction)
  adapter training     one LN+head adapter (headline recipe, 22 HFC subjects), timed
  FastSurfer (labels)  experiments/fastsurfer_zenodo_batch.log, sequential runs,
                       --seg_only --no_cereb --no_hypothal, 3 threads, CPU
  SynthSeg+ (pseudo)   logs/synthseg_batch.log (4 parallel jobs x 3 threads; contention)

Output: experiments/r3_e6_time_budget/results.json
"""

import json
import re
import time
from datetime import datetime, timedelta

import numpy as np
import torch

from common import DS_DIR, EXP, ROOT, lcs, provenance, save_json, subjects_and_gt

OUT = EXP / "r3_e6_time_budget" / "results.json"


def parse_batch(path, start_pat, end_pat):
    starts, durs = {}, {}
    for line in path.read_text().splitlines():
        m = re.search(start_pat, line)
        if m:
            starts[m.group(1)] = datetime.strptime(m.group(2), "%H:%M:%S")
        m = re.search(end_pat, line)
        if m and m.group(1) in starts:
            t = datetime.strptime(m.group(2), "%H:%M:%S")
            d = t - starts[m.group(1)]
            if d.total_seconds() < 0:
                d += timedelta(days=1)
            durs[m.group(1)] = d.total_seconds()
    return durs


def stats_s(v):
    v = np.array(list(v), float)
    return {"n": len(v), "median_s": round(float(np.median(v)), 1), "min_s": round(float(v.min()), 1),
            "max_s": round(float(v.max()), 1), "median_min": round(float(np.median(v) / 60), 1)}


def main():
    subjects, gt = subjects_and_gt()
    res = {"experiment": "r3_e6_time_budget", "answers": ["R2.12"], "cpu": None,
           "provenance": provenance("scripts/round3/e6_time_budget.py")}
    lat = json.loads((EXP / "inference_latency" / "results.json").read_text())
    res["cpu"] = {"model": lat.get("cpu_model"), "torch_threads": torch.get_num_threads()}

    # acquisition
    acq = []
    for s in subjects:
        for ses in ("HFC", "HFE"):
            js = DS_DIR / s / f"ses-{ses}" / "anat" / f"{s}_ses-{ses}_acq-axi_T2w.json"
            acq.append(json.loads(js.read_text()).get("AcquisitionDuration"))
    res["acquisition_axial_T2w"] = {"source": "BIDS sidecar AcquisitionDuration (MEASURED by scanner)", **stats_s(acq)}

    # load + preprocess, end-to-end
    model = lcs.build_base_model().eval()
    pre, e2e = [], []
    with torch.no_grad():
        model(torch.zeros(1, 1, 64, 64, 64))
        for s in subjects:
            for ses in ("HFC", "HFE"):
                t0 = time.perf_counter()
                v = lcs.load_scan(s, ses)
                t1 = time.perf_counter()
                model(torch.tensor(v)[None, None].float())
                t2 = time.perf_counter()
                pre.append(t1 - t0); e2e.append(t2 - t0)
    res["load_preprocess_per_scan"] = {"source": "MEASURED here (nibabel load, percentile norm, scipy zoom to 64^3)",
                                       "median_ms": round(1000 * float(np.median(pre)), 1), "iqr_ms": round(1000 * float(np.subtract(*np.percentile(pre, [75, 25]))), 1), "n": len(pre)}
    res["end_to_end_scan_to_prediction"] = {"source": "MEASURED here (load + preprocess + forward, warm process)",
                                            "median_ms": round(1000 * float(np.median(e2e)), 1), "n": len(e2e)}
    res["forward_pass_cold"] = {"source": "experiments/inference_latency/results.json (MEASURED, random 64^3 input, cold)",
                                "median_ms": lat["cold_full_volume_forward"]["median_ms"]}

    # adapter training
    t0 = time.perf_counter()
    lcs.train_adapter(model, subjects[1:], gt, 0)
    res["adapter_training_one_fold"] = {"source": "MEASURED here (headline recipe: 22 HFC scans x 6, 80 epochs, includes NIfTI loading)",
                                        "seconds": round(time.perf_counter() - t0, 1)}

    # FastSurfer
    fs = parse_batch(EXP / "fastsurfer_zenodo_batch.log", r"\[RUN \] (sub-\S+) - started (\d\d:\d\d:\d\d)",
                     r"\[OK  \] (sub-\S+) - complete (\d\d:\d\d:\d\d)")
    res["fastsurfer_reference_labels"] = {"source": "LOG-DERIVED experiments/fastsurfer_zenodo_batch.log (sequential, --seg_only --no_cereb --no_hypothal, CPU, 3 threads, deepmi/fastsurfer:cpu-v2.3.3)",
                                          **stats_s(fs.values()),
                                          "note": "ds006557 FastSurfer runs also executed HypVINN (no --no_hypothal) under 4-way contention and are not used for timing."}
    ss = parse_batch(ROOT / "logs" / "synthseg_batch.log", r"\[RUN \] (sub-\S+) . started (\d\d:\d\d:\d\d)",
                     r"\[(?:OK  |DONE)\] (sub-\S+) . complete (\d\d:\d\d:\d\d)")
    res["synthseg_plus_pseudolabels"] = {"source": "LOG-DERIVED logs/synthseg_batch.log (freesurfer 7.4.1 mri_synthseg --robust, 64 mT T2w input, 4 parallel jobs x 3 threads — wall time under contention)",
                                         **(stats_s(ss.values()) if ss else {"error": "no parsable entries"})}
    res["summary_table"] = [
        {"step": "64 mT axial T2w acquisition", "time": f"{res['acquisition_axial_T2w']['median_s']:.0f} s", "needed_for": "every scan"},
        {"step": "NIfTI load + normalise + resize", "time": f"{res['load_preprocess_per_scan']['median_ms']} ms", "needed_for": "every scan"},
        {"step": "ViT3D forward pass (CPU, cold)", "time": f"{res['forward_pass_cold']['median_ms']} ms", "needed_for": "every scan"},
        {"step": "Adapter training (once per site, 22 labelled scans)", "time": f"{res['adapter_training_one_fold']['seconds']:.0f} s", "needed_for": "once per site"},
        {"step": "FastSurfer reference labels (3T MPRAGE)", "time": f"{res['fastsurfer_reference_labels']['median_min']} min/subject", "needed_for": "adaptation labels only (once per site)"},
        {"step": "SynthSeg+ pseudo-labels (64 mT T2w)", "time": f"{res['synthseg_plus_pseudolabels'].get('median_min')} min/subject (contended)", "needed_for": "alternative adaptation labels only"},
    ]
    save_json(OUT, res)
    for r in res["summary_table"]:
        print(r)


if __name__ == "__main__":
    main()
