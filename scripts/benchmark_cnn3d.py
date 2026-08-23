"""
Inference latency benchmark for BaselineCNN3D — companion to inference_benchmark.py
==================================================================================
Why this exists: fig10 claims "~95 s" for CNN3D against "47 ms" for ViT3D. Both were
hardcoded, neither measured (integrity item I10). The CNN3D figure is not merely
unmeasured, it is implausible: CNN3D is an 8,222,337-parameter CNN doing one forward
pass over a 64^3 volume, and ViT3D (4,225,537 params) measures 4.53 ms cold on the
same hardware. A ~20,000x gap between two small networks on the same input has no
mechanism behind it.

Same protocol as inference_benchmark.py so the two numbers are comparable: cold
full-volume forward pass, freshly-allocated input each repetition, eval() + no_grad(),
batch = 1. Merges its result into experiments/inference_latency/results.json.

Run:  python scripts/benchmark_cnn3d.py
"""

import sys, json, time, platform
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
import warnings; warnings.filterwarnings("ignore")
from models.baselines import BaselineCNN3D

OUT = project_root / "experiments" / "inference_latency" / "results.json"
SHAPE = (64, 64, 64)
N, N_WARMUP = 100, 10


def cpu_model():
    try:
        for line in open("/proc/cpuinfo"):
            if "model name" in line:
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def timeit(model, cold):
    ts = []
    x = torch.from_numpy(np.random.rand(1, 1, *SHAPE).astype(np.float32))
    with torch.no_grad():
        for _ in range(N_WARMUP):
            model(x)
        for _ in range(N):
            if cold:                       # fresh allocation each rep
                x = torch.from_numpy(np.random.rand(1, 1, *SHAPE).astype(np.float32))
            t0 = time.perf_counter()
            model(x)
            ts.append((time.perf_counter() - t0) * 1000)
    a = np.array(ts)
    return {
        "median_ms": round(float(np.median(a)), 2),
        "mean_ms":   round(float(a.mean()), 2),
        "iqr_ms":    round(float(np.percentile(a, 75) - np.percentile(a, 25)), 2),
        "p25_ms":    round(float(np.percentile(a, 25)), 2),
        "p75_ms":    round(float(np.percentile(a, 75)), 2),
        "min_ms":    round(float(a.min()), 2),
        "max_ms":    round(float(a.max()), 2),
    }


if __name__ == "__main__":
    model = BaselineCNN3D(input_shape=SHAPE, num_classes=1).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"BaselineCNN3D params: {n_params:,}")
    assert n_params == 8222337, f"unexpected param count {n_params:,} — expected 8,222,337"

    cold   = timeit(model, cold=True)
    warmed = timeit(model, cold=False)
    print(f"  cold   median {cold['median_ms']} ms (IQR {cold['iqr_ms']})")
    print(f"  warmed median {warmed['median_ms']} ms")

    d = json.load(open(OUT)) if OUT.exists() else {}
    d["cnn3d"] = {
        "model": "BaselineCNN3D (scalar regression head)",
        "n_params": n_params,
        "input_shape": list(SHAPE),
        "batch_size": 1,
        "device": "cpu",
        "cpu_model": cpu_model(),
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "n_repetitions": N,
        "n_warmup": N_WARMUP,
        "cold_full_volume_forward": cold,
        "warmed_reused_tensor": warmed,
        "headline_ms": cold["median_ms"],
        "note": ("Replaces the unmeasured '~95 s' in fig10, which was implausible: a "
                 "forward pass of an 8.2M-parameter CNN over a 64^3 volume cannot take "
                 "20,000x the time of a 4.2M-parameter ViT on the same input."),
    }
    json.dump(d, open(OUT, "w"), indent=2)
    print(f"\nmerged into {OUT.relative_to(project_root)}")
