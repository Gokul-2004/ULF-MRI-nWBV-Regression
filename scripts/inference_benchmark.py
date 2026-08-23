"""
Inference latency benchmark for ViT3D (Reviewer R3 / manuscript 47 ms claim)
============================================================================
Measures a COLD full-volume forward pass: a freshly-allocated 64^3 input each
repetition (no reused/cached tensor), model in eval() + torch.no_grad(),
single-scan batch. Reports median, IQR, N, input shape, thread count, CPU model.

Also reports a WARMED loop (reused tensor) for comparison, to explain any gap
vs. the manuscript's stated figure — a warmed loop is optimistic; cold is the
honest deployment number.

Output: experiments/inference_latency/results.json
"""

import sys, json, time, platform, subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
import warnings; warnings.filterwarnings("ignore")
from models.baselines import BaselineViT3D

OUT = project_root / "experiments" / "inference_latency"
SHAPE = (64, 64, 64)
N = 100          # timed repetitions
N_WARMUP = 10


def cpu_model():
    try:
        for line in open("/proc/cpuinfo"):
            if "model name" in line:
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def build():
    m = BaselineViT3D(img_size=SHAPE, patch_size=16, num_classes=4,
                      embed_dim=256, num_layers=4, num_heads=8)
    m.head = nn.Linear(m.head.in_features, 1)   # deployed scalar-regression head
    m.eval()
    return m


def time_cold(model, n):
    """Fresh input tensor EACH repetition — no cache reuse; honest cold latency."""
    ts = []
    with torch.no_grad():
        for _ in range(n):
            x = torch.randn(1, 1, *SHAPE)       # allocated inside the loop
            t0 = time.perf_counter()
            _ = model(x)
            ts.append((time.perf_counter() - t0) * 1000.0)
    return np.array(ts)


def time_warmed(model, n):
    """Single reused tensor — optimistic; for comparison only."""
    x = torch.randn(1, 1, *SHAPE)
    ts = []
    with torch.no_grad():
        for _ in range(n):
            t0 = time.perf_counter()
            _ = model(x)
            ts.append((time.perf_counter() - t0) * 1000.0)
    return np.array(ts)


def stats(a):
    return {"median_ms": round(float(np.median(a)), 2),
            "mean_ms": round(float(np.mean(a)), 2),
            "iqr_ms": round(float(np.percentile(a, 75) - np.percentile(a, 25)), 2),
            "p25_ms": round(float(np.percentile(a, 25)), 2),
            "p75_ms": round(float(np.percentile(a, 75)), 2),
            "min_ms": round(float(a.min()), 2),
            "max_ms": round(float(a.max()), 2)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    n_threads = torch.get_num_threads()
    model = build()
    n_params = sum(p.numel() for p in model.parameters())

    # warmup (excluded from timing) — loads kernels, allocates buffers
    with torch.no_grad():
        xw = torch.randn(1, 1, *SHAPE)
        for _ in range(N_WARMUP):
            _ = model(xw)

    cold = time_cold(model, N)
    warmed = time_warmed(model, N)

    result = {
        "experiment": "inference_latency",
        "model": "BaselineViT3D (scalar regression head)",
        "n_params": n_params,
        "input_shape": list(SHAPE),
        "batch_size": 1,
        "device": "cpu",
        "cpu_model": cpu_model(),
        "torch_version": torch.__version__,
        "torch_threads": n_threads,
        "n_repetitions": N,
        "n_warmup": N_WARMUP,
        "cold_full_volume_forward": stats(cold),
        "warmed_reused_tensor": stats(warmed),
        "headline_ms": round(float(np.median(cold)), 2),
        "note": ("Headline is the COLD median: a freshly-allocated 64^3 input "
                 "each repetition, model.eval()+no_grad, batch=1, on CPU. The "
                 "warmed (reused-tensor) figure is lower and optimistic. "
                 "Manuscript cites 47 ms on a standard GPU; this file measures CPU."),
    }
    (OUT / "results.json").write_text(json.dumps(result, indent=2))

    print("=== ViT3D inference latency (CPU) ===")
    print(f"  CPU: {result['cpu_model']}  | torch threads: {n_threads}")
    print(f"  input {SHAPE}, batch=1, N={N} cold reps")
    print(f"  COLD  : median={result['cold_full_volume_forward']['median_ms']}ms "
          f"IQR={result['cold_full_volume_forward']['iqr_ms']}ms "
          f"[{result['cold_full_volume_forward']['p25_ms']}-{result['cold_full_volume_forward']['p75_ms']}]")
    print(f"  WARMED: median={result['warmed_reused_tensor']['median_ms']}ms "
          f"(optimistic, reused tensor)")
    print(f"  -> headline (cold median): {result['headline_ms']} ms")
    print(f"  Saved -> {OUT/'results.json'}")


if __name__ == "__main__":
    main()
