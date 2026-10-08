"""
E8 — OASIS-1 test metrics on fixed, seeded simulated inputs (R3.3)
=================================================================
Every OASIS-1 test number in the submitted manuscript was computed on 64 mT
inputs simulated on the fly (FieldConverter 'hyperfine', unseeded Rician noise),
so re-evaluating the SAME checkpoint gives different numbers (CNN3D r 0.877 on
one evaluation, 0.909 on another; experiments/r3_e2c_cnn3d_target).

This script evaluates every saved OASIS-fine-tuned checkpoint on the 38 seed-42
test subjects under K = 10 fixed noise realisations. Realisation k uses
np.random.seed(salt_k + i) for subject i, salt_k = 99_000 + 1_000 k, so k = 0
is the exact input set used by E4. Preprocessing is identical to
scripts/finetune_oasis.OASISDataset (min-max, trilinear zoom to 64^3, simulate).

Checkpoints: oasis_finetuned.pt (headline ViT3D), r3_cnn3d_oasis.pt (CNN3D,
Table-II recipe), swin_oasis.pt (Swin-UNETR), dino_oasis_finetuned.pt (DINO
ViT3D). UNETR, MAE, SimMIM and contrastive fine-tuned weights were not saved
and cannot be re-evaluated.

Output: experiments/r3_e8_oasis_fixed_inputs/results.json
"""

import json

import numpy as np
import torch
from scipy import stats

from common import EXP, ROOT, provenance, save_json
from e2_nonneural_baselines import swin_model
from e4_matched_physics_vs_blur import fixed_inputs, oasis_split
from utils.field_conversion import FieldConverter
from models.baselines import BaselineCNN3D, BaselineViT3D

OUT = EXP / "r3_e8_oasis_fixed_inputs" / "results.json"
K = 10
SALTS = [99_000 + 1_000 * k for k in range(K)]


def vit(ckpt):
    m = BaselineViT3D(img_size=(64, 64, 64), patch_size=16, num_classes=1,
                      embed_dim=256, num_layers=4, num_heads=8)
    m.load_state_dict(torch.load(ROOT / "checkpoints" / ckpt, map_location="cpu",
                                 weights_only=False)["model_state_dict"])
    return m.eval()


def cnn():
    m = BaselineCNN3D(input_shape=(64, 64, 64), num_classes=1)
    m.load_state_dict(torch.load(ROOT / "checkpoints" / "r3_cnn3d_oasis.pt", map_location="cpu",
                                 weights_only=False)["model_state_dict"])
    return m.eval()


def swin():
    m = swin_model()
    m.forward = lambda x: m.head(m.gap(m.backbone(x)).flatten(1))
    return m


MODELS = {"vit3d_headline": lambda: vit("oasis_finetuned.pt"),
          "cnn3d": cnn,
          "swin_unetr": swin,
          "vit3d_dino": lambda: vit("dino_oasis_finetuned.pt")}


@torch.no_grad()
def predict(m, x):
    return np.concatenate([m(x[i:i + 4]).numpy().ravel() for i in range(0, len(x), 4)])


def main():
    torch.manual_seed(42)
    _, _, te = oasis_split()
    conv = FieldConverter({})
    inputs = []
    for s in SALTS:
        x, y = fixed_inputs(te, "physics", conv, salt=s)
        inputs.append(x)
    res = {"experiment": "r3_e8_oasis_fixed_inputs", "answers": ["R3.3"],
           "design": f"K={K} fixed noise realisations, salt_k = 99000 + 1000k, np.random.seed(salt_k + i) per subject",
           "test_subjects": [r["subj_id"] for r in te], "true_nwbv": [float(v) for v in y],
           "provenance": provenance("scripts/round3/e8_oasis_fixed_inputs.py"), "models": {}}
    for name, build in MODELS.items():
        m = build()
        runs = []
        for k, x in enumerate(inputs):
            p = predict(m, x)
            runs.append({"k": k, "mae": round(float(np.mean(np.abs(p - y))), 5),
                         "r": round(float(stats.pearsonr(p, y)[0]), 4),
                         "bias": round(float(np.mean(p - y)), 5),
                         "pred": [round(float(v), 5) for v in p]})
        maes = np.array([r["mae"] for r in runs]); rs = np.array([r["r"] for r in runs])
        res["models"][name] = {"mae_mean": round(float(maes.mean()), 5), "mae_sd": round(float(maes.std(ddof=1)), 5),
                               "mae_range": [float(maes.min()), float(maes.max())],
                               "r_mean": round(float(rs.mean()), 4), "r_sd": round(float(rs.std(ddof=1)), 4),
                               "r_range": [float(rs.min()), float(rs.max())], "runs": runs}
        print(f"{name:16s} MAE {maes.mean():.4f} ± {maes.std(ddof=1):.4f}  r {rs.mean():.3f} ± {rs.std(ddof=1):.3f}")
    save_json(OUT, res)


if __name__ == "__main__":
    main()
