"""
E2c — CNN3D through the target-domain cross-session protocol (R2.8, R4.2)
========================================================================
No CNN3D checkpoint was ever saved, and the only CNN3D 64 mT predictions
(experiments/ablation_vit_vs_cnn) come from a CNN trained on ALL OASIS
subjects without a hold-out. This script:

 1. Retrains CNN3D with the exact Table-II recipe (scripts/cnn3d_oasis_comparison.py:
    seed-42 300/37/38 split, AdamW 5e-5, wd, augmentation, early stop on val r),
    saves checkpoints/r3_cnn3d_oasis.pt and evaluates the 38 OASIS test subjects.
 2. Runs the frozen CNN3D on both ds006557 sessions and evaluates, under the
    headline cross-session design (fit on HFC of 22, predict held-out HFE):
      unadapted output, LOO offset, LOO affine recalibration,
      nested ridge on the 256-d global-pool features,
      nested ridge on the 128-d penultimate features (linear-head analogue).
 3. ICC(C,1) of the frozen CNN across sessions (fold-independent).

Output: experiments/r3_e2c_cnn3d_target/results.json
"""

import json
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

from common import EXP, ROOT, icc_c1, lcs, provenance, save_json, subjects_and_gt
from e2_nonneural_baselines import cross_session, metrics

sys.path.insert(0, str(ROOT / "scripts"))
import cnn3d_oasis_comparison as cc  # noqa: E402

OUT = EXP / "r3_e2c_cnn3d_target" / "results.json"
CKPT = ROOT / "checkpoints" / "r3_cnn3d_oasis.pt"


@torch.no_grad()
def cnn_feats(m, vol):
    x = torch.tensor(vol)[None, None].float()
    h = m.pool1(F.relu(m.bn1(m.conv1(x))))
    h = m.layer3(m.layer2(m.layer1(h)))
    g = m.global_pool(h).flatten(1)
    pen = m.fc[1](m.fc[0](g))
    return g.numpy().ravel(), pen.numpy().ravel(), float(m.fc[3](m.fc[2](pen)))


def main():
    res = {"experiment": "r3_e2c_cnn3d_target_domain", "answers": ["R2.8", "R4.2"],
           "provenance": provenance("scripts/round3/e2c_cnn3d_target_domain.py")}
    torch.manual_seed(cc.SEED); np.random.seed(cc.SEED)
    tr, va, te = cc.make_split()
    if CKPT.exists():
        from models.baselines import BaselineCNN3D
        m = BaselineCNN3D(input_shape=cc.TARGET_SHAPE, num_classes=1)
        ck = torch.load(CKPT, map_location="cpu", weights_only=False)
        m.load_state_dict(ck["model_state_dict"]); best_r = ck.get("best_val_r")
    else:
        m, best_r, hist = cc.train_cnn(tr, va)
        torch.save({"model_state_dict": m.state_dict(), "best_val_r": best_r, "recipe": "cnn3d_oasis_comparison.train_cnn, seed 42"}, CKPT)
    m.eval()
    res["oasis_test"] = cc.evaluate(m, te)
    res["oasis_best_val_r"] = best_r

    subjects, gt = subjects_and_gt()
    y = np.array([gt[s] for s in subjects])
    Fd = {}
    for ses in ("HFC", "HFE"):
        g, p, o = zip(*[cnn_feats(m, lcs.load_scan(s, ses)) for s in subjects])
        Fd[ses] = (np.array(g), np.array(p), np.array(o))
    head = json.loads((EXP / "loocv_cross_session" / "results.json").read_text())["per_subject"]
    head_err = np.abs(np.array([r["pred_hfe"] for r in head]) - y)
    loo_mean = np.array([np.delete(y, i).mean() for i in range(len(y))])
    arms = {}
    per_subject = [{"subject": s, "true_nwbv": float(t), "cnn_unadapted_hfe": round(float(u), 5)}
                   for s, t, u in zip(subjects, y, Fd["HFE"][2])]
    arms["cnn_unadapted_hfe"] = metrics(Fd["HFE"][2], y, head_err, loo_mean)
    for name, Xc, Xe in [("cnn_frozen_offset", Fd["HFC"][2][:, None], Fd["HFE"][2][:, None]),
                         ("cnn_frozen_affine", Fd["HFC"][2][:, None], Fd["HFE"][2][:, None]),
                         ("ridge_cnn_globalpool256", Fd["HFC"][0], Fd["HFE"][0]),
                         ("ridge_cnn_penultimate128", Fd["HFC"][1], Fd["HFE"][1])]:
        p, _ = cross_session(name.replace("cnn_frozen", "vit_frozen"), Xc, Xe, y)
        arms[name] = metrics(p, y, head_err, loo_mean)
        for row, v in zip(per_subject, p):
            row[name] = round(float(v), 5)
        print(f"{name:28s} MAE {arms[name]['mae']:.4f} r {arms[name].get('pearson_r', float('nan')):+.3f} skill {arms[name]['mse_skill_vs_loo_mean']:+.3f}")
    res["cross_session_arms"] = arms
    res["per_subject"] = per_subject
    res["frozen_cnn_icc_c1_hfc_vs_hfe"] = round(icc_c1(Fd["HFC"][2], Fd["HFE"][2]), 4)
    res["frozen_cnn_hfe_vs_truth_r"] = round(float(stats.pearsonr(Fd["HFE"][2], y)[0]), 4)
    save_json(OUT, res)


if __name__ == "__main__":
    main()
