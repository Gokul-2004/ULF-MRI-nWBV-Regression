"""
E2 — Non-neural / linear baselines under the headline cross-session design (R4.2, R2.8)
=====================================================================================
Every model here is evaluated exactly like the headline LN+head adapter:
for held-out subject i, fit on the HFC scans of the other 22 subjects and
predict the HFE scan of subject i. Nothing about subject i (either session,
or its label) enters fitting or hyper-parameter selection.

Arms
  constant_loo_mean        mean of the 22 training labels
  vit_frozen_offset        frozen Stage-2 ViT3D output + offset fitted on 22 HFC
  vit_frozen_affine        frozen Stage-2 ViT3D output, affine (a + b*pred) on 22 HFC
  ridge_vit_cls_prenorm    ridge on the 256-d CLS token before the final LayerNorm
                           (the representation the LN+head adapter operates on)
  ridge_vit_cls_postnorm   ridge on the frozen post-LayerNorm CLS token (linear head-only analogue)
  ridge_vit_meantokens     ridge on mean-pooled post-norm patch tokens
  ridge_image_stats        ridge on hand-crafted image statistics (no network)
  ridge_swin_gap           ridge on the 8-d GAP features of the OASIS-trained Swin-UNETR (R2.8)
Ridge: features standardised on the training fold; alpha chosen by inner
leave-one-out on the 22 training subjects (nested), grid 10^-3..10^4.

Reference: headline LN+head LOOCV (experiments/loocv_cross_session).

Output: experiments/r3_e2_nonneural_baselines/results.json
"""

import json

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from scipy import stats

from common import (EXP, ROOT, SEED, lcs, provenance, save_json, scan_path,
                    subjects_and_gt)

OUT = EXP / "r3_e2_nonneural_baselines" / "results.json"
FEAT_CACHE = EXP / "r3_e2_nonneural_baselines" / "features.npz"
ALPHAS = np.logspace(-3, 4, 15)
N_BOOT = 10_000


# ── features ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def vit_features(model, vol):
    x = torch.tensor(vol)[None, None].float()
    h = model.patch_embed(x).flatten(2).transpose(1, 2)
    h = torch.cat([model.cls_token.expand(1, -1, -1), h], dim=1) + model.pos_embed
    for blk in model.blocks:
        h = blk(h)
    pre = h[:, 0].clone()
    hn = model.norm(h)
    pred = model.head(hn[:, 0])
    return pre.numpy().ravel(), hn[:, 0].numpy().ravel(), hn[:, 1:].mean(1).numpy().ravel(), float(pred)


def image_stats(vol_norm, path):
    """Hand-crafted statistics: normalised-intensity percentiles inside a head
    mask, mask fraction, and raw-image QC (position, noise)."""
    img = nib.load(str(path))
    raw = img.get_fdata(dtype=np.float32)
    raw = raw[..., 0] if raw.ndim == 4 else raw
    m = vol_norm > 0.1
    pv = np.percentile(vol_norm[m], [10, 25, 50, 75, 90]) if m.any() else np.zeros(5)
    from scipy import ndimage
    rm = raw > np.percentile(raw, 60)
    com = np.array(ndimage.center_of_mass(rm)) - (np.array(raw.shape) - 1) / 2
    com_mm = com * np.array(img.header.get_zooms()[:3])
    return np.concatenate([pv, [m.mean(), vol_norm.mean(), vol_norm.std()], com_mm])


def swin_model():
    from monai.networks.nets import SwinUNETR

    class SwinReg(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = SwinUNETR(in_channels=1, out_channels=8, feature_size=48, spatial_dims=3)
            self.gap = nn.AdaptiveAvgPool3d(1)
            self.head = nn.Linear(8, 1)

    m = SwinReg()
    ck = torch.load(ROOT / "checkpoints" / "swin_oasis.pt", map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    return m


def extract(subjects):
    if FEAT_CACHE.exists():
        print(f"Using cached features {FEAT_CACHE.relative_to(ROOT)}")
        return dict(np.load(FEAT_CACHE))
    vit = lcs.build_base_model().eval()
    try:
        swin = swin_model()
    except Exception as e:  # monai missing etc.
        print(f"Swin unavailable: {e}")
        swin = None
    F = {}
    for ses in ("HFC", "HFE"):
        pre, post, mean, pred, ims, sw = [], [], [], [], [], []
        for s in subjects:
            vol = lcs.load_scan(s, ses)
            a, b, c, p = vit_features(vit, vol)
            pre.append(a); post.append(b); mean.append(c); pred.append(p)
            ims.append(image_stats(vol, scan_path(s, ses)))
            if swin is not None:
                with torch.no_grad():
                    sw.append(swin.gap(swin.backbone(torch.tensor(vol)[None, None].float())).flatten().numpy())
            print(f"  {ses} {s}")
        F[f"{ses}_pre"], F[f"{ses}_post"], F[f"{ses}_mean"] = map(np.array, (pre, post, mean))
        F[f"{ses}_pred"], F[f"{ses}_img"] = np.array(pred), np.array(ims)
        if sw:
            F[f"{ses}_swin"] = np.array(sw)
    FEAT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(FEAT_CACHE, **F)
    return F


# ── models ────────────────────────────────────────────────────────────────────

def ridge_fit_predict(Xtr, ytr, Xte, alpha):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1
    Z = (Xtr - mu) / sd
    ym = ytr.mean()
    A = Z.T @ Z + alpha * np.eye(Z.shape[1])
    w = np.linalg.solve(A, Z.T @ (ytr - ym))
    return ((Xte - mu) / sd) @ w + ym


def nested_alpha(X, y):
    best, best_a = np.inf, None
    for a in ALPHAS:
        err = [abs(ridge_fit_predict(np.delete(X, j, 0), np.delete(y, j), X[j:j + 1], a)[0] - y[j])
               for j in range(len(y))]
        if np.mean(err) < best:
            best, best_a = np.mean(err), a
    return best_a


def cross_session(name, Xc, Xe, y):
    n = len(y)
    preds, alphas = np.zeros(n), []
    for i in range(n):
        tr = np.arange(n) != i
        if name == "constant_loo_mean":
            preds[i] = y[tr].mean()
        elif name == "vit_frozen_offset":
            preds[i] = Xe[i, 0] + np.mean(y[tr] - Xc[tr, 0])
        elif name == "vit_frozen_affine":
            b, a = np.polyfit(Xc[tr, 0], y[tr], 1)
            preds[i] = a + b * Xe[i, 0]
        else:
            al = nested_alpha(Xc[tr], y[tr])
            alphas.append(float(al))
            preds[i] = ridge_fit_predict(Xc[tr], y[tr], Xe[i:i + 1], al)[0]
    return preds, alphas


def metrics(pred, y, ref_err=None, loo_mean=None):
    err = pred - y
    ae = np.abs(err)
    rng = np.random.default_rng(SEED)
    boots = [ae[rng.integers(0, len(ae), len(ae))].mean() for _ in range(N_BOOT)]
    out = {"mae": round(float(ae.mean()), 5),
           "mae_ci95": [round(float(np.percentile(boots, 2.5)), 5), round(float(np.percentile(boots, 97.5)), 5)],
           "rmse": round(float(np.sqrt(np.mean(err ** 2))), 5),
           "bias": round(float(err.mean()), 5),
           "n_below_0.020": int((ae < 0.020).sum())}
    if np.std(pred) > 0:
        r, p = stats.pearsonr(pred, y)
        slope = stats.linregress(pred, y)
        out.update({"pearson_r": round(float(r), 4), "pearson_p": round(float(p), 4),
                    "calibration_slope_true_on_pred": round(float(slope.slope), 3),
                    "calibration_slope_se": round(float(slope.stderr), 3)})
    if loo_mean is not None:
        out["mse_skill_vs_loo_mean"] = round(float(1 - np.mean(err ** 2) / np.mean((loo_mean - y) ** 2)), 4)
        out["mae_skill_vs_loo_mean"] = round(float(1 - ae.mean() / np.mean(np.abs(loo_mean - y))), 4)
    if ref_err is not None:
        d = ae - ref_err
        rng = np.random.default_rng(SEED)
        bd = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)]
        out["vs_headline_delta_mae"] = round(float(d.mean()), 5)
        out["vs_headline_delta_mae_ci95"] = [round(float(np.percentile(bd, 2.5)), 5), round(float(np.percentile(bd, 97.5)), 5)]
        out["vs_headline_wilcoxon_p"] = round(float(stats.wilcoxon(ae, ref_err).pvalue), 4)
    return out


def main():
    subjects, gt = subjects_and_gt()
    y = np.array([gt[s] for s in subjects])
    F = extract(subjects)

    head = json.loads((EXP / "loocv_cross_session" / "results.json").read_text())["per_subject"]
    assert [r["subject"] for r in head] == subjects
    head_pred = np.array([r["pred_hfe"] for r in head])
    head_err = np.abs(head_pred - y)
    loo_mean = np.array([np.delete(y, i).mean() for i in range(len(y))])

    # sanity: our forward reproduces the frozen predictions from E1
    e1 = json.loads((EXP / "r3_frozen_predictions" / "predictions.json").read_text())["per_subject"]
    assert np.allclose([r["pred_hfe"] for r in e1], F["HFE_pred"], atol=1e-4)

    arms = {
        "constant_loo_mean": (F["HFC_pred"][:, None], F["HFE_pred"][:, None]),
        "vit_frozen_offset": (F["HFC_pred"][:, None], F["HFE_pred"][:, None]),
        "vit_frozen_affine": (F["HFC_pred"][:, None], F["HFE_pred"][:, None]),
        "ridge_vit_cls_prenorm": (F["HFC_pre"], F["HFE_pre"]),
        "ridge_vit_cls_postnorm": (F["HFC_post"], F["HFE_post"]),
        "ridge_vit_meantokens": (F["HFC_mean"], F["HFE_mean"]),
        "ridge_image_stats": (F["HFC_img"], F["HFE_img"]),
    }
    if "HFC_swin" in F:
        arms["ridge_swin_gap"] = (F["HFC_swin"], F["HFE_swin"])

    res = {"experiment": "r3_e2_nonneural_baselines", "answers": ["R4.2", "R2.8"],
           "design": "cross-session LOOCV identical to headline: fit on HFC of 22, predict HFE of held-out subject",
           "ridge": {"standardise": "per training fold", "alpha_grid": [float(a) for a in ALPHAS],
                     "alpha_selection": "inner leave-one-out on the 22 training subjects (nested)"},
           "image_stats_features": ["p10", "p25", "p50", "p75", "p90 (normalised intensity in mask>0.1)",
                                    "mask fraction", "mean", "sd", "centre-of-mass offset x/y/z (mm, raw image)"],
           "headline_ln_head_reference": metrics(head_pred, y, loo_mean=loo_mean),
           "arms": {}, "per_subject": [],
           "provenance": provenance("scripts/round3/e2_nonneural_baselines.py")}
    preds_all = {}
    for name, (Xc, Xe) in arms.items():
        p, al = cross_session(name, Xc, Xe, y)
        preds_all[name] = p
        res["arms"][name] = metrics(p, y, ref_err=head_err, loo_mean=loo_mean)
        if al:
            res["arms"][name]["alpha_median"] = float(np.median(al))
            res["arms"][name]["feature_dim"] = int(Xc.shape[1])
        m = res["arms"][name]
        print(f"{name:26s} MAE {m['mae']:.4f} {m['mae_ci95']}  r {m.get('pearson_r', float('nan')):+.3f}  "
              f"skill(MSE) {m['mse_skill_vs_loo_mean']:+.3f}  vs headline d={m['vs_headline_delta_mae']:+.4f} p={m['vs_headline_wilcoxon_p']}")
    h = res["headline_ln_head_reference"]
    print(f"{'headline LN+head':26s} MAE {h['mae']:.4f} {h['mae_ci95']}  r {h['pearson_r']:+.3f}  skill(MSE) {h['mse_skill_vs_loo_mean']:+.3f}")
    for i, s in enumerate(subjects):
        res["per_subject"].append({"subject": s, "true_nwbv": float(y[i]), "headline_ln_head": float(head_pred[i]),
                                   **{k: round(float(v[i]), 5) for k, v in preds_all.items()}})
    save_json(OUT, res)


if __name__ == "__main__":
    main()
