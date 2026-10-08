"""
E4 — Matched, paired physics-simulation vs Gaussian-blur rerun (R2.7)
====================================================================
R2.7: "A matched-split, paired rerun with equal model budgets is required
before any statement that modeled physics specifically improves nWBV recovery."

The submitted ablation (scripts/ablation_gaussianblur.py) is NOT the headline
pipeline (intensity-regression pretext, clean OASIS fine-tuning, 80 IXI) and
its two arms used different random OASIS test splits. Here BOTH arms run the
exact headline pipeline and differ ONLY in the degradation operator:

  Stage 1  experiments/stage1/run_stage1.train_baseline_model on the 156-volume
           Stage-1 set (utils.data_utils.create_data_loaders, deterministic
           sorted split, 5 augmentations/volume, Adam 5e-4, <=30 epochs,
           patience 10), 4 proxy targets. Inputs:
             physics: data/low_field/*_low_field.npy (FieldConverter 'combined',
                      the files the headline Stage 1 used)
             blur:    data/low_field_blur/*_low_field.npy (Gaussian blur sigma 2
                      + N(0, 0.04) noise — the blur operator of the submitted
                      ablation), generated from the same data/high_field volumes
  Stage 2  scripts/finetune_oasis.py recipe: head swapped to 1 output, full
           fine-tune, Adam 5e-5, cosine T_max 50, <=50 epochs, early stop on
           val Pearson r (patience 10), the seed-42 300/37/38 split. On-the-fly
           degradation: physics = FieldConverter 'hyperfine' (as headline);
           blur = the same blur operator.
  Stage 3  loocv_cross_session.train_adapter (LN+head, 769 params) under the
           cross-session LOOCV on ds006557 (HFC of 22 -> HFE of held-out).

Equal budgets: same volumes, same splits, same epochs/patience, same optimiser,
same augmentation, same seeds per arm. Seeds: 42, 1, 7.

Endpoints (fixed before running):
  primary    paired per-subject |error| on real 64 mT HFE under the LOOCV (n=23)
  secondary  paired per-subject |error| on the 38 OASIS test subjects, with
             FIXED test inputs (identical noise realisation for both arms),
             evaluated on physics-simulated inputs AND on blur inputs.

Usage:
  python e4_matched_physics_vs_blur.py --arm physics --seed 42
  python e4_matched_physics_vs_blur.py --aggregate
Output: experiments/r3_e4_physics_vs_blur/{arm}_seed{seed}.json, summary.json
"""

import argparse
import copy
import json
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from scipy import stats
from scipy.ndimage import gaussian_filter, zoom

from common import EXP, ROOT, lcs, provenance, save_json, subjects_and_gt

sys.path.insert(0, str(ROOT / "experiments" / "stage1"))
import finetune_oasis as fo  # noqa: E402
from run_stage1 import train_baseline_model  # noqa: E402
from utils.data_utils import create_data_loaders  # noqa: E402
from utils.field_conversion import FieldConverter  # noqa: E402
from models.baselines import BaselineViT3D  # noqa: E402

OUT_DIR = EXP / "r3_e4_physics_vs_blur"
CKPT_DIR = ROOT / "checkpoints" / "r3_e4"
LF_PHYS = ROOT / "data" / "low_field"
LF_BLUR = ROOT / "data" / "low_field_blur"
HF = ROOT / "data" / "high_field"
SEEDS = [42, 1, 7]
DEV = torch.device("cpu")


def blur_degrade(vol, rng=np.random):
    """Blur operator of the submitted ablation: Gaussian sigma=2 + N(0,0.04), clipped to [0,1]."""
    b = gaussian_filter(vol.astype(np.float32), sigma=2.0)
    return np.clip(b + rng.normal(0, 0.04, vol.shape).astype(np.float32), 0, 1).astype(np.float32)


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def make_blur_lf():
    """Blur counterpart of data/low_field, one file per Stage-1 volume, fixed noise seed."""
    LF_BLUR.mkdir(parents=True, exist_ok=True)
    phys = sorted(LF_PHYS.glob("*_low_field.npy"))
    rng = np.random.default_rng(12345)
    for p in phys:
        out = LF_BLUR / p.name
        if out.exists():
            continue
        hf = np.load(HF / (p.name.replace("_low_field.npy", ".npy"))).astype(np.float32)
        if hf.ndim == 4:
            hf = hf[0]
        lo, hi = hf.min(), hf.max()
        hf = (hf - lo) / (hi - lo) if hi > lo else hf
        np.save(out, blur_degrade(hf, rng))
    # the biomarker label cache lives with high_field; nothing else to copy
    print(f"blur LF set: {len(list(LF_BLUR.glob('*_low_field.npy')))} volumes")


# ── Stage 1 ───────────────────────────────────────────────────────────────────

def stage1(arm, seed):
    ck = CKPT_DIR / f"{arm}_seed{seed}_stage1.pt"
    if ck.exists():
        return ck, json.loads(ck.with_suffix(".json").read_text())
    cfg = yaml.safe_load(open(ROOT / "configs" / "config.yaml"))
    cfg["data"]["high_field_dir"] = str(HF)
    cfg["data"]["low_field_dir"] = str(LF_PHYS if arm == "physics" else LF_BLUR)
    seed_all(seed)
    t0 = time.time()
    tr, va, te = create_data_loaders(cfg)
    vit = BaselineViT3D(tuple(cfg["data"]["image_size"]), patch_size=16,
                        num_classes=cfg["biomarkers"]["num_classes"], embed_dim=256, num_layers=4, num_heads=8)
    vit = train_baseline_model(cfg, vit, f"ViT[{arm},{seed}]", tr, va, DEV)
    vit.eval()
    se, n = 0.0, 0
    with torch.no_grad():
        for b in te:
            se += float(((vit(b["volume"]) - b["label"]) ** 2).sum()); n += b["label"].numel()
    info = {"n_train_samples": len(tr.dataset), "n_val": len(va.dataset), "n_test": len(te.dataset),
            "stage1_test_mse_4targets": se / n, "seconds": round(time.time() - t0, 1)}
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": vit.state_dict(), "arm": arm, "seed": seed}, ck)
    ck.with_suffix(".json").write_text(json.dumps(info))
    print(f"[stage1 {arm} {seed}] {info}")
    return ck, info


# ── Stage 2 ───────────────────────────────────────────────────────────────────

class OASISDeg(torch.utils.data.Dataset):
    def __init__(self, recs, arm, conv):
        self.recs, self.arm, self.conv = recs, arm, conv

    def __len__(self):
        return len(self.recs)

    def base(self, i):
        import nibabel as nib
        v = nib.load(str(self.recs[i]["t1w"])).get_fdata(dtype=np.float32)
        v = v[..., 0] if v.ndim == 4 else v
        lo, hi = v.min(), v.max()
        v = (v - lo) / (hi - lo) if hi > lo else v
        return zoom(v, [t / s for t, s in zip(fo.TARGET_SHAPE, v.shape)], order=1)

    def __getitem__(self, i):
        v = self.base(i)
        lf = self.conv.convert(v, method="hyperfine") if self.arm == "physics" else blur_degrade(v)
        return (torch.from_numpy(np.asarray(lf, np.float32)).unsqueeze(0),
                torch.tensor([self.recs[i]["nwbv"]], dtype=torch.float32))


def oasis_split():
    import pandas as pd
    df = pd.read_excel(fo.CSV_PATH)
    df["subj_id"] = df["ID"].str.extract(r"(OAS1_\d{4}_MR\d)")
    df = df.set_index("subj_id")
    recs = fo.build_records(fo.find_oasis_scans(), df)
    n = len(recs); ntr = int(0.8 * n); nva = int(0.1 * n)
    st = np.random.get_state()
    np.random.seed(42)
    idx = np.random.permutation(n)
    np.random.set_state(st)
    return [recs[i] for i in idx[:ntr]], [recs[i] for i in idx[ntr:ntr + nva]], [recs[i] for i in idx[ntr + nva:]]


def _winit(wid):
    np.random.seed((torch.initial_seed() + wid) % 2 ** 32)


def stage2(arm, seed, s1_ck):
    ck = CKPT_DIR / f"{arm}_seed{seed}_stage2.pt"
    if ck.exists():
        return ck, json.loads(ck.with_suffix(".json").read_text())
    seed_all(seed)
    tr, va, te = oasis_split()
    conv = FieldConverter({})
    g = torch.Generator(); g.manual_seed(seed)
    trl = torch.utils.data.DataLoader(OASISDeg(tr, arm, conv), batch_size=fo.BATCH_SIZE, shuffle=True,
                                      num_workers=3, worker_init_fn=_winit, generator=g, persistent_workers=True)
    val_x, val_y = fixed_inputs(va, arm, conv, salt=777)
    m = BaselineViT3D(img_size=fo.TARGET_SHAPE, patch_size=16, num_classes=4, embed_dim=256, num_layers=4, num_heads=8)
    m.load_state_dict(torch.load(s1_ck, map_location=DEV, weights_only=False)["model_state_dict"])
    m.head = nn.Linear(m.head.in_features, 1)
    opt = torch.optim.Adam(m.parameters(), lr=fo.LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=fo.EPOCHS)
    crit = nn.MSELoss()
    best_r, best_state, pat, hist = -np.inf, None, 0, []
    t0 = time.time()
    for ep in range(fo.EPOCHS):
        m.train()
        tl = 0.0
        for x, y in trl:
            opt.zero_grad(); loss = crit(m(x), y); loss.backward(); opt.step(); tl += loss.item()
        sch.step()
        m.eval()
        with torch.no_grad():
            vp = m(val_x).numpy().ravel()
        r = float(stats.pearsonr(val_y, vp)[0]) if np.std(vp) > 0 else 0.0
        hist.append({"epoch": ep + 1, "train_loss": tl / len(trl), "val_r": round(r, 4)})
        print(f"  [stage2 {arm} {seed}] ep {ep+1} loss {tl/len(trl):.5f} val_r {r:.3f}  ({time.time()-t0:.0f}s)", flush=True)
        if r > best_r:
            best_r, best_state, pat = r, copy.deepcopy(m.state_dict()), 0
        else:
            pat += 1
            if pat >= fo.PATIENCE:
                break
    torch.save({"model_state_dict": best_state, "arm": arm, "seed": seed}, ck)
    info = {"best_val_r": round(best_r, 4), "epochs_run": len(hist), "seconds": round(time.time() - t0, 1), "history": hist}
    ck.with_suffix(".json").write_text(json.dumps(info))
    return ck, info


def fixed_inputs(recs, kind, conv, salt):
    """Deterministic degraded inputs: same noise realisation for a given (subject, kind)."""
    ds = OASISDeg(recs, kind, conv)
    xs = []
    st = np.random.get_state()
    for i in range(len(recs)):
        np.random.seed(salt + i)
        v = ds.base(i)
        lf = conv.convert(v, method="hyperfine") if kind == "physics" else blur_degrade(v)
        xs.append(np.asarray(lf, np.float32))
    np.random.set_state(st)
    return torch.from_numpy(np.stack(xs)).unsqueeze(1), np.array([r["nwbv"] for r in recs])


# ── Stage 3 + evaluation ──────────────────────────────────────────────────────

def evaluate(arm, seed, s2_ck):
    _, _, te = oasis_split()
    conv = FieldConverter({})
    m = BaselineViT3D(img_size=fo.TARGET_SHAPE, patch_size=16, num_classes=4, embed_dim=256, num_layers=4, num_heads=8)
    m.head = nn.Linear(m.head.in_features, 1)
    m.load_state_dict(torch.load(s2_ck, map_location=DEV, weights_only=False)["model_state_dict"])
    m.eval()
    out = {"oasis_test_subjects": [r["subj_id"] for r in te]}
    for kind in ("physics", "blur"):
        x, y = fixed_inputs(te, kind, conv, salt=99_000)
        with torch.no_grad():
            p = m(x).numpy().ravel()
        out[f"oasis_test_on_{kind}_inputs"] = {"pred": [round(float(v), 5) for v in p], "true": [float(v) for v in y],
                                               "mae": round(float(np.mean(np.abs(p - y))), 5),
                                               "r": round(float(stats.pearsonr(p, y)[0]), 4)}
    # LOOCV on real 64 mT
    lcs.VIT_CKPT = s2_ck
    subjects, gt = subjects_and_gt()
    base = lcs.build_base_model()
    seed_all(seed)
    recs = []
    t0 = time.time()
    for i, s in enumerate(subjects):
        tr = [t for t in subjects if t != s]
        a = lcs.train_adapter(base, tr, gt, i)
        recs.append({"subject": s, "true_nwbv": gt[s], "pred_hfe": round(lcs.predict(a, s, "HFE"), 5),
                     "pred_hfc": round(lcs.predict(a, s, "HFC"), 5)})
        print(f"  [loocv {arm} {seed}] {i+1}/23 {s} err {recs[-1]['pred_hfe']-gt[s]:+.4f} ({time.time()-t0:.0f}s)", flush=True)
    y = np.array([r["true_nwbv"] for r in recs]); p = np.array([r["pred_hfe"] for r in recs])
    out["loocv_real_64mt"] = {"per_subject": recs, "mae": round(float(np.mean(np.abs(p - y))), 5),
                              "r": round(float(stats.pearsonr(p, y)[0]), 4), "bias": round(float(np.mean(p - y)), 5),
                              "seconds": round(time.time() - t0, 1)}
    return out


def run(arm, seed):
    out_path = OUT_DIR / f"{arm}_seed{seed}.json"
    if out_path.exists():
        print(f"done already: {out_path}"); return
    if arm == "blur":
        make_blur_lf()
    s1, i1 = stage1(arm, seed)
    s2, i2 = stage2(arm, seed, s1)
    ev = evaluate(arm, seed, s2)
    save_json(out_path, {"arm": arm, "seed": seed, "stage1": i1, "stage2": i2, **ev,
                         "provenance": provenance("scripts/round3/e4_matched_physics_vs_blur.py")})


def paired(a, b, n_boot=10_000):
    a, b = np.asarray(a), np.asarray(b)
    d = a - b
    rng = np.random.default_rng(42)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    return {"mean_abs_err_physics": round(float(a.mean()), 5), "mean_abs_err_blur": round(float(b.mean()), 5),
            "delta_mae_physics_minus_blur": round(float(d.mean()), 5),
            "delta_ci95": [round(float(np.percentile(bs, 2.5)), 5), round(float(np.percentile(bs, 97.5)), 5)],
            "wilcoxon_p": round(float(stats.wilcoxon(a, b).pvalue), 4), "n": len(a)}


def aggregate():
    res = {"experiment": "r3_e4_matched_physics_vs_blur", "answers": ["R2.7"], "per_seed": {}, "pooled_over_seeds": {}}
    E = {}
    for arm in ("physics", "blur"):
        for s in SEEDS:
            f = OUT_DIR / f"{arm}_seed{s}.json"
            if f.exists():
                E[(arm, s)] = json.loads(f.read_text())
    seeds = [s for s in SEEDS if ("physics", s) in E and ("blur", s) in E]
    keys = {"loocv_real_64mt": lambda e: np.abs(np.array([r["pred_hfe"] - r["true_nwbv"] for r in e["loocv_real_64mt"]["per_subject"]])),
            "oasis_test_on_physics_inputs": lambda e: np.abs(np.array(e["oasis_test_on_physics_inputs"]["pred"]) - np.array(e["oasis_test_on_physics_inputs"]["true"])),
            "oasis_test_on_blur_inputs": lambda e: np.abs(np.array(e["oasis_test_on_blur_inputs"]["pred"]) - np.array(e["oasis_test_on_blur_inputs"]["true"]))}
    for s in seeds:
        res["per_seed"][str(s)] = {k: paired(f(E[("physics", s)]), f(E[("blur", s)])) for k, f in keys.items()}
    if seeds:
        for k, f in keys.items():
            a = np.mean([f(E[("physics", s)]) for s in seeds], axis=0)
            b = np.mean([f(E[("blur", s)]) for s in seeds], axis=0)
            res["pooled_over_seeds"][k] = paired(a, b) | {"note": "per-subject |error| averaged over seeds, then paired across subjects"}
    res["seeds_complete"] = seeds
    save_json(OUT_DIR / "summary.json", res)
    print(json.dumps(res["pooled_over_seeds"], indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["physics", "blur"])
    ap.add_argument("--seed", type=int)
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    if a.aggregate:
        aggregate()
    else:
        run(a.arm, a.seed)
