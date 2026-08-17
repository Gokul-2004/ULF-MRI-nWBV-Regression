"""
DINO headline test: DINO-pretrained encoder -> OASIS finetune (SAVED) -> 64mT LOOCV
===================================================================================
Tests whether DINO's OASIS advantage (MAE 0.027 vs denoising 0.058) transfers to
the real 64 mT cross-session LOOCV target (the headline metric; denoising gives
0.0134). Per the Swin transfer probe, high-field gains may NOT survive the
domain gap — this settles it for the pretraining objective.

Stage A: rebuild the DINO->OASIS finetuned regressor and SAVE it as
         checkpoints/dino_oasis_finetuned.pt in the format loocv_cross_session
         expects ({"model_state_dict": ...}, head shape [1,256]).
Stage B: run the identical cross-session LOOCV (LN+head adapter) on that
         checkpoint, reporting 64mT MAE / ICC — directly comparable to the
         denoising headline 0.0134.

    python scripts/dino_headline_test.py

Output: experiments/dino_headline_loocv/results.json
"""

import sys, json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

import warnings; warnings.filterwarnings("ignore")

from models.baselines import BaselineViT3D
import ssl_comparators_extra as sse
import finetune_oasis as fo

DEVICE = torch.device("cpu")
TARGET = (64, 64, 64)
EMBED  = 256
SEED   = 42

DINO_ENC   = project_root / "checkpoints" / "dino_pretrained_encoder.pt"
DINO_OASIS = project_root / "checkpoints" / "dino_oasis_finetuned.pt"
OUT_DIR    = project_root / "experiments" / "dino_headline_loocv"


# ── Stage A: DINO -> OASIS finetune, SAVE the regressor ─────────────────────
def finetune_and_save():
    print("=" * 62); print("Stage A: DINO -> OASIS finetune (saving checkpoint)"); print("=" * 62, flush=True)
    torch.manual_seed(SEED); np.random.seed(SEED)

    scans = fo.find_oasis_scans()
    import pandas as pd
    df = pd.read_excel(fo.CSV_PATH); df["subj_id"] = df["ID"].str.extract(r"(OAS1_\d{4}_MR\d)")
    df = df.set_index("subj_id")
    records = fo.build_records(scans, df)
    n = len(records); n_tr = int(0.8 * n); n_va = int(0.1 * n)
    np.random.seed(42); idx = np.random.permutation(n)
    tr = [records[i] for i in idx[:n_tr]]
    va = [records[i] for i in idx[n_tr:n_tr + n_va]]
    te = [records[i] for i in idx[n_tr + n_va:]]

    conv = fo.FieldConverter({})
    from torch.utils.data import DataLoader
    tl = DataLoader(fo.OASISDataset(tr, conv), batch_size=4, shuffle=True, num_workers=2)
    vl = DataLoader(fo.OASISDataset(va, conv), batch_size=4, shuffle=False, num_workers=2)
    tel = DataLoader(fo.OASISDataset(te, conv), batch_size=4, shuffle=False, num_workers=2)

    # start from the DINO-pretrained encoder (same as the SSL run did)
    model = BaselineViT3D(img_size=TARGET, patch_size=16, num_classes=4,
                          embed_dim=EMBED, num_layers=4, num_heads=8)
    model.head = nn.Linear(model.head.in_features, 1)
    model.load_state_dict(torch.load(DINO_ENC, weights_only=False)["model_state_dict"])
    model.to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=5e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
    crit = nn.MSELoss()
    best_r, best_state, patience = -np.inf, None, 0

    for ep in range(50):
        model.train(); tot = 0.0
        for x, y in tl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step(); tot += loss.item()
        model.eval(); vp, vy = [], []
        with torch.no_grad():
            for x, y in vl: vp.append(model(x.to(DEVICE)).cpu().numpy()); vy.append(y.numpy())
        sched.step()
        vp = np.concatenate(vp).flatten(); vy = np.concatenate(vy).flatten()
        try: r = float(pearsonr(vy, vp)[0])
        except Exception: r = 0.0
        print(f"  ft ep {ep+1:2d}/50  train={tot/len(tl):.5f}  val_r={r:.3f}", flush=True)
        if r > best_r:
            best_r = r; best_state = {k: v.clone() for k, v in model.state_dict().items()}; patience = 0
        else:
            patience += 1
            if patience >= 10: print(f"early stop ep {ep+1}", flush=True); break

    # test (sanity, should match the SSL run ~0.027)
    model.load_state_dict(best_state); model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in tel: preds.append(model(x.to(DEVICE)).cpu().numpy()); trues.append(y.numpy())
    preds = np.concatenate(preds).flatten(); trues = np.concatenate(trues).flatten()
    oasis_mae = float(np.mean(np.abs(preds - trues)))
    print(f"  DINO->OASIS test MAE={oasis_mae:.4f} (SSL run reported 0.027)", flush=True)

    # SAVE in loocv-compatible format
    torch.save({"epoch": 0, "model_name": "dino_oasis_finetuned",
                "model_state_dict": best_state, "val_loss": 0.0, "train_loss": 0.0},
               DINO_OASIS)
    print(f"Saved -> {DINO_OASIS}", flush=True)
    return oasis_mae


# ── Stage B: 64mT cross-session LOOCV on the DINO-OASIS checkpoint ───────────
def run_loocv():
    print("\n" + "=" * 62); print("Stage B: 64mT cross-session LOOCV on DINO encoder"); print("=" * 62, flush=True)
    import loocv_cross_session as lc
    import random
    # point LOOCV's base model at the DINO checkpoint
    lc.VIT_CKPT = DINO_OASIS

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    subjects, gt = lc.load_subjects()
    base = lc.build_base_model()

    records = []
    for i, held in enumerate(subjects):
        train = [s for s in subjects if s != held]
        m = lc.train_adapter(base, train, gt, i)
        pred_hfe = lc.predict(m, held, "HFE")
        pred_hfc = lc.predict(m, held, "HFC")
        if pred_hfe is None: continue
        records.append({"subject": held, "true_nwbv": round(gt[held], 4),
                        "pred_hfe": round(pred_hfe, 4),
                        "pred_hfc": round(pred_hfc, 4) if pred_hfc is not None else None,
                        "abs_error_hfe": round(abs(pred_hfe - gt[held]), 4)})
        print(f"  [{i+1:2d}/23] {held}  err={abs(pred_hfe-gt[held]):.4f}", flush=True)

    trues = np.array([r["true_nwbv"] for r in records])
    preds = np.array([r["pred_hfe"] for r in records])
    errs = np.abs(preds - trues)
    mae = float(np.mean(errs))
    bias = float(np.mean(preds - trues))
    n_below = int(np.sum(errs < 0.020))

    icc = None
    if all(r["pred_hfc"] is not None for r in records) and len(records) == len(subjects):
        hfc = np.array([r["pred_hfc"] for r in records])
        icc = float(lc.compute_icc31(hfc, preds))

    print("\n" + "=" * 62)
    print(f"DINO 64mT LOOCV:  MAE={mae:.4f}  bias={bias:+.4f}  {n_below}/23 below 0.020  ICC={icc}")
    print(f"  vs DENOISING headline: MAE=0.0134, ICC=0.615")
    print("=" * 62, flush=True)
    return records, mae, bias, n_below, icc


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    oasis_mae = finetune_and_save()
    records, mae, bias, n_below, icc = run_loocv()

    verdict = ("DINO improves the 64mT headline" if mae < 0.0134 - 0.0005 else
               "DINO comparable on 64mT (high-field gain does NOT transfer)"
               if abs(mae - 0.0134) <= 0.0005 else
               "DINO worse on 64mT than denoising")

    out = {
        "experiment": "dino_headline_loocv",
        "addresses": ["R2.3 headline decision"],
        "dino_oasis_test_mae": round(oasis_mae, 4),
        "dino_64mt_loocv_mae": round(mae, 4),
        "dino_64mt_bias": round(bias, 4),
        "dino_64mt_icc": round(icc, 4) if icc is not None else None,
        "n_below_threshold": n_below,
        "denoising_headline": {"loocv_mae": 0.0134, "icc": 0.615},
        "verdict": verdict,
        "per_subject": records,
    }
    (OUT_DIR / "results.json").write_text(json.dumps(out, indent=2))
    print(f"\nVERDICT: {verdict}")
    print(f"Saved -> {OUT_DIR / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
