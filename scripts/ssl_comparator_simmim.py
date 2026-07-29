"""
Self-Supervised Pretraining Comparator: SimMIM (Reviewer R2.3)
===============================================================
Compares a SimMIM (masked image modelling) pretraining objective against the
paper's physics-grounded *denoising* Stage-1 objective, then fine-tunes both
identically on OASIS-1.

To isolate the OBJECTIVE (not the data or architecture), SimMIM uses:
  - the SAME ViT3D encoder (4.23M params, 64^3 input, 16^3 patches),
  - the SAME physics-simulated 64 mT IXI inputs (FieldConverter on IXI volumes),
  - the SAME OASIS-1 seed-42 fine-tuning protocol.
Only the pretraining loss differs: SimMIM reconstructs randomly masked patches
(mask ratio 0.5) instead of denoising the full volume to its high-field target.

Two stages:
  Stage A — SimMIM pretrain the ViT3D encoder on physics-sim IXI (25 epochs).
  Stage B — Transfer encoder, fine-tune on OASIS-1 for nWBV (same as ViT3D).

Output: experiments/ssl_comparator_simmim/results.json
"""

import sys, json, pickle, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import zoom
from scipy.stats import pearsonr

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

from models.baselines import BaselineViT3D
from utils.field_conversion import FieldConverter
import finetune_oasis as fo

import warnings; warnings.filterwarnings("ignore")

DEVICE       = torch.device("cpu")
TARGET_SHAPE = (64, 64, 64)
PATCH        = 16
N_PATCH_SIDE = 4                      # 64/16
N_PATCHES    = N_PATCH_SIDE ** 3      # 64
EMBED        = 256
MASK_RATIO   = 0.5
SEED         = 42

# Stage A (pretrain) — reduced per plan
PRETRAIN_EPOCHS = 25
PRETRAIN_LR     = 1e-4
PRETRAIN_BATCH  = 4
IXI_DIR = project_root / "data" / "ixi_raw" / "IXI_data" / "Train"
MAX_IXI = 156                          # match paper's Stage-1 IXI n

# Stage B (OASIS fine-tune) — same as ViT3D
FT_EPOCHS = 50
FT_LR     = 5e-5
FT_BATCH  = 4
FT_PATIENCE = 10

OUT_DIR   = project_root / "experiments" / "ssl_comparator_simmim"
ENC_CKPT  = project_root / "checkpoints" / "simmim_pretrained_encoder.pt"


# ── IXI loading + physics simulation (same simulator as Stage-1) ────────────
def load_ixi_simulated(max_n=MAX_IXI):
    conv = FieldConverter({})
    files = sorted(IXI_DIR.glob("subject_*.pkl"))[:max_n]
    vols = []
    for f in files:
        vol, _seg = pickle.load(open(f, "rb"))
        vol = vol.astype(np.float32)
        vmin, vmax = vol.min(), vol.max()
        if vmax > vmin:
            vol = (vol - vmin) / (vmax - vmin)
        if vol.shape != TARGET_SHAPE:
            zf = [t / s for t, s in zip(TARGET_SHAPE, vol.shape)]
            vol = zoom(vol, zf, order=1)
        lf = conv.convert(vol.astype(np.float32), method="hyperfine")  # physics 64mT
        vols.append(lf.astype(np.float32))
    return np.stack(vols)


# ── SimMIM wrapper around the ViT3D encoder ─────────────────────────────────
class SimMIM(nn.Module):
    """ViT3D patch encoder + light decoder that reconstructs masked patches."""
    def __init__(self):
        super().__init__()
        base = BaselineViT3D(img_size=TARGET_SHAPE, patch_size=PATCH,
                             num_classes=4, embed_dim=EMBED,
                             num_layers=4, num_heads=8)
        self.patch_embed = base.patch_embed          # Conv3d 1->256 stride16
        self.blocks      = base.blocks
        self.norm        = base.norm
        self.mask_token  = nn.Parameter(torch.zeros(1, 1, EMBED))
        # decoder: token -> patch voxels (16^3 = 4096)
        self.decoder = nn.Sequential(
            nn.Linear(EMBED, 512), nn.GELU(), nn.Linear(512, PATCH ** 3))
        self._base_ref = base                          # keep for transfer

    def forward(self, x):
        B = x.shape[0]
        pe = self.patch_embed(x)                        # (B,256,4,4,4)
        tok = pe.flatten(2).transpose(1, 2)             # (B,64,256)
        # random mask
        n_mask = int(N_PATCHES * MASK_RATIO)
        mask = torch.zeros(B, N_PATCHES, dtype=torch.bool, device=x.device)
        for b in range(B):
            idx = torch.randperm(N_PATCHES)[:n_mask]
            mask[b, idx] = True
        tok_in = tok.clone()
        tok_in[mask] = self.mask_token.to(tok.dtype)
        h = tok_in
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        rec = self.decoder(h)                           # (B,64,4096) predicted patch voxels
        # target patches from input
        tgt = self._patchify(x)                         # (B,64,4096)
        loss = ((rec - tgt) ** 2)[mask].mean()          # loss on masked patches only
        return loss

    @staticmethod
    def _patchify(x):
        B = x.shape[0]
        p = PATCH
        # (B,1,64,64,64) -> (B, 64patches, 4096)
        x = x.unfold(2, p, p).unfold(3, p, p).unfold(4, p, p)   # B,1,4,4,4,16,16,16
        x = x.contiguous().view(B, 1, N_PATCHES, p * p * p)
        return x.squeeze(1)


def pretrain():
    print("=" * 62); print("SimMIM PRETRAIN on physics-sim IXI"); print("=" * 62, flush=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    data = load_ixi_simulated()
    print(f"IXI simulated volumes: {data.shape}", flush=True)
    X = torch.tensor(data).unsqueeze(1).float()

    model = SimMIM().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=PRETRAIN_LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=PRETRAIN_EPOCHS)

    model.train()
    for ep in range(PRETRAIN_EPOCHS):
        perm = torch.randperm(len(X))
        tot, nb = 0.0, 0
        for i in range(0, len(X), PRETRAIN_BATCH):
            xb = X[perm[i:i + PRETRAIN_BATCH]].to(DEVICE)
            opt.zero_grad()
            loss = model(xb)
            loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        print(f"  pretrain ep {ep+1:2d}/{PRETRAIN_EPOCHS}  loss={tot/max(nb,1):.5f}", flush=True)

    # save the transferable encoder (patch_embed + blocks + norm) via a full ViT3D
    base = model._base_ref
    base.patch_embed.load_state_dict(model.patch_embed.state_dict())
    base.blocks.load_state_dict(model.blocks.state_dict())
    base.norm.load_state_dict(model.norm.state_dict())
    base.head = nn.Linear(base.head.in_features, 1)
    torch.save({"model_state_dict": base.state_dict()}, ENC_CKPT)
    print(f"Saved SimMIM-pretrained encoder → {ENC_CKPT}", flush=True)


def finetune_oasis_from_ckpt():
    print("=" * 62); print("OASIS FINE-TUNE from SimMIM encoder"); print("=" * 62, flush=True)
    torch.manual_seed(SEED); np.random.seed(SEED)

    # identical seed-42 split
    scans = fo.find_oasis_scans()
    import pandas as pd
    df = pd.read_excel(fo.CSV_PATH); df["subj_id"] = df["ID"].str.extract(r"(OAS1_\d{4}_MR\d)")
    df = df.set_index("subj_id")
    records = fo.build_records(scans, df)
    n = len(records); n_tr = int(0.8*n); n_va = int(0.1*n)
    np.random.seed(42); idx = np.random.permutation(n)
    tr = [records[i] for i in idx[:n_tr]]
    va = [records[i] for i in idx[n_tr:n_tr+n_va]]
    te = [records[i] for i in idx[n_tr+n_va:]]

    conv = fo.FieldConverter({})
    from torch.utils.data import DataLoader
    tl = DataLoader(fo.OASISDataset(tr,conv), batch_size=FT_BATCH, shuffle=True,  num_workers=2)
    vl = DataLoader(fo.OASISDataset(va,conv), batch_size=FT_BATCH, shuffle=False, num_workers=2)
    tel= DataLoader(fo.OASISDataset(te,conv), batch_size=FT_BATCH, shuffle=False, num_workers=2)

    model = BaselineViT3D(img_size=TARGET_SHAPE, patch_size=PATCH, num_classes=4,
                          embed_dim=EMBED, num_layers=4, num_heads=8)
    model.head = nn.Linear(model.head.in_features, 1)
    ck = torch.load(ENC_CKPT, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    model.to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=FT_LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=FT_EPOCHS)
    crit = nn.MSELoss()
    best_r, best_state, patience, hist = -np.inf, None, 0, []

    for ep in range(FT_EPOCHS):
        model.train(); tot=0.0
        for x,y in tl:
            x,y=x.to(DEVICE),y.to(DEVICE)
            opt.zero_grad(); loss=crit(model(x),y); loss.backward(); opt.step(); tot+=loss.item()
        tot/=len(tl)
        model.eval(); vp,vy=[],[]
        with torch.no_grad():
            for x,y in vl: vp.append(model(x.to(DEVICE)).cpu().numpy()); vy.append(y.numpy())
        sched.step()
        vp=np.concatenate(vp).flatten(); vy=np.concatenate(vy).flatten()
        try: r=float(pearsonr(vy,vp)[0])
        except Exception: r=0.0
        hist.append({"epoch":ep+1,"train_loss":round(tot,6),"val_r":round(r,4)})
        print(f"  ft ep {ep+1:2d}/{FT_EPOCHS}  train={tot:.5f}  val_r={r:.3f}", flush=True)
        if r>best_r:
            best_r=r; best_state={k:v.clone() for k,v in model.state_dict().items()}; patience=0
        else:
            patience+=1
            if patience>=FT_PATIENCE: print(f"early stop ep {ep+1}", flush=True); break

    model.load_state_dict(best_state); model.eval()
    preds,trues=[],[]
    with torch.no_grad():
        for x,y in tel: preds.append(model(x.to(DEVICE)).cpu().numpy()); trues.append(y.numpy())
    preds=np.concatenate(preds).flatten(); trues=np.concatenate(trues).flatten()
    mae=float(np.mean(np.abs(preds-trues)))
    r_test=float(pearsonr(trues,preds)[0]) if len(trues)>2 else 0.0

    print("\n"+"="*62)
    print(f"SimMIM→OASIS TEST:  MAE={mae:.4f}  Pearson r={r_test:.3f}  (n_test={len(trues)})")
    print("="*62, flush=True)

    out = {
        "experiment":"ssl_comparator_simmim",
        "addresses":["R2.3"],
        "objective":"SimMIM masked-patch reconstruction (mask 0.5) vs paper denoising Stage-1",
        "same_as_paper":"ViT3D encoder, physics-sim IXI input, OASIS seed-42 finetune",
        "pretrain_epochs":PRETRAIN_EPOCHS,
        "n_test":len(trues),
        "mae":round(mae,4),
        "pearson_r":round(r_test,4),
        "best_val_r":round(best_r,4),
        "finetune_history":hist,
        "per_test":[{"true":round(float(t),4),"pred":round(float(p),4)} for t,p in zip(trues,preds)],
        "note":"Compare MAE/r against the paper's denoising-pretrained ViT3D on the same OASIS test split.",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR/"results.json","w") as f: json.dump(out,f,indent=2)
    print(f"Saved → {OUT_DIR/'results.json'}", flush=True)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pretrain()
    finetune_oasis_from_ckpt()
