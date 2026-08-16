"""
Additional SSL Pretraining Comparators: MAE, DINO, Contrastive (Reviewer R2.3)
==============================================================================
Extends the SimMIM comparison to the other self-supervised methods Reviewer 2
named: Masked Autoencoders (MAE), DINO (self-distillation), and contrastive
learning (SimCLR-style InfoNCE).

To isolate the OBJECTIVE, all three use — identically to the SimMIM run:
  - the SAME ViT3D encoder (4.23M params, 64^3 input, 16^3 patches),
  - the SAME physics-simulated 64 mT IXI inputs (FieldConverter on IXI volumes),
  - the SAME OASIS-1 seed-42 fine-tuning protocol (reused verbatim from
    ssl_comparator_simmim.finetune, via a shared encoder checkpoint path).
Only the Stage-A pretraining loss differs.

Usage:
    python scripts/ssl_comparators_extra.py mae
    python scripts/ssl_comparators_extra.py dino
    python scripts/ssl_comparators_extra.py contrastive

Each writes experiments/ssl_comparator_<method>/results.json and a pretrained
encoder checkpoint, then fine-tunes on OASIS exactly as SimMIM did.
"""

import sys, json, pickle, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
N_PATCH_SIDE = 4
N_PATCHES    = N_PATCH_SIDE ** 3
EMBED        = 256
SEED         = 42

PRETRAIN_EPOCHS = 25
PRETRAIN_LR     = 1e-4
PRETRAIN_BATCH  = 4
IXI_DIR = project_root / "data" / "ixi_raw" / "IXI_data" / "Train"
MAX_IXI = 156

FT_EPOCHS = 50
FT_LR     = 5e-5
FT_BATCH  = 4
FT_PATIENCE = 10


# ── IXI loading (identical to SimMIM) ───────────────────────────────────────
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
        lf = conv.convert(vol.astype(np.float32), method="hyperfine")
        vols.append(lf.astype(np.float32))
    return np.stack(vols)


def base_encoder():
    m = BaselineViT3D(img_size=TARGET_SHAPE, patch_size=PATCH, num_classes=4,
                      embed_dim=EMBED, num_layers=4, num_heads=8)
    return m


def encode_tokens(enc, x):
    """Run patch_embed + blocks + norm -> (B, N_PATCHES, EMBED) tokens."""
    pe = enc.patch_embed(x)                    # (B,256,4,4,4)
    tok = pe.flatten(2).transpose(1, 2)        # (B,64,256)
    h = tok
    for blk in enc.blocks:
        h = blk(h)
    return enc.norm(h)                         # (B,64,256)


def patchify(x):
    B = x.shape[0]; p = PATCH
    x = x.unfold(2, p, p).unfold(3, p, p).unfold(4, p, p)
    return x.contiguous().view(B, 1, N_PATCHES, p * p * p).squeeze(1)  # (B,64,4096)


# ── augmentation for DINO / contrastive (two views) ─────────────────────────
def augment_view(x):
    """Light 3D augmentation: noise + intensity scale + random flip."""
    v = x + torch.randn_like(x) * 0.02
    v = v * (0.9 + 0.2 * torch.rand(x.shape[0], 1, 1, 1, 1))
    if torch.rand(1).item() > 0.5:
        v = torch.flip(v, dims=[2 + int(torch.randint(0, 3, (1,)).item())])
    return v.clamp(0, 1)


# ── MAE: encoder-only mask, reconstruct masked patches ──────────────────────
class MAEModel(nn.Module):
    """MAE-style: mask a high fraction, encode visible, reconstruct masked."""
    def __init__(self, mask_ratio=0.75):
        super().__init__()
        e = base_encoder()
        self.patch_embed = e.patch_embed
        self.blocks = e.blocks
        self.norm = e.norm
        self._enc = e
        self.mask_token = nn.Parameter(torch.zeros(1, 1, EMBED))
        self.decoder = nn.Sequential(nn.Linear(EMBED, 512), nn.GELU(),
                                     nn.Linear(512, PATCH ** 3))
        self.mask_ratio = mask_ratio

    def forward(self, x):
        B = x.shape[0]
        pe = self.patch_embed(x)
        tok = pe.flatten(2).transpose(1, 2)          # (B,64,256)
        n_mask = int(N_PATCHES * self.mask_ratio)
        mask = torch.zeros(B, N_PATCHES, dtype=torch.bool, device=x.device)
        for b in range(B):
            mask[b, torch.randperm(N_PATCHES)[:n_mask]] = True
        tok_in = tok.clone()
        tok_in[mask] = self.mask_token.to(tok.dtype)
        h = tok_in
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        rec = self.decoder(h)
        tgt = patchify(x)
        return ((rec - tgt) ** 2)[mask].mean()

    def transfer_into(self, base):
        base.patch_embed.load_state_dict(self.patch_embed.state_dict())
        base.blocks.load_state_dict(self.blocks.state_dict())
        base.norm.load_state_dict(self.norm.state_dict())


# ── DINO: student/teacher self-distillation, CLS-pooled ─────────────────────
class DINOHead(nn.Module):
    def __init__(self, out_dim=1024):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(EMBED, 512), nn.GELU(),
                                 nn.Linear(512, out_dim))
    def forward(self, z):
        return self.mlp(z)


class DINOModel(nn.Module):
    """Self-distillation: student sees view1, teacher (EMA) sees view2."""
    def __init__(self, out_dim=1024, tau_s=0.1, tau_t=0.04, m=0.996):
        super().__init__()
        self.student = base_encoder()
        self.teacher = base_encoder()
        self.s_head = DINOHead(out_dim)
        self.t_head = DINOHead(out_dim)
        self.teacher.load_state_dict(self.student.state_dict())
        self.t_head.load_state_dict(self.s_head.state_dict())
        for p in list(self.teacher.parameters()) + list(self.t_head.parameters()):
            p.requires_grad = False
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.tau_s, self.tau_t, self.m = tau_s, tau_t, m

    def _emb(self, enc, head, x):
        h = encode_tokens(enc, x).mean(dim=1)     # mean-pool tokens (B,256)
        return head(h)

    def forward(self, x):
        v1, v2 = augment_view(x), augment_view(x)
        s1 = self._emb(self.student, self.s_head, v1)
        s2 = self._emb(self.student, self.s_head, v2)
        with torch.no_grad():
            t1 = self._emb(self.teacher, self.t_head, v1)
            t2 = self._emb(self.teacher, self.t_head, v2)
            tc1 = F.softmax((t1 - self.center) / self.tau_t, dim=-1)
            tc2 = F.softmax((t2 - self.center) / self.tau_t, dim=-1)
        ls1 = F.log_softmax(s1 / self.tau_s, dim=-1)
        ls2 = F.log_softmax(s2 / self.tau_s, dim=-1)
        loss = -(tc2 * ls1).sum(-1).mean() / 2 - (tc1 * ls2).sum(-1).mean() / 2
        # update center + EMA teacher
        with torch.no_grad():
            self.center = 0.9 * self.center + 0.1 * torch.cat([t1, t2]).mean(0, keepdim=True)
        return loss

    @torch.no_grad()
    def ema_update(self):
        for ps, pt in zip(self.student.parameters(), self.teacher.parameters()):
            pt.data = self.m * pt.data + (1 - self.m) * ps.data
        for ps, pt in zip(self.s_head.parameters(), self.t_head.parameters()):
            pt.data = self.m * pt.data + (1 - self.m) * ps.data

    def transfer_into(self, base):
        base.patch_embed.load_state_dict(self.student.patch_embed.state_dict())
        base.blocks.load_state_dict(self.student.blocks.state_dict())
        base.norm.load_state_dict(self.student.norm.state_dict())


# ── Contrastive: SimCLR InfoNCE on two views ────────────────────────────────
class ContrastiveModel(nn.Module):
    def __init__(self, proj_dim=128, temp=0.2):
        super().__init__()
        self.enc = base_encoder()
        self.proj = nn.Sequential(nn.Linear(EMBED, 256), nn.GELU(),
                                  nn.Linear(256, proj_dim))
        self.temp = temp

    def _z(self, x):
        h = encode_tokens(self.enc, x).mean(dim=1)
        return F.normalize(self.proj(h), dim=-1)

    def forward(self, x):
        z1, z2 = self._z(augment_view(x)), self._z(augment_view(x))
        B = z1.shape[0]
        z = torch.cat([z1, z2], 0)                    # (2B, d)
        sim = z @ z.t() / self.temp                   # (2B,2B)
        sim.fill_diagonal_(-9e15)
        targets = torch.cat([torch.arange(B) + B, torch.arange(B)]).to(x.device)
        return F.cross_entropy(sim, targets)

    def transfer_into(self, base):
        base.patch_embed.load_state_dict(self.enc.patch_embed.state_dict())
        base.blocks.load_state_dict(self.enc.blocks.state_dict())
        base.norm.load_state_dict(self.enc.norm.state_dict())


BUILDERS = {"mae": MAEModel, "dino": DINOModel, "contrastive": ContrastiveModel}


def pretrain(method, enc_ckpt):
    print("=" * 62); print(f"{method.upper()} PRETRAIN on physics-sim IXI"); print("=" * 62, flush=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    data = load_ixi_simulated()
    print(f"IXI simulated volumes: {data.shape}", flush=True)
    X = torch.tensor(data).unsqueeze(1).float()

    model = BUILDERS[method]().to(DEVICE)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=PRETRAIN_LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=PRETRAIN_EPOCHS)
    model.train()
    for ep in range(PRETRAIN_EPOCHS):
        perm = torch.randperm(len(X)); tot, nb = 0.0, 0
        for i in range(0, len(X), PRETRAIN_BATCH):
            xb = X[perm[i:i + PRETRAIN_BATCH]].to(DEVICE)
            opt.zero_grad(); loss = model(xb); loss.backward(); opt.step()
            if method == "dino":
                model.ema_update()
            tot += loss.item(); nb += 1
        sched.step()
        print(f"  {method} pretrain ep {ep+1:2d}/{PRETRAIN_EPOCHS}  loss={tot/max(nb,1):.5f}", flush=True)

    base = base_encoder()
    base.head = nn.Linear(base.head.in_features, 1)
    model.transfer_into(base)
    torch.save({"model_state_dict": base.state_dict()}, enc_ckpt)
    print(f"Saved {method}-pretrained encoder -> {enc_ckpt}", flush=True)


def finetune(method, enc_ckpt):
    """OASIS fine-tune — identical protocol to the SimMIM run."""
    print("=" * 62); print(f"OASIS FINE-TUNE from {method} encoder"); print("=" * 62, flush=True)
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
    tl = DataLoader(fo.OASISDataset(tr, conv), batch_size=FT_BATCH, shuffle=True, num_workers=2)
    vl = DataLoader(fo.OASISDataset(va, conv), batch_size=FT_BATCH, shuffle=False, num_workers=2)
    tel = DataLoader(fo.OASISDataset(te, conv), batch_size=FT_BATCH, shuffle=False, num_workers=2)

    model = base_encoder(); model.head = nn.Linear(model.head.in_features, 1)
    model.load_state_dict(torch.load(enc_ckpt, weights_only=False)["model_state_dict"])
    model.to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=FT_LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=FT_EPOCHS)
    crit = nn.MSELoss()
    best_r, best_state, patience, hist = -np.inf, None, 0, []

    for ep in range(FT_EPOCHS):
        model.train(); tot = 0.0
        for x, y in tl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step(); tot += loss.item()
        tot /= len(tl)
        model.eval(); vp, vy = [], []
        with torch.no_grad():
            for x, y in vl: vp.append(model(x.to(DEVICE)).cpu().numpy()); vy.append(y.numpy())
        sched.step()
        vp = np.concatenate(vp).flatten(); vy = np.concatenate(vy).flatten()
        try: r = float(pearsonr(vy, vp)[0])
        except Exception: r = 0.0
        hist.append({"epoch": ep + 1, "train_loss": round(tot, 6), "val_r": round(r, 4)})
        print(f"  ft ep {ep+1:2d}/{FT_EPOCHS}  train={tot:.5f}  val_r={r:.3f}", flush=True)
        if r > best_r:
            best_r = r; best_state = {k: v.clone() for k, v in model.state_dict().items()}; patience = 0
        else:
            patience += 1
            if patience >= FT_PATIENCE: print(f"early stop ep {ep+1}", flush=True); break

    model.load_state_dict(best_state); model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in tel: preds.append(model(x.to(DEVICE)).cpu().numpy()); trues.append(y.numpy())
    preds = np.concatenate(preds).flatten(); trues = np.concatenate(trues).flatten()
    mae = float(np.mean(np.abs(preds - trues)))
    r_test = float(pearsonr(trues, preds)[0]) if len(trues) > 2 else 0.0

    print("\n" + "=" * 62)
    print(f"{method.upper()}->OASIS TEST:  MAE={mae:.4f}  Pearson r={r_test:.3f}  (n_test={len(trues)})")
    print("=" * 62, flush=True)

    out_dir = project_root / "experiments" / f"ssl_comparator_{method}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "experiment": f"ssl_comparator_{method}",
        "addresses": ["R2.3"],
        "objective": {"mae": "Masked Autoencoder (mask 0.75, reconstruct masked patches)",
                      "dino": "DINO self-distillation (student/teacher, EMA, centering)",
                      "contrastive": "SimCLR InfoNCE on two augmented views"}[method],
        "same_as_paper": "ViT3D encoder, physics-sim IXI input, OASIS seed-42 finetune",
        "pretrain_epochs": PRETRAIN_EPOCHS,
        "n_test": len(trues), "mae": round(mae, 4), "pearson_r": round(r_test, 4),
        "best_val_r": round(best_r, 4), "finetune_history": hist,
        "per_test": [{"true": round(float(t), 4), "pred": round(float(p), 4)} for t, p in zip(trues, preds)],
        "note": "Compare against denoising ViT3D (MAE 0.058) and SimMIM (0.054) on the same OASIS test split.",
    }
    (out_dir / "results.json").write_text(json.dumps(out, indent=2))
    print(f"Saved -> {out_dir / 'results.json'}", flush=True)


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in BUILDERS:
        sys.exit(f"usage: python {sys.argv[0]} [mae|dino|contrastive]")
    method = sys.argv[1]
    enc_ckpt = project_root / "checkpoints" / f"{method}_pretrained_encoder.pt"
    pretrain(method, enc_ckpt)
    finetune(method, enc_ckpt)


if __name__ == "__main__":
    main()
