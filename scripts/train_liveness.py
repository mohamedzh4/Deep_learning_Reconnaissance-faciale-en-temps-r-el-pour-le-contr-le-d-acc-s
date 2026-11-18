"""Train the liveness (anti-spoofing) classifier locally (on Mac MPS/CPU or CUDA).

Reads: data/liveness/<source>/<live|spoof>/<group>/*.jpg
Outputs: models/liveness/antispoof.onnx + models/liveness/antispoof.json

Usage:
  python scripts/train_liveness.py                 # train on data/liveness/own
  python scripts/train_liveness.py --epochs 15     # custom epochs
  python scripts/train_liveness.py --sources own celeba_spoof  # multiple sources
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve

from src.config import ROOT, load_config

# Check PyTorch availability
try:
    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.transforms as T
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    sys.exit("PyTorch is required: pip install torch torchvision")


# ── Dataset Indexation & Splitting ───────────────────────────────────────────
def index_dataset(root):
    rows = []
    root = Path(root)
    if not root.exists():
        return pd.DataFrame(columns=["source", "label", "group", "key", "path"])
    for src in sorted(p for p in root.iterdir() if p.is_dir()):
        for lab_name, lab in (("live", 1), ("spoof", 0)):
            target_dir = src / lab_name
            if not target_dir.exists():
                continue
            for gdir in sorted(target_dir.glob("*")):
                if not gdir.is_dir():
                    continue
                for p in sorted(gdir.glob("*.jpg")):
                    rows.append((src.name, lab, gdir.name, f"{src.name}/{gdir.name}", str(p)))
    return pd.DataFrame(rows, columns=["source", "label", "group", "key", "path"])


def assign_splits(df, train_sources, val=0.15, test=0.2, seed=0):
    rng = np.random.default_rng(seed)
    split = {}
    for src in train_sources:
        d = df[df.source == src]
        kl = d.groupby("key")["label"].agg(lambda s: tuple(sorted(set(s)))).to_dict()
        for lset in sorted(set(kl.values())):
            ks = sorted(k for k, v in kl.items() if v == lset)
            rng.shuffle(ks)
            n = len(ks)
            if n < 3:
                print(f"⚠️  {src} labels={lset}: {n} group(s) -> all placed in train. Collect >=3 distinct groups.")
                for k in ks:
                    split[k] = "train"
                continue
            n_test, n_val = max(1, round(n * test)), max(1, round(n * val))
            for i, k in enumerate(ks):
                split[k] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
    df = df.copy()
    df["split"] = df["key"].map(split).fillna("unused")
    return df


def eer_threshold(scores, labels):
    fpr, tpr, thr = roc_curve(labels, scores)
    i = int(np.nanargmin(np.abs(fpr - (1 - tpr))))
    return float(thr[i]) if np.isfinite(thr[i]) else 0.5


def metrics_at(scores, labels, t):
    scores, labels = np.asarray(scores), np.asarray(labels)
    pred_live = scores >= t
    apcer = float(pred_live[labels == 0].mean()) if (labels == 0).any() else float("nan")
    bpcer = float((~pred_live[labels == 1]).mean()) if (labels == 1).any() else float("nan")
    return dict(APCER=apcer, BPCER=bpcer, ACER=(apcer + bpcer) / 2)


# ── PyTorch Dataset & Transforms ─────────────────────────────────────────────
class LivenessDataset(Dataset):
    def __init__(self, df, size, is_train=True):
        self.paths = df.path.tolist()
        self.labels = df.label.tolist()
        self.size = size
        if is_train:
            self.transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.3, 0.3, 0.3, 0.05),
                T.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                T.RandomApply([T.GaussianBlur(3, (0.1, 1.2))], p=0.2),
                T.ToTensor(),
            ])
        else:
            self.transform = T.ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        if img.size != (self.size, self.size):
            img = img.resize((self.size, self.size))
        return self.transform(img), self.labels[i]


# ── Model Architecture (MobileNetV3-Small) ───────────────────────────────────
class AntiSpoofNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        m = torchvision.models.mobilenet_v3_small(weights=weights)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 2)
        self.net = m
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


class AntiSpoofExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, x):
        return torch.softmax(self.m(x), dim=1)[:, 1]  # P(live)


@torch.no_grad()
def predict_scores(model, loader, device):
    model.eval()
    scores = []
    for x, _ in loader:
        x = x.to(device)
        logits = model(x)
        p = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
        scores.append(p)
    return np.concatenate(scores) if scores else np.array([])


# ── Main Training Loop ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train Anti-Spoofing Liveness Model locally")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sources", nargs="+", default=["own"])
    parser.add_argument("--config", help="path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    lc = cfg.get("liveness", {})
    size = int(lc.get("size", 128))
    crop_scale = float(lc.get("crop_scale", 2.0))

    # Detect best available device (MPS on Apple Silicon, CUDA on Nvidia, or CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using NVIDIA GPU (CUDA): {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    live_dir = ROOT / "data" / "liveness"
    print(f"Scanning liveness data in {live_dir}...")
    df = index_dataset(live_dir)
    if len(df) == 0:
        sys.exit(
            f"No images found under {live_dir}.\n"
            "Collect data first using:\n"
            "  python scripts/collect_liveness_data.py --label live --group s1\n"
            "  python scripts/collect_liveness_data.py --label spoof --group s1"
        )

    print(f"Found {len(df)} total images across {df.group.nunique()} groups.")
    df = assign_splits(df, args.sources)
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0).rename(columns={0: "spoof", 1: "live"}))

    d_train = df[df.split == "train"]
    d_val = df[df.split == "val"]
    if len(d_val) == 0 or d_val.label.nunique() < 2:
        print("Val set too small, using 20% random split of training set for validation...")
        from sklearn.model_selection import train_test_split
        d_train, d_val = train_test_split(df, test_size=0.2, random_state=42, stratify=df.label)

    train_loader = DataLoader(LivenessDataset(d_train, size, is_train=True), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(LivenessDataset(d_val, size, is_train=False), batch_size=args.batch_size, shuffle=False)

    model = AntiSpoofNet(pretrained=True).to(device)
    counts = np.bincount(d_train.label, minlength=2)
    weights = torch.tensor(len(d_train) / (2.0 * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.epochs * len(train_loader))

    print(f"\nTraining for {args.epochs} epochs...")
    best_auc = -1.0
    best_state = None

    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * len(x)

        val_scores = predict_scores(model, val_loader, device)
        val_auc = roc_auc_score(d_val.label, val_scores)
        m = metrics_at(val_scores, d_val.label, 0.5)
        dt = time.time() - t0
        print(f"Epoch {ep:02d}/{args.epochs:02d} | Loss: {total_loss / len(d_train):.4f} | Val AUC: {val_auc:.4f} | ACER@0.5: {m['ACER']:.3f} | {dt:.1f}s")
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)

    # Threshold calibration on validation set
    final_val_scores = predict_scores(model, val_loader, device)
    thr = eer_threshold(final_val_scores, d_val.label.values)
    print(f"\nCalibrated optimal threshold (EER): {thr:.4f}")

    # Export to ONNX
    out_dir = ROOT / "models" / "liveness"
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "antispoof.onnx"
    json_path = out_dir / "antispoof.json"

    export_net = AntiSpoofExport(model.to("cpu").eval())
    dummy_input = torch.rand(1, 3, size, size)
    kw = dict(
        input_names=["input"],
        output_names=["prob_live"],
        dynamic_axes={"input": {0: "batch"}, "prob_live": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exporting ONNX model to {onnx_path}...")
    torch.onnx.export(export_net, dummy_input, str(onnx_path), **kw)

    meta = {
        "size": size,
        "scale": crop_scale,
        "threshold": round(thr, 4),
        "output": "prob_live (1 = live)",
        "best_val_auc": round(best_auc, 4),
        "trained_on": args.sources,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    json_path.write_text(json.dumps(meta, indent=2))
    print(f"Saved metadata to {json_path}")
    print("\n✅ Training and export complete! You can now enable liveness in Settings or config.yaml.")


if __name__ == "__main__":
    main()

