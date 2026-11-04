"""ROC + threshold selection on your own data.
Probes = images not used for templates + all images of '_*' impostor folders.
Usage: python scripts/calibrate_threshold.py --far 0.01 [--write]"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve

from src.config import ROOT, load_config
from src.face_engine import FaceEngine
from src.gallery import Gallery

EXT = {".jpg", ".jpeg", ".png"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--far", type=float, default=1e-2)
    ap.add_argument("--write", action="store_true", help="write the threshold into config.yaml")
    ap.add_argument("--config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    engine = FaceEngine(cfg)
    gallery = Gallery.load(ROOT / cfg["paths"]["gallery"])
    if len(gallery) == 0:
        sys.exit("Gallery is empty: enroll people first.")
    n_t = cfg["enroll"]["n_templates"]

    X, y = [], []
    for pdir in sorted(p for p in (ROOT / cfg["paths"]["own_dataset"]).iterdir() if p.is_dir()):
        skip = 0 if pdir.name.startswith("_") else n_t
        for p in sorted(q for q in pdir.glob("*") if q.suffix.lower() in EXT)[skip:]:
            e = engine.embed(cv2.imread(str(p)))
            if e is not None:
                X.append(e)
                y.append(pdir.name)
    if not X:
        sys.exit("No probe images found.")
    X, y = np.array(X), np.array(y)

    S = X @ gallery.embs.T
    genuine = y[:, None] == np.array(gallery.names)[None, :]
    scores, labels = S.ravel(), genuine.ravel().astype(int)
    n_gen, n_imp = int(labels.sum()), int((1 - labels).sum())
    if n_gen == 0 or n_imp == 0:
        sys.exit("Need both genuine and impostor pairs (>= 2 enrolled people, or an _impostors folder).")

    fpr, tpr, thr = roc_curve(labels, scores)
    idx = max(int(np.searchsorted(fpr, args.far, side="right")) - 1, 0)
    t = float(thr[idx])
    print(f"genuine pairs: {n_gen} | impostor pairs: {n_imp}")
    print(f"AUC={auc(fpr, tpr):.4f} | threshold={t:.3f} @ FAR={fpr[idx]:.4f} -> TAR={tpr[idx]:.4f}, FRR={1 - tpr[idx]:.4f}")
    if n_imp < 1 / args.far:
        print("WARNING: too few impostor pairs to trust this FAR; add more people / impostor images.")

    out = ROOT / cfg["paths"]["results"]
    out.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4))
    plt.semilogx(np.maximum(fpr, 1e-6), tpr)
    plt.xlabel("FAR")
    plt.ylabel("TAR")
    plt.grid(True, which="both", alpha=0.3)
    plt.title("ROC")
    plt.savefig(out / "roc.png", dpi=150, bbox_inches="tight")
    print(f"ROC saved to {out / 'roc.png'}")

    if args.write:
        cfg_path = Path(args.config) if args.config else ROOT / "config.yaml"
        text = cfg_path.read_text(encoding="utf-8")
        text = re.sub(r"(^\s*threshold:\s*)[0-9.]+", lambda m: f"{m.group(1)}{t:.3f}", text, count=1, flags=re.M)
        cfg_path.write_text(text, encoding="utf-8")
        print(f"threshold written to {cfg_path}")


if __name__ == "__main__":
    main()
