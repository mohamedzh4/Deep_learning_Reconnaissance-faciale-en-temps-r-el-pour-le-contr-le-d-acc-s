"""Build the gallery from data/own_dataset/<person>/*.jpg (e.g. images prepared in Colab).
Folders starting with '_' (impostors) are skipped. Usage: python scripts/enroll_from_folders.py"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from src.config import ROOT, load_config
from src.face_engine import FaceEngine
from src.gallery import Gallery

EXT = {".jpg", ".jpeg", ".png"}


def list_images(d):
    return sorted(p for p in Path(d).glob("*") if p.suffix.lower() in EXT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    engine = FaceEngine(cfg)
    gallery = Gallery()
    for pdir in sorted(p for p in (ROOT / cfg["paths"]["own_dataset"]).iterdir() if p.is_dir()):
        if pdir.name.startswith("_"):
            continue
        embs = [e for e in (engine.embed(cv2.imread(str(p))) for p in list_images(pdir)[: cfg["enroll"]["n_templates"]]) if e is not None]
        if embs:
            gallery.add(pdir.name, embs)
            print(f"{pdir.name}: {len(embs)} images used")
    gallery.save(ROOT / cfg["paths"]["gallery"])
    print(f"Gallery size: {len(gallery)}")


if __name__ == "__main__":
    main()
