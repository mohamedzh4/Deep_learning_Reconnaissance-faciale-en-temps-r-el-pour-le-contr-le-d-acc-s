"""Images -> normalized tensors. Normalization is fixed: (x - 127.5) / 127.5, RGB, CHW (range [-1, 1]).
The exact same formula is used at inference by the access-control project (src/embedder.py)."""
import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def load_rgb(path, size=112):
    """Read an image as an HxWx3 uint8 RGB array (resized if it is not already size x size)."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.size != (size, size):
            im = im.resize((size, size), Image.BICUBIC)
        return np.asarray(im)


def to_tensor(img_uint8_rgb):
    """HxWx3 uint8 -> 3xHxW float32 in [-1, 1]."""
    x = torch.from_numpy(np.ascontiguousarray(img_uint8_rgb)).permute(2, 0, 1).float()
    return (x - 127.5) / 127.5


def batch_to_tensor(imgs_uint8_rgb):
    """NxHxWx3 uint8 -> Nx3xHxW float32 in [-1, 1]."""
    x = torch.from_numpy(np.ascontiguousarray(imgs_uint8_rgb)).permute(0, 3, 1, 2).float()
    return (x - 127.5) / 127.5


def random_flip(img):
    return img[:, ::-1] if random.random() < 0.5 else img


def photometric(img):
    """Mild brightness/contrast jitter + occasional random erasing (simulates glasses / partial occlusion)."""
    out = img.astype(np.float32) * random.uniform(0.8, 1.2) + random.uniform(-20, 20)
    out = np.clip(out, 0, 255)
    if random.random() < 0.2:
        h, w = out.shape[:2]
        eh, ew = int(h * random.uniform(0.1, 0.4)), int(w * random.uniform(0.1, 0.4))
        y, x = random.randint(0, h - eh), random.randint(0, w - ew)
        out[y:y + eh, x:x + ew] = 127.0
    return out.astype(np.uint8)


class ManifestDataset(Dataset):
    """Reads the manifest written by step2_clean_and_index.py (path,label,identity)."""

    def __init__(self, root, manifest_csv, augment=False, size=112):
        self.root, self.size, self.augment = Path(root), size, augment
        with open(manifest_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.paths = [r["path"] for r in rows]
        self.labels = [int(r["label"]) for r in rows]
        self.n_classes = max(self.labels) + 1 if self.labels else 0

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        for _ in range(5):  # an unreadable file must not kill a long training run
            try:
                img = load_rgb(self.root / self.paths[i], self.size)
                break
            except Exception:
                i = random.randrange(len(self.paths))
        else:
            raise RuntimeError("Too many unreadable images in a row")
        img = random_flip(img)
        if self.augment:
            img = photometric(img)
        return to_tensor(img), self.labels[i]
