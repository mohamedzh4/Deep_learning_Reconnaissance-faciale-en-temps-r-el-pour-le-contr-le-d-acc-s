"""Look at the data: image -> array -> normalized tensor -> (untrained) embedding vector. Saves a 4x4 sample grid."""
import random

import torch
from PIL import Image, ImageDraw

from fr.data import ManifestDataset, load_rgb, to_tensor
from fr.model import build_backbone
from fr.paths import INDEX_DIR, load_paths


def main():
    ds = ManifestDataset(load_paths()["train_root"], INDEX_DIR / "train_manifest.csv")
    print(f"{len(ds)} images, {ds.n_classes} identities")

    img = load_rgb(ds.root / ds.paths[0])
    print("1. as an array :", img.shape, img.dtype, "min", img.min(), "max", img.max())
    t = to_tensor(img)
    print("2. as a tensor :", tuple(t.shape), t.dtype, "min %.2f max %.2f mean %.2f" % (t.min(), t.max(), t.mean()))
    net = build_backbone(512).eval()
    with torch.no_grad():
        v = net(t[None])
    print("3. as a vector :", tuple(v.shape), "(random until the network is trained)")

    idx = random.Random(0).sample(range(len(ds)), 16)
    grid = Image.new("RGB", (4 * 112, 4 * 112))
    for k, i in enumerate(idx):
        tile = Image.fromarray(load_rgb(ds.root / ds.paths[i]))
        ImageDraw.Draw(tile).text((3, 3), str(ds.labels[i]), fill=(255, 255, 0))
        grid.paste(tile, ((k % 4) * 112, (k // 4) * 112))
    out = INDEX_DIR / "samples.png"
    grid.save(out)
    print("sample grid saved to", out)


if __name__ == "__main__":
    main()
