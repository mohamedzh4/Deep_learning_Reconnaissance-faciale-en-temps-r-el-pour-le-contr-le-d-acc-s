"""STEP 2 - inspect, clean, index and check the normalization statistics.

  python step2_clean_and_index.py                         # whole dataset (a few minutes)
  python step2_clean_and_index.py --max-ids 1000 --max-images 30   # small subset for quick experiments

What it does: decodes every image, drops unreadable / corrupt / blank / wrong-size files, removes exact duplicates and
images that appear under two identities, drops identities with too few images, and writes
data/index/{train_manifest.csv, classes.json, removed.csv, stats.json, images_per_identity.png}.
"""
import argparse
import csv
import json
import os
from pathlib import Path

from fr.cleaning import build_index, compute_mean_std
from fr.paths import INDEX_DIR, load_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-images", type=int, default=10, help="drop identities with fewer images")
    ap.add_argument("--max-ids", type=int, default=None, help="keep a random subset of identities")
    ap.add_argument("--max-images", type=int, default=None, help="cap images per identity")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--fast", action="store_true", help="skip full decoding (only sizes and hashes)")
    ap.add_argument("--resize", action="store_true", help="keep wrong-size images (they are resized when loaded)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(load_paths()["train_root"])
    print("training images:", root)
    stats, counts = build_index(root, INDEX_DIR, args.min_images, args.max_ids, args.max_images,
                                workers=args.workers, full=not args.fast, resize=args.resize,
                                seed=args.seed, show_progress=True)

    print(f"\nscanned {stats['images_scanned']} images")
    print("removed:", stats["removed"] or "nothing")
    print(f"kept {stats['n_images']} images of {stats['n_identities']} identities")
    print("images per identity:", stats["images_per_identity"])

    with open(INDEX_DIR / "train_manifest.csv", newline="", encoding="utf-8") as f:
        rel_paths = [r["path"] for r in csv.DictReader(f)]
    if rel_paths:
        mean, std = compute_mean_std(root, rel_paths)
        stats["pixel_mean_std_0_1"] = {"mean": mean, "std": std}
        (INDEX_DIR / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print("\nmeasured pixel mean (RGB, 0-1):", [round(m, 3) for m in mean], "std:", [round(s, 3) for s in std])
        print("Normalization used everywhere: (x - 127.5) / 127.5  ->  values in [-1, 1].")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 4))
        plt.hist(counts, bins=50)
        plt.xlabel("images per identity")
        plt.ylabel("number of identities")
        plt.title("Class balance")
        plt.savefig(INDEX_DIR / "images_per_identity.png", dpi=120, bbox_inches="tight")
        print("histogram saved to", INDEX_DIR / "images_per_identity.png")
    except ImportError:
        pass
    print("Next: python show_samples.py   then   python step3_train.py --selftest")


if __name__ == "__main__":
    main()
