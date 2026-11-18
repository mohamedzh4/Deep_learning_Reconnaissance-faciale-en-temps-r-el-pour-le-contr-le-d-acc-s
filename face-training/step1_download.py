"""STEP 1 - get the data.

Default: download the aligned (112x112) CASIA-WebFace training set and the LFW / CALFW / CPLFW / AgeDB-30
validation sets from Kaggle (unofficial mirrors prepared by the authors of the UniFace project).

  python step1_download.py                    # train + validation
  python step1_download.py --only val         # only the validation sets
  python step1_download.py --only train --train-root /path/to/webface   # data you already downloaded

Kaggle needs an account + API token: kaggle.com -> Settings -> API -> Create New Token, then put kaggle.json in
~/.kaggle/ (or set KAGGLE_USERNAME / KAGGLE_KEY). See the kagglehub documentation if authentication fails.
The official CASIA-WebFace release requires an application (see its website); point --train-root at it once approved.
"""
import argparse
import json
import os

from fr.cleaning import find_class_root, find_val_root
from fr.paths import PATHS_JSON, ROOT

TRAIN_SLUG = "yakhyokhuja/webface-112x112"
VAL_SLUG = "yakhyokhuja/agedb-30-calfw-cplfw-lfw-aligned-112x112"


def download(slug, target):
    """Download into the project (data/train or data/val). Older kagglehub versions without `output_dir`
    fall back to a cache folder inside the project (data/kaggle) instead of ~/.cache/kagglehub."""
    os.environ.setdefault("KAGGLEHUB_CACHE", str(ROOT / "data" / "kaggle"))
    try:
        import kagglehub
    except ImportError:
        raise SystemExit("kagglehub is not installed: pip install kagglehub")
    print(f"Downloading {slug} into {target} (can take a while)...")
    try:
        try:
            return kagglehub.dataset_download(slug, output_dir=str(target))
        except TypeError:
            return kagglehub.dataset_download(slug)
    except Exception as e:  # network, authentication, rate limit...
        raise SystemExit(
            f"Kaggle download failed: {e}\n"
            "Fix your Kaggle credentials, or download the dataset manually from kaggle.com/datasets/" + slug +
            " and run again with --train-root / --val-root."
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["train", "val", "both"], default="both")
    ap.add_argument("--train-root", help="folder with the training images (skips the download)")
    ap.add_argument("--val-root", help="folder with the validation sets (skips the download)")
    args = ap.parse_args()

    paths = json.loads(PATHS_JSON.read_text(encoding="utf-8")) if PATHS_JSON.exists() else {}
    if args.only in ("train", "both"):
        paths["train_root"] = str(find_class_root(args.train_root or download(TRAIN_SLUG, ROOT / "data" / "train")))
    if args.only in ("val", "both"):
        paths["val_root"] = str(find_val_root(args.val_root or download(VAL_SLUG, ROOT / "data" / "val")))

    PATHS_JSON.parent.mkdir(parents=True, exist_ok=True)
    PATHS_JSON.write_text(json.dumps(paths, indent=2), encoding="utf-8")
    print("\nSaved", PATHS_JSON)
    for k, v in paths.items():
        print(f"  {k}: {v}")
    print("Next: python step2_clean_and_index.py")


if __name__ == "__main__":
    main()