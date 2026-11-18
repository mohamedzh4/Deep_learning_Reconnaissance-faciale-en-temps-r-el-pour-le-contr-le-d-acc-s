"""Dataset inspection, cleaning and indexing. Needs only numpy + Pillow (no torch)."""
import csv
import hashlib
import io
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png"}


def _has_images(d, limit=50):
    for n, p in enumerate(d.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            return True
        if n >= limit:
            break
    return False


def find_class_root(path, min_ids=2, max_depth=4):
    """Find the folder that directly contains one sub-folder per identity (each holding images)."""
    queue = [(Path(path), 0)]
    while queue:
        d, depth = queue.pop(0)
        subs = sorted(s for s in d.iterdir() if s.is_dir())
        if len(subs) >= min_ids and any(_has_images(s) for s in subs[:5]):
            return d
        if depth < max_depth:
            queue.extend((s, depth + 1) for s in subs[:20])
    raise FileNotFoundError(f"No identity folders (one sub-folder per person with images) found under {path}")


def find_val_root(path):
    """Folder containing the *_ann.txt pair files (image paths inside them are relative to this folder)."""
    ann = sorted(Path(path).rglob("*_ann.txt"))
    if not ann:
        raise FileNotFoundError(f"No *_ann.txt validation files found under {path}")
    return ann[0].parent


def inspect_file(args):
    """Returns (path, status, md5, width, height); status in ok / unreadable / corrupt / blank / bad_size."""
    path, expected, full = args
    try:
        data = Path(path).read_bytes()
    except OSError:
        return path, "unreadable", "", 0, 0
    digest = hashlib.md5(data).hexdigest()
    try:
        with Image.open(io.BytesIO(data)) as im:
            w, h = im.size
            if full:
                arr = np.asarray(im.convert("RGB"))
                if float(arr.std()) < 2.0:
                    return path, "blank", digest, w, h
    except Exception:
        return path, "corrupt", digest, 0, 0
    return path, ("ok" if (w, h) == (expected, expected) else "bad_size"), digest, w, h


def build_index(train_root, out_dir, min_images=10, max_ids=None, max_images=None, expected=112,
                workers=None, full=True, resize=False, seed=0, show_progress=False):
    """Inspect every image, drop bad ones, de-duplicate, filter small identities, write the manifest.

    Writes out_dir/train_manifest.csv (path,label,identity), classes.json, removed.csv, stats.json.
    Returns (stats, images_per_identity_list).
    """
    train_root, out_dir = Path(train_root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    owner, jobs = {}, []
    for name in sorted(d.name for d in train_root.iterdir() if d.is_dir()):
        for p in sorted(f for f in (train_root / name).rglob("*") if f.is_file() and f.suffix.lower() in IMG_EXT):
            jobs.append((str(p), expected, full))
            owner[str(p)] = name

    if workers == 1 or len(jobs) < 500:
        results = map(inspect_file, jobs)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        results = pool.map(inspect_file, jobs, chunksize=256)
    if show_progress:
        from tqdm import tqdm

        results = tqdm(results, total=len(jobs), desc="inspecting images")
    results = list(results)
    if pool is not None:
        pool.shutdown()

    removed, keep = [], []  # keep: (identity, rel_path, md5)
    for path, status, digest, _, _ in results:
        rel = Path(path).relative_to(train_root).as_posix()
        if status == "bad_size" and resize:
            status = "ok"
        if status == "ok":
            keep.append((owner[path], rel, digest))
        else:
            removed.append((rel, status))

    by_hash = defaultdict(list)
    for name, rel, digest in keep:
        by_hash[digest].append((name, rel))
    dropped = set()
    for items in by_hash.values():
        if len(items) < 2:
            continue
        if len({n for n, _ in items}) > 1:  # same image under several identities: ambiguous label
            for _, rel in items:
                dropped.add(rel)
                removed.append((rel, "label_conflict"))
        else:                                # exact duplicate inside one identity: keep the first
            for _, rel in items[1:]:
                dropped.add(rel)
                removed.append((rel, "duplicate"))
    keep = [k for k in keep if k[1] not in dropped]

    per_id = defaultdict(list)
    for name, rel, _ in keep:
        per_id[name].append(rel)
    for name in [n for n, files in per_id.items() if len(files) < min_images]:
        removed.extend((rel, "small_identity") for rel in per_id.pop(name))

    rng = random.Random(seed)
    names = sorted(per_id)
    if max_ids and len(names) > max_ids:
        names = sorted(rng.sample(names, max_ids))
    rows = []
    for label, name in enumerate(names):
        files = sorted(per_id[name])
        if max_images and len(files) > max_images:
            rng.shuffle(files)
            files = sorted(files[:max_images])
        rows.extend((rel, label, name) for rel in files)

    with open(out_dir / "train_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "identity"])
        w.writerows(rows)
    with open(out_dir / "removed.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "reason"])
        w.writerows(sorted(removed))
    (out_dir / "classes.json").write_text(json.dumps(names), encoding="utf-8")

    counts = list(Counter(r[1] for r in rows).values())
    stats = {
        "train_root": str(train_root),
        "images_scanned": len(jobs),
        "removed": dict(Counter(reason for _, reason in removed)),
        "n_identities": len(names),
        "n_images": len(rows),
        "images_per_identity": {
            "min": int(min(counts)) if counts else 0,
            "median": float(np.median(counts)) if counts else 0,
            "mean": float(np.mean(counts)) if counts else 0,
            "max": int(max(counts)) if counts else 0,
        },
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats, counts


def compute_mean_std(root, rel_paths, n=2000, seed=0):
    """Per-channel mean/std (0..1 scale) on a random sample of images."""
    rng = random.Random(seed)
    sample = rng.sample(list(rel_paths), min(n, len(rel_paths)))
    s, s2, cnt = np.zeros(3), np.zeros(3), 0
    for rel in sample:
        a = np.asarray(Image.open(Path(root) / rel).convert("RGB"), np.float64) / 255.0
        s += a.sum((0, 1))
        s2 += (a ** 2).sum((0, 1))
        cnt += a.shape[0] * a.shape[1]
    mean = s / cnt
    std = np.sqrt(np.maximum(s2 / cnt - mean ** 2, 0))
    return mean.tolist(), std.tolist()
