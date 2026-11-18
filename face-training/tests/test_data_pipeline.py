"""Tests that need only numpy + Pillow (no torch)."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fr.cleaning import build_index, compute_mean_std, find_class_root, find_val_root
from fr.pairs import kfold_accuracy, read_pairs


def save_noise(path, seed, size=112):
    arr = np.random.default_rng(seed).integers(0, 255, (size, size, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "train"
    for ident, n in (("A", 12), ("B", 12), ("C", 4)):
        (root / ident).mkdir(parents=True)
        for i in range(n):
            save_noise(root / ident / f"img{i:02d}.png", seed=hash((ident, i)) % 10_000)
    (root / "A" / "corrupt.jpg").write_bytes(b"this is not an image")
    Image.fromarray(np.full((112, 112, 3), 128, np.uint8)).save(root / "A" / "blank.png")
    shutil.copy(root / "A" / "img00.png", root / "A" / "dup00.png")        # duplicate inside one identity
    shutil.copy(root / "B" / "img01.png", root / "A" / "stolen.png")       # same image under two identities
    save_noise(root / "B" / "small.png", seed=1, size=64)                   # wrong size
    return root


def test_build_index_cleans_and_labels(dataset, tmp_path):
    stats, counts = build_index(dataset, tmp_path / "idx", min_images=10, workers=1)
    assert stats["removed"] == {"corrupt": 1, "blank": 1, "duplicate": 1, "label_conflict": 2,
                                "bad_size": 1, "small_identity": 4}
    assert stats["n_identities"] == 2 and stats["n_images"] == 23 and sorted(counts) == [11, 12]
    assert json.loads((tmp_path / "idx" / "classes.json").read_text()) == ["A", "B"]
    rows = (tmp_path / "idx" / "train_manifest.csv").read_text().splitlines()
    assert rows[0] == "path,label,identity" and len(rows) == 24
    assert all(r.split(",")[1] in ("0", "1") for r in rows[1:])


def test_subset_options(dataset, tmp_path):
    stats, _ = build_index(dataset, tmp_path / "idx", min_images=10, max_ids=1, max_images=5, workers=1)
    assert stats["n_identities"] == 1 and stats["n_images"] == 5


def test_resize_keeps_wrong_size_images(dataset, tmp_path):
    stats, _ = build_index(dataset, tmp_path / "idx", min_images=10, workers=1, resize=True)
    assert "bad_size" not in stats["removed"] and stats["n_images"] == 24


def test_mean_std_of_noise_is_about_half(dataset):
    mean, std = compute_mean_std(dataset, ["A/img00.png", "A/img01.png", "B/img00.png"], n=3)
    assert all(0.45 < m < 0.55 for m in mean) and all(0.25 < s < 0.33 for s in std)


def test_find_roots(tmp_path):
    nested = tmp_path / "kaggle" / "versions" / "1" / "webface_112x112"
    for ident in ("0001", "0002"):
        (nested / ident).mkdir(parents=True)
        save_noise(nested / ident / "a.png", 1)
    assert find_class_root(tmp_path / "kaggle") == nested
    val = tmp_path / "val_dl" / "val"
    val.mkdir(parents=True)
    (val / "lfw_ann.txt").write_text("6000\n")
    assert find_val_root(tmp_path / "val_dl") == val
    with pytest.raises(FileNotFoundError):
        find_val_root(tmp_path / "kaggle")


def test_read_pairs_skips_header_and_resolves_paths(tmp_path):
    ann = tmp_path / "lfw_ann.txt"
    ann.write_text("6000\n1 a/1.jpg a/2.jpg\n0 a/1.jpg b/1.jpg\nbroken line\n")
    pairs = read_pairs(ann, tmp_path)
    assert pairs == [(str(tmp_path / "a/1.jpg"), str(tmp_path / "a/2.jpg"), True),
                     (str(tmp_path / "a/1.jpg"), str(tmp_path / "b/1.jpg"), False)]


def test_kfold_accuracy_separable_and_random():
    rng = np.random.default_rng(0)
    issame = np.tile([True, False], 300)
    sims = np.where(issame, 0.7, 0.1) + rng.normal(0, 0.05, 600)
    acc, std, thr = kfold_accuracy(sims, issame)
    assert acc > 0.99 and 0.2 < thr < 0.6
    acc_random, _, _ = kfold_accuracy(rng.normal(0, 0.3, 600), issame)
    assert 0.4 < acc_random < 0.62
