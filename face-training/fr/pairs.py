"""Verification pairs (LFW-style) and the 10-fold accuracy protocol. Needs only numpy."""
from pathlib import Path

import numpy as np


def read_pairs(ann_path, root=None):
    """Lines look like `is_same path1 path2` (paths relative to the val root). Header/odd lines are skipped."""
    root = Path(root) if root is not None else Path(ann_path).parent
    pairs = []
    with open(ann_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 3 and parts[0] in ("0", "1"):
                pairs.append((str(root / parts[1]), str(root / parts[2]), parts[0] == "1"))
    if not pairs:
        raise ValueError(f"No valid pairs found in {ann_path}")
    return pairs


def kfold_accuracy(sims, issame, n_folds=10):
    """Same-person if cosine similarity > threshold. Best threshold chosen on 9 folds, tested on the 10th.
    Returns (mean accuracy, std, mean threshold)."""
    sims, issame = np.asarray(sims, np.float64), np.asarray(issame, bool)
    thresholds = np.arange(-1.0, 1.0, 0.005)
    folds = np.array_split(np.arange(len(sims)), n_folds)
    accs, ths = [], []
    for k in range(n_folds):
        test = folds[k]
        train = np.concatenate([folds[j] for j in range(n_folds) if j != k])
        acc_train = ((sims[train][:, None] > thresholds[None, :]) == issame[train][:, None]).mean(0)
        best = thresholds[int(np.argmax(acc_train))]
        accs.append(float(np.mean((sims[test] > best) == issame[test])))
        ths.append(float(best))
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(ths))
