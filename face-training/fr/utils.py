import os
import random
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # let unsupported ops fall back to CPU on Apple GPUs

import numpy as np
import torch

from .paths import INDEX_DIR, PATHS_JSON, ROOT, load_paths  # noqa: F401 (re-exported)


def get_device(pref="auto"):
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AverageMeter:
    def __init__(self):
        self.sum, self.n = 0.0, 0

    def update(self, value, n=1):
        self.sum += float(value) * n
        self.n += n

    @property
    def avg(self):
        return self.sum / max(self.n, 1)
