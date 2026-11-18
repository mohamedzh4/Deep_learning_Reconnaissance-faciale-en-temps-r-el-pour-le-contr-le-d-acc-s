"""Verification accuracy on LFW / CALFW / CPLFW / AgeDB-30 (same protocol as the reference training repo)."""
import numpy as np
import torch

from .data import batch_to_tensor, load_rgb
from .pairs import kfold_accuracy, read_pairs


def torch_feature_fn(net, device, batch=128):
    """Embeddings = concat(f(x), f(flip(x))) for each image path."""

    @torch.no_grad()
    def fn(paths):
        net.eval()
        feats = []
        for i in range(0, len(paths), batch):
            x = batch_to_tensor(np.stack([load_rgb(p) for p in paths[i:i + batch]])).to(device)
            feats.append(torch.cat([net(x), net(torch.flip(x, dims=[3]))], dim=1).float().cpu())
        return torch.cat(feats).numpy()

    return fn


def onnx_feature_fn(session, batch=64):
    """Same as torch_feature_fn but runs an exported ONNX model (checks the deployed artifact)."""
    name = session.get_inputs()[0].name

    def fn(paths):
        feats = []
        for i in range(0, len(paths), batch):
            x = batch_to_tensor(np.stack([load_rgb(p) for p in paths[i:i + batch]])).numpy()
            a = session.run([], {name: x})[0]
            b = session.run([], {name: np.ascontiguousarray(x[:, :, :, ::-1])})[0]
            feats.append(np.concatenate([a, b], axis=1))
        return np.concatenate(feats)

    return fn


def evaluate_ann(feature_fn, val_root, ann_name):
    """Returns (mean accuracy, std, mean threshold)."""
    pairs = read_pairs(f"{val_root}/{ann_name}", val_root)
    uniq = sorted({p for a, b, _ in pairs for p in (a, b)})
    index = {p: i for i, p in enumerate(uniq)}
    feats = np.asarray(feature_fn(uniq), np.float64)
    feats /= np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12
    f1 = feats[[index[a] for a, _, _ in pairs]]
    f2 = feats[[index[b] for _, b, _ in pairs]]
    sims = (f1 * f2).sum(1)
    return kfold_accuracy(sims, np.array([s for _, _, s in pairs]))
