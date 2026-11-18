"""Model / loss / init tests. Skipped automatically when torch is not installed."""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch.nn.functional as F  # noqa: E402

from fr.data import batch_to_tensor, to_tensor  # noqa: E402
from fr.model import ArcFace, MobileFaceNet, build_backbone, count_params  # noqa: E402


def test_backbone_output_shape_and_param_count():
    net = build_backbone(512).eval()
    with torch.no_grad():
        out = net(torch.randn(2, 3, 112, 112))
    assert out.shape == (2, 512) and 0.5e6 < count_params(net) < 10e6


def test_init_weights_batchnorm_and_linear():
    net = build_backbone(128)
    bn = net.fc[2]
    assert torch.all(bn.weight == 1) and torch.all(bn.bias == 0)
    assert net.features[0][0].weight.std() > 0


def test_arcface_margin_makes_true_class_logit_smaller():
    torch.manual_seed(0)
    head = ArcFace(16, 5, 64.0, 0.0)
    emb, y = torch.randn(8, 16), torch.randint(0, 5, (8,))
    plain = head(emb, y)
    head.set_margin(0.5)
    margin = head(emb, y)
    idx = torch.arange(8)
    assert torch.all(margin[idx, y] <= plain[idx, y] + 1e-5)
    mask = F.one_hot(y, 5).bool()
    assert torch.allclose(margin[~mask], plain[~mask])


def test_normalization_range():
    import numpy as np

    img = np.zeros((112, 112, 3), np.uint8)
    img[0, 0] = 255
    t = to_tensor(img)
    assert t.shape == (3, 112, 112) and t.min().item() == -1.0 and t.max().item() == 1.0
    assert batch_to_tensor(np.stack([img, img])).shape == (2, 3, 112, 112)


def test_overfits_a_tiny_batch():
    torch.manual_seed(0)
    net, head = MobileFaceNet(64), ArcFace(64, 4, 32.0, 0.1)
    x, y = torch.randn(16, 3, 112, 112), torch.arange(4).repeat_interleave(4)
    opt = torch.optim.SGD(list(net.parameters()) + list(head.parameters()), lr=0.05, momentum=0.9)
    losses = []
    for _ in range(30):
        loss = F.cross_entropy(head(net(x), y), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.9 * losses[0]
