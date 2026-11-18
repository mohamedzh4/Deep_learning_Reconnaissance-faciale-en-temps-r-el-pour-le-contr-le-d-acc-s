"""MobileFaceNet backbone + ArcFace head + weight initialization."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBN(nn.Sequential):
    """Convolution + BatchNorm (+ PReLU)."""

    def __init__(self, cin, cout, k=1, stride=1, groups=1, act=True):
        layers = [nn.Conv2d(cin, cout, k, stride, k // 2, groups=groups, bias=False), nn.BatchNorm2d(cout)]
        if act:
            layers.append(nn.PReLU(cout))
        super().__init__(*layers)


class Bottleneck(nn.Module):
    """1x1 expand -> 3x3 depthwise -> 1x1 linear project, with a skip connection when shapes match."""

    def __init__(self, cin, cout, stride, t):
        super().__init__()
        mid = cin * t
        self.use_res = stride == 1 and cin == cout
        self.block = nn.Sequential(
            ConvBN(cin, mid, 1),
            ConvBN(mid, mid, 3, stride, groups=mid),
            ConvBN(mid, cout, 1, act=False),
        )

    def forward(self, x):
        y = self.block(x)
        return x + y if self.use_res else y


class MobileFaceNet(nn.Module):
    """112x112x3 image -> emb_dim-d vector (not normalized; the ArcFace head / cosine comparison normalize it)."""

    CFG = [(2, 64, 5, 2), (4, 128, 1, 2), (2, 128, 6, 1), (4, 128, 1, 2), (2, 128, 2, 1)]  # t, channels, repeats, stride

    def __init__(self, emb_dim=512):
        super().__init__()
        layers = [ConvBN(3, 64, 3, 2), ConvBN(64, 64, 3, 1, groups=64)]
        cin = 64
        for t, c, n, s in self.CFG:
            for i in range(n):
                layers.append(Bottleneck(cin, c, s if i == 0 else 1, t))
                cin = c
        layers.append(ConvBN(cin, 512, 1))
        self.features = nn.Sequential(*layers)  # 112x112 -> 7x7x512
        self.gdc = nn.Sequential(nn.Conv2d(512, 512, 7, groups=512, bias=False), nn.BatchNorm2d(512))  # global depthwise conv
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(512, emb_dim, bias=False), nn.BatchNorm1d(emb_dim))

    def forward(self, x):
        return self.fc(self.gdc(self.features(x)))


class ArcFace(nn.Module):
    """Training-only head. logit_j = s*cos(theta_j); for the true class s*cos(theta_y + m)."""

    def __init__(self, emb_dim, n_classes, scale=64.0, margin=0.5):
        super().__init__()
        self.W = nn.Parameter(torch.empty(n_classes, emb_dim))
        nn.init.normal_(self.W, std=0.01)
        self.s = scale
        self.set_margin(margin)

    def set_margin(self, m):
        self.m = m
        self.cos_m, self.sin_m = math.cos(m), math.sin(m)
        self.th = math.cos(math.pi - m)  # beyond this angle cos(theta + m) stops being monotonic
        self.mm = math.sin(math.pi - m) * m

    def cos_logits(self, emb):
        """s * cos(theta_j), no margin: used to monitor training accuracy."""
        return F.linear(F.normalize(emb), F.normalize(self.W)).clamp(-1 + 1e-7, 1 - 1e-7) * self.s

    def forward(self, emb, labels):
        cos = self.cos_logits(emb) / self.s
        sin = torch.sqrt(1.0 - cos * cos)
        phi = cos * self.cos_m - sin * self.sin_m  # cos(theta + m)
        phi = torch.where(cos > self.th, phi, cos - self.mm)
        onehot = F.one_hot(labels, cos.size(1)).bool()
        return torch.where(onehot, phi, cos) * self.s


def init_weights(m):
    """Variable initialization: He init for convolutions (PReLU slope 0.25), BN gamma=1 / beta=0, Xavier for Linear."""
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu", a=0.25)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)


def build_backbone(emb_dim=512):
    net = MobileFaceNet(emb_dim)
    net.apply(init_weights)
    return net


def count_params(module):
    return sum(p.numel() for p in module.parameters())
