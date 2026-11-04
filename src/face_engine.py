from dataclasses import dataclass

import numpy as np

from .config import ROOT


@dataclass
class Face:
    bbox: np.ndarray        # x1, y1, x2, y2
    kps: np.ndarray         # 5 landmarks
    det_score: float
    embedding: np.ndarray   # 512-d, L2-normalized


def largest_face(faces):
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


class FaceEngine:
    """Detection + 5-point alignment + ArcFace embedding (InsightFace, pretrained)."""

    def __init__(self, cfg):
        from insightface.app import FaceAnalysis  # lazy: keeps tests/imports light

        m = cfg["model"]
        gpu = bool(m.get("use_gpu"))
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu else ["CPUExecutionProvider"]
        self.app = FaceAnalysis(
            name=m["name"],
            root=str(ROOT),  # weights are cached in ./models/<name>
            providers=providers,
            allowed_modules=["detection", "recognition"],
        )
        self.app.prepare(ctx_id=0 if gpu else -1, det_size=tuple(m["det_size"]))

    def analyze(self, frame_bgr):
        return [Face(f.bbox, f.kps, float(f.det_score), f.normed_embedding) for f in self.app.get(frame_bgr)]

    def embed(self, frame_bgr):
        faces = self.analyze(frame_bgr)
        return largest_face(faces).embedding if faces else None
