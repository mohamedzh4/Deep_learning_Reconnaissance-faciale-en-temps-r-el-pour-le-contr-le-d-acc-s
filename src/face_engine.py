"""Unified FaceEngine — detector (InsightFace buffalo) + optional custom ONNX recognizer.

model_id tracks which recognizer built the embeddings so the gallery can refuse
to mix embeddings from different models.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .config import ROOT


@dataclass
class Face:
    bbox: np.ndarray          # x1, y1, x2, y2
    kps: np.ndarray           # 5 landmarks
    det_score: float
    embedding: Optional[np.ndarray]   # 512-d (or D-d), L2-normalised; None in detection-only mode


def largest_face(faces):
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


class FaceEngine:
    """Detection + 5-point alignment + ArcFace embedding (InsightFace, pretrained).

    If cfg["model"]["recognizer"] points to an existing .onnx file the buffalo detector is kept
    and YOUR model produces the embeddings (see src/embedder.py for the input contract).
    model_id identifies the recogniser so galleries built with a different model are rejected.
    """

    def __init__(self, cfg, detection_only: bool = False):
        from insightface.app import FaceAnalysis  # lazy import

        m = cfg["model"]
        gpu = bool(m.get("use_gpu"))
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu else ["CPUExecutionProvider"]
        )
        rec_path_str = m.get("recognizer", "") or ""

        self.app = FaceAnalysis(
            name=m["name"],
            root=str(ROOT),
            providers=providers,
            # skip the buffalo recogniser when we have a custom one (saves memory/time)
            allowed_modules=(
                ["detection"]
                if (detection_only or rec_path_str)
                else ["detection", "recognition"]
            ),
        )
        self.app.prepare(ctx_id=0 if gpu else -1, det_size=tuple(m["det_size"]))

        self.embedder = None
        if rec_path_str and not detection_only:
            from .embedder import OnnxEmbedder

            self.embedder = OnnxEmbedder(rec_path_str, providers)

        self.model_id = (
            f"custom:{Path(rec_path_str).name}" if rec_path_str else m["name"]
        )

    def analyze(self, frame_bgr):
        faces = []
        for f in self.app.get(frame_bgr):
            if self.embedder is not None:
                from .align import align_face
                emb = self.embedder(align_face(frame_bgr, f.kps, self.embedder.size))
            else:
                emb = getattr(f, "normed_embedding", None)
            faces.append(Face(f.bbox, f.kps, float(f.det_score), emb))
        return faces

    def embed(self, frame_bgr):
        faces = self.analyze(frame_bgr)
        return largest_face(faces).embedding if faces else None

