"""Passive anti-spoofing (ONNX, outputs P(live)).

Disabled (always live) unless `liveness.enabled: true` in config.yaml. When enabled and the model file is
missing it raises instead of silently accepting everyone.
"""
import json
from pathlib import Path

import cv2
import numpy as np

from .config import ROOT


def crop_face(frame_bgr, bbox, scale=2.0, size=128):
    """Square crop centred on the face box, `scale` x its longest side (keeps context such as screen
    bezels and paper edges), zero-padded outside the frame, resized to size x size.
    MUST stay identical between data collection, training (Colab) and inference."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * scale / 2
    l, t = int(round(cx - half)), int(round(cy - half))
    r, b = int(round(cx + half)), int(round(cy + half))
    crop = frame_bgr[max(t, 0):min(b, h), max(l, 0):min(r, w)]
    if crop.size == 0:
        return np.zeros((size, size, 3), np.uint8)
    crop = cv2.copyMakeBorder(
        crop, max(0, -t), max(0, b - h), max(0, -l), max(0, r - w), cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


class Liveness:
    """Passive anti-spoofing (ONNX, outputs P(live)).

    Disabled (always live) unless `liveness.enabled: true` in config.yaml. When enabled and the model file is
    missing it raises instead of silently accepting everyone."""

    def __init__(self, cfg=None):
        lc = (cfg or {}).get("liveness") or {}
        self.enabled = bool(lc.get("enabled", False))
        self.session = None
        if not self.enabled:
            return
        model_path = ROOT / lc.get("model", "models/liveness/antispoof.onnx")
        if not model_path.exists():
            raise FileNotFoundError(
                f"Liveness enabled but model not found: {model_path}. Train it (notebooks/liveness_training.ipynb) "
                "or set liveness.enabled: false."
            )
        meta_path = model_path.with_suffix(".json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self.size = int(meta.get("size", lc.get("size", 128)))
        self.scale = float(meta.get("scale", lc.get("crop_scale", 2.0)))
        thr = lc.get("threshold")
        self.threshold = float(thr if thr is not None else meta.get("threshold", 0.5))

        import onnxruntime as ort

        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def reset(self):
        """No-op for stateless liveness; override in stateful subclasses."""
        pass

    def score(self, frame_bgr, face):
        """P(live) in [0, 1]."""
        crop = crop_face(frame_bgr, face.bbox, self.scale, self.size)
        x = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
        out = self.session.run(None, {self.input_name: x})[0]
        return float(np.asarray(out).reshape(-1)[0])

    def evaluate(self, frame_bgr, face):
        """Return (is_live, score or None)."""
        if not self.enabled:
            return True, None
        s = self.score(frame_bgr, face)
        return s >= self.threshold, s

    def check(self, frame_bgr, face):
        """Compatibility alias — returns bool."""
        return self.evaluate(frame_bgr, face)[0]

