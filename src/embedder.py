from pathlib import Path

import numpy as np

from .config import ROOT


class OnnxEmbedder:
    """Your own recognizer exported to ONNX.

    Contract: input  = aligned face, RGB, (x - 127.5) / 127.5, NCHW, e.g. 1x3x112x112
              output = embedding vector (any dimension D); it is L2-normalized here.
    """

    def __init__(self, path, providers=("CPUExecutionProvider",)):
        import onnxruntime as ort

        path = Path(path)
        path = path if path.is_absolute() else ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"Recognizer model not found: {path}")
        self.session = ort.InferenceSession(str(path), providers=list(providers))
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.size = int(inp.shape[2]) if isinstance(inp.shape[2], int) else 112

    def __call__(self, aligned_bgr):
        rgb = aligned_bgr[:, :, ::-1]
        x = ((rgb.astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[None]
        e = np.asarray(self.session.run([], {self.input_name: x})[0], np.float32).reshape(-1)
        return e / (np.linalg.norm(e) + 1e-12)

