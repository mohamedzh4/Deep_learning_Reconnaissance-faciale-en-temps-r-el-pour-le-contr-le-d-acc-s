"""STEP 5 - export the trained backbone to ONNX for the access-control project.

  python step5_export_onnx.py --ckpt runs/exp1/best.pt --out exports/my_model.onnx

Then copy the file to  <access-control>/models/recognition/my_model.onnx  and set in its config.yaml:
  model:
    recognizer: models/recognition/my_model.onnx
Re-enroll everyone and re-run scripts/calibrate_threshold.py --write (embeddings of different models are not comparable).
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
from pathlib import Path

import torch

from fr.evaluate import evaluate_ann, onnx_feature_fn
from fr.export import export_onnx, parity
from fr.model import build_backbone
from fr.paths import load_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="exports/my_model.onnx")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    net = build_backbone(ck["args"]["emb"])
    net.load_state_dict(ck["net"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    export_onnx(net, args.out)
    size = Path(args.out).stat().st_size / 1e6
    print(f"exported {args.out} ({size:.1f} MB)")
    try:
        print(f"parity torch vs onnxruntime: max abs diff {parity(net, args.out):.2e} (should be < 1e-3)")
        val_root = load_paths().get("val_root")
        if val_root:
            import onnxruntime as ort

            fn = onnx_feature_fn(ort.InferenceSession(args.out, providers=["CPUExecutionProvider"]))
            acc, std, thr = evaluate_ann(fn, val_root, "lfw_ann.txt")
            print(f"LFW accuracy of the ONNX file: {acc * 100:.2f}% +- {std * 100:.2f}% (cosine threshold {thr:.3f})")
    except ImportError:
        print("install onnxruntime to run the parity check: pip install onnxruntime")


if __name__ == "__main__":
    main()
