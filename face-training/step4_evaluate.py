"""STEP 4 - verification accuracy on LFW, CALFW, CPLFW and AgeDB-30.

  python step4_evaluate.py --ckpt runs/exp1/best.pt
  python step4_evaluate.py --onnx exports/my_model.onnx     # evaluates the exported file (what you will deploy)
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse

import torch

from fr.evaluate import evaluate_ann, onnx_feature_fn, torch_feature_fn
from fr.model import build_backbone
from fr.utils import get_device, load_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--onnx")
    ap.add_argument("--sets", default="lfw,calfw,cplfw,agedb_30")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    if not (args.ckpt or args.onnx):
        raise SystemExit("give --ckpt or --onnx")
    val_root = load_paths().get("val_root")
    if not val_root:
        raise SystemExit("No validation sets: run  python step1_download.py --only val")

    if args.onnx:
        import onnxruntime as ort

        fn = onnx_feature_fn(ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"]))
    else:
        device = get_device(args.device)
        ck = torch.load(args.ckpt, map_location="cpu")
        net = build_backbone(ck["args"]["emb"])
        net.load_state_dict(ck["net"])
        fn = torch_feature_fn(net.to(device), device)

    print(f"{'set':<10} {'accuracy':>9} {'std':>7} {'threshold (cosine)':>20}")
    for name in args.sets.split(","):
        acc, std, thr = evaluate_ann(fn, val_root, f"{name}_ann.txt")
        print(f"{name:<10} {acc * 100:8.2f}% {std * 100:6.2f}% {thr:20.3f}")


if __name__ == "__main__":
    main()
