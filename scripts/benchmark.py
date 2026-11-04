"""Latency / FPS of detect+align+embed. Usage: python scripts/benchmark.py [--image path.jpg] [--n 50]
Without --image, one webcam frame is used."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from src.camera import open_camera
from src.config import load_config
from src.face_engine import FaceEngine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    engine = FaceEngine(cfg)

    if args.image:
        frame = cv2.imread(args.image)
    else:
        cap = open_camera(cfg)
        for _ in range(10):  # let exposure settle
            ok, frame = cap.read()
        cap.release()
    if frame is None:
        sys.exit("No image.")

    for _ in range(5):
        engine.analyze(frame)
    t0 = time.perf_counter()
    for _ in range(args.n):
        engine.analyze(frame)
    dt = (time.perf_counter() - t0) / args.n
    print(f"model={cfg['model']['name']} gpu={cfg['model']['use_gpu']}: {dt * 1000:.1f} ms/frame -> {1 / dt:.1f} FPS")


if __name__ == "__main__":
    main()
