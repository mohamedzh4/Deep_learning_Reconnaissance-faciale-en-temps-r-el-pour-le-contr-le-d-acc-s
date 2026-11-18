"""Capture face crops for the liveness dataset (same crop as inference).

  python scripts/collect_liveness_data.py --label live  --group alice_s1          # real face, one session
  python scripts/collect_liveness_data.py --label spoof --group phone_photo_s1    # photo on a phone screen
  python scripts/collect_liveness_data.py --label spoof --group phone_replay_s1   # video replay on a phone
  python scripts/collect_liveness_data.py --label spoof --group print_a4_s1       # printed photo

Output: data/liveness/own/<label>/<group>/NNNN.jpg. Group = one session / one attack type: the training
notebook splits train/val/test BY GROUP, so use a new group name for every new session (lighting, day, place).
Only capture yourself or consenting people. q = stop."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from src.camera import open_camera
from src.config import ROOT, load_config
from src.face_engine import FaceEngine, largest_face
from src.liveness import crop_face

HINTS = {
    "live": "Real face: vary distance (0.4-1.5 m), head pose, expression, glasses on/off.",
    "spoof": "Attack: show the photo / video / print to the camera. Vary distance, angle and light.",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", choices=["live", "spoof"], required=True)
    ap.add_argument("--group", required=True, help="session / attack name, e.g. alice_s1, phone_photo_s1")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--interval", type=float, default=0.15)
    ap.add_argument("--source", default="own")
    ap.add_argument("--config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    lc = cfg.get("liveness") or {}
    size, scale = int(lc.get("size", 128)), float(lc.get("crop_scale", 2.0))
    engine = FaceEngine(cfg, detection_only=True)
    cap = open_camera(cfg)
    out = ROOT / "data/liveness" / args.source / args.label / args.group
    out.mkdir(parents=True, exist_ok=True)
    idx, last_t = len(list(out.glob("*.jpg"))), 0.0
    target = idx + args.n

    while idx < target:
        ok, frame = cap.read()
        if not ok:
            break
        faces = engine.analyze(frame)
        show = frame.copy()
        if faces:
            f = largest_face(faces)
            x1, y1, x2, y2 = f.bbox.astype(int)
            good = f.det_score > 0.7
            cv2.rectangle(show, (x1, y1), (x2, y2), (0, 200, 0) if good else (0, 165, 255), 2)
            if good and time.time() - last_t >= args.interval:
                idx += 1
                last_t = time.time()
                cv2.imwrite(str(out / f"{idx:04d}.jpg"), crop_face(frame, f.bbox, scale, size), [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.putText(show, f"{args.label}/{args.group}: {idx - (target - args.n)}/{args.n}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(show, HINTS[args.label], (10, show.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.imshow("Liveness data (q to stop)", show)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    cap.release()
    cv2.destroyAllWindows()
    print(f"{len(list(out.glob('*.jpg')))} images in {out}")


if __name__ == "__main__":
    main()
