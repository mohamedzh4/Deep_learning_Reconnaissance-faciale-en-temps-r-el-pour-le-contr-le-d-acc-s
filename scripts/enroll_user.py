"""Enroll a person from the webcam. Usage: python scripts/enroll_user.py --name mohamed [--n 15] [--no-images]

Captures N images automatically (one face, good detection score), saves them to data/own_dataset/<name>/
(used later for threshold calibration) and builds the template from the first `enroll.n_templates` captures.
Only enroll people who gave their consent.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from src.camera import open_camera
from src.config import ROOT, load_config
from src.face_engine import FaceEngine
from src.gallery import Gallery


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--no-images", action="store_true", help="do not keep the captured images")
    ap.add_argument("--config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    engine = FaceEngine(cfg)
    cap = open_camera(cfg)
    out_dir = ROOT / cfg["paths"]["own_dataset"] / args.name
    if not args.no_images:
        out_dir.mkdir(parents=True, exist_ok=True)
    start_idx = len(list(out_dir.glob("*.jpg"))) if out_dir.exists() else 0

    embs, last_t = [], 0.0
    while len(embs) < args.n:
        ok, frame = cap.read()
        if not ok:
            break
        faces = engine.analyze(frame)
        show = frame.copy()
        if len(faces) == 1 and faces[0].det_score > 0.7:
            x1, y1, x2, y2 = faces[0].bbox.astype(int)
            cv2.rectangle(show, (x1, y1), (x2, y2), (0, 200, 0), 2)
            if time.time() - last_t > 0.7:
                embs.append(faces[0].embedding)
                last_t = time.time()
                if not args.no_images:
                    cv2.imwrite(str(out_dir / f"{start_idx + len(embs):03d}.jpg"), frame)
        msg = f"{args.name}: {len(embs)}/{args.n} - turn your head slightly, vary expression"
        if len(faces) != 1:
            msg = "Exactly one face must be visible"
        cv2.putText(show, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Enrollment (q to cancel)", show)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    cap.release()
    cv2.destroyAllWindows()

    if not embs:
        sys.exit("Nothing captured.")
    gallery_path = ROOT / cfg["paths"]["gallery"]
    gallery = Gallery.load(gallery_path)
    gallery.add(args.name, embs[: cfg["enroll"]["n_templates"]])
    gallery.save(gallery_path)
    print(f"Enrolled '{args.name}' ({len(embs)} captures). Gallery size: {len(gallery)}")


if __name__ == "__main__":
    main()
