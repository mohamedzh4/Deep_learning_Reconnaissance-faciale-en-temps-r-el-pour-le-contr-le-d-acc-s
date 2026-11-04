# """Real-time access control. Usage: python scripts/run_realtime.py [--camera 0] [--threshold 0.4]. Press q to quit."""
# import argparse
# import sys
# import time
# from pathlib import Path

# sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# import cv2

# from src.camera import open_camera
# from src.config import load_config
# from src.pipeline import AccessPipeline, draw_overlay


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--config")
#     ap.add_argument("--camera", type=int)
#     ap.add_argument("--threshold", type=float)
#     args = ap.parse_args()

#     cfg = load_config(args.config)
#     if args.camera is not None:
#         cfg["camera"]["index"] = args.camera
#     if args.threshold is not None:
#         cfg["matching"]["threshold"] = args.threshold

#     pipe = AccessPipeline(cfg)
#     if len(pipe.gallery) == 0:
#         sys.exit("Gallery is empty. Run: python scripts/enroll_user.py --name <name>")
#     cap = open_camera(cfg)
#     every = max(1, cfg["runtime"]["process_every"])

#     last, banner, n, fps, t_prev = None, None, 0, 0.0, time.perf_counter()
#     while True:
#         ok, frame = cap.read()
#         if not ok:
#             break
#         if n % every == 0:
#             last = pipe.process(frame)
#             if last and last.event:
#                 col = (0, 200, 0) if last.event == "GRANTED" else (0, 0, 255)
#                 banner = (f"{last.event}: {last.name or 'unknown'}", col, time.time() + 2.0)
#         n += 1
#         now = time.perf_counter()
#         fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
#         t_prev = now
#         shown = banner if banner and banner[2] > time.time() else None
#         cv2.imshow("Access control (q to quit)", draw_overlay(frame, last, pipe.door.is_open, fps, shown))
#         if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     pipe.audit.close()


# if __name__ == "__main__":
#     main()

"""Real-time access control. Usage: python scripts/run_realtime.py [--camera 0] [--threshold 0.4]. Press q to quit."""
import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from src.camera import open_camera
from src.config import load_config
from src.pipeline import AccessPipeline, draw_overlay


class InferenceWorker(threading.Thread):
    """Runs the pipeline on the most recent frame only, so video display never waits for the model."""

    def __init__(self, pipe):
        super().__init__(daemon=True)
        self.pipe = pipe
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.running = True
        self._frame = None
        self._result = None
        self._result_ts = 0.0
        self._banner = None

    def submit(self, frame):
        with self.lock:
            self._frame = frame
        self.wake.set()

    def latest(self, max_age=0.8):
        with self.lock:
            fresh = (time.time() - self._result_ts) < max_age
            banner = self._banner if self._banner and self._banner[2] > time.time() else None
            return (self._result if fresh else None), banner

    def stop(self):
        self.running = False
        self.wake.set()
        self.join(timeout=2)

    def run(self):
        while self.running:
            self.wake.wait(0.1)
            self.wake.clear()
            with self.lock:
                frame, self._frame = self._frame, None
            if frame is None:
                continue
            res = self.pipe.process(frame)
            with self.lock:
                self._result, self._result_ts = res, time.time()
                if res and res.event:
                    col = (0, 200, 0) if res.event == "GRANTED" else (0, 0, 255)
                    self._banner = (f"{res.event}: {res.name or 'unknown'}", col, time.time() + 2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--camera", type=int)
    ap.add_argument("--threshold", type=float)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.camera is not None:
        cfg["camera"]["index"] = args.camera
    if args.threshold is not None:
        cfg["matching"]["threshold"] = args.threshold

    pipe = AccessPipeline(cfg)
    if len(pipe.gallery) == 0:
        sys.exit("Gallery is empty. Run: python scripts/enroll_user.py --name <name>")
    cap = open_camera(cfg)
    worker = InferenceWorker(pipe)
    worker.start()

    fps, t_prev = 0.0, time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            worker.submit(frame.copy())
            res, banner = worker.latest()
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now
            cv2.imshow("Access control (q to quit)", draw_overlay(frame, res, pipe.door.is_open, fps, banner))
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        worker.stop()
        cap.release()
        cv2.destroyAllWindows()
        pipe.audit.close()


if __name__ == "__main__":
    main()