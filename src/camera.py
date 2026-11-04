# import sys

# import cv2


# def open_camera(cfg):
#     c = cfg["camera"]
#     cap = cv2.VideoCapture(c["index"], cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(c["index"])
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, c["width"])
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c["height"])
#     if not cap.isOpened():
#         raise SystemExit("Cannot open camera. Try another index with --camera 1.")
#     return cap
import sys
import time

import cv2


def open_camera(cfg):
    c = cfg["camera"]
    if sys.platform == "darwin":
        backend = cv2.CAP_AVFOUNDATION
    elif sys.platform == "win32":
        backend = cv2.CAP_DSHOW
    else:
        backend = cv2.CAP_ANY
    cap = cv2.VideoCapture(c["index"], backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, c["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c["height"])

    # first frames can fail while the camera starts or the permission prompt is pending
    deadline = time.time() + 5
    while time.time() < deadline:
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap
        time.sleep(0.2)
    cap.release()

    hint = ""
    if sys.platform == "darwin":
        hint = ("\nmacOS: System Settings > Privacy & Security > Camera > enable your terminal app "
                "(Terminal / iTerm / VS Code / Cursor), then quit and reopen it.")
    raise SystemExit(f"Cannot read from camera index {c['index']}. "
                     f"Set camera.index in config.yaml to try another one.{hint}")