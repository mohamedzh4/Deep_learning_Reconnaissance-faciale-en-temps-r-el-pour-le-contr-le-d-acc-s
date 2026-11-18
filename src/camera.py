import sys
import time

import cv2

BACKENDS = {
    "avfoundation": cv2.CAP_AVFOUNDATION,  # macOS
    "dshow": cv2.CAP_DSHOW,                # Windows (DirectShow)
    "msmf": cv2.CAP_MSMF,                  # Windows (Media Foundation)
    "v4l2": cv2.CAP_V4L2,                  # Linux
    "any": cv2.CAP_ANY,
}


def _auto_backends():
    if sys.platform == "darwin":
        return [cv2.CAP_AVFOUNDATION]
    if sys.platform == "win32":
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF]  # DirectShow first, then Media Foundation
    return [cv2.CAP_V4L2, cv2.CAP_ANY]


def open_camera(cfg):
    c = cfg["camera"]
    name = str(c.get("backend", "auto")).lower()
    backends = _auto_backends() if name == "auto" else [BACKENDS[name]]
    for backend in backends:
        cap = cv2.VideoCapture(c["index"], backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, c["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c["height"])
        # first frames can fail while the camera starts or a permission prompt is pending
        deadline = time.time() + 5
        while time.time() < deadline:
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    return cap
            time.sleep(0.2)
        cap.release()

    hints = {
        "darwin": "macOS: System Settings > Privacy & Security > Camera > enable your terminal app, then quit and reopen it.",
        "win32": "Windows: Settings > Privacy & security > Camera > allow camera access and 'Let desktop apps access your camera'. "
                 "Close Teams/Zoom/browser tabs using the camera, or set camera.backend to msmf / dshow.",
    }
    raise SystemExit(f"Cannot read from camera index {c['index']}. Try another index (--camera 1).\n"
                     + hints.get(sys.platform, "Linux: check /dev/video* and that your user is in the 'video' group."))

