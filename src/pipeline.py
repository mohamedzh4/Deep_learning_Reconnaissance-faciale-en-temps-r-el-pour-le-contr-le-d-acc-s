from dataclasses import dataclass
from typing import Optional

import cv2

from .audit_log import AuditLog
from .config import ROOT
from .decision import AccessDecision
from .door import DoorController
from .face_engine import Face, FaceEngine, largest_face
from .gallery import Gallery
from .liveness import Liveness
from .matcher import identify


@dataclass
class FrameResult:
    face: Face
    name: Optional[str]
    score: float
    live: bool
    event: Optional[str] = None  # "GRANTED" / "DENIED" when a decision was just taken


class AccessPipeline:
    """frame -> detect/align/embed -> liveness -> match -> vote -> door + audit log."""

    def __init__(self, cfg, engine=None):
        self.engine = engine or FaceEngine(cfg)
        self.gallery = Gallery.load(ROOT / cfg["paths"]["gallery"])
        self.threshold = cfg["matching"]["threshold"]
        self.liveness = Liveness()
        d = cfg["decision"]
        self.decision = AccessDecision(d["window"], d["min_votes"], d["cooldown_s"])
        self.door = DoorController(cfg["door"]["open_seconds"])
        self.audit = AuditLog(ROOT / cfg["paths"]["log_db"])

    def process(self, frame):
        faces = self.engine.analyze(frame)
        if not faces:
            self.decision.reset()
            return None
        face = largest_face(faces)
        live = self.liveness.check(frame, face)
        name, score = identify(face.embedding, self.gallery, self.threshold) if live else (None, 0.0)
        event, who = self.decision.update(name)
        if event == "GRANTED":
            self.door.open()
        if event:
            self.audit.log(event, who, score, "" if live else "spoof_suspected")
        return FrameResult(face, name, score, live, event)


def draw_overlay(frame, res, door_open, fps=0.0, banner=None):
    if res is not None:
        x1, y1, x2, y2 = res.face.bbox.astype(int)
        color = (0, 200, 0) if res.name else (0, 0, 255)
        if not res.live:
            color = (0, 200, 255)
        label = f"{res.name or 'unknown'} {res.score:.2f}" if res.live else "SPOOF?"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    door_txt, door_col = ("DOOR: OPEN", (0, 200, 0)) if door_open else ("DOOR: LOCKED", (0, 0, 255))
    cv2.putText(frame, door_txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, door_col, 2)
    cv2.putText(frame, f"{fps:.1f} FPS", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if banner:
        cv2.putText(frame, banner[0], (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, banner[1], 2)
    return frame
