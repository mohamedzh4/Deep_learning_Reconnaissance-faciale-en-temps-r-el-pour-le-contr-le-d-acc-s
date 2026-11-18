"""Unified access pipeline: detect → liveness → recognise → vote → door + audit."""
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
    name: Optional[str]              # accepted identity (None = unknown or liveness failed)
    score: float                     # best cosine similarity
    live: bool
    event: Optional[str] = None      # "GRANTED" / "DENIED" when a decision was just taken
    reason: str = ""                 # liveness failure reason (empty when live)
    claimed: Optional[str] = None    # match even if liveness failed (for audit)
    live_score: Optional[float] = None  # P(real); None when liveness disabled


class AccessPipeline:
    """frame -> detect/align/embed -> liveness -> match -> vote -> door + audit log."""

    def __init__(self, cfg, engine=None, liveness=None):
        self.engine = engine or FaceEngine(cfg)
        self.gallery = Gallery.load(ROOT / cfg["paths"]["gallery"])
        self.gallery.check_model(getattr(self.engine, "model_id", None))
        self.threshold = cfg["matching"]["threshold"]
        self.liveness = liveness or Liveness(cfg)
        d = cfg["decision"]
        self.decision = AccessDecision(d["window"], d["min_votes"], d["cooldown_s"])
        self.door = DoorController(cfg["door"]["open_seconds"])
        self.audit = AuditLog(ROOT / cfg["paths"]["log_db"])

    def process(self, frame):
        faces = self.engine.analyze(frame)
        if not faces:
            self.decision.reset()
            self.liveness.reset()
            return None
        face = largest_face(faces)
        live, live_score = self.liveness.evaluate(frame, face)
        claimed, score = identify(face.embedding, self.gallery, self.threshold)
        name = claimed if live else None
        event, who = self.decision.update(name)
        if event == "GRANTED":
            self.door.open()
        if event:
            note = "" if live else f"liveness_fail;claimed={claimed or 'unknown'}"
            if live_score is not None:
                note += f";live_p={live_score:.2f}"
            self.audit.log(event, who, score, note)
        return FrameResult(face, name, score, live, event, "", claimed, live_score)


def draw_overlay(frame, res, door_open, fps=0.0, banner=None):
    if res is not None:
        x1, y1, x2, y2 = res.face.bbox.astype(int)
        if not res.live:
            color = (0, 200, 255)
            label = f"SPOOF? {res.live_score:.2f}" if res.live_score is not None else "SPOOF?"
        else:
            color = (0, 200, 0) if res.name else (0, 0, 255)
            label = f"{res.name or 'unknown'} {res.score:.2f}"
            if res.live_score is not None:
                label += f" | live {res.live_score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    door_txt, door_col = ("DOOR: OPEN", (0, 200, 0)) if door_open else ("DOOR: LOCKED", (0, 0, 255))
    cv2.putText(frame, door_txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, door_col, 2)
    cv2.putText(frame, f"{fps:.1f} FPS", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if banner:
        cv2.putText(frame, banner[0], (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, banner[1], 2)
    return frame

