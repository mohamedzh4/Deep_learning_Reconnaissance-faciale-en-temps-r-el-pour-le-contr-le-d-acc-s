import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.decision import AccessDecision
from src.face_engine import Face
from src.gallery import Gallery
from src.matcher import identify
from src.pipeline import AccessPipeline


def unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


def make_gallery():
    rng = np.random.default_rng(0)
    a, b = unit(rng.normal(size=512)), unit(rng.normal(size=512))
    g = Gallery()
    g.add("alice", [a])
    g.add("bob", [b])
    return g, a, b


def test_gallery_roundtrip(tmp_path):
    g, a, _ = make_gallery()
    g.save(tmp_path / "g.npz")
    g2 = Gallery.load(tmp_path / "g.npz")
    assert g2.names == ["alice", "bob"] and np.allclose(g2.embs, g.embs)
    g2.remove("alice")
    assert g2.names == ["bob"] and g2.embs.shape == (1, 512)


def test_identify():
    g, a, _ = make_gallery()
    assert identify(a, g, 0.4)[0] == "alice"
    stranger = unit(np.random.default_rng(1).normal(size=512))
    assert identify(stranger, g, 0.4)[0] is None


def test_decision_voting_and_cooldown():
    d = AccessDecision(window=5, min_votes=3, cooldown_s=60)
    assert [d.update("alice") for _ in range(2)] == [(None, None)] * 2
    assert d.update("alice") == ("GRANTED", "alice")
    assert [d.update("alice") for _ in range(3)][-1] == (None, None)  # cooldown
    d2 = AccessDecision(5, 3, 0)
    assert [d2.update(None) for _ in range(3)][-1] == ("DENIED", None)


class FakeEngine:
    def __init__(self, emb):
        self.emb = emb

    def analyze(self, frame):
        return [Face(np.array([0, 0, 10, 10]), np.zeros((5, 2)), 0.99, self.emb)]


def test_pipeline_grants(tmp_path):
    g, a, _ = make_gallery()
    g.save(tmp_path / "g.npz")
    cfg = {
        "matching": {"threshold": 0.4},
        "decision": {"window": 5, "min_votes": 3, "cooldown_s": 0},
        "door": {"open_seconds": 3},
        "paths": {"gallery": str(tmp_path / "g.npz"), "log_db": str(tmp_path / "log.sqlite")},
    }
    pipe = AccessPipeline(cfg, engine=FakeEngine(a))
    events = [pipe.process(None).event for _ in range(3)]
    assert events == [None, None, "GRANTED"] and pipe.door.is_open
    row = pipe.audit.db.execute("SELECT event, name FROM events").fetchone()
    assert row == ("GRANTED", "alice")
