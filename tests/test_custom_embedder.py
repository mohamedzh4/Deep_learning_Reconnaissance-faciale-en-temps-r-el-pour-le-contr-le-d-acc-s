import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.align import ARCFACE_DST, align_face, similarity_transform
from src.gallery import Gallery

onnx = pytest.importorskip("onnx")


def make_dummy_model(path, dim=8, size=112):
    """3-channel global average -> linear layer -> `dim`-d vector (stand-in for a trained recognizer)."""
    from onnx import TensorProto, helper, numpy_helper

    w = numpy_helper.from_array(np.random.default_rng(0).normal(size=(3, dim)).astype(np.float32), "W")
    nodes = [
        helper.make_node("GlobalAveragePool", ["x"], ["g"]),
        helper.make_node("Flatten", ["g"], ["f"], axis=1),
        helper.make_node("MatMul", ["f", "W"], ["y"]),
    ]
    graph = helper.make_graph(
        nodes, "dummy",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, size, size])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, dim])],
        [w],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))


def test_similarity_transform_recovers_known_transform():
    th, s, t = 0.3, 1.7, np.array([5.0, -3.0])
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    src = (ARCFACE_DST - t) @ R / s   # inverse of dst = s * R @ src + t
    M = similarity_transform(src, ARCFACE_DST)
    assert np.allclose(src @ M[:, :2].T + M[:, 2], ARCFACE_DST, atol=1e-6)


def test_align_face_output_shape():
    img = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    kps = ARCFACE_DST + np.array([200.0, 100.0])
    out = align_face(img, kps, 112)
    assert out.shape == (112, 112, 3)
    assert np.array_equal(out[50:60, 40:50], img[150:160, 240:250])  # identity transform + translation only


def test_embedder_returns_unit_vector(tmp_path):
    from src.embedder import OnnxEmbedder

    p = tmp_path / "m.onnx"
    make_dummy_model(p, dim=8)
    e = OnnxEmbedder(p)(np.full((112, 112, 3), 90, np.uint8))
    assert e.shape == (8,) and abs(np.linalg.norm(e) - 1.0) < 1e-5
    with pytest.raises(FileNotFoundError):
        OnnxEmbedder(tmp_path / "missing.onnx")


def test_engine_uses_custom_recognizer(tmp_path, monkeypatch):
    p = tmp_path / "m.onnx"
    make_dummy_model(p, dim=8)

    class FakeFace:
        bbox = np.array([100, 100, 200, 200], np.float32)
        kps = ARCFACE_DST.astype(np.float32) * 0.9 + np.array([110.0, 105.0], np.float32)
        det_score = 0.99

    class FakeAnalysis:
        def __init__(self, name, root, providers, allowed_modules):
            FakeAnalysis.modules = allowed_modules

        def prepare(self, ctx_id, det_size):
            pass

        def get(self, frame):
            return [FakeFace()]

    fake_app = types.ModuleType("insightface.app")
    fake_app.FaceAnalysis = FakeAnalysis
    monkeypatch.setitem(sys.modules, "insightface", types.ModuleType("insightface"))
    monkeypatch.setitem(sys.modules, "insightface.app", fake_app)

    from src.face_engine import FaceEngine

    cfg = {"model": {"name": "buffalo_s", "det_size": [320, 320], "use_gpu": False, "recognizer": str(p)}}
    eng = FaceEngine(cfg)
    faces = eng.analyze(np.full((400, 400, 3), 128, np.uint8))
    assert FakeAnalysis.modules == ["detection"] and eng.model_id == "custom:m.onnx"
    assert faces[0].embedding.shape == (8,)
    cfg["model"]["recognizer"] = None
    assert FaceEngine(cfg).model_id == "buffalo_s" and FakeAnalysis.modules == ["detection", "recognition"]


def test_gallery_model_tag_and_other_dimensions(tmp_path):
    g = Gallery(model_id="custom:m.onnx")
    g.add("alice", [np.ones(8, np.float32)])
    g.add("bob", [np.arange(8, dtype=np.float32) + 1])
    g.save(tmp_path / "g.npz")
    g2 = Gallery.load(tmp_path / "g.npz")
    assert g2.embs.shape == (2, 8) and g2.model_id == "custom:m.onnx"
    g2.check_model("custom:m.onnx")
    with pytest.raises(SystemExit):
        g2.check_model("buffalo_s")
    legacy = Gallery(["a"], np.ones((1, 512), np.float32))  # no model_id -> accepted
    legacy.check_model("buffalo_s")
