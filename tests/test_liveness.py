import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.face_engine import Face
from src.liveness import Liveness, crop_face


def face_at(x1, y1, x2, y2):
    return Face(np.array([x1, y1, x2, y2], np.float32), np.zeros((5, 2)), 0.99, None)


def test_crop_shape_and_edges():
    frame = np.full((100, 100, 3), 255, np.uint8)
    assert crop_face(frame, [40, 40, 60, 60], 2.0, 64).shape == (64, 64, 3)
    # face at the frame corner: padded with zeros, still the requested size
    c = crop_face(frame, [0, 0, 20, 20], 2.0, 64)
    assert c.shape == (64, 64, 3) and c[0, 0].sum() == 0 and c[-1, -1].sum() > 0
    # box completely outside the frame -> black image, no crash
    assert crop_face(frame, [500, 500, 520, 520], 2.0, 32).sum() == 0


def test_disabled_always_live():
    lv = Liveness({})
    assert lv.evaluate(np.zeros((10, 10, 3), np.uint8), face_at(0, 0, 5, 5)) == (True, None)
    assert Liveness({"liveness": {"enabled": False}}).check(None, None) is True


def test_enabled_without_model_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        Liveness({"liveness": {"enabled": True, "model": str(tmp_path / "missing.onnx")}})


def test_onnx_inference_path(tmp_path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper

    # tiny stand-in model: P(live) = mean pixel value of the crop (bright = live, dark = spoof)
    node = helper.make_node("ReduceMean", ["input"], ["prob_live"], axes=[1, 2, 3], keepdims=0)
    graph = helper.make_graph(
        [node], "fake",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, 128, 128])],
        [helper.make_tensor_value_info("prob_live", TensorProto.FLOAT, ["batch"])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    path = tmp_path / "antispoof.onnx"
    onnx.save(model, str(path))
    path.with_suffix(".json").write_text(json.dumps({"size": 128, "scale": 2.0, "threshold": 0.5}))

    lv = Liveness({"liveness": {"enabled": True, "model": str(path)}})
    bright = np.full((200, 200, 3), 255, np.uint8)
    dark = np.zeros((200, 200, 3), np.uint8)
    f = face_at(80, 80, 120, 120)
    live, s = lv.evaluate(bright, f)
    assert live and s > 0.9
    live, s = lv.evaluate(dark, f)
    assert (not live) and s < 0.1
