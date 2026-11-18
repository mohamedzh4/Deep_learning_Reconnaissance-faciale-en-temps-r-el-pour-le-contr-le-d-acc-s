import numpy as np
import torch


def export_onnx(net, path, opset=17):
    """Export the backbone: input 1x3x112x112 float in [-1, 1] (batch is dynamic), output embedding."""
    net = net.eval().cpu()
    dummy = torch.randn(1, 3, 112, 112)
    kwargs = dict(
        input_names=["input"], output_names=["embedding"],
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}}, opset_version=opset,
    )
    try:
        torch.onnx.export(net, dummy, str(path), dynamo=False, **kwargs)
    except TypeError:  # older PyTorch without the `dynamo` argument
        torch.onnx.export(net, dummy, str(path), **kwargs)
    return path


def parity(net, path, n=4):
    """Max absolute difference between PyTorch and onnxruntime outputs on random inputs."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = torch.randn(n, 3, 112, 112)
    with torch.no_grad():
        a = net.eval().cpu()(x).numpy()
    b = sess.run([], {"input": x.numpy()})[0]
    return float(np.abs(a - b).max())
