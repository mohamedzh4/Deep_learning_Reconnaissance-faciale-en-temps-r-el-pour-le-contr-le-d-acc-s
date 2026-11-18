"""5-point face alignment to the standard ArcFace 112x112 template.

IMPORTANT: use this same function to prepare your training data, otherwise a model you train
sees different crops at training time and at inference time.
"""
import cv2
import numpy as np

ARCFACE_DST = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float64,
)  # left eye, right eye, nose, left mouth corner, right mouth corner (for size 112)


def similarity_transform(src, dst):
    """Least-squares similarity transform (scale + rotation + translation), Umeyama method. Returns a 2x3 matrix."""
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    n = len(src)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(d0.T @ s0 / n)
    D = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[1, 1] = -1
    R = U @ D @ Vt
    scale = np.trace(np.diag(S) @ D) / ((s0 ** 2).sum() / n)
    M = np.zeros((2, 3))
    M[:, :2] = scale * R
    M[:, 2] = mu_d - scale * R @ mu_s
    return M


def align_face(img_bgr, kps, size=112):
    """Warp the face so its 5 landmarks match the ArcFace template; returns a size x size BGR crop."""
    M = similarity_transform(kps, ARCFACE_DST * (size / 112.0))
    return cv2.warpAffine(img_bgr, M, (size, size), borderValue=0.0)

