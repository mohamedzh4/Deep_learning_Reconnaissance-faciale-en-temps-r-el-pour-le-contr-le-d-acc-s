class Liveness:
    """Phase 2 placeholder: always accepts.

    Replace `check` with an anti-spoofing model (trained on CelebA-Spoof / OULU-NPU,
    evaluated cross-dataset) that returns False for print / replay / mask attacks.
    """

    def check(self, frame_bgr, face):
        return True
