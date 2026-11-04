import time


class DoorController:
    """Simulated door. For real hardware, drive a relay/GPIO inside open()."""

    def __init__(self, open_seconds=3):
        self.open_seconds = open_seconds
        self._until = 0.0

    def open(self):
        self._until = time.time() + self.open_seconds

    @property
    def is_open(self):
        return time.time() < self._until
