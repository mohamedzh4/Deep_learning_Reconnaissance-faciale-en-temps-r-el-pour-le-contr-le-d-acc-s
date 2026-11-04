import time
from collections import Counter, deque


class AccessDecision:
    """Multi-frame voting + per-identity cooldown. update() returns (event, name)."""

    def __init__(self, window=5, min_votes=3, cooldown_s=5):
        self.votes = deque(maxlen=window)
        self.min_votes = min_votes
        self.cooldown_s = cooldown_s
        self._last = {}

    def reset(self):
        self.votes.clear()

    def update(self, name):
        """name: matched identity, or None for unknown / spoof."""
        self.votes.append(name or "unknown")
        top, n = Counter(self.votes).most_common(1)[0]
        if n < self.min_votes:
            return None, None
        now = time.time()
        if now - self._last.get(top, 0.0) < self.cooldown_s:
            return None, None
        self._last[top] = now
        self.votes.clear()
        if top == "unknown":
            return "DENIED", None
        return "GRANTED", top
