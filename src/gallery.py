from pathlib import Path

import numpy as np


class Gallery:
    """Enrolled templates: one L2-normalized embedding per person (embeddings only, no images)."""

    def __init__(self, names=None, embs=None):
        self.names = list(names) if names is not None else []
        self.embs = np.asarray(embs, np.float32) if embs is not None else np.zeros((0, 512), np.float32)

    def __len__(self):
        return len(self.names)

    @classmethod
    def load(cls, path):
        path = Path(path)
        if not path.exists():
            return cls()
        d = np.load(path)
        return cls(d["names"], d["embs"])

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, names=np.array(self.names), embs=self.embs)

    def add(self, name, embeddings):
        """Average the embeddings into one template; replaces an existing person of the same name."""
        m = np.mean(np.asarray(embeddings, np.float32), axis=0)
        m /= np.linalg.norm(m)
        if name in self.names:
            self.embs[self.names.index(name)] = m
        else:
            self.names.append(name)
            self.embs = np.vstack([self.embs, m[None, :]])

    def remove(self, name):
        if name in self.names:
            i = self.names.index(name)
            self.names.pop(i)
            self.embs = np.delete(self.embs, i, axis=0)
