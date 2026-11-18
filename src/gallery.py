"""Enrolled templates: one L2-normalized embedding per person.

Tracks `model_id` — the recognizer that produced the embeddings — so galleries
built with a different model cannot accidentally be mixed (embeddings of different
models are not comparable).
"""
from pathlib import Path

import numpy as np


class Gallery:
    """Enrolled templates: one L2-normalized embedding per person (embeddings only, no images)."""

    def __init__(self, names=None, embs=None, model_id=None):
        self.names = list(names) if names is not None else []
        self.embs = np.asarray(embs, np.float32) if embs is not None else np.zeros((0, 512), np.float32)
        self.model_id = model_id or ""

    def __len__(self):
        return len(self.names)

    @classmethod
    def load(cls, path):
        path = Path(path)
        if not path.exists():
            return cls()
        d = np.load(path)
        model_id = str(d["model_id"]) if "model_id" in d.files else ""
        return cls(d["names"], d["embs"], model_id or None)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, names=np.array(self.names), embs=self.embs, model_id=np.array(self.model_id or ""))

    def add(self, name, embeddings):
        """Average the embeddings into one template; replaces an existing person of the same name."""
        m = np.mean(np.asarray(embeddings, np.float32), axis=0)
        m /= np.linalg.norm(m)
        if not self.names:
            self.embs = m[None, :]
            self.names = [name]
        elif name in self.names:
            self.embs[self.names.index(name)] = m
        else:
            self.names.append(name)
            self.embs = np.vstack([self.embs, m[None, :]])

    def remove(self, name):
        if name in self.names:
            i = self.names.index(name)
            self.names.pop(i)
            self.embs = np.delete(self.embs, i, axis=0)

    def check_model(self, model_id):
        """Embeddings of different models are not comparable: refuse to mix them."""
        if model_id and self.model_id and len(self) and self.model_id != model_id:
            raise SystemExit(
                f"Gallery was built with '{self.model_id}' but the current model is '{model_id}'. "
                "Delete data/enrollment/gallery.npz and re-enroll (then recalibrate the threshold)."
            )

