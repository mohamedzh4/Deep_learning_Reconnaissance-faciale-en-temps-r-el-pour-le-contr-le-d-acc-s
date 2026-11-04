import numpy as np


def identify(embedding, gallery, threshold):
    """Return (name or None, best cosine similarity)."""
    if len(gallery) == 0:
        return None, 0.0
    sims = gallery.embs @ embedding
    i = int(np.argmax(sims))
    score = float(sims[i])
    return (gallery.names[i] if score >= threshold else None), score
