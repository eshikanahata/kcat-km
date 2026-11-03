"""Per-target blend weights for combining two or more prediction sets (fitted on OOF)."""
import itertools

import numpy as np

from . import TARGETS, metrics


def simplex(k, step=0.05):
    n = int(round(1 / step))
    for c in itertools.product(range(n + 1), repeat=k - 1):
        if sum(c) <= n:
            yield np.array([*c, n - sum(c)], dtype=float) / n


def tune(Y, preds, step=0.05):
    """Y: (n, T); preds: list of (n, T). Returns W (len(preds), T) maximising each target's Spearman."""
    W = np.zeros((len(preds), len(TARGETS)))
    for j in range(len(TARGETS)):
        stack = np.stack([p[:, j] for p in preds])
        best_w, best_s = None, -np.inf
        for w in simplex(len(preds), step):
            s = metrics.scc(Y[:, j], w @ stack)
            if s > best_s + 1e-9:
                best_w, best_s = w, s
        W[:, j] = best_w
    return W


def apply(W, preds):
    return sum(W[i] * p for i, p in enumerate(preds))
