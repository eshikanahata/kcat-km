"""Feature registry. A feature spec in a config is either a name or {name: {params}}.

    features:
      - seq_len
      - aac
      - morgan: {radius: 2, n_bits: 2048}

To add a feature: subclass Extractor in protein.py / ligand.py (or a new
module imported below), decorate with @register, and reference it by name.
"""
import numpy as np

from . import ligand, plm, protein  # noqa: F401  (registers extractors; torch is imported lazily)
from .base import REGISTRY, Extractor, register  # noqa: F401


def make_extractors(specs) -> list:
    out = []
    for spec in specs:
        if isinstance(spec, str):
            name, params = spec, {}
        else:
            (name, params), = spec.items()
            params = params or {}
        if name not in REGISTRY:
            raise KeyError(f"unknown feature {name!r}; available: {sorted(REGISTRY)}")
        out.append(REGISTRY[name](**params))
    return out


def build_matrix(df, extractors, use_cache: bool = True):
    """Returns (X float32 [n_rows, D], feature_names)."""
    # Compute time-adaptive extractors (ESM) LAST so they can use all the time that is
    # left before an inference deadline; columns keep the configured order.
    order = sorted(range(len(extractors)), key=lambda i: bool(getattr(extractors[i], "adaptive", False)))
    blocks = [None] * len(extractors)
    for i in order:
        ex = extractors[i]
        blocks[i] = ex.transform(df[ex.entity].tolist(), use_cache=use_cache)
    names = [n for ex in extractors for n in ex.feature_names()]
    X = np.hstack(blocks).astype(np.float32)
    assert X.shape[1] == len(names), "feature_names() length must match compute() width"
    return X, names
