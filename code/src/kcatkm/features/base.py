"""Feature extractor interface + on-disk cache.

Rules that keep this reusable:
  * An extractor is *stateless*: it maps one input string (a sequence or a
    SMILES) to a fixed-length vector. Anything that must be fitted (scaling,
    PCA, target encoding) belongs in the model pipeline, not here.
  * Vectors are computed once per *unique* string and cached under a key built
    from (extractor name, version, params). Re-running an experiment, adding
    val inputs, or changing the model never recomputes features.
  * Bump `version` when you change an extractor's code, so stale caches are
    not reused.
"""
import hashlib
import json
import os

import numpy as np

from .. import paths

REGISTRY = {}
_MEMO = {}  # in-process feature memo, enabled with KCATKM_MEMO=1 (used by predict.py)
TIMINGS = {}  # extractor name -> seconds spent in transform() in this process (used by predict.py)


def register(cls):
    REGISTRY[cls.name] = cls
    return cls


def _sha(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()


class Extractor:
    name = "base"
    entity = "sequence"   # which input column it reads: "sequence" or "smiles"
    version = 1
    chunk_size = None     # set for slow extractors: cache is saved after every chunk,
                          # so an interrupted run resumes where it stopped

    def __init__(self, **params):
        self.params = params

    # --- implement these -------------------------------------------------
    def compute(self, values: list) -> np.ndarray:
        raise NotImplementedError

    def feature_names(self) -> list:
        raise NotImplementedError

    # --- caching ------------------------------------------------------------
    @property
    def key(self) -> str:
        blob = json.dumps({"n": self.name, "v": self.version, "p": self.params}, sort_keys=True)
        return f"{self.name}_{_sha(blob)[:10]}"

    def transform(self, values, use_cache: bool = True) -> np.ndarray:
        """values: list of strings (may repeat). Returns (len(values), d)."""
        import time
        t0 = time.time()
        try:
            return self._transform(values, use_cache)
        finally:
            TIMINGS[self.name] = TIMINGS.get(self.name, 0.0) + time.time() - t0

    def _transform(self, values, use_cache):
        uniq, inv = np.unique(np.asarray(values, dtype=object).astype(str), return_inverse=True)
        if not use_cache or os.environ.get("KCATKM_NO_CACHE"):
            if os.environ.get("KCATKM_MEMO"):  # inference: stages share features in memory
                mk = (self.key, _sha("\x00".join(uniq)))
                if mk not in _MEMO:
                    _MEMO[mk] = np.asarray(self.compute(list(uniq)), dtype=np.float32)
                return _MEMO[mk][inv]
            return self.compute(list(uniq))[inv]
        X = self._cached(list(uniq))
        return X[inv]

    def _cached(self, uniq: list) -> np.ndarray:
        d = paths.FEATURE_CACHE / self.key
        kfile, xfile = d / "keys.npy", d / "X.npy"
        hashes = np.array([_sha(v) for v in uniq])
        if kfile.exists():
            keys, X = np.load(kfile), np.load(xfile)
        else:
            keys, X = np.array([], dtype=hashes.dtype), None
        pos = {k: i for i, k in enumerate(keys)}
        missing = [i for i, h in enumerate(hashes) if h not in pos]
        step = self.chunk_size or max(len(missing), 1)
        for s in range(0, len(missing), step):
            part = missing[s:s + step]
            new = np.asarray(self.compute([uniq[i] for i in part]), dtype=np.float32)
            X = new if X is None else np.vstack([X, new])
            keys = np.concatenate([keys, hashes[part]])
            d.mkdir(parents=True, exist_ok=True)
            np.save(kfile, keys)
            np.save(xfile, X)
            (d / "spec.json").write_text(json.dumps(
                {"name": self.name, "version": self.version, "params": self.params}, indent=2))
            if self.chunk_size:
                print(f"    {self.name}: {min(s + step, len(missing))}/{len(missing)} new inputs cached", flush=True)
        if missing:
            pos = {k: i for i, k in enumerate(keys)}
        return X[[pos[h] for h in hashes]]

    def assets(self) -> list:
        """Local directories inference needs (e.g. pretrained weights); bundled into submissions."""
        return []

    def __getstate__(self):
        # never pickle loaded models/tokenizers (attributes starting with "_")
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def __repr__(self):
        return f"{self.name}({', '.join(f'{k}={v}' for k, v in self.params.items())})"
