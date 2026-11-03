"""Label-derived ("neighbour") features, fitted inside each training fold.

    label_features:
      emb_prefix: esm2_t12_35M_mean_   # which X columns hold the enzyme embedding
      inner_folds: 5
      smiles_te:   {smoothing: 5}        # substrate target encoding
      enzyme_knn:  {k: 10, tau: 0.1}     # labels of the most similar training enzymes
      pair_nn:     {tau: 0.1}            # same substrate, most similar training enzyme
      canonical_smiles: false            # match substrates on RDKit-canonical SMILES
      substrate_knn: {k: 10, tau: 0.1, prefix: ChemBERTa}   # labels of the most similar OTHER substrates

Features (per target where relevant):
  smiles_te   smoothed mean label of training rows with the exact same SMILES,
              plus log(1 + count). Unseen substrates get the global mean, count 0.
  enzyme_knn  similarity-weighted mean of per-enzyme mean labels over the k
              nearest training enzymes (cosine on centred ESM embeddings), plus
              the top and average similarity.
  pair_nn     among training rows with the same SMILES, a similarity-weighted
              label of the closest enzymes, the top similarity and the count.
              NaN when the substrate was never seen.
  substrate_knn  similarity-weighted mean of per-substrate mean labels over the k
              most similar training substrates (cosine on centred ChemBERTa
              embeddings), EXCLUDING the exact same substrate (smiles_te covers
              that), plus the top and average similarity. Helps substrates that
              are new or rare in training.

Leakage control
  * fit(X, Y, meta) uses only the rows it is given (one outer training fold).
  * Features for those training rows are computed out-of-fold with inner
    GroupKFold on meta["group"] (the same enzyme clusters as the CV split), so
    the model sees them under the same "unseen enzyme" conditions as test rows.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from . import TARGETS

T = len(TARGETS)
_CANON = {}  # raw SMILES -> RDKit canonical SMILES (process-wide cache)


def canonical(smiles_list):
    """RDKit canonical SMILES, so different spellings of one molecule match. Unparseable
    SMILES are kept as they are."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    out = []
    for s in smiles_list:
        c = _CANON.get(s)
        if c is None:
            m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
            c = Chem.MolToSmiles(m) if m is not None else s
            _CANON[s] = c
        out.append(c)
    return np.asarray(out, dtype=object)


def _wmean(w, V, fill):
    """Weighted mean over the last-but-one axis that skips missing labels: w (q, k), V (q, k, T).
    Rows whose neighbours all miss a target get `fill`. Identical to w @ V without NaNs."""
    ok = np.isfinite(V)
    num = np.einsum("qk,qkt->qt", w, np.where(ok, V, 0.0))
    den = np.einsum("qk,qkt->qt", w, ok.astype(w.dtype))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), fill)


def _norm(E):
    return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-8)


class LabelFeatures:
    def __init__(self, spec, feature_names, seed=42):
        self.spec = dict(spec or {})
        self.inner_folds = int(self.spec.get("inner_folds", 5))
        self.canonical = bool(self.spec.get("canonical_smiles", False))
        self.seed = seed
        prefix = self.spec.get("emb_prefix")
        self.emb_cols = np.array([i for i, n in enumerate(feature_names or []) if prefix and n.startswith(prefix)],
                                 dtype=int)
        if ("enzyme_knn" in self.spec or "pair_nn" in self.spec) and len(self.emb_cols) == 0:
            raise ValueError(f"label_features needs enzyme embedding columns; none start with {prefix!r}")
        sk = self.spec.get("substrate_knn")
        if sk is not None:
            sp = (sk or {}).get("prefix", "ChemBERTa")
            self.smi_cols = np.array([i for i, n in enumerate(feature_names or []) if n.startswith(sp)], dtype=int)
            if len(self.smi_cols) == 0:
                raise ValueError(f"substrate_knn needs substrate embedding columns; none start with {sp!r}")

    # ------------------------------------------------------------------ names
    def names(self):
        short = [t.replace("log10_", "") for t in TARGETS]
        out = []
        if "smiles_te" in self.spec:
            out += [f"lf_smi_{s}" for s in short] + ["lf_smi_logn"]
        if "enzyme_knn" in self.spec:
            out += [f"lf_knn_{s}" for s in short] + ["lf_knn_maxsim", "lf_knn_meansim"]
        if "pair_nn" in self.spec:
            out += [f"lf_pair_{s}" for s in short] + ["lf_pair_maxsim", "lf_pair_logn"]
        if "substrate_knn" in self.spec:
            out += [f"lf_sknn_{s}" for s in short] + ["lf_sknn_maxsim", "lf_sknn_meansim"]
        return out

    def _smiles_keys(self, meta):
        raw = meta["smiles"].to_numpy()
        return canonical(raw) if getattr(self, "canonical", False) else raw

    # -------------------------------------------------------------- reference
    def _ref(self, X, Y, meta):
        r = {"gmean": np.nan_to_num(np.nanmean(Y, 0)) if np.isnan(Y).any() else Y.mean(0)}
        smiles = self._smiles_keys(meta)
        if "smiles_te" in self.spec:
            g = pd.DataFrame(Y, columns=list(range(T))).groupby(smiles)
            r["smi_sum"], r["smi_n"], r["smi_cnt"] = g.sum(), g.size(), g.count()  # count: per target
        if "substrate_knn" in self.spec:
            self._ref_substrates(r, X, Y, smiles)
        if len(self.emb_cols):
            seqs = meta["sequence"].to_numpy()
            uniq, first, inv = np.unique(seqs, return_index=True, return_inverse=True)
            E = X[first][:, self.emb_cols].astype(np.float32)
            r["mu"] = E.mean(0)
            r["E"] = _norm(E - r["mu"])
            r["seq_mean"] = (pd.DataFrame(Y).groupby(inv).mean().sort_index().to_numpy())
            if "pair_nn" in self.spec:
                r["by_smiles"] = {s: (inv[idx], Y[idx]) for s, idx in pd.Series(np.arange(len(smiles)))
                                  .groupby(smiles).indices.items()}
        return r

    def _ref_substrates(self, r, X, Y, smiles):
        uniq, first, inv = np.unique(smiles.astype(str), return_index=True, return_inverse=True)
        F = X[first][:, self.smi_cols].astype(np.float32)
        r["smu"] = F.mean(0)
        r["F"] = _norm(F - r["smu"])
        r["smi_keys"] = uniq
        r["smi_mean"] = pd.DataFrame(Y).groupby(inv).mean().sort_index().to_numpy()

    def _query_emb(self, ref, X):
        return _norm(X[:, self.emb_cols].astype(np.float32) - ref["mu"])

    # --------------------------------------------------------------- features
    def _features(self, ref, X, meta):
        cols = []
        smiles = self._smiles_keys(meta)
        if "smiles_te" in self.spec:
            m = float(self.spec["smiles_te"].get("smoothing", 5)) if isinstance(self.spec["smiles_te"], dict) else 5.0
            n = ref["smi_n"].reindex(smiles).fillna(0).to_numpy()
            s = ref["smi_sum"].reindex(smiles).fillna(0).to_numpy()
            c = ref["smi_cnt"].reindex(smiles).fillna(0).to_numpy() if "smi_cnt" in ref else n[:, None]
            cols += [((s + m * ref["gmean"]) / (c + m)), np.log1p(n)[:, None]]

        if len(self.emb_cols):
            seqs = meta["sequence"].to_numpy()
            uq, first, inv = np.unique(seqs, return_index=True, return_inverse=True)
            Q = self._query_emb(ref, X[first])  # one embedding per unique query enzyme

        if "enzyme_knn" in self.spec:
            p = self.spec["enzyme_knn"] or {}
            k, tau = min(int(p.get("k", 10)), len(ref["E"])), float(p.get("tau", 0.1))
            out = np.zeros((len(uq), T + 2), dtype=np.float32)
            for b in range(0, len(uq), 1024):
                S = Q[b:b + 1024] @ ref["E"].T
                idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
                s = np.take_along_axis(S, idx, axis=1)
                w = np.exp((s - s.max(1, keepdims=True)) / tau)
                w /= w.sum(1, keepdims=True)
                out[b:b + 1024, :T] = _wmean(w, ref["seq_mean"][idx], ref["gmean"])
                out[b:b + 1024, T] = s.max(1)
                out[b:b + 1024, T + 1] = s.mean(1)
            cols.append(out[inv])

        if "pair_nn" in self.spec:
            tau = float((self.spec["pair_nn"] or {}).get("tau", 0.1))
            out = np.full((len(X), T + 2), np.nan, dtype=np.float32)
            out[:, T + 1] = 0.0
            for smi, rows in pd.Series(np.arange(len(X))).groupby(smiles).indices.items():
                hit = ref["by_smiles"].get(smi)
                if hit is None:
                    continue
                seq_idx, Yr = hit
                S = Q[inv[rows]] @ ref["E"][seq_idx].T          # (rows, same-substrate training rows)
                w = np.exp((S - S.max(1, keepdims=True)) / tau)
                w /= w.sum(1, keepdims=True)
                out[rows, :T] = _wmean(w, np.broadcast_to(Yr, (len(rows), *Yr.shape)), np.nan)
                out[rows, T] = S.max(1)
                out[rows, T + 1] = np.log1p(len(seq_idx))
            cols.append(out)

        if "substrate_knn" in self.spec:
            p = self.spec["substrate_knn"] or {}
            tau = float(p.get("tau", 0.1))
            k = min(int(p.get("k", 10)), max(len(ref["F"]) - 1, 1))
            us, first, inv = np.unique(smiles.astype(str), return_index=True, return_inverse=True)
            Qs = _norm(X[first][:, self.smi_cols].astype(np.float32) - ref["smu"])
            self_idx = pd.Series(np.arange(len(ref["smi_keys"])), index=ref["smi_keys"]).reindex(us).to_numpy()
            out = np.zeros((len(us), T + 2), dtype=np.float32)
            for b in range(0, len(us), 1024):
                S = Qs[b:b + 1024] @ ref["F"].T
                hit = self_idx[b:b + 1024]
                rows = np.where(~np.isnan(hit))[0]
                S[rows, hit[rows].astype(int)] = -np.inf  # never the exact same substrate
                idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
                sv = np.take_along_axis(S, idx, axis=1)
                sv = np.where(np.isfinite(sv), sv, -1.0)
                w = np.exp((sv - sv.max(1, keepdims=True)) / tau)
                w /= w.sum(1, keepdims=True)
                out[b:b + 1024, :T] = _wmean(w, ref["smi_mean"][idx], ref["gmean"])
                out[b:b + 1024, T] = sv.max(1)
                out[b:b + 1024, T + 1] = sv.mean(1)
            cols.append(out[inv])

        return np.hstack(cols).astype(np.float32) if cols else np.zeros((len(X), 0), np.float32)

    # ------------------------------------------------------------------- API
    def fit(self, X, Y, meta):
        meta = meta.reset_index(drop=True)
        self.ref_ = self._ref(X, Y, meta)
        groups = meta["group"].to_numpy() if "group" in meta else pd.factorize(meta["sequence"])[0]
        n_splits = min(self.inner_folds, len(np.unique(groups)))
        F = np.zeros((len(X), len(self.names())), dtype=np.float32)
        for tr, va in GroupKFold(n_splits, shuffle=True, random_state=self.seed).split(X, groups=groups):
            ref = self._ref(X[tr], Y[tr], meta.iloc[tr].reset_index(drop=True))
            F[va] = self._features(ref, X[va], meta.iloc[va].reset_index(drop=True))
        self.train_features_ = F
        return self

    def transform(self, X, meta):
        return self._features(self.ref_, X, meta.reset_index(drop=True))
