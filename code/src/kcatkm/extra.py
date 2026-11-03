"""Extra training data (e.g. the public Leak-CURBER kinetic-parameter records), used for
TRAINING ONLY: CV keeps the original folds and scores only the original rows, so a run
with extra data is directly comparable (paired, fold by fold) with every earlier run.

    data/raw/extra_train.parquet   id, smiles, sequence, log10_kcat, log10_km
                                   (written by scripts/import_extra.py)

    config:
      data: {extra_data: true}
      data: {extra_data: true, extra_file: extra_public.parquet}   # another file in data/raw/

Leakage control inside CV
  An extra row is left out of a fold's training data when its enzyme is similar
  (k-mer cosine >= the split threshold) to ANY enzyme in that fold's validation part,
  i.e. extra rows obey the same "unseen enzyme" rule as the CV split itself. Extra
  rows whose enzyme is unlike every original enzyme are used in every fold.
  Pairs (SMILES, sequence) already in the training data are dropped on import.
"""
import hashlib

import numpy as np
import pandas as pd

from . import TARGETS, paths, similarity, splits

EXTRA_FILE = paths.RAW / "extra_train.parquet"
PREFIX = "ex_"


def wanted(cfg) -> bool:
    return bool((cfg.get("data") or {}).get("extra_data"))


def file_of(cfg):
    """data.extra_file: which file in data/raw/ (default extra_train.parquet)."""
    name = (cfg.get("data") or {}).get("extra_file")
    return paths.RAW / name if name else EXTRA_FILE


def targets_of(cfg):
    """data.extra_targets: which labels of the extra rows to use (default both). The others
    are set missing, so those rows train only the chosen targets' models."""
    t = (cfg.get("data") or {}).get("extra_targets") or list(TARGETS)
    bad = set(t) - set(TARGETS)
    if bad:
        raise ValueError(f"unknown extra_targets {sorted(bad)}")
    return list(t)


def load(base: pd.DataFrame, targets=None, path=None) -> pd.DataFrame:
    path = path or EXTRA_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run scripts/import_extra.py / source_overlap.py --write first")
    ex = pd.read_parquet(path)
    targets = list(targets or TARGETS)
    # a row is used if it has at least one of the chosen targets; the rest stay missing and
    # those models skip it (e.g. kcat from one source, Km from another)
    ex = ex.dropna(subset=["smiles", "sequence"])
    ex = ex[ex[targets].notna().any(axis=1)].copy()
    for t in TARGETS:
        if t not in targets:
            ex[t] = np.nan
    pairs = set(zip(base["smiles"], base["sequence"]))
    ex = ex[[p not in pairs for p in zip(ex["smiles"], ex["sequence"])]]
    ex = ex.drop_duplicates(["smiles", "sequence"]).reset_index(drop=True)
    ex["id"] = PREFIX + ex["id"].astype(str)
    return ex[["id", "smiles", "sequence", *TARGETS]]


def fingerprint(path=None) -> str:
    path = path or EXTRA_FILE
    if not path.exists():
        return "none"
    return hashlib.sha1(path.read_bytes()).hexdigest()[:8]


def is_extra(ids) -> np.ndarray:
    return pd.Series(ids).astype(str).str.startswith(PREFIX).to_numpy()


def _link(ex_seqs, base_seqs, base_fold, base_group, n_folds, threshold, k=3, batch=512):
    """For each unique extra sequence: which folds it must be kept out of, and the group of
    its most similar original sequence (or None when it is unlike all of them)."""
    uq, inv = np.unique(np.asarray(ex_seqs, dtype=object).astype(str), return_inverse=True)
    bq, binv = np.unique(np.asarray(base_seqs, dtype=object).astype(str), return_inverse=True)
    # fold / group of each unique base sequence (a sequence belongs to one group, hence one fold)
    bfold = np.empty(len(bq), dtype=int)
    bgroup = np.empty(len(bq), dtype=np.int64)
    bfold[binv], bgroup[binv] = base_fold, base_group
    Q, R = similarity.kmer_matrix(uq, k), similarity.kmer_matrix(bq, k)
    block = np.zeros((len(uq), n_folds), dtype=bool)
    near_group = np.full(len(uq), -1, dtype=np.int64)
    for s in range(0, len(uq), batch):
        S = (Q[s:s + batch] @ R.T).toarray()
        hit = S >= threshold
        for f in range(n_folds):
            block[s:s + batch, f] = hit[:, bfold == f].any(axis=1)
        best = S.argmax(axis=1)
        ok = S[np.arange(len(best)), best] >= threshold
        near_group[s:s + batch][ok] = bgroup[best[ok]]
    return block[inv], near_group[inv]


def attach(base: pd.DataFrame, folds: pd.DataFrame, split_spec: dict, targets=None, path=None):
    """base + extra rows. Returns (df_all, folds_all, block) where folds_all has fold -1 for
    extra rows and block[i, f] says extra row i must not be trained on in fold f."""
    ex = load(base, targets, path)
    n_folds = int(folds["fold"].max()) + 1
    threshold = float(split_spec.get("threshold", 0.3))
    block, near = _link(ex["sequence"], base["sequence"], folds["fold"].to_numpy(), folds["group"].to_numpy(),
                        n_folds, threshold, k=split_spec.get("k", 3))
    # groups for the label features' inner folds: the most similar original cluster, else a new one
    new_ids = pd.factorize(ex["sequence"])[0] + int(folds["group"].max()) + 1
    groups = np.where(near >= 0, near, new_ids)
    df_all = pd.concat([base, ex], ignore_index=True)
    folds_all = pd.concat([folds, pd.DataFrame({"id": ex["id"], "group": groups, "fold": -1})], ignore_index=True)
    return df_all, folds_all, block


def training_frame(cfg, base: pd.DataFrame):
    """For a full-data refit: (rows to train on, enzyme-cluster groups for inner folds)."""
    folds = splits.get_folds(base, cfg["split"])
    if not wanted(cfg):
        return base, folds["group"].to_numpy()
    df_all, folds_all, _ = attach(base, folds, cfg["split"], targets_of(cfg), file_of(cfg))
    return df_all, folds_all["group"].to_numpy()


def describe(df_all, block) -> str:
    ex = is_extra(df_all["id"])
    n = int(ex.sum())
    always = int((~block.any(axis=1)).sum())
    per_fold = ", ".join(f"{k}: {int((~block[:, k]).sum())}" for k in range(block.shape[1]))
    return (f"{n} extra rows ({df_all.loc[ex, 'sequence'].nunique()} enzymes); {always} unlike every original "
            f"enzyme (used in every fold); extra rows trained on per fold {{{per_fold}}}")
