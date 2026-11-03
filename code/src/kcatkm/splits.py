"""Cached, leak-aware K-fold assignments.

Every experiment that names the same split spec gets the *same* folds, read
from disk, so scores across runs are comparable and clustering runs once.

Strategies
  random      plain KFold on rows (leaky; only as a reference point)
  smiles      GroupKFold by exact SMILES
  seq_exact   GroupKFold by exact sequence
  seq_kmer    GroupKFold by k-mer-cosine clusters   (params: threshold, k)
  mmseqs      GroupKFold by MMseqs2 clusters        (params: min_seq_id, coverage)
"""
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from . import paths, similarity


def _hash(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:10]


def _groups(df: pd.DataFrame, strategy: str, params: dict) -> np.ndarray:
    if strategy == "random":
        return np.arange(len(df))
    if strategy == "smiles":
        return pd.factorize(df["smiles"])[0]
    uniq, inv = np.unique(df["sequence"].to_numpy(), return_inverse=True)
    if strategy == "seq_exact":
        return inv
    if strategy == "seq_kmer":
        labels = similarity.kmer_clusters(uniq, threshold=params.get("threshold", 0.5), k=params.get("k", 3))
        return labels[inv]
    if strategy == "mmseqs":
        labels = similarity.mmseqs_clusters(uniq, min_seq_id=params.get("min_seq_id", 0.3),
                                            coverage=params.get("coverage", 0.8))
        return labels[inv]
    raise ValueError(f"unknown split strategy {strategy!r}")


def get_folds(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """spec: {strategy, n_folds, seed, ...params}. Returns DataFrame[id, group, fold] aligned to df."""
    spec = dict(spec)
    strategy = spec.pop("strategy")
    n_folds = spec.pop("n_folds", 5)
    seed = spec.pop("seed", 42)
    data_key = hashlib.sha1("".join(df["id"]).encode()).hexdigest()[:10]
    fname = f"{strategy}_k{n_folds}_s{seed}_{_hash(spec)}_{data_key}.parquet"
    path = paths.SPLIT_CACHE / fname
    if path.exists():
        folds = pd.read_parquet(path)
        if list(folds["id"]) == list(df["id"]):
            return folds

    groups = _groups(df, strategy, spec)
    sizes = pd.Series(groups).value_counts()
    if len(sizes) < n_folds or sizes.iloc[0] > 0.5 * len(df):
        raise ValueError(f"{strategy} {spec}: degenerate grouping ({len(sizes)} groups, largest holds "
                         f"{sizes.iloc[0] / len(df):.0%} of rows); clusters chain together at this setting. "
                         "Use a higher threshold.")
    fold = np.empty(len(df), dtype=int)
    if strategy == "random":
        splitter = KFold(n_folds, shuffle=True, random_state=seed).split(df)
    else:
        splitter = GroupKFold(n_folds, shuffle=True, random_state=seed).split(df, groups=groups)
    for k, (_, va) in enumerate(splitter):
        fold[va] = k
    folds = pd.DataFrame({"id": df["id"].to_numpy(), "group": groups, "fold": fold})
    path.parent.mkdir(parents=True, exist_ok=True)
    folds.to_parquet(path, index=False)
    return folds


def label(spec: dict) -> str:
    """Short name used in runs/index.csv, e.g. 'seq_kmer(threshold=0.3)/5'."""
    params = {k: v for k, v in spec.items() if k not in ("strategy", "n_folds", "seed")}
    p = ",".join(f"{k}={v}" for k, v in sorted(params.items()))
    seed = spec.get("seed", 42)
    return f"{spec['strategy']}{f'({p})' if p else ''}/{spec.get('n_folds', 5)}" + ("" if seed == 42 else f"/s{seed}")


def describe(folds: pd.DataFrame) -> str:
    g = folds.groupby("group").size()
    sizes = folds.groupby("fold").size().to_dict()
    return (f"{folds['group'].nunique()} groups (largest {g.max()} rows, "
            f"{(g == 1).mean():.0%} singletons); rows per fold {sizes}")
