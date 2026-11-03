"""Sequence similarity utilities used for leak-aware splits and diagnostics.

The hidden val/test sets are separated from train by enzyme-sequence
similarity, so a random row split leaks near-duplicate enzymes and inflates CV.
These helpers give two ways to measure/cluster similarity:

* k-mer cosine (pure numpy/scipy, always available): a fast *proxy* for
  sequence identity. Good enough for grouping and for comparing distributions.
* MMseqs2 (if the `mmseqs` binary is on PATH): real identity-based clustering.
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer


def kmer_matrix(seqs, k: int = 3, n_features: int = 2**20):
    vec = HashingVectorizer(analyzer="char", ngram_range=(k, k), n_features=n_features,
                            alternate_sign=False, norm="l2", lowercase=False)
    return vec.transform(list(seqs)).tocsr().astype(np.float32)


def nearest_similarity(query, ref, k: int = 3, batch: int = 1024, exclude_self: bool = False):
    """For each query sequence, the max k-mer cosine similarity to any ref sequence.

    exclude_self=True when query and ref are the same list (ignores the diagonal).
    """
    Q, R = kmer_matrix(query, k), kmer_matrix(ref, k)
    out = np.empty(Q.shape[0], dtype=np.float32)
    for s in range(0, Q.shape[0], batch):
        S = (Q[s:s + batch] @ R.T).toarray()
        if exclude_self:
            idx = np.arange(s, min(s + batch, Q.shape[0]))
            S[idx - s, idx] = -1
        out[s:s + batch] = S.max(axis=1)
    return out


def kmer_clusters(seqs, threshold: float = 0.5, k: int = 3, batch: int = 1024) -> np.ndarray:
    """Connected components of the graph 'k-mer cosine >= threshold'. Returns a label per seq."""
    X = kmer_matrix(seqs, k)
    n = X.shape[0]
    rows, cols = [], []
    for s in range(0, n, batch):
        S = (X[s:s + batch] @ X.T).toarray()
        r, c = np.nonzero(S >= threshold)
        rows.append(r + s)
        cols.append(c)
    r, c = np.concatenate(rows), np.concatenate(cols)
    G = sparse.coo_matrix((np.ones_like(r, dtype=np.int8), (r, c)), shape=(n, n))
    _, labels = connected_components(G, directed=False)
    return labels


def mmseqs_available() -> bool:
    return shutil.which("mmseqs") is not None


def mmseqs_clusters(seqs, min_seq_id: float = 0.3, coverage: float = 0.8, threads: int = 4) -> np.ndarray:
    """Cluster unique sequences with MMseqs2 easy-cluster. Returns a label per seq."""
    if not mmseqs_available():
        raise RuntimeError("mmseqs not found on PATH (conda install -c bioconda mmseqs2)")
    seqs = list(seqs)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        with open(tmp / "in.fa", "w") as f:
            for i, s in enumerate(seqs):
                f.write(f">s{i}\n{s}\n")
        subprocess.run(["mmseqs", "easy-cluster", str(tmp / "in.fa"), str(tmp / "out"), str(tmp / "work"),
                        "--min-seq-id", str(min_seq_id), "-c", str(coverage), "--threads", str(threads)],
                       check=True, stdout=subprocess.DEVNULL)
        tsv = pd.read_csv(tmp / "out_cluster.tsv", sep="\t", header=None, names=["rep", "member"])
    rep_of = dict(zip(tsv.member, tsv.rep))
    reps = {r: j for j, r in enumerate(sorted(set(rep_of.values())))}
    return np.array([reps[rep_of[f"s{i}"]] for i in range(len(seqs))])
