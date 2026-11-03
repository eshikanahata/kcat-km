"""Compare two versions of the training data (e.g. before/after the 23 Oct refresh).

    python scripts/compare_data_versions.py data/raw_old data/raw

Each argument is a folder containing train.parquet (and optionally val_inputs.parquet).
Reports sizes, label distributions, how labels changed for (SMILES, sequence) pairs
present in both, and how much the validation inputs overlap.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import _bootstrap  # noqa: F401
from kcatkm import TARGETS


def describe(name, df):
    print(f"\n{name}: {len(df)} rows, {df['sequence'].nunique()} sequences, {df['smiles'].nunique()} SMILES, "
          f"{df.duplicated(['smiles', 'sequence']).sum()} duplicated (smiles, sequence) pairs")
    for t in TARGETS:
        v = df[t]
        top = v.round(6).value_counts().head(3)
        print(f"  {t}: min {v.min():.2f}  max {v.max():.2f}  mean {v.mean():.3f}  std {v.std():.3f}  "
              f"most common values {', '.join(f'{k:g} (x{n})' for k, n in top.items())}")


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    old_dir, new_dir = Path(sys.argv[1]), Path(sys.argv[2])
    old, new = pd.read_parquet(old_dir / "train.parquet"), pd.read_parquet(new_dir / "train.parquet")
    describe(f"OLD {old_dir}", old)
    describe(f"NEW {new_dir}", new)

    key = ["smiles", "sequence"]
    o = old.groupby(key)[list(TARGETS)].mean()
    n = new.groupby(key)[list(TARGETS)].mean()
    shared = o.join(n, lsuffix="_old", rsuffix="_new", how="inner")
    print(f"\n(smiles, sequence) pairs: old {len(o)}, new {len(n)}, shared {len(shared)}, "
          f"only old {len(o) - len(shared)}, only new {len(n) - len(shared)}")
    for t in TARGETS:
        a, b = shared[f"{t}_old"], shared[f"{t}_new"]
        d = (b - a).abs()
        print(f"  {t} on shared pairs: spearman old vs new {spearmanr(a, b).statistic:.3f}, "
              f"identical (|diff|<1e-6) {np.mean(d < 1e-6):.1%}, |diff|>0.5 {np.mean(d > 0.5):.1%}, "
              f"mean shift (new - old) {np.mean(b - a):+.3f}")

    vo, vn = old_dir / "val_inputs.parquet", new_dir / "val_inputs.parquet"
    if vo.exists() and vn.exists():
        vo, vn = pd.read_parquet(vo), pd.read_parquet(vn)
        po, pn = set(zip(vo.smiles, vo.sequence)), set(zip(vn.smiles, vn.sequence))
        print(f"\nval inputs: old {len(vo)} rows, new {len(vn)} rows; shared pairs {len(po & pn)}; "
              f"new val pairs present in NEW train: {len(pn & set(zip(new.smiles, new.sequence)))} (should be 0)")
        print(f"  new val sequences in new train: {vn['sequence'].isin(set(new['sequence'])).mean():.1%}; "
              f"new val SMILES in new train: {vn['smiles'].isin(set(new['smiles'])).mean():.1%}")


if __name__ == "__main__":
    main()
