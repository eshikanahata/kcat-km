"""Which local CV split looks most like the hidden validation set?

For the public val inputs, measures how close each row is to the training data
(nearest k-mer similarity of its enzyme, and whether its exact sequence /
SMILES / pair appear in train). Then measures the same for local folds under
several split strategies. Pick the strategy whose numbers match the public val
best and set it in configs/base.yaml. Run once on the real data; the report is
written to notes/split_check.md.

    python scripts/split_check.py
    python scripts/split_check.py --thresholds 0.3 0.5 0.7 --mmseqs 0.3 0.5
"""
import argparse

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from kcatkm import data, paths, similarity, splits

Q = [0.1, 0.25, 0.5, 0.75, 0.9]


def closeness(query: pd.DataFrame, ref: pd.DataFrame) -> dict:
    uq = query["sequence"].unique()
    sim = pd.Series(similarity.nearest_similarity(uq, ref["sequence"].unique()), index=uq)
    row_sim = query["sequence"].map(sim).to_numpy()
    ref_pairs = set(zip(ref["smiles"], ref["sequence"]))
    out = {f"sim_q{int(q * 100)}": float(np.quantile(row_sim, q)) for q in Q}
    out["seq_seen"] = float(query["sequence"].isin(set(ref["sequence"])).mean())
    out["smiles_seen"] = float(query["smiles"].isin(set(ref["smiles"])).mean())
    out["pair_seen"] = float(np.mean([p in ref_pairs for p in zip(query["smiles"], query["sequence"])]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds", nargs="*", type=float, default=[0.3, 0.5, 0.7])
    ap.add_argument("--mmseqs", nargs="*", type=float, default=[0.3, 0.5],
                    help="MMseqs2 min_seq_id values to try (skipped if mmseqs is not installed)")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--folds-to-check", type=int, default=2, help="average over this many folds")
    args = ap.parse_args()

    train, val = data.load_train(), data.load_val_inputs()
    print(f"train {len(train)} rows / {train['sequence'].nunique()} seqs / {train['smiles'].nunique()} smiles")
    print(f"val   {len(val)} rows / {val['sequence'].nunique()} seqs / {val['smiles'].nunique()} smiles")

    rows = [{"split": "PUBLIC VAL vs train", **closeness(val, train)}]
    candidates = [{"strategy": "random"}, {"strategy": "smiles"}, {"strategy": "seq_exact"}]
    candidates += [{"strategy": "seq_kmer", "threshold": t} for t in args.thresholds]
    if similarity.mmseqs_available():
        candidates += [{"strategy": "mmseqs", "min_seq_id": s} for s in args.mmseqs]
    else:
        print("(mmseqs not on PATH: skipping MMseqs2 splits)")

    for spec in candidates:
        spec = {**spec, "n_folds": args.n_folds, "seed": 42}
        try:
            folds = splits.get_folds(train, spec)
        except ValueError as e:
            print(f"  skipped: {e}")
            continue
        stats = []
        for k in range(args.folds_to_check):
            va = (folds["fold"] == k).to_numpy()
            stats.append(closeness(train[va], train[~va]))
        label = spec["strategy"] + "".join(f" {k}={v}" for k, v in spec.items()
                                           if k not in ("strategy", "n_folds", "seed"))
        rows.append({"split": label, **pd.DataFrame(stats).mean().to_dict(),
                     "largest_group_frac": folds.groupby("group").size().max() / len(folds)})
        print(f"  done {label}: {splits.describe(folds)}")

    table = pd.DataFrame(rows)
    ref = table.iloc[0]
    sim_cols = [c for c in table if c.startswith("sim_q")]
    table["distance_to_val"] = (table[sim_cols] - ref[sim_cols]).abs().mean(axis=1) + \
        (table[["seq_seen", "smiles_seen"]] - ref[["seq_seen", "smiles_seen"]]).abs().mean(axis=1)
    text = table.to_string(index=False, float_format=lambda v: f"{v:.3f}")
    print("\n" + text)
    best = table.iloc[1:].sort_values("distance_to_val").iloc[0]["split"]
    print(f"\nclosest to public val: {best}  -> set it under `split:` in configs/base.yaml")

    paths.NOTES.mkdir(exist_ok=True)
    (paths.NOTES / "split_check.md").write_text(
        "# Split check\n\nsim_qXX = quantiles of nearest k-mer cosine similarity to the reference set; "
        "*_seen = fraction of rows whose exact sequence/SMILES/pair occurs in the reference set.\n\n"
        f"```\n{text}\n```\n\nClosest to public val: **{best}**\n")


if __name__ == "__main__":
    main()
