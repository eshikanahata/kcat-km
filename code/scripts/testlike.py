"""Does CV rank models the way the test set would?

The leaderboard's test set has ~2 rows per enzyme (1,714 rows / 848 enzymes); the training
data has many enzymes with dozens or hundreds of rows, and those dominate an ordinary CV
score. The "test-like" score (metrics.testlike) caps every enzyme at 2 rows (averaged over
random draws), so it weighs enzymes the way the test set does.

    python scripts/testlike.py          # back-fill tl_* for existing runs, then rank both ways

New runs get tl_* automatically. compare.py shows tl_mean_scc, and --vs its per-fold delta.
"""
import json

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from kcatkm import TARGETS, data, metrics, paths, tracking, valset


def profile(name, seqs):
    c = pd.Series(seqs).value_counts()
    print(f"  {name:<12} {len(seqs):>6} rows, {len(c):>6} enzymes, {len(seqs) / len(c):5.1f} rows/enzyme; "
          f"enzymes with 1 row {np.mean(c == 1):.0%}; rows from enzymes with >20 rows {c[c > 20].sum() / len(seqs):.0%}")


def main():
    train = data.load_train()
    print("rows per enzyme:")
    profile("train", train["sequence"])
    for f in sorted(paths.RAW.glob("*.parquet")):
        if f.name != "train.parquet":
            profile(f.stem, pd.read_parquet(f, columns=["sequence"])["sequence"])

    seq_of = train.set_index("id")["sequence"]
    idx = tracking.read_index()
    comp = tracking.comparable(idx)
    rows = []
    for rid in comp["run_id"]:
        d = paths.RUNS / rid
        if not (d / "oof.parquet").exists():
            continue
        oof = pd.read_parquet(d / "oof.parquet")
        oof = oof[~valset.is_val_row(oof["id"])]
        seqs = seq_of.reindex(oof["id"]).to_numpy()
        if pd.isna(seqs).any():
            continue
        Y = oof[list(TARGETS)].to_numpy()
        P = oof[[f"pred_{t}" for t in TARGETS]].to_numpy()
        tl = metrics.testlike(Y, P, seqs)
        m = json.loads((d / "metrics.json").read_text())
        for k, fold in enumerate(sorted(oof["fold"].unique())):
            sel = (oof["fold"] == fold).to_numpy()
            if k < len(m["folds"]):
                m["folds"][k]["tl_mean_scc"] = metrics.testlike(Y[sel], P[sel], seqs[sel])["mean_scc"]
        for k, v in tl.items():
            m["overall"][f"tl_{k}"] = v
            idx.loc[idx["run_id"] == rid, f"tl_{k}"] = round(v, 5)
        (d / "metrics.json").write_text(json.dumps(m, indent=2, default=str))
        rows.append({"run_id": rid, "mean_scc": m["overall"]["mean_scc"], "tl_mean_scc": tl["mean_scc"],
                     "tl_kcat_scc": tl["kcat_scc"], "tl_km_scc": tl["km_scc"]})
    idx.to_csv(paths.RUNS_INDEX, index=False)
    if not rows:
        print("no comparable runs")
        return
    out = pd.DataFrame(rows)
    out["cv_rank"] = out["mean_scc"].rank(ascending=False).astype(int)
    out["tl_rank"] = out["tl_mean_scc"].rank(ascending=False).astype(int)
    out = out.sort_values("tl_mean_scc", ascending=False)
    with pd.option_context("display.width", 200):
        print("\n" + out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    r = metrics.scc(out["mean_scc"], out["tl_mean_scc"])
    print(f"\nrank agreement between CV and test-like scores across runs: Spearman {r:.2f}")
    print("For reference, leaderboard: ens_lf_esm8_tuned 0.4833 (CV 0.5640); 8M+35M blend 0.5040 (CV 0.5770)")


if __name__ == "__main__":
    main()
