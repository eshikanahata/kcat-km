"""Error analysis for one run (or a head-to-head of two).

    python scripts/analyze.py best
    python scripts/analyze.py 20261022-1130_lgbm_handcrafted --vs ridge_starter

Breaks OOF performance down by how novel the enzyme is (nearest similarity to
the other folds), sequence length, whether the SMILES was seen in other folds,
and target range. Writes a scatter/residual plot into the run folder when
matplotlib is installed.
"""
import argparse

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from kcatkm import TARGETS, data, metrics, similarity, tracking


def load_oof(run):
    d = tracking.find_run(run)
    return d, pd.read_parquet(d / "oof.parquet")


def add_context(oof: pd.DataFrame) -> pd.DataFrame:
    train = data.load_train().set_index("id").loc[oof["id"]]
    oof = oof.copy()
    oof["sequence"], oof["smiles"] = train["sequence"].to_numpy(), train["smiles"].to_numpy()
    oof["seq_len"] = oof["sequence"].str.len()
    sim = np.empty(len(oof))
    smiles_seen = np.empty(len(oof), dtype=bool)
    for k in oof["fold"].unique():
        va = (oof["fold"] == k).to_numpy()
        uq = oof.loc[va, "sequence"].unique()
        s = pd.Series(similarity.nearest_similarity(uq, oof.loc[~va, "sequence"].unique()), index=uq)
        sim[va] = oof.loc[va, "sequence"].map(s).to_numpy()
        smiles_seen[va] = oof.loc[va, "smiles"].isin(set(oof.loc[~va, "smiles"]))
    oof["nearest_sim"], oof["smiles_seen"] = sim, smiles_seen
    return oof


def breakdown(oof: pd.DataFrame, by: pd.Series, label: str):
    rows = []
    for key, g in oof.groupby(by, observed=True):
        s = metrics.score(g[list(TARGETS)].to_numpy(), g[[f"pred_{t}" for t in TARGETS]].to_numpy())
        rows.append({label: key, "n": len(g), **{k: s[k] for k in ("mean_scc", "kcat_scc", "km_scc", "mean_rmse")}})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--vs", help="second run to compare against (same split)")
    args = ap.parse_args()

    d, oof = load_oof(args.run)
    print(f"run {d.name}")
    oof = add_context(oof)
    fmt = lambda v: f"{v:.3f}"  # noqa: E731
    reports = {
        "enzyme novelty (nearest k-mer sim to other folds)": breakdown(
            oof, pd.qcut(oof["nearest_sim"], 5, duplicates="drop"), "nearest_sim"),
        "sequence length": breakdown(oof, pd.qcut(oof["seq_len"], 4, duplicates="drop"), "seq_len"),
        "SMILES seen in other folds": breakdown(oof, oof["smiles_seen"], "smiles_seen"),
        "per fold": breakdown(oof, oof["fold"], "fold"),
    }
    out = []
    for title, tab in reports.items():
        block = f"## {title}\n{tab.to_string(index=False, float_format=fmt)}\n"
        print("\n" + block)
        out.append(block)

    for t in TARGETS:
        r = oof[f"pred_{t}"] - oof[t]
        print(f"{t}: pred std {oof[f'pred_{t}'].std():.3f} vs true std {oof[t].std():.3f}; "
              f"bias {r.mean():+.3f}; worst |resid| {r.abs().max():.2f}")

    if args.vs:
        d2, oof2 = load_oof(args.vs)
        m = oof.merge(oof2, on="id", suffixes=("", "_b"))
        print(f"\nhead-to-head vs {d2.name} on {len(m)} shared rows:")
        for t in TARGETS:
            a = metrics.scc(m[t], m[f"pred_{t}"])
            b = metrics.scc(m[t], m[f"pred_{t}_b"])
            corr = np.corrcoef(m[f"pred_{t}"], m[f"pred_{t}_b"])[0, 1]
            blend = metrics.scc(m[t], m[f"pred_{t}"] + m[f"pred_{t}_b"])
            print(f"  {t}: this {a:.4f} | other {b:.4f} | 50/50 blend {blend:.4f} | pred corr {corr:.3f}")

    (d / "analysis.md").write_text("\n".join(out))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        for ax, t in zip(axes, TARGETS):
            ax.scatter(oof[t], oof[f"pred_{t}"], s=3, alpha=0.25)
            lo, hi = oof[t].min(), oof[t].max()
            ax.plot([lo, hi], [lo, hi], lw=1, color="k")
            ax.set(xlabel=f"true {t}", ylabel="OOF prediction",
                   title=f"{t}  SCC={metrics.scc(oof[t], oof[f'pred_{t}']):.3f}")
        fig.tight_layout()
        fig.savefig(d / "oof_scatter.png", dpi=120)
        print(f"\nplots + analysis.md written to {d}")
    except ImportError:
        print(f"\nanalysis.md written to {d} (pip install matplotlib for plots)")


if __name__ == "__main__":
    main()
