"""Show the experiment log, best first.

    python scripts/compare.py              # all runs on the current default split
    python scripts/compare.py --all        # every split
    python scripts/compare.py --top 10 --grep lgbm
    python scripts/compare.py --vs lgbm_esm35m   # paired per-fold comparison against a reference run

--vs: all runs on a split share the same folds, so the right question is not
"is the gap bigger than the fold std?" but "does it win fold by fold?". For
each run this prints the mean per-fold difference in mean SCC vs the
reference, its standard error across folds, and how many folds it won.
Rough reading with 5 folds: 5/5 wins and delta > 2 x se -> real; mixed wins -> noise.
"""
import argparse
import json

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from kcatkm import config, data, paths, splits, tracking

COLS = ["run_id", "model", "mean_scc", "tl_mean_scc", "mean_scc_fold_std", "kcat_scc", "km_scc", "mean_rmse",
        "val_mean_scc", "val_kcat_scc", "val_km_scc", "val_unseen_mean_scc", "fit_seconds", "notes"]


def fold_scores(run_id: str) -> pd.DataFrame:
    return pd.DataFrame(json.loads((paths.RUNS / run_id / "metrics.json").read_text())["folds"])


def paired(idx: pd.DataFrame, ref_id: str) -> pd.DataFrame:
    ref = fold_scores(ref_id)
    rows = []
    for rid in idx["run_id"]:
        f = fold_scores(rid)
        if len(f) != len(ref):
            rows.append({"run_id": rid})
            continue
        row = {"run_id": rid}
        for m in ("mean_scc", "kcat_scc", "km_scc", "tl_mean_scc"):
            if m not in f or m not in ref:
                continue
            d = (f[m] - ref[m]).to_numpy()
            row[f"d_{m}"] = d.mean()
            if m == "mean_scc":
                row["se"] = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
                row["n_win"], row["n_folds"] = int((d > 0).sum()), len(d)
                row["wins"] = f"{row['n_win']}/{len(d)}"
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="include runs on other splits")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--grep", help="substring filter on run_id/features/model/notes")
    ap.add_argument("--vs", metavar="RUN", help="paired per-fold comparison against this run")
    args = ap.parse_args()

    idx = tracking.read_index()
    if idx.empty:
        print("no runs yet")
        return
    if not args.all or args.vs:
        split = splits.label(config.load("base.yaml")["split"])
        n_before = len(idx)
        idx = tracking.comparable(idx)
        hidden = n_before - len(idx)
        print(f"split = {split}, training data = {data.current_fingerprint()}"
              + (f"  ({hidden} runs on other splits/data hidden; --all shows them)" if hidden and not args.vs else "")
              + "\n")
    if args.grep:
        hay = idx[["run_id", "features", "model", "notes"]].astype(str).agg(" ".join, axis=1)
        idx = idx[hay.str.contains(args.grep, case=False)]
    full = idx
    idx = idx.sort_values("mean_scc", ascending=False).head(args.top).copy()
    if args.vs:  # --top must never drop the reference run itself
        ref_id = tracking.find_run(args.vs).name
        if ref_id not in set(idx["run_id"]) and ref_id in set(full["run_id"]):
            idx = pd.concat([idx, full[full["run_id"] == ref_id]])
    idx["notes"] = idx["notes"].fillna("").astype(str).str.slice(0, 50)
    fmt = lambda v: f"{v:.4f}"  # noqa: E731

    if args.vs:
        ref = tracking.find_run(args.vs).name
        if ref not in set(idx["run_id"]):
            raise SystemExit(f"{ref} is not on the current split; --vs needs runs that share folds")
        p = paired(idx, ref)
        extra = [c for c in ("val_mean_scc", "val_unseen_mean_scc") if c in idx]
        tl = [c for c in ("tl_mean_scc",) if c in idx]
        tl_d = [c for c in ("d_tl_mean_scc",) if c in p]
        out = idx[["run_id", "mean_scc", *tl, *extra, "fit_seconds"]].merge(p, on="run_id")
        clear_win = (out["d_mean_scc"] > 2 * out["se"]) & (out["n_win"] == out["n_folds"])
        clear_loss = (out["d_mean_scc"] < -2 * out["se"]) & (out["n_win"] == 0)
        out["verdict"] = np.select([out["run_id"] == ref, clear_win, clear_loss],
                                   ["reference", "better", "worse"], default="unclear")
        print(f"paired vs {ref} (d_ = mean per-fold difference; se = its standard error)\n")
        with pd.option_context("display.width", 250):
            print(out[["run_id", "mean_scc", "d_mean_scc", "se", "wins", "verdict",
                       "d_kcat_scc", "d_km_scc", *tl, *tl_d, *extra, "fit_seconds"]].to_string(index=False, float_format=fmt))
        if extra:
            print("\nval_* = instructor's labelled validation set (21 Oct labels; pairs already in train dropped). "
                  "Runs without it: python scripts/val_eval.py RUN")
        return

    with pd.option_context("display.width", 250, "display.max_colwidth", 50, "display.max_columns", 20):
        print(idx[[c for c in COLS if c in idx]].to_string(index=False, float_format=fmt))


if __name__ == "__main__":
    main()
