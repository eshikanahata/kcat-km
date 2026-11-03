"""Score a blend of existing runs from their saved OOF predictions (no retraining).

    python scripts/blend.py RUN_A RUN_B RUN_C                  # equal weights
    python scripts/blend.py RUN_A RUN_B --weights 0.7 0.3
    python scripts/blend.py RUN_A RUN_B --rank                 # average ranks instead of values
    python scripts/blend.py RUN_A RUN_B --tune                 # best weights per target (grid)

All runs must share the same split and data (same ids/folds). --tune also prints
how many folds the tuned blend beats the best single run on: with one or two
weights per target the overfitting risk is small, but a gain that does not hold
on most folds is noise.
"""
import argparse
import itertools

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import _bootstrap  # noqa: F401
from kcatkm import TARGETS, metrics, tracking


def simplex(k, step):
    n = int(round(1 / step))
    for c in itertools.product(range(n + 1), repeat=k - 1):
        if sum(c) <= n:
            yield np.array([*c, n - sum(c)], dtype=float) / n


def fold_mean_scc(Y, P, folds):
    return np.array([metrics.score(Y[folds == f], P[folds == f])["mean_scc"] for f in np.unique(folds)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--weights", nargs="*", type=float)
    ap.add_argument("--rank", action="store_true")
    ap.add_argument("--tune", action="store_true", help="grid-search weights per target")
    ap.add_argument("--step", type=float, default=0.05)
    args = ap.parse_args()

    frames, names = [], []
    for r in args.runs:
        d = tracking.find_run(r)
        names.append(d.name)
        frames.append(pd.read_parquet(d / "oof.parquet").set_index("id").sort_index())
    base = frames[0]
    for f in frames[1:]:
        if not (f.index.equals(base.index) and (f["fold"] == base["fold"]).all()):
            raise SystemExit("runs do not share the same rows/folds; rerun them on one split and data")
    from kcatkm import valset
    keep = ~valset.is_val_row(base.index)  # labelled-val training rows are not CV rows
    frames = [f[keep] for f in frames]
    base = frames[0]
    Y = base[list(TARGETS)].to_numpy()
    folds = base["fold"].to_numpy()
    preds = [f[[f"pred_{t}" for t in TARGETS]].to_numpy() for f in frames]
    if args.rank:
        preds = [np.column_stack([rankdata(p[:, j]) / len(p) for j in range(p.shape[1])]) for p in preds]

    singles = []
    for n, p in zip(names, preds):
        s = metrics.score(Y, p)
        singles.append(s["mean_scc"])
        print(f"{n}: mean_scc={s['mean_scc']:.4f} kcat={s['kcat_scc']:.4f} km={s['km_scc']:.4f}")
    best = int(np.argmax(singles))

    if args.tune:
        if len(preds) > 4:
            raise SystemExit("--tune grid is meant for <= 4 runs")
        W = np.zeros((len(preds), len(TARGETS)))
        P = np.zeros_like(Y)
        for j, t in enumerate(TARGETS):
            stack = np.stack([p[:, j] for p in preds])
            best_w, best_s = None, -np.inf
            for w in simplex(len(preds), args.step):
                s = metrics.scc(Y[:, j], w @ stack)
                if s > best_s + 1e-9:
                    best_w, best_s = w, s
            W[:, j], P[:, j] = best_w, best_w @ stack
            print(f"  {t}: weights {', '.join(f'{n.split('_', 1)[1]}={x:.2f}' for n, x in zip(names, best_w))}"
                  f" -> {best_s:.4f}")
    else:
        w = np.array(args.weights or [1.0] * len(preds))
        w = w / w.sum()
        P = sum(wi * p for wi, p in zip(w, preds))

    s = metrics.score(Y, P)
    d = fold_mean_scc(Y, P, folds) - fold_mean_scc(Y, preds[best], folds)
    print(f"\nblend{' (rank)' if args.rank else ''}: mean_scc={s['mean_scc']:.4f} kcat={s['kcat_scc']:.4f} "
          f"km={s['km_scc']:.4f}  | vs best single ({names[best]}): {np.mean(d):+.4f} per fold, "
          f"better on {int((d > 0).sum())}/{len(d)} folds")


if __name__ == "__main__":
    main()
