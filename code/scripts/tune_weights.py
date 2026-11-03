"""Fit per-target ensemble weights from an ensemble run's saved member OOFs (no retraining).

    python scripts/tune_weights.py ens_lgbm_mlp
    python scripts/tune_weights.py ens_lgbm_mlp --step 0.05 --name ens_lgbm_mlp_tuned

Searches the weights that maximise each target's overall OOF Spearman, then
records the result as a NEW run (config with the tuned weights + re-weighted
OOF predictions), so compare.py / analyze.py / build_submission.py work on it
directly. The members are not retrained: their OOF predictions are reused.

Overfitting risk is tiny (one weight per target per extra member, fitted on
~37k OOF rows), but the paired per-fold table shows whether the gain holds on
every fold rather than only overall.
"""
import argparse
import copy
import itertools
import json

import numpy as np
import pandas as pd
import yaml

import _bootstrap  # noqa: F401
from kcatkm import TARGETS, config, data, metrics, paths, splits, tracking


def simplex_grid(k, step):
    n = int(round(1 / step))
    for combo in itertools.product(range(n + 1), repeat=k - 1):
        if sum(combo) <= n:
            yield np.array([*combo, n - sum(combo)], dtype=float) / n


def best_weights(P, y, step):
    """P: (k, n) member predictions for one target. Exhaustive simplex grid (fine for k <= 5)."""
    best_w, best_s = None, -np.inf
    for w in simplex_grid(P.shape[0], step):
        s = metrics.scc(y, w @ P)
        if s > best_s + 1e-9:
            best_w, best_s = w, s
    return best_w, best_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="an ensemble run (id / unique part of its name)")
    ap.add_argument("--step", type=float, default=0.05, help="weight grid resolution")
    ap.add_argument("--name", help="name of the new run (default: <run name>_tuned)")
    args = ap.parse_args()

    src = tracking.find_run(args.run)
    cfg = yaml.safe_load((src / "config.yaml").read_text())
    if cfg["model"]["name"] != "ensemble":
        raise SystemExit(f"{src.name} is not an ensemble run")
    oof = pd.read_parquet(src / "oof.parquet")
    k = len(cfg["model"]["members"])
    if not all(f"member{i}_pred_{TARGETS[0]}" in oof for i in range(k)):
        raise SystemExit("this run has no saved member predictions; rerun it with the current code")
    if k > 5:
        raise SystemExit("grid search is meant for <= 5 members")
    if k == 5:
        print("5 members: the weight grid takes a few minutes")

    from kcatkm import valset
    lv = valset.is_val_row(oof["id"])  # labelled-val rows used for training: scored separately
    full_oof, oof = oof, oof[~lv].reset_index(drop=True)
    Y = oof[list(TARGETS)].to_numpy()
    names = [f"{i}:{m['name']}" for i, m in enumerate(cfg["model"]["members"])]
    W = np.zeros((k, len(TARGETS)))
    blended = np.zeros_like(Y)
    print(f"tuning {src.name} ({', '.join(names)}), grid step {args.step}\n")
    for j, t in enumerate(TARGETS):
        P = np.stack([oof[f"member{i}_pred_{t}"].to_numpy() for i in range(k)])
        w, s = best_weights(P, Y[:, j], args.step)
        W[:, j] = w
        blended[:, j] = w @ P
        alone = "  ".join(f"{n}={metrics.scc(Y[:, j], P[i]):.4f}" for i, n in enumerate(names))
        print(f"{t}: members alone {alone} | current blend {metrics.scc(Y[:, j], oof[f'pred_{t}']):.4f}"
              f" -> tuned {s:.4f} with weights " + ", ".join(f"{n}={x:.2f}" for n, x in zip(names, w)) + "")

    # new config + run
    new = copy.deepcopy(cfg)
    new["name"] = args.name or f"{cfg['name']}_tuned"
    for i, m in enumerate(new["model"]["members"]):
        m["weight"] = {t: round(float(W[i, j]), 4) for j, t in enumerate(TARGETS)}
    new["notes"] = f"per-target weights fitted on OOF of {src.name} (members not retrained)"

    fold_scores = []
    for f in sorted(oof["fold"].unique()):
        m = (oof["fold"] == f).to_numpy()
        fold_scores.append(metrics.score(Y[m], blended[m]))
    overall = metrics.score(Y, blended)
    overall["mean_scc_fold_std"] = float(pd.DataFrame(fold_scores)["mean_scc"].std())

    old_folds = pd.DataFrame(json.loads((src / "metrics.json").read_text())["folds"])
    d = pd.DataFrame(fold_scores)["mean_scc"] - old_folds["mean_scc"]
    print(f"\nmean_scc {metrics.score(Y, oof[[f'pred_{t}' for t in TARGETS]].to_numpy())['mean_scc']:.4f}"
          f" -> {overall['mean_scc']:.4f}; per fold delta {np.round(d.to_numpy(), 4).tolist()}"
          f" ({int((d > 0).sum())}/{len(d)} folds better)")

    run_id, rd = tracking.new_run_dir(new["name"])
    tracking.save_yaml(new, rd / "config.yaml")
    src_metrics = json.loads((src / "metrics.json").read_text())
    tracking.save_json({"overall": overall, "folds": fold_scores, "fingerprint": config.fingerprint(new),
                        "derived_from": src.name, "n_rows": len(oof),
                        "n_features": src_metrics.get("n_features")}, rd / "metrics.json")
    out = full_oof.copy()
    for j, t in enumerate(TARGETS):
        P_all = np.stack([full_oof[f"member{i}_pred_{t}"].to_numpy() for i in range(k)])
        out[f"pred_{t}"] = W[:, j] @ P_all
    out.to_parquet(rd / "oof.parquet", index=False)
    val_cols = {}
    if lv.any():
        vdf, _ = valset.load(train=data.load_train())
        vpred = out.set_index("id").loc[valset.PREFIX + vdf["id"].astype(str)]
        vs = valset.score(vdf, vpred[[f"pred_{t}" for t in TARGETS]].to_numpy())
        print(f"labelled val (out-of-fold): {valset.fmt(vs)}")
        val_cols = {"val_mean_scc": round(vs["mean_scc"], 5), "val_kcat_scc": round(vs["kcat_scc"], 5),
                    "val_km_scc": round(vs["km_scc"], 5),
                    "val_unseen_mean_scc": round(vs["unseen_enzyme_mean_scc"], 5)}

    src_row = tracking.read_index().set_index("run_id").loc[src.name]
    tracking.append_index({
        "run_id": run_id, "name": new["name"], "fingerprint": config.fingerprint(new),
        "git": tracking.git_commit(), "split": splits.label(new["split"]), "data": src_row.get("data"),
        "features": src_row["features"], "model": "ensemble", "n_features": src_row["n_features"],
        "seeds": src_row["seeds"], **{kk: round(v, 5) for kk, v in overall.items()},
        "extra_train": src_row.get("extra_train", ""), **val_cols,
        "fit_seconds": src_row["fit_seconds"], "notes": new["notes"],
    })
    print(f"saved runs/{run_id}  ->  python scripts/build_submission.py {new['name']}")


if __name__ == "__main__":
    main()
