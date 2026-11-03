"""Merge ensemble runs that share features, split and data into ONE run, without retraining.

    python scripts/merge_runs.py ens3_lf_l8_et ens4_lf_l8 --name ens5_lf_l8
    python scripts/tune_weights.py ens5_lf_l8

Members that appear in several runs with the same settings and seed (e.g. the LightGBM and
MLP every ensemble has) are kept once. The merged config gives each member its original
seed (seed_offset), so build_submission.py refits exactly the models whose out-of-fold
predictions the weights were tuned on. Members tuned to weight 0 are skipped at refit.
"""
import argparse
import copy
import json

import numpy as np
import pandas as pd
import yaml

import _bootstrap  # noqa: F401
from kcatkm import TARGETS, config, metrics, splits, tracking, valset


def strip(cfg):
    c = copy.deepcopy(cfg)
    for k in ("name", "notes", "model", "seeds"):
        c.pop(k, None)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="untuned ensemble runs (their member predictions are used)")
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    dirs = [tracking.find_run(r) for r in args.runs]
    cfgs = [yaml.safe_load((d / "config.yaml").read_text()) for d in dirs]
    oofs = [pd.read_parquet(d / "oof.parquet") for d in dirs]
    for d, c, o in zip(dirs, cfgs, oofs):
        if c["model"]["name"] != "ensemble":
            raise SystemExit(f"{d.name} is not an ensemble run")
        if strip(c) != strip(cfgs[0]):
            raise SystemExit(f"{d.name} differs from {dirs[0].name} in features/split/data/label features")
        if c.get("seeds", [42]) != cfgs[0].get("seeds", [42]):
            raise SystemExit("runs use different seeds")
        if not (o["id"].equals(oofs[0]["id"]) and o["fold"].equals(oofs[0]["fold"])):
            raise SystemExit(f"{d.name} has different rows or folds")

    members, cols = [], []
    for d, c, o in zip(dirs, cfgs, oofs):
        for i, m in enumerate(c["model"]["members"]):
            spec = {k: v for k, v in m.items() if k != "weight"}
            spec["seed_offset"] = int(m.get("seed_offset", i))
            pred = np.column_stack([o[f"member{i}_pred_{t}"].to_numpy() for t in TARGETS])
            dup = next((j for j, s in enumerate(members) if s == spec), None)
            if dup is not None:
                if not np.allclose(pred, cols[dup], atol=1e-3):
                    print(f"  note: {d.name} member {i} ({m['name']}) has the same settings but different "
                          "predictions; keeping the first")
                continue
            members.append(spec)
            cols.append(pred)
            print(f"  member {len(members) - 1}: {m['name']} (from {d.name})")
    if len(members) > 5:
        raise SystemExit(f"{len(members)} members; tune_weights handles at most 5")

    new = copy.deepcopy(cfgs[0])
    new["name"] = args.name
    new["notes"] = "merged from " + ", ".join(d.name for d in dirs) + " (members not retrained)"
    new["model"] = {"name": "ensemble", "members": [{**m, "weight": 1.0} for m in members]}

    base = oofs[0]
    out = base[["id", "fold", "group", *TARGETS]].copy()
    P = np.mean(cols, axis=0)
    for i, c in enumerate(cols):
        for j, t in enumerate(TARGETS):
            out[f"member{i}_pred_{t}"] = c[:, j]
    for j, t in enumerate(TARGETS):
        out[f"pred_{t}"] = P[:, j]
    keep = ~valset.is_val_row(out["id"])
    Y = out.loc[keep, list(TARGETS)].to_numpy()
    folds = []
    for f in sorted(out["fold"].unique()):
        m = keep & (out["fold"] == f).to_numpy()
        folds.append(metrics.score(out.loc[m, list(TARGETS)].to_numpy(), P[m]))
    overall = metrics.score(Y, P[keep])
    overall["mean_scc_fold_std"] = float(pd.DataFrame(folds)["mean_scc"].std())
    print(f"equal-weight merge: mean_scc={overall['mean_scc']:.4f} kcat={overall['kcat_scc']:.4f} "
          f"km={overall['km_scc']:.4f}")

    idx = tracking.read_index().set_index("run_id")
    src = idx.loc[dirs[0].name]
    run_id, rd = tracking.new_run_dir(new["name"])
    tracking.save_yaml(new, rd / "config.yaml")
    out.to_parquet(rd / "oof.parquet", index=False)
    fit_s = float(sum(idx.loc[d.name, "fit_seconds"] for d in dirs))
    tracking.save_json({"overall": overall, "folds": folds, "fingerprint": config.fingerprint(new),
                        "derived_from": [d.name for d in dirs], "n_rows": len(out),
                        "n_features": src["n_features"], "fit_seconds": fit_s}, rd / "metrics.json")
    tracking.append_index({
        "run_id": run_id, "name": new["name"], "fingerprint": config.fingerprint(new),
        "git": tracking.git_commit(), "split": splits.label(new["split"]), "data": src.get("data"),
        "features": src["features"], "model": "ensemble", "n_features": src["n_features"],
        "seeds": src["seeds"], **{k: round(v, 5) for k, v in overall.items()},
        "extra_train": src.get("extra_train", ""), "fit_seconds": round(fit_s, 1), "notes": new["notes"],
    })
    print(f"saved runs/{run_id}  ->  python scripts/tune_weights.py {new['name']}")


if __name__ == "__main__":
    main()
