"""Score runs or built submissions on the instructor's labelled validation set (data/labels/).

    python scripts/val_eval.py RUN [RUN ...]          # refit each run's config on ALL training rows, score
    python scripts/val_eval.py --submission s08_...   # run the built package end to end (fallbacks,
                                                      # time gate, blend) exactly as the grader would

New runs are scored automatically by run.py; this is for older runs and for packages.
Labels are the 21 Oct version: compare models with it, but do not expect the exact
leaderboard number (the grader now uses the 23 Oct labels).
"""
import argparse
import json

import pandas as pd
import yaml

import _bootstrap  # noqa: F401
from kcatkm import TARGETS, data, paths, splits, tracking, valset
from kcatkm.pipeline import KcatKmPipeline


def eval_run(run):
    d = tracking.find_run(run)
    cfg = yaml.safe_load((d / "config.yaml").read_text())
    if (cfg.get("data") or {}).get("extra_train"):
        print(f"{d.name}: trained on the labelled val rows; its out-of-fold val score is already in the index")
        return
    train = data.load_train()
    vdf, info = valset.load(train=train, verbose=True)
    from kcatkm import extra as extra_data
    train, groups = extra_data.training_frame(cfg, train)
    import time
    m = json.loads((d / "metrics.json").read_text())
    per_fold = m.get("fit_seconds", 0) / max(len(m.get("folds", [])), 1)
    print(f"{d.name}: refitting on all {len(train)} rows (no output until done; ~{per_fold * 1.3 / 60:.0f} min)...",
          flush=True)
    t0 = time.time()
    pipe = KcatKmPipeline(cfg["features"], cfg["model"], seed=cfg.get("seeds", [42])[0],
                          label_features=cfg.get("label_features"),
                          sample_weight=cfg.get("sample_weight"))
    pipe.fit(train, use_cache=True, groups=groups)
    print(f"  refit done in {time.time() - t0:.0f}s; computing features for the labelled rows "
          "(new enzymes need ESM embeddings)...", flush=True)
    P = pipe.predict_x(pipe.features(vdf, use_cache=True), KcatKmPipeline.make_meta(vdf))
    s = valset.score(vdf, P)
    print(f"{d.name}: {valset.fmt(s)}", flush=True)
    m["labelled_val"] = s
    (d / "metrics.json").write_text(json.dumps(m, indent=2, default=str))
    idx = tracking.read_index()
    row = idx["run_id"] == d.name
    for col, key in [("val_mean_scc", "mean_scc"), ("val_kcat_scc", "kcat_scc"), ("val_km_scc", "km_scc"),
                     ("val_unseen_mean_scc", "unseen_enzyme_mean_scc")]:
        idx.loc[row, col] = round(s[key], 5)
    idx.to_csv(paths.RUNS_INDEX, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--submission", help="name of a built package in submissions/")
    args = ap.parse_args()
    if not valset.available():
        raise SystemExit(f"put val_labels.csv and the matching val_inputs.parquet in {valset.LABELS}")
    for r in args.runs:
        eval_run(r)
    if args.submission:
        sub = paths.SUBMISSIONS / args.submission
        s, stage, dt = valset.score_package(sub, paths.SUBMISSIONS / f"{args.submission}__labelled_val")
        print(f"{args.submission} (stage: {stage}, {dt:.0f}s): {valset.fmt(s)}")


if __name__ == "__main__":
    main()
