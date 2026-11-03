"""Collect the course deliverables for ONE built submission: readable code, model weights and
the records that produced them.

    python scripts/make_deliverables.py s16_ens_lf_esm8_pub_tuned          # -> deliverables/<name>/
    python scripts/make_deliverables.py s16_ens_lf_esm8_pub_tuned --zip    # + one zip per part

deliverables/<name>/
  code/       the project's readable source (the uploaded package holds compiled bytecode only):
              src/kcatkm, scripts, configs, submission_template, tests, requirements-infer.txt
  weights/    exactly the files the submission predicts with: model/*.joblib (+ blend.json)
              and the pretrained encoders it bundles (ESM-2, ChemBERTa)
  records/    the submission's build record, both runs' config + metrics, the run index rows,
              the submission ledger, and the extra training-data file the runs used
  SUBMISSION.md   what this submission is and how to rebuild it
"""
import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import yaml

import _bootstrap  # noqa: F401
from kcatkm import paths, tracking

IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")


def copy(src: Path, dst: Path):
    if src.is_dir():
        shutil.copytree(src, dst, ignore=IGNORE, dirs_exist_ok=True)
    elif src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def size(p: Path) -> str:
    n = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return f"{n / 2**20:.1f} MiB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission", help="local submission name, e.g. s16_ens_lf_esm8_pub_tuned")
    ap.add_argument("--out", default="deliverables")
    ap.add_argument("--zip", action="store_true", help="also write code.zip, weights.zip, records.zip")
    args = ap.parse_args()

    sub = paths.SUBMISSIONS / args.submission
    meta_file = paths.SUBMISSIONS / f"{args.submission}.json"
    if not sub.is_dir() or not meta_file.exists():
        raise SystemExit(f"{sub} or {meta_file} not found: build the submission first (build_submission.py)")
    meta = json.loads(meta_file.read_text())
    out = paths.ROOT / args.out / args.submission
    if out.exists():
        shutil.rmtree(out)

    # ---- code (readable)
    code = out / "code"
    for name in ("src", "scripts", "configs", "submission_template", "tests", "requirements-infer.txt"):
        copy(paths.ROOT / name, code / name)

    # ---- weights (what the package predicts with)
    weights = out / "weights"
    copy(sub / "model", weights / "model")
    copy(sub / "pretrained", weights / "pretrained")

    # ---- records
    rec = out / "records"
    copy(meta_file, rec / meta_file.name)
    run_ids = [meta["run_id"]] + ([meta["second"]["run_id"]] if meta.get("second") else [])
    extra_files = set()
    for rid in run_ids:
        d = paths.RUNS / rid
        for f in ("config.yaml", "metrics.json"):
            copy(d / f, rec / "runs" / rid / f)
        cfg = yaml.safe_load((d / "config.yaml").read_text())
        data = cfg.get("data") or {}
        if data.get("extra_data"):
            extra_files.add(data.get("extra_file") or "extra_train.parquet")
    idx = tracking.read_index()
    if not idx.empty:
        idx[idx["run_id"].isin(run_ids)].to_csv(rec / "runs_index_rows.csv", index=False)
    copy(paths.LEDGER, rec / "submissions_ledger.csv")
    for f in sorted(extra_files):
        copy(paths.RAW / f, rec / "data" / f)

    # ---- summary
    second = meta.get("second") or {}
    lines = [
        f"# Submission {args.submission}", "",
        f"Built {meta.get('built')} (code commit {meta.get('git') or 'n/a'}).", "",
        "## Models",
        f"* first model: run `{meta['run_id']}` (config in records/runs/{meta['run_id']}/config.yaml)",
    ]
    if second:
        lines.append(f"* second model: run `{second['run_id']}`, blended with per-target weights on the second "
                     f"model {second.get('weights')}")
    lines += ["", "## Cross-validated score of the submitted blend (5-fold, enzyme-cluster split)", ""]
    cv = meta.get("cv") or {}
    for k in ("mean_scc", "kcat_scc", "km_scc", "mean_rmse"):
        if k in cv:
            lines.append(f"* {k}: {cv[k]:.4f}")
    if extra_files:
        lines += ["", "## Extra training data",
                  f"records/data/{', '.join(sorted(extra_files))}: rows used for TRAINING ONLY (never scored); "
                  "see scripts/import_extra.py and scripts/source_overlap.py for how they were built and checked."]
    lines += ["", "## Rebuild", "", "```bash",
              *(f"python scripts/run.py configs/{yaml.safe_load((paths.RUNS / r / 'config.yaml').read_text())['name']}"
                ".yaml --no-val  # (tuned runs: then python scripts/tune_weights.py <name>)" for r in run_ids),
              f"python scripts/build_submission.py {meta['run_id']}"
              + (f" --second {second['run_id']}" if second else "") + " --compiled --docker", "```", "",
              "Inference entry point: `python predict.py --input INPUT.parquet --output OUTPUT.csv` "
              "(code/submission_template/predict.py; staged, fail-safe, 10-minute budget on 2 CPUs).", ""]
    (out / "SUBMISSION.md").write_text("\n".join(lines))

    print(f"deliverables for {args.submission} in {out.relative_to(paths.ROOT)}/")
    for part in ("code", "weights", "records"):
        print(f"  {part:8s} {size(out / part)}")
        if args.zip:
            z = shutil.make_archive(str(out.parent / f"{args.submission}_{part}"), "zip", out, part)
            print(f"           -> {Path(z).relative_to(paths.ROOT)}")


if __name__ == "__main__":
    main()
