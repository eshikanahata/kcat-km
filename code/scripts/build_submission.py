"""Turn a run into a leaderboard-ready HF model repo, verify it, optionally upload.

    python scripts/build_submission.py best                        # build + local check
    python scripts/build_submission.py best --docker               # + exact grader docker run
    python scripts/build_submission.py best --docker --upload YOU/kcat-km-model

Steps
  1. refit the run's config on ALL training rows
  2. assemble submissions/<name>/ : predict.py, Dockerfile, requirements.txt,
     bundled kcatkm package, model/pipeline.joblib, neutral README.md
     (config + CV details go to submissions/<name>.json, which is NOT uploaded)
     plus two fallback models (model/fallback1.joblib, fallback2.joblib) that
     predict.py uses if the full model fails on the grader (see predict.py)
  3. local check: run predict.py on val_inputs in a separate process, validate
     the CSV exactly like the grader, report runtime (limit: 10 min on 2 vCPU)
  4. --docker: build the image and run it with the grader's flags
     (--network none --cpus 2 --memory 16g)
  5. --upload: push to a public HF model repo; the commit hash is printed and
     stored in notes/submissions.csv -- enter it as the revision on the leaderboard
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import pandas as pd
import yaml

import _bootstrap  # noqa: F401
from kcatkm import data, ledger, paths, splits, tracking
from kcatkm.pipeline import KcatKmPipeline

LIMIT_SECONDS = 600

# Fallbacks bundled into every submission (predict.py runs them first, cheapest first).
# They avoid torch entirely, so a torch / pretrained-model problem still yields a score.
FALLBACKS = {
    "fallback1": {  # ~ lgbm_handcrafted, CV ~0.38
        "features": ["seq_len", "aac", "dpc", "seq_physchem",
                     {"morgan": {"radius": 2, "n_bits": 1024, "counts": True}}, "rdkit_desc"],
        "model": {"name": "lgbm", "params": {"n_estimators": 1500, "learning_rate": 0.03, "num_leaves": 63,
                                             "min_child_samples": 20, "subsample": 0.8, "subsample_freq": 1,
                                             "colsample_bytree": 0.3, "reg_lambda": 1.0}},
    },
    "fallback2": {  # ~ ridge_starter, CV ~0.32; numpy + sklearn only
        "features": ["seq_len", "aac", "smiles_chars"],
        # alpha 12 (not 10) on purpose: it makes this stage's leaderboard score differ from the
        # earlier builds', so an exact 0.40950557 can only come from fallback1 (see notes, 26 Oct)
        "model": {"name": "ridge", "params": {"alpha": 12.0}},
    },
}


def dir_size(p):
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def final_stage(stderr: str) -> str:
    for line in reversed(stderr.splitlines()):
        if "final predictions from:" in line:
            return line.rsplit(":", 1)[1].strip()
    return "unknown"


EXPECTED_STAGE = "full"  # "full+second" when a second model is bundled (set in main)


def require_full(stderr: str, label: str):
    stage = final_stage(stderr)
    if stage != EXPECTED_STAGE:
        tail = "\n".join(stderr.splitlines()[-40:])
        raise SystemExit(f"\n[{label}] expected the final predictions from '{EXPECTED_STAGE}', got '{stage}'.\n"
                         f"A fallback or the time gate hid a problem; fix it before uploading. predict.py log:\n{tail}")
    print(f"  [{label}] final predictions came from the {EXPECTED_STAGE.replace('+', ' + ')} model")


def check_output(csv_path, inputs, label):
    pred = pd.read_csv(csv_path)
    data.check_predictions(pred, inputs)
    print(f"  [{label}] output valid: {len(pred)} rows, every input id exactly once, all finite")
    return pred


def esm_cost(pipe):
    """Relative ESM inference cost: transformer layers x hidden size^2 (summed over ESM extractors)."""
    from kcatkm.features.plm import pretrained_dir
    total = 0.0
    for ex in pipe.extractors:
        if ex.name != "esm":
            continue
        c = json.loads((pretrained_dir(ex.params["model"]) / "config.json").read_text())
        layers = ex.params.get("num_layers") or c["num_hidden_layers"]
        total += layers * c["hidden_size"] ** 2 * (0.6 if ex.params.get("quantize") else 1.0)
    return total


def esm_factor(first, second, margin=1.3):
    a, b = esm_cost(first), esm_cost(second)
    if a <= 0:
        return 5.0
    return round(max(1.0, b / a * margin), 2)


def assemble(sub_dir, pipe, run_dir, cfg, metrics, fallbacks, second=None):
    if sub_dir.exists():
        shutil.rmtree(sub_dir)
    shutil.copytree(paths.TEMPLATE, sub_dir)
    shutil.copytree(paths.ROOT / "src" / "kcatkm", sub_dir / "kcatkm",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy(paths.ROOT / "requirements-infer.txt", sub_dir / "requirements.txt")
    (sub_dir / "model").mkdir()
    pipe.save(sub_dir / "model" / "pipeline.joblib")
    for stage, fb in fallbacks.items():
        fb.save(sub_dir / "model" / f"{stage}.joblib")
    extractors = list(pipe.extractors)
    if second is not None:
        second["pipe"].save(sub_dir / "model" / "second.joblib")
        factor = esm_factor(pipe, second["pipe"])
        print(f"  second model's ESM should take ~{factor:.1f}x the first's (incl. 30% margin): used by the time gate")
        (sub_dir / "model" / "blend.json").write_text(json.dumps({"weights_second": second["weights"],
                                                                  "esm_factor": factor}, indent=2))
        extractors += list(second["pipe"].extractors)
    for ex in extractors:
        for asset in ex.assets():
            dest = sub_dir / "pretrained" / asset.name
            if not dest.exists():
                shutil.copytree(asset, dest, ignore=shutil.ignore_patterns(".cache"))
                print(f"  bundled {asset.name} ({dir_size(dest) / 2**20:.0f} MiB)")
    # Details stay LOCAL (next to the folder, not inside it): the HF repo must be public.
    meta = {"submission": sub_dir.name, "run_id": run_dir.name, "built": ledger.now(),
            "second": None if second is None else {"run_id": second["run_id"], "weights": second["weights"]},
            "git": tracking.git_commit(), "cv": metrics, "config": cfg, "pipeline_meta": pipe.meta}
    (sub_dir.parent / f"{sub_dir.name}.json").write_text(json.dumps(meta, indent=2, default=str))
    (sub_dir / "README.md").write_text(
        "---\ntags: [leak-curber, kcat, km]\nlicense: mit\n---\n"
        "# Leak CURBER kcat–Km submission\n\n"
        "Entry point: `python /app/predict.py --input INPUT --output OUTPUT`.\n")


GRADER_IMAGE = "python:3.12-slim-bookworm"  # must match the Dockerfile's base image
STUB = '''"""Entry point: python /app/predict.py --input INPUT --output OUTPUT"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _runner  # noqa: E402

if __name__ == "__main__":
    _runner.main()
'''


def _strip_docstrings(src: str) -> str:
    import ast
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            b = node.body
            if b and isinstance(b[0], ast.Expr) and isinstance(getattr(b[0], "value", None), ast.Constant) \
                    and isinstance(b[0].value.value, str):
                node.body = b[1:] or [ast.Pass()]
    return ast.unparse(tree)  # comments are dropped too


def compile_package(sub_dir):
    """Ship bytecode instead of source: predict.py becomes a 9-line stub, the inference logic
    (_runner) and the kcatkm package become .pyc files without docstrings or comments.

    The bytecode must match the grader image's Python (3.12), so it is compiled by
    $KCATKM_COMPILE_PYTHON if set, else a local python3.12, else inside the grader's docker image.
    This raises the effort needed to read the approach; it cannot make it impossible."""
    # the local checks ran this package with the local Python, which left __pycache__ files behind
    for d in list(sub_dir.rglob("__pycache__")):
        shutil.rmtree(d, ignore_errors=True)
    if (sub_dir / "predict.py").read_text() != STUB:  # (else: resuming an interrupted compile)
        (sub_dir / "predict.py").rename(sub_dir / "_runner.py")
        (sub_dir / "predict.py").write_text(STUB)
    sources = sorted(p for p in sub_dir.rglob("*.py") if p.name != "predict.py")
    if not sources:
        print("  already compiled")
        return
    for p in sources:
        p.write_text(_strip_docstrings(p.read_text()))
    targets = [str(p.relative_to(sub_dir)) for p in sources]
    py = os.environ.get("KCATKM_COMPILE_PYTHON") or shutil.which("python3.12")
    if py:
        cmd = [py, "-m", "py_compile", *targets]
        where = py
    else:
        uid = f"{os.getuid()}:{os.getgid()}" if hasattr(os, "getuid") else "0:0"
        cmd = ["docker", "run", "--rm", "--network", "none", "--user", uid, "-v", f"{sub_dir.resolve()}:/w",
               "-w", "/w", GRADER_IMAGE, "python", "-m", "py_compile", *targets]
        where = f"docker {GRADER_IMAGE}"
    # py_compile writes __pycache__/x.cpython-3XX.pyc; sourceless imports need x.pyc next to the module
    r = subprocess.run(cmd, cwd=sub_dir, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"compiling the package failed ({where}):\n{r.stderr[-3000:]}")
    for p in sources:
        cache = sorted((p.parent / "__pycache__").glob(f"{p.stem}.*.pyc"))
        if len(cache) != 1:
            raise SystemExit(f"expected one compiled file for {p}, found {cache}")
        cache[0].replace(p.with_suffix(".pyc"))
        p.unlink()
    for d in sub_dir.rglob("__pycache__"):
        shutil.rmtree(d, ignore_errors=True)
    di = sub_dir / ".dockerignore"
    di.write_text("\n".join(ln for ln in di.read_text().splitlines() if ln.strip() != "*.pyc") + "\n")
    left = [p.name for p in sub_dir.rglob("*.py") if p.name != "predict.py"]
    if left:
        raise SystemExit(f"source files left in the compiled package: {left}")
    print(f"  compiled the code to bytecode with {where}: {len(sources)} modules, source removed")


def local_check(sub_dir, out_csv, inputs):
    # Run like the grader: a clean environment (predict.py must set everything it needs itself),
    # only the submission folder, 2 threads, no network.
    keep = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "SYSTEMROOT")
    env = {k: v for k, v in os.environ.items() if k in keep}
    t0 = time.time()
    r = subprocess.run([sys.executable, "-I", str(sub_dir / "predict.py"), "--input", str(paths.VAL_INPUTS_FILE),
                        "--output", str(out_csv)], env=env, cwd="/", capture_output=True, text=True)
    dt = time.time() - t0
    if r.returncode != 0:
        raise SystemExit(f"[local] predict.py crashed:\n{r.stderr[-4000:]}")
    require_full(r.stderr, "local")
    flag = "OK" if dt < LIMIT_SECONDS * 0.5 else "WARNING: close to the 10-minute limit"
    print(f"  [local] inference on {len(inputs)} rows took {dt:.1f}s ({flag}; the grader has 2 vCPUs)")
    return check_output(out_csv, inputs, "local")


def docker_check(sub_dir, inputs):
    tag = f"kcatkm-{sub_dir.name.lower()}"
    out = sub_dir.parent / f"{sub_dir.name}__docker_output"
    out.mkdir(exist_ok=True)
    t0 = time.time()
    subprocess.run(["docker", "build", "--platform", "linux/amd64", "-t", tag, str(sub_dir)], check=True)
    print(f"  [docker] build took {time.time() - t0:.0f}s (limit 30 min)")
    out.chmod(0o777)  # the container may run as any user
    t0 = time.time()
    # Stricter than the organisers' sample command: read-only root filesystem (only /tmp
    # writable) and an arbitrary non-root user, in case the grader does either.
    r = subprocess.run(["docker", "run", "--rm", "--platform", "linux/amd64", "--network", "none",
                        "--cpus", "2", "--memory", "16g", "--read-only", "--tmpfs", "/tmp",
                        "--user", "12345:12345",
                        "-v", f"{paths.RAW}:/input:ro", "-v", f"{out}:/output", tag,
                        "python", "/app/predict.py", "--input", "/input/val_inputs.parquet",
                        "--output", "/output/predictions.csv"], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"[docker] predict.py crashed:\n{r.stderr[-4000:]}")
    require_full(r.stderr, "docker")
    print(f"  [docker] inference took {time.time() - t0:.1f}s (limit 600s)")
    check_output(out / "predictions.csv", inputs, "docker")


def upload(sub_dir, repo_id, public_name):
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
    if api.model_info(repo_id).private:
        raise SystemExit(f"{repo_id} is PRIVATE. Make it public on HF before submitting.")
    info = api.upload_folder(folder_path=str(sub_dir), repo_id=repo_id, repo_type="model",
                             commit_message=public_name,  # neutral: commit history is public too
                             delete_patterns="*",  # remove files left over from older submissions
                             ignore_patterns=["output/**", "predictions.csv", ".cache/**", "**/__pycache__/**"])
    return info.oid


def fit_second(run, first_dir, train, seed):
    """Refit a second run on all data and tune per-target blend weights on both runs' OOF."""
    from kcatkm import TARGETS, blending
    from kcatkm import metrics as M
    d2 = tracking.find_run(run)
    cfg2 = yaml.safe_load((d2 / "config.yaml").read_text())
    cfg1 = yaml.safe_load((first_dir / "config.yaml").read_text())
    if (cfg1.get("data") or {}).get("extra_train") != (cfg2.get("data") or {}).get("extra_train"):
        raise SystemExit("the two runs must use the same data.extra_train setting")
    o1 = pd.read_parquet(first_dir / "oof.parquet").set_index("id").sort_index()
    o2 = pd.read_parquet(d2 / "oof.parquet").set_index("id").sort_index()
    if not (o1.index.equals(o2.index) and (o1["fold"] == o2["fold"]).all()):
        raise SystemExit(f"{first_dir.name} and {d2.name} do not share rows/folds; cannot tune blend weights")
    Y = o1[list(TARGETS)].to_numpy()
    P1, P2 = (o[[f"pred_{t}" for t in TARGETS]].to_numpy() for o in (o1, o2))
    W = blending.tune(Y, [P1, P2])
    blended = M.score(Y, blending.apply(W, [P1, P2]))
    weights = {t: round(float(W[1, j]), 4) for j, t in enumerate(TARGETS)}
    print(f"  second model {d2.name}: blend weights {weights}; OOF mean_scc first {M.score(Y, P1)['mean_scc']:.4f}, "
          f"second {M.score(Y, P2)['mean_scc']:.4f}, blend {blended['mean_scc']:.4f}")
    t0 = time.time()
    from kcatkm import extra as extra_data
    train, groups = extra_data.training_frame(cfg2, train)  # + extra rows if that run used them
    pipe2 = KcatKmPipeline(cfg2["features"], cfg2["model"], seed=cfg2.get("seeds", [42])[0],
                           label_features=cfg2.get("label_features"), sample_weight=cfg2.get("sample_weight"))
    pipe2.fit(train, use_cache=True, groups=groups)
    print(f"  refit second model in {time.time() - t0:.0f}s")
    return {"pipe": pipe2, "weights": weights, "run_id": d2.name, "metrics": blended}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="run id / prefix / 'best' / 'last'")
    ap.add_argument("--name", help="local submission name (default: sNN_<run name>)")
    ap.add_argument("--public-name", help="name to type on the leaderboard / commit message (default: sub-NN)")
    ap.add_argument("--docker", action="store_true", help="also verify with the grader's docker command")
    ap.add_argument("--upload", metavar="USER/REPO", help="push to this public HF model repo")
    ap.add_argument("--notes", default="")
    ap.add_argument("--no-fallback", action="store_true", help="do not bundle fallback models")
    ap.add_argument("--skip-val", action="store_true", help="do not score the package on the labelled val set")
    ap.add_argument("--second", metavar="RUN",
                    help="also bundle this run's model (e.g. a bigger ESM); it runs at inference only if the "
                         "measured time allows, and is blended with per-target weights tuned on OOF")
    ap.add_argument("--compiled", action="store_true",
                    help="upload bytecode instead of readable source (needs --docker to verify it)")
    ap.add_argument("--probe", action="store_true",
                    help="diagnostic build: the leaderboard score reveals which stage ran (never nominate it)")
    args = ap.parse_args()

    if args.compiled and not (args.docker or os.environ.get("KCATKM_COMPILE_PYTHON")):
        raise SystemExit("--compiled needs --docker: the compiled package only runs on the grader's Python 3.12")
    run_dir = tracking.find_run(args.run)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())["overall"]
    if cfg.get("data", {}).get("subsample"):
        print("note: run used a subsample; the submission is refit on ALL training rows")
        cfg["data"]["subsample"] = None

    n_prev = len(ledger.read())
    name = args.name or f"s{n_prev + 1:02d}_{cfg['name']}"
    public_name = args.public_name or f"sub-{n_prev + 1:02d}"
    sub_dir = paths.SUBMISSIONS / name
    print(f"building {name} from {run_dir.name} (CV mean_scc {metrics['mean_scc']:.4f})")
    print(ledger.quota())

    t0 = time.time()
    from kcatkm import valset
    train, vdf_extra = valset.training_data(cfg)
    base_train = train
    from kcatkm import extra as extra_data
    if vdf_extra is not None:
        print(f"  training includes {len(vdf_extra)} labelled-val rows (data.extra_train)")
    seed = cfg.get("seeds", [42])[0]
    run_metrics = json.loads((run_dir / "metrics.json").read_text())
    per_fold = run_metrics.get("fit_seconds", 0) / max(len(run_metrics.get("folds", [])), 1)
    print(f"  [1/4] refitting on all {len(train)} rows (no output until done; "
          f"expect ~{per_fold * 1.3 / 60:.0f} min, a bit longer than one CV fold)...", flush=True)
    # inner folds (label features, cross-target models) use the same enzyme clusters as CV (cached)
    train, groups = extra_data.training_frame(cfg, train)
    if extra_data.wanted(cfg):
        print(f"  training includes {int(extra_data.is_extra(train['id']).sum())} extra rows (data.extra_data)")
    pipe = KcatKmPipeline(cfg["features"], cfg["model"], seed=seed, label_features=cfg.get("label_features"),
                          sample_weight=cfg.get("sample_weight"))
    pipe.fit(train, use_cache=True, groups=groups)
    print(f"  refit on {len(train)} rows in {time.time() - t0:.0f}s")
    second = None
    if args.second:
        second = fit_second(args.second, run_dir, base_train, seed)
        global EXPECTED_STAGE
        EXPECTED_STAGE = "full+second"
        metrics = second["metrics"]
    fallbacks = {}
    if not args.no_fallback:
        for stage, spec in FALLBACKS.items():
            t1 = time.time()
            fallbacks[stage] = KcatKmPipeline(spec["features"], spec["model"], seed=seed).fit(train, use_cache=True)
            print(f"  fitted {stage} in {time.time() - t1:.0f}s")
    assemble(sub_dir, pipe, run_dir, cfg, metrics, fallbacks, second)
    if args.probe:
        (sub_dir / "model" / "PROBE").write_text(
            "Diagnostic build. Leaderboard mean SCC reveals the stage that produced the predictions:\n"
            "  > 0.3            full model\n  about +0.13      fallback1 (kcat sign-flipped)\n"
            "  about -0.12      fallback2 (Km sign-flipped)\n  0                constant\n")
        print("  PROBE build: a fallback's predictions are sign-flipped on purpose; do NOT nominate this one")
    size = dir_size(sub_dir)
    print(f"  assembled {sub_dir.relative_to(paths.ROOT)} ({size / 2**20:.1f} MiB; limit 4 GiB)")

    inputs = data.load_val_inputs()
    print(f"  [2/4] local inference on {len(inputs)} val rows with 2 threads, as the grader runs it "
          "(includes computing all features from scratch, e.g. ESM)...", flush=True)
    pred = local_check(sub_dir, run_dir / "val_predictions.csv", inputs)
    print(f"  val prediction spread: kcat std {pred['log10_kcat'].std():.3f}, km std {pred['log10_km'].std():.3f}")
    val_note = ""
    if vdf_extra is not None:
        print("  [labelled val] skipped: those rows are part of this model's training data "
              "(see the run's out-of-fold val score instead)")
    elif valset.available() and not args.skip_val:
        s, stage, dt = valset.score_package(sub_dir, paths.SUBMISSIONS / f"{name}__labelled_val")
        print(f"  [labelled val] package scored end to end (stage: {stage}, {dt:.0f}s): {valset.fmt(s)}")
        val_note = f"labelled val {s['mean_scc']:.4f}. "
    if args.compiled:
        compile_package(sub_dir)
        if os.environ.get("KCATKM_COMPILE_PYTHON"):  # compiled for this interpreter (tests): rerun locally
            again = local_check(sub_dir, run_dir / "val_predictions_compiled.csv", inputs)
            if not again[["log10_kcat", "log10_km"]].round(4).equals(pred[["log10_kcat", "log10_km"]].round(4)):
                raise SystemExit("[compiled] predictions differ from the source build")
            print("  [compiled] same predictions as the source build")
    if args.docker:
        print("  [3/4] docker build + grader-style run (first build downloads torch; later builds are cached)...",
              flush=True)
        docker_check(sub_dir, inputs)
    if args.upload:
        print(f"  [4/4] uploading to {args.upload}...", flush=True)

    fields = dict(run_id=run_dir.name, cv_mean_scc=round(metrics["mean_scc"], 5),
                  cv_kcat_scc=round(metrics["kcat_scc"], 5), cv_km_scc=round(metrics["km_scc"], 5),
                  status="built", public_name=public_name,
                  notes=("PROBE build. " if args.probe else "")
                  + (f"second={second['run_id']} " if second else "") + ("compiled. " if args.compiled else "")
                  + val_note + args.notes)
    ledger.upsert(name, **fields)  # recorded before uploading, so a failed upload still leaves a row
    if args.upload:
        rev = upload(sub_dir, args.upload, public_name)
        fields.update(repo_id=args.upload, revision=rev, status="uploaded")
        print(f"\nuploaded to https://huggingface.co/{args.upload} at commit {rev}")
        print("leaderboard form ->  repository: "
              f"{args.upload}   revision: {rev}   model name: {public_name}")
        print(f"ONLY after pressing submit on the leaderboard:  python scripts/ledger.py submitted {name}")
        if args.probe:
            print("Reading the score: > 0.3 full model | about +0.13 fallback1 | about -0.12 fallback2 | 0 constant")
        ledger.upsert(name, **fields)


if __name__ == "__main__":
    main()
