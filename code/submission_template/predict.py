"""Leaderboard entry point: python /app/predict.py --input INPUT --output OUTPUT

Fail-safe, staged inference. The grader reports only "inference failed" and the
test set is scored after the deadline, so this script makes sure a valid file
is always written:

  1. fallback2  ridge on lengths + composition       (pure numpy, ~1 s)
  2. fallback1  LightGBM on hand-crafted + RDKit     (no torch)
  3. full       the real model (ESM, ChemBERTa, ...) in a child process
  4. second     optional bigger-ESM model, run only if the measured time of stage 3 says it
                fits; its predictions are blended with stage 3's (model/blend.json)

The fallbacks run FIRST: the grader's CPUs turned out to be much slower than a
laptop, and a full model that runs out of time must leave the good fallback,
not the ridge one. The features fallback1 computed (RDKit etc.) are handed to
the full model's child through a file in /tmp, so they are not computed twice.
Inside the full model ESM is computed last and gets all the time that is left.

The fallbacks run in this process (no torch). The full model runs in a CHILD
process with a timeout, so even a hard crash there (segfault, out-of-memory
kill, illegal CPU instruction, hang) leaves the fallback predictions in place
and this process still exits cleanly. After every success the output file is
overwritten with the better predictions. A global deadline (default 540 s,
under the 600 s limit) is passed to the slow extractors, which stop early
rather than overrun. stderr ends with `final predictions from: <stage>`.
"""
import argparse
import os
import sys
import time
import traceback
from pathlib import Path

T0 = time.time()
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # find the bundled kcatkm package even under `python -I`

# Do not rely on the Dockerfile's ENV reaching this process: set everything here,
# before numpy/torch are imported. setdefault keeps explicit overrides working.
for key, value in {
    "KCATKM_PRETRAINED": str(HERE / "pretrained"),  # bundled weights
    "KCATKM_NO_CACHE": "1",                          # never write feature caches at inference
    "KCATKM_MEMO": "1",                              # share features between stages in memory
    "KCATKM_ESM_BUDGET": "420",                      # cap in seconds; the deadline usually binds first
    "KCATKM_BATCH_TOKENS": "4000",                   # smaller ESM batches: ~4x less peak memory
    "KCATKM_TIME_LIMIT": "540",                      # our own deadline, below the grader's 600 s
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
    "TOKENIZERS_PARALLELISM": "false", "RAYON_NUM_THREADS": "2",
}.items():
    os.environ.setdefault(key, value)
os.environ.setdefault("KCATKM_DEADLINE", str(T0 + float(os.environ["KCATKM_TIME_LIMIT"])))
DEADLINE = float(os.environ["KCATKM_DEADLINE"])  # a child process inherits its parent's deadline

STAGES = [("fallback2", "fallback2.joblib"), ("fallback1", "fallback1.joblib"), ("full", "pipeline.joblib"),
          ("second", "second.joblib")]
CHILD_STAGES = ("full", "second")  # torch-based stages run in a child process


def pick_scratch(output_path):
    """First writable directory for temporary files: $TMPDIR, /tmp, then the output folder.
    Do not assume /tmp is writable on the grader (KCATKM_SCRATCH_CANDIDATES overrides, for tests)."""
    import tempfile
    env = os.environ.get("KCATKM_SCRATCH_CANDIDATES")
    candidates = env.split(":") if env is not None else [os.environ.get("TMPDIR"), "/tmp"]
    candidates.append(os.path.dirname(os.path.abspath(output_path)))
    for base in filter(None, candidates):
        try:
            d = tempfile.mkdtemp(prefix=".kcatkm_", dir=base)
            with open(os.path.join(d, "probe"), "w") as f:
                f.write("ok")
            os.remove(os.path.join(d, "probe"))
            return d
        except Exception:
            continue
    raise RuntimeError("no writable scratch directory")


def log(msg):
    print(f"[predict {time.time() - T0:6.1f}s] {msg}", file=sys.stderr, flush=True)


def write(frame, preds, path):
    import numpy as np
    import pandas as pd
    from kcatkm import TARGETS
    from kcatkm.data import check_predictions
    preds = np.clip(np.nan_to_num(np.asarray(preds, dtype=float), nan=0.0, posinf=15, neginf=-15), -1e6, 1e6)
    out = pd.DataFrame(preds, columns=list(TARGETS))
    out.insert(0, "id", frame["id"].to_numpy())
    check_predictions(out, frame)
    tmp = Path(str(path) + ".tmp")
    out.to_csv(tmp, index=False)
    os.replace(tmp, path)  # atomic: the file is never half-written


def limit_torch_threads():
    try:
        import torch
        torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def run_stage(stage, fname, inputs, frame, output):
    """Run one stage in this process. Raises on any failure."""
    def hooked(var):  # test hooks; never set on the grader
        return stage in set(filter(None, os.environ.get(var, "").split(",")))
    if hooked("KCATKM_FAIL_STAGES"):
        raise RuntimeError(f"forced failure of {stage} (KCATKM_FAIL_STAGES)")
    if hooked("KCATKM_KILL_STAGES"):   # simulates a segfault / out-of-memory kill
        os.kill(os.getpid(), 9)
    if hooked("KCATKM_HANG_STAGES"):   # simulates a hang / far too slow model
        time.sleep(10**6)
    from kcatkm.features import base
    if stage in CHILD_STAGES:
        limit_torch_threads()
        memo = os.environ.get("KCATKM_MEMO_FILE")
        if memo and os.path.exists(memo):  # reuse features computed earlier (RDKit, Morgan, ChemBERTa, ...)
            import pickle
            with open(memo, "rb") as f:
                base._MEMO.update(pickle.load(f))
            log(f"{stage}: reusing {len(base._MEMO)} feature blocks computed earlier")
    t0 = time.time()
    from kcatkm.pipeline import KcatKmPipeline
    pipe = KcatKmPipeline.load(HERE / "model" / fname)
    write(frame, pipe.predict(inputs, use_cache=False), output)
    if stage in CHILD_STAGES:  # report back to the parent: timings, truncation, computed features
        import json
        import pickle
        from kcatkm.features import plm
        status = {"seconds": time.time() - t0, "timings": base.TIMINGS,
                  "truncated": plm.STATE["truncated"]}
        with open(output + ".status.json", "w") as f:
            json.dump(status, f)
        with open(output + ".memo.pkl", "wb") as f:
            pickle.dump(base._MEMO, f, protocol=pickle.HIGHEST_PROTOCOL)


def run_child(stage, args, scratch, frame):
    """Run a heavy stage in a child process (a crash or hang there cannot take this process down).
    Returns (predictions array, status dict)."""
    import json
    import pickle
    import subprocess
    import pandas as pd
    from kcatkm.features import base
    if scratch is None:
        raise RuntimeError(f"no scratch directory for the {stage} stage")
    tmp = os.path.join(scratch, f"{stage}_predictions.csv")
    memo_file = os.path.join(scratch, "memo.pkl")
    env = dict(os.environ)
    try:  # hand over every feature already computed (by the fallbacks or an earlier child)
        with open(memo_file, "wb") as f:
            pickle.dump(base._MEMO, f, protocol=pickle.HIGHEST_PROTOCOL)
        env["KCATKM_MEMO_FILE"] = memo_file
    except Exception:
        log(f"could not share features with the {stage} stage:\n{traceback.format_exc()}")
    # the entry script, not __file__: in a compiled build this code lives in _runner.pyc
    r = subprocess.run([sys.executable, "-I", str(HERE / "predict.py"), "--input", args.input,
                        "--output", tmp, "--stage", stage],
                       timeout=max(DEADLINE - time.time(), 1), env=env)
    if r.returncode != 0:
        raise RuntimeError(f"{stage} child process exited with code {r.returncode}")
    preds = pd.read_csv(tmp)[["log10_kcat", "log10_km"]].to_numpy()
    status = {}
    try:
        with open(tmp + ".status.json") as f:
            status = json.load(f)
        with open(tmp + ".memo.pkl", "rb") as f:  # keep the child's features for the next stage
            base._MEMO.update(pickle.load(f))
    except Exception:
        log(f"{stage}: no status from the child\n{traceback.format_exc()}")
    return preds, status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--stage", help=argparse.SUPPRESS)  # internal: child process runs one stage
    args = ap.parse_args()

    import pandas as pd
    from kcatkm.data import check_inputs
    frame = pd.read_parquet(args.input)
    check_inputs(frame)
    frame = frame.reset_index(drop=True)
    inputs = frame[["id", "smiles", "sequence"]]

    if args.stage:  # child mode
        run_stage(args.stage, dict(STAGES)[args.stage], inputs, frame, args.output)
        return

    log(f"{len(frame)} rows, {frame['sequence'].nunique()} unique sequences, {frame['smiles'].nunique()} unique SMILES")
    try:
        scratch = pick_scratch(args.output)
        # every library that writes temp/cache files uses the scratch dir (inherited by the child)
        os.environ.update(TMPDIR=scratch, HF_HOME=os.path.join(scratch, "hf"))
        log(f"scratch directory: {scratch}")
    except Exception:
        scratch = None
        log(f"no writable scratch directory; the full model will be skipped\n{traceback.format_exc()}")
    try:
        final = run_all(args, frame, inputs, scratch)
    finally:
        if scratch:
            import shutil
            shutil.rmtree(scratch, ignore_errors=True)

    if final is None:  # last resort: a valid file scores 0 instead of failing
        import numpy as np
        write(frame, np.zeros((len(frame), 2)), args.output)
        final = "constant"
    if (HERE / "model" / "PROBE").exists():
        probe_mark(final, frame, args.output)
    log(f"final predictions from: {final}")


def probe_mark(final, frame, output):
    """Diagnostic builds only (model/PROBE exists): make the stage readable from the leaderboard
    score. full: unchanged (score ~0.5); fallback1: kcat sign-flipped (score ~ +0.13);
    fallback2: Km sign-flipped (score ~ -0.12); constant: 0. Never nominate a probe build."""
    flip = {"fallback1": "log10_kcat", "fallback2": "log10_km"}.get(final)
    if flip is None:
        return
    import pandas as pd
    out = pd.read_csv(output)
    out[flip] = -out[flip]
    write(frame, out[["log10_kcat", "log10_km"]].to_numpy(), output)
    log(f"PROBE build: {flip} sign-flipped to mark stage {final}")


def second_stage_estimate(status):
    """Seconds the second (bigger-ESM) model should need, from the first model's measured times:
    everything except ESM and ChemBERTa is redone, ESM costs esm_factor times more (from
    model/blend.json, set at build time from the two ESM models' sizes; ChemBERTa and RDKit
    features are reused)."""
    t = status.get("timings", {})
    esm, chem = t.get("esm", 0.0), t.get("chemberta", 0.0)
    rest = max(status.get("seconds", 0.0) - esm - chem, 0.0)
    import json
    factor = os.environ.get("KCATKM_SECOND_FACTOR")
    if factor is None:  # written by build_submission.py from the two models' ESM sizes
        try:
            factor = json.loads((HERE / "model" / "blend.json").read_text()).get("esm_factor", 5.0)
        except Exception:
            factor = 5.0
    return float(factor) * esm + rest


def run_all(args, frame, inputs, scratch):
    import json
    import numpy as np
    final, full_preds, full_status = None, None, {}
    for stage, fname in STAGES:
        if not (HERE / "model" / fname).exists():
            continue
        left = DEADLINE - time.time()
        if left < 20:
            log(f"{stage}: skipped, out of time")
            break
        try:
            if stage == "full":
                full_preds, full_status = run_child("full", args, scratch, frame)
                write(frame, full_preds, args.output)
                log(f"full: ESM took {full_status.get('timings', {}).get('esm', float('nan')):.0f}s"
                    + (" (sequences shortened by the time guard)" if full_status.get("truncated") else ""))
            elif stage == "second":
                if final != "full":
                    log("second: skipped (the first full model did not finish)")
                    continue
                est = second_stage_estimate(full_status)
                if est > left - 30:
                    log(f"second: skipped, estimated {est:.0f}s but only {left:.0f}s left")
                    continue
                log(f"second: estimated {est:.0f}s, {left:.0f}s left")
                t_second = time.time()
                preds2, status2 = run_child("second", args, scratch, frame)
                log(f"second: took {time.time() - t_second:.0f}s (estimated {est:.0f}s)")
                if status2.get("truncated"):
                    log("second: its ESM had to shorten sequences; keeping the first model alone")
                    continue
                W = json.loads((HERE / "model" / "blend.json").read_text())["weights_second"]
                w = np.array([W["log10_kcat"], W["log10_km"]])
                write(frame, (1 - w) * full_preds + w * preds2, args.output)
                stage = "full+second"
            else:
                run_stage(stage, fname, inputs, frame, args.output)
            final = stage
            log(f"{stage}: ok (elapsed {time.time() - T0:.0f}s)")
        except Exception:
            log(f"{stage}: FAILED\n{traceback.format_exc()}")
    return final


if __name__ == "__main__":
    main()
