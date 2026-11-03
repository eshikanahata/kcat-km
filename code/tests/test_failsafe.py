"""Fail-safe inference: every failure mode of the full model must still yield a valid file."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture(scope="module")
def submission(tmp_path_factory):
    import fake_data
    import test_esm
    tmp = tmp_path_factory.mktemp("proj")
    for name in ("src", "configs", "scripts", "submission_template", "requirements-infer.txt"):
        (tmp / name).symlink_to(ROOT / name)
    raw = tmp / "data" / "raw"
    raw.mkdir(parents=True)
    tr, va = fake_data.make(n_train=400, n_val=60)
    tr.to_parquet(raw / "train.parquet", index=False)
    va.to_parquet(raw / "val_inputs.parquet", index=False)
    test_esm.make_tiny_esm(tmp / "pretrained" / "esm_tiny")
    test_esm.make_tiny_roberta(tmp / "pretrained" / "chem_tiny")
    env = {**os.environ, "KCATKM_ROOT": str(tmp), "KCATKM_PRETRAINED": str(tmp / "pretrained")}

    def run(*a):
        r = subprocess.run([sys.executable, *a], env=env, cwd=tmp, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        return r.stdout + r.stderr

    run("scripts/run.py", "configs/ens_lgbm_mlp.yaml", "--set",
        "features=[seq_len, morgan, {esm: {model: x/esm_tiny}}, {chemberta: {model: x/chem_tiny}}]",
        "model.members.0.params.n_estimators=30", "model.members.1.params.epochs=3")
    out = run("scripts/build_submission.py", "ens_lgbm_mlp")
    assert "final predictions came from the full model" in out and "fitted fallback1" in out
    sub = next(p for p in (tmp / "submissions").iterdir() if p.is_dir())
    return sub, raw / "val_inputs.parquet", va


def predict(submission, tmp_path, **env_extra):
    sub, inp, va = submission
    out = tmp_path / "pred.csv"
    keep = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "LANG")}
    r = subprocess.run([sys.executable, "-I", str(sub / "predict.py"), "--input", str(inp), "--output", str(out)],
                       env={**keep, **env_extra}, cwd="/", capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr  # the process must ALWAYS exit cleanly
    pred = pd.read_csv(out)
    from kcatkm.data import check_predictions
    check_predictions(pred, va)
    stage = [ln for ln in r.stderr.splitlines() if "final predictions from:" in ln][-1].rsplit(":", 1)[1].strip()
    predict.last_log = r.stderr
    return stage, pred


def test_normal_run_runs_fallbacks_first_then_full(submission, tmp_path):
    assert predict(submission, tmp_path)[0] == "full"
    log = predict.last_log
    assert log.index("fallback2: ok") < log.index("fallback1: ok") < log.index("full: ok")


@pytest.mark.parametrize("env,expected", [
    ({"KCATKM_FAIL_STAGES": "full"}, "fallback1"),
    ({"KCATKM_KILL_STAGES": "full"}, "fallback1"),                       # hard crash in the child
    ({"KCATKM_FAIL_STAGES": "full,fallback1"}, "fallback2"),
    ({"KCATKM_FAIL_STAGES": "full,fallback1,fallback2"}, "constant"),
])
def test_failures_fall_back(submission, tmp_path, env, expected):
    assert predict(submission, tmp_path, **env)[0] == expected


def test_hang_is_cut_off_by_deadline(submission, tmp_path):
    import time
    t0 = time.time()
    # fallback1 already ran before the full model; a hanging full model is cut at the deadline
    stage, _ = predict(submission, tmp_path, KCATKM_HANG_STAGES="full", KCATKM_TIME_LIMIT="60")
    assert stage == "fallback1" and time.time() - t0 < 90


def test_full_model_reuses_fallback_features(submission, tmp_path):
    # with the parent's features handed over, the child must not need RDKit at all
    stage, _ = predict(submission, tmp_path)
    import re
    assert stage == "full" and "could not share features" not in predict.last_log
    m = re.search(r"full: reusing (\d+) feature blocks", predict.last_log)
    assert m and int(m.group(1)) >= 2, predict.last_log[-2000:]  # seq_len + morgan at least


def test_full_predictions_differ_from_fallback(submission, tmp_path):
    _, full = predict(submission, tmp_path)
    _, fb = predict(submission, tmp_path, KCATKM_FAIL_STAGES="full")
    assert not full.equals(fb)


def test_full_model_runs_when_tmp_is_not_writable(submission, tmp_path):
    # the grader's /tmp may not be writable: scratch must fall back to the output folder
    out_dir = tmp_path / "outdir"
    out_dir.mkdir()
    stage, _ = predict(submission, out_dir, KCATKM_SCRATCH_CANDIDATES="/nonexistent/a:/proc/no")
    assert stage == "full", predict.last_log[-2000:]
    assert f"scratch directory: {out_dir}" in predict.last_log
    assert sorted(p.name for p in out_dir.iterdir()) == ["pred.csv"]  # scratch cleaned up


def test_probe_marks_the_stage(submission, tmp_path):
    import shutil
    sub, inp, va = submission
    _, fb1 = predict(submission, tmp_path, KCATKM_FAIL_STAGES="full")
    _, ridge = predict(submission, tmp_path, KCATKM_FAIL_STAGES="full,fallback1")
    _, full = predict(submission, tmp_path)
    (sub / "model" / "PROBE").write_text("probe")
    try:
        _, p_fb1 = predict(submission, tmp_path, KCATKM_FAIL_STAGES="full")
        close = lambda a, b: np.allclose(a, b, atol=1e-9)  # noqa: E731  (CSV round trips are not bit-exact)
        assert close(p_fb1["log10_kcat"], -fb1["log10_kcat"]) and close(p_fb1["log10_km"], fb1["log10_km"])
        _, p_ridge = predict(submission, tmp_path, KCATKM_FAIL_STAGES="full,fallback1")
        assert close(p_ridge["log10_km"], -ridge["log10_km"]) and close(p_ridge["log10_kcat"], ridge["log10_kcat"])
        _, p_full = predict(submission, tmp_path)
        pd.testing.assert_frame_equal(p_full, full, rtol=1e-5, atol=1e-6)  # never altered (thread scheduling can move the last digits)
    finally:
        (sub / "model" / "PROBE").unlink()
