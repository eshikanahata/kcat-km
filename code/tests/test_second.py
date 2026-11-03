"""Gated second model: runs and blends when time allows, never makes things worse when it doesn't."""
import json
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
    tmp = tmp_path_factory.mktemp("proj2")
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

    common = ["model.members.0.params.n_estimators=30", "model.members.1.params.epochs=3",
              "label_features.emb_prefix=esm_tiny_mean_"]
    run("scripts/run.py", "configs/ens_lf_esm8.yaml", "--name", "first", "--set",
        "features=[seq_len, morgan, {esm: {model: x/esm_tiny, window: 50}}, {chemberta: {model: x/chem_tiny}}]",
        *common)
    run("scripts/run.py", "configs/ens_lf.yaml", "--name", "second", "--set",
        "features=[seq_len, morgan, {esm: {model: x/esm_tiny}}, {chemberta: {model: x/chem_tiny}}]", *common)
    out = run("scripts/build_submission.py", "first", "--second", "second")
    assert "blend weights" in out and "final predictions came from the full + second model" in out
    sub = next(p for p in (tmp / "submissions").iterdir() if p.is_dir())
    assert (sub / "model" / "second.joblib").exists() and (sub / "model" / "blend.json").exists()
    return sub, raw / "val_inputs.parquet", va


def predict(submission, tmp_path, **env_extra):
    sub, inp, va = submission
    out = tmp_path / "pred.csv"
    keep = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "LANG")}
    r = subprocess.run([sys.executable, "-I", str(sub / "predict.py"), "--input", str(inp), "--output", str(out)],
                       env={**keep, **env_extra}, cwd="/", capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    from kcatkm.data import check_predictions
    pred = pd.read_csv(out)
    check_predictions(pred, va)
    stage = [ln for ln in r.stderr.splitlines() if "final predictions from:" in ln][-1].rsplit(":", 1)[1].strip()
    predict.last_log = r.stderr
    return stage, pred


def test_blends_when_time_allows(submission, tmp_path):
    stage, blended = predict(submission, tmp_path)
    assert stage == "full+second", predict.last_log[-2000:]
    assert "second: reusing" in predict.last_log  # features from earlier stages are handed over
    _, first_only = predict(submission, tmp_path, KCATKM_SECOND_FACTOR="1e9")
    assert not np.allclose(blended[["log10_kcat", "log10_km"]], first_only[["log10_kcat", "log10_km"]])


@pytest.mark.parametrize("env", [
    {"KCATKM_SECOND_FACTOR": "1e9"},          # time gate says it will not fit
    {"KCATKM_FAIL_STAGES": "second"},          # exception in the second model
    {"KCATKM_KILL_STAGES": "second"},          # hard crash in its child process
])
def test_first_model_stands_when_second_cannot_run(submission, tmp_path, env):
    stage, pred = predict(submission, tmp_path, **env)
    assert stage == "full", predict.last_log[-2000:]


def test_second_skipped_if_first_failed(submission, tmp_path):
    stage, _ = predict(submission, tmp_path, KCATKM_FAIL_STAGES="full")
    assert stage == "fallback1" and "second: skipped (the first full model did not finish)" in predict.last_log


def test_blend_uses_the_tuned_weights(submission, tmp_path):
    sub, _, _ = submission
    w = json.loads((sub / "model" / "blend.json").read_text())["weights_second"]
    assert set(w) == {"log10_kcat", "log10_km"} and all(0 <= v <= 1 for v in w.values())


def test_truncated_second_model_is_not_blended(submission, tmp_path):
    # an impossible ESM budget forces the time guard to shorten sequences in both models;
    # the first model's (shortened) predictions are still used, the second's are not blended in
    stage, _ = predict(submission, tmp_path, KCATKM_ESM_BUDGET="1e-9", KCATKM_BATCH_TOKENS="300",
                       KCATKM_SECOND_FACTOR="0")
    assert stage == "full", predict.last_log[-2000:]
    assert "second: its ESM had to shorten sequences" in predict.last_log


def test_time_gate_uses_the_factor_from_the_build(submission, tmp_path):
    import json
    sub, _, _ = submission
    f = sub / "model" / "blend.json"
    meta = json.loads(f.read_text())
    assert meta["esm_factor"] >= 1.0  # same tiny ESM in both models: about 1.3 (cost ratio 1 x margin)
    try:
        f.write_text(json.dumps({**meta, "esm_factor": 1e9}))
        stage, _ = predict(submission, tmp_path)
        assert stage == "full" and "second: skipped, estimated" in predict.last_log
    finally:
        f.write_text(json.dumps(meta))


def test_esm_factor_from_model_sizes():
    sys.path.insert(0, str(ROOT / "scripts"))
    import types
    import build_submission as b
    from unittest import mock

    def pipe(layers):
        ex = types.SimpleNamespace(name="esm", params={"model": "facebook/esm2_t12_35M_UR50D", "num_layers": layers})
        return types.SimpleNamespace(extractors=[ex])
    cfgs = {"esm2_t12_35M_UR50D": {"num_hidden_layers": 12, "hidden_size": 480}}
    with mock.patch.object(b, "esm_cost", side_effect=lambda p: sum(
            (x.params["num_layers"] or 12) * 480 ** 2 for x in p.extractors)):
        assert b.esm_factor(pipe(6), pipe(12)) == 2.6  # 2x the layers, + 30% margin
    assert cfgs  # (the real esm_cost reads pretrained/<model>/config.json)
