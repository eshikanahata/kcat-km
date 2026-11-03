"""End-to-end smoke test on synthetic data in a throwaway project root.

    pytest -q tests/
"""
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
import fake_data  # noqa: E402


def _setup(tmp: Path):
    for name in ("src", "configs", "scripts", "submission_template", "requirements-infer.txt"):
        (tmp / name).symlink_to(ROOT / name)
    raw = tmp / "data" / "raw"
    raw.mkdir(parents=True)
    tr, va = fake_data.make()
    tr.to_parquet(raw / "train.parquet", index=False)
    va.to_parquet(raw / "val_inputs.parquet", index=False)
    return {**os.environ, "KCATKM_ROOT": str(tmp)}


def _run(env, *args):
    r = subprocess.run([sys.executable, *args], env=env, cwd=env["KCATKM_ROOT"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout + r.stderr


def test_end_to_end(tmp_path):
    env = _setup(tmp_path)
    s = "scripts/"
    out = _run(env, s + "run.py", "configs/ridge_starter.yaml")
    assert "OOF mean_scc" in out
    out = _run(env, s + "run.py", "configs/debug.yaml", "--set", "data.subsample=null", "split.n_folds=5")
    assert "OOF mean_scc" in out
    # identical config is skipped, not recomputed
    assert "already run" in _run(env, s + "run.py", "configs/ridge_starter.yaml")
    # ...but NOT when the training data changed (e.g. the organisers refresh the dataset)
    raw = tmp_path / "data" / "raw" / "train.parquet"
    df = pd.read_parquet(raw)
    df.loc[0, "log10_km"] += 1.0
    df.to_parquet(raw, index=False)
    out = _run(env, s + "run.py", "configs/ridge_starter.yaml")
    assert "already run" not in out and "OOF mean_scc" in out
    df.loc[0, "log10_km"] -= 1.0
    df.to_parquet(raw, index=False)

    idx = pd.read_csv(tmp_path / "runs" / "index.csv")
    assert len(idx) == 3 and idx["mean_scc"].notna().all()  # 2 on the original data + 1 on the changed data

    _run(env, s + "compare.py")
    _run(env, s + "analyze.py", "best", "--vs", "ridge_starter")
    _run(env, s + "blend.py", idx["run_id"][0], idx["run_id"][1])
    _run(env, s + "split_check.py", "--thresholds", "0.5", "--folds-to-check", "1")

    out = _run(env, s + "build_submission.py", "best")
    assert "output valid" in out
    sub = next((tmp_path / "submissions").iterdir())
    for f in ("predict.py", "Dockerfile", "requirements.txt", "model/pipeline.joblib", "kcatkm/pipeline.py"):
        assert (sub / f).exists(), f
    _run(env, s + "ledger.py", "submitted", sub.name)
    out = _run(env, s + "ledger.py", "score", sub.name, "--mean", "0.3")
    assert "1/30 used since" in out
