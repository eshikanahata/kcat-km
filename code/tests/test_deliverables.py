"""make_deliverables.py collects readable code, the weights and the records of a built submission."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))


def test_deliverables(tmp_path):
    import fake_data
    for name in ("src", "configs", "scripts", "submission_template", "requirements-infer.txt", "tests"):
        (tmp_path / name).symlink_to(ROOT / name)
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    tr, va = fake_data.make(n_train=400, n_val=30)
    tr.to_parquet(raw / "train.parquet", index=False)
    va.to_parquet(raw / "val_inputs.parquet", index=False)
    env = {**os.environ, "KCATKM_ROOT": str(tmp_path)}

    def run(*a):
        r = subprocess.run([sys.executable, *a], env=env, cwd=tmp_path, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        return r.stdout + r.stderr

    run("scripts/run.py", "configs/lgbm_handcrafted.yaml", "--no-val", "--set", "model.params.n_estimators=20")
    run("scripts/build_submission.py", "lgbm_handcrafted", "--name", "sX", "--skip-val")
    out = run("scripts/make_deliverables.py", "sX", "--zip")
    d = tmp_path / "deliverables" / "sX"
    assert (d / "code" / "src" / "kcatkm" / "pipeline.py").exists()
    assert (d / "code" / "submission_template" / "predict.py").exists()
    assert not list(d.rglob("__pycache__"))
    assert (d / "weights" / "model" / "pipeline.joblib").exists()
    assert list((d / "records" / "runs").glob("*/config.yaml"))
    assert "lgbm_handcrafted" in (d / "SUBMISSION.md").read_text()
    for part in ("code", "weights", "records"):
        assert (tmp_path / "deliverables" / f"sX_{part}.zip").exists(), out
