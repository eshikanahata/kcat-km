"""--compiled: the package ships bytecode only and predicts exactly like the source build."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))


def test_compiled_package(tmp_path):
    import fake_data
    for name in ("src", "configs", "scripts", "submission_template", "requirements-infer.txt"):
        (tmp_path / name).symlink_to(ROOT / name)
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    tr, va = fake_data.make(n_train=400, n_val=40)
    tr.to_parquet(raw / "train.parquet", index=False)
    va.to_parquet(raw / "val_inputs.parquet", index=False)
    env = {**os.environ, "KCATKM_ROOT": str(tmp_path), "KCATKM_COMPILE_PYTHON": sys.executable}

    def run(*a, ok=True):
        r = subprocess.run([sys.executable, *a], env=env, cwd=tmp_path, capture_output=True, text=True)
        assert (r.returncode == 0) == ok, r.stdout + r.stderr
        return r.stdout + r.stderr

    run("scripts/run.py", "configs/lgbm_handcrafted.yaml", "--set", "model.params.n_estimators=30", "--no-val")
    out = run("scripts/build_submission.py", "lgbm_handcrafted", "--compiled", "--name", "c1")
    assert "same predictions as the source build" in out
    sub = tmp_path / "submissions" / "c1"
    assert sorted(p.name for p in sub.rglob("*.py")) == ["predict.py"]
    assert len((sub / "predict.py").read_text().splitlines()) < 12
    assert (sub / "_runner.pyc").exists() and (sub / "kcatkm" / "pipeline.pyc").exists()
    assert "*.pyc" not in (sub / ".dockerignore").read_text()
    assert not list(sub.rglob("__pycache__"))
    # docstrings are gone from the bytecode
    assert b"Fail-safe, staged inference" not in (sub / "_runner.pyc").read_bytes()
    # the ledger row exists although nothing was uploaded
    import pandas as pd
    led = pd.read_csv(tmp_path / "notes" / "submissions.csv")
    assert (led["sub_name"] == "c1").any() and "compiled" in led.set_index("sub_name").loc["c1", "notes"]

    # without a compiler override, --compiled insists on the docker check
    env.pop("KCATKM_COMPILE_PYTHON")
    assert "needs --docker" in run("scripts/build_submission.py", "lgbm_handcrafted", "--compiled", ok=False)


def test_compile_ignores_stray_caches_and_resumes(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "src"))
    import shutil
    import build_submission as b
    sub = tmp_path / "pkg"
    shutil.copytree(ROOT / "submission_template", sub)
    shutil.copytree(ROOT / "src" / "kcatkm", sub / "kcatkm", ignore=shutil.ignore_patterns("__pycache__"))
    for d in (sub / "kcatkm", sub / "kcatkm" / "features"):  # caches left by a local check on another Python
        (d / "__pycache__").mkdir(exist_ok=True)
        (d / "__pycache__" / "__init__.cpython-399.pyc").write_bytes(b"x")
    os.environ["KCATKM_COMPILE_PYTHON"] = sys.executable
    try:
        b.compile_package(sub)
        # simulate an interrupted compile: one module left as source, then resume
        (sub / "kcatkm" / "extra.py").write_text("X = 1\n")
        b.compile_package(sub)
    finally:
        os.environ.pop("KCATKM_COMPILE_PYTHON")
    assert sorted(p.name for p in sub.rglob("*.py")) == ["predict.py"]
    assert (sub / "kcatkm" / "extra.pyc").exists() and (sub / "_runner.pyc").exists()
    assert not list(sub.rglob("__pycache__"))
