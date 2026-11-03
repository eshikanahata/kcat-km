"""Multi-task MLP and ensemble tests (synthetic data, CPU, seconds)."""
import os
import pickle
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


def _data(n=600, d=20, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    X[rng.random(X.shape) < 0.02] = np.nan       # missing values must be handled
    X[:, 3] *= 1e4                               # badly scaled column must not dominate
    Xc = np.nan_to_num(X)
    y1 = 2 * Xc[:, 0] - Xc[:, 1] + 0.3 * rng.normal(size=n)
    y2 = 0.5 * y1 + Xc[:, 2] + 0.3 * rng.normal(size=n)
    return X, np.column_stack([y1, y2])


def test_mlp_learns_and_pickles():
    from kcatkm.metrics import score
    from kcatkm.models import make_model
    X, Y = _data()
    spec = {"name": "mlp", "params": {"hidden": [32], "epochs": 30, "dropout": 0.1, "batch_size": 32}}
    m = make_model(spec, seed=1)
    m.fit(X[:450], Y[:450])
    P = m.predict(X[450:])
    assert P.shape == (150, 2) and np.isfinite(P).all()
    assert score(Y[450:], P)["mean_scc"] > 0.7
    m2 = pickle.loads(pickle.dumps(m))
    np.testing.assert_allclose(m2.predict(X[450:]), P, atol=1e-5)
    # same seed -> same model
    again = make_model(spec, seed=1)
    np.testing.assert_allclose(again.fit(X[:450], Y[:450]).predict(X[450:]), P, atol=1e-4)


def test_ensemble_average():
    from kcatkm.models import make_model
    X, Y = _data()
    spec = {"name": "ensemble", "members": [
        {"name": "ridge", "weight": 3, "params": {"alpha": 1.0}},
        {"name": "mlp", "weight": 1, "params": {"hidden": [16], "epochs": 5}}]}
    m = make_model(spec).fit(X, Y)
    a, b = m.member_predictions(X)
    np.testing.assert_allclose(m.predict(X), 0.75 * a + 0.25 * b, atol=1e-6)


def test_ensemble_cv_and_submission(tmp_path):
    import fake_data
    for name in ("src", "configs", "scripts", "submission_template", "requirements-infer.txt"):
        (tmp_path / name).symlink_to(ROOT / name)
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    tr, va = fake_data.make(n_train=500, n_val=60)
    tr.to_parquet(raw / "train.parquet", index=False)
    va.to_parquet(raw / "val_inputs.parquet", index=False)
    env = {**os.environ, "KCATKM_ROOT": str(tmp_path)}
    sets = ["features=[seq_len, aac, morgan, rdkit_desc]",
            "model.members.0.params.n_estimators=50", "model.members.1.params.epochs=5"]

    def run(*a):
        r = subprocess.run([sys.executable, *a], env=env, cwd=tmp_path, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        return r.stdout + r.stderr

    out = run("scripts/run.py", "configs/ens_lgbm_mlp.yaml", "--set", *sets)
    assert "member 0 (lgbm) alone" in out and "member 1 (mlp) alone" in out
    run_dir = next((tmp_path / "runs").glob("*_ens_lgbm_mlp"))
    oof = pd.read_parquet(run_dir / "oof.parquet")
    assert {"member0_pred_log10_kcat", "member1_pred_log10_km"} <= set(oof.columns)
    out = run("scripts/tune_weights.py", "ens_lgbm_mlp", "--step", "0.1")
    assert "tuned" in out and "folds better" in out
    tuned_cfg = next((tmp_path / "runs").glob("*_ens_lgbm_mlp_tuned")) / "config.yaml"
    assert isinstance(__import__("yaml").safe_load(tuned_cfg.read_text())["model"]["members"][0]["weight"], dict)
    out = run("scripts/compare.py", "--vs", "ens_lgbm_mlp")
    assert "ens_lgbm_mlp_tuned" in out
    out = run("scripts/build_submission.py", "ens_lgbm_mlp_tuned")
    assert "output valid" in out


def test_ensemble_per_target_weights():
    from kcatkm.models import make_model
    X, Y = _data()
    spec = {"name": "ensemble", "members": [
        {"name": "ridge", "weight": {"log10_kcat": 1.0, "log10_km": 0.0}},
        {"name": "ridge", "weight": {"log10_kcat": 0.0, "log10_km": 2.0}, "params": {"alpha": 1e4}}]}
    m = make_model(spec).fit(X, Y)
    a, b = m.member_predictions(X)
    P = m.predict(X)
    np.testing.assert_allclose(P[:, 0], a[:, 0], atol=1e-6)
    np.testing.assert_allclose(P[:, 1], b[:, 1], atol=1e-6)
    with pytest.raises(ValueError):
        make_model({"name": "ensemble", "members": [{"name": "ridge", "weight": 0}]})


def test_fusion_groups_columns_by_name_and_pickles():
    import pickle
    import numpy as np
    import pandas as pd
    from kcatkm.pipeline import KcatKmPipeline
    rng = np.random.default_rng(0)
    n = 300
    names = [f"esm2_t6_8M_mean_{i}" for i in range(6)] + [f"morgan_{i}" for i in range(5)] + ["lf_smi_kcat"]
    X = rng.normal(size=(n, len(names)))
    Y = np.column_stack([X[:, 0] * X[:, 6] + 0.1 * rng.normal(size=n), X[:, 1]])
    meta = pd.DataFrame({"smiles": ["C"] * n, "sequence": [f"S{i}" for i in range(n)]})
    for spec in [{"name": "fusion", "params": {"epochs": 3, "width": 8, "other_width": 4, "hidden": [8]}},
                 {"name": "ensemble", "members": [{"name": "ridge"}, {"name": "fusion", "params": {"epochs": 2}}]}]:
        pipe = KcatKmPipeline([], spec).fit_xy(X, Y, meta, names=names)
        P = pipe.predict_x(X)
        assert P.shape == (n, 2) and np.isfinite(P).all()
        assert np.allclose(pickle.loads(pickle.dumps(pipe)).predict_x(X), P, atol=1e-5)
    fus = KcatKmPipeline([], {"name": "fusion", "params": {"epochs": 1}}).fit_xy(X, Y, meta, names=names).model
    assert fus._groups(len(names)) == (list(range(6)), list(range(6, 11)), [11])


def test_catboost_member():
    import numpy as np
    from kcatkm.models import make_model
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    Y = np.column_stack([X[:, 0], X[:, 1]]) + 0.1 * rng.normal(size=(200, 2))
    m = make_model({"name": "catboost", "params": {"iterations": 50, "loss_function": "Huber:delta=1.0"}})
    P = m.fit(X, Y, sample_weight=np.ones(200)).predict(X)
    assert P.shape == (200, 2) and np.corrcoef(P[:, 0], Y[:, 0])[0, 1] > 0.8
