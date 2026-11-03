"""Every config in configs/ must resolve and its model must fit on a tiny dataset.

Catches inheritance mistakes (e.g. one model's hyper-parameters leaking into
another's) before a long CV run does.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
CONFIGS = sorted((ROOT / "configs").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=[p.stem for p in CONFIGS])
def test_config_model_fits(path):
    from kcatkm import config
    from kcatkm.features import make_extractors
    from kcatkm.models import make_model

    cfg = config.load(path)
    assert {"split", "features", "model"} <= set(cfg), f"{path.name} is missing a section"
    make_extractors(cfg["features"])  # unknown feature names / bad params fail here

    spec = cfg["model"]
    # keep the check fast: shrink iteration counts, never change anything else
    for m in [spec, *spec.get("members", [])]:
        p = m.get("params") or {}
        if "n_estimators" in p:
            p["n_estimators"] = 5
        if "epochs" in p:
            p["epochs"] = 1
        if "iterations" in p:
            p["iterations"] = 5
    rng = np.random.default_rng(0)
    X, Y = rng.normal(size=(80, 12)).astype(np.float32), rng.normal(size=(80, 2))
    model = make_model(spec)
    if hasattr(model, "set_feature_names"):  # as the pipeline does
        model.set_feature_names([f"esm2_x_{i}" for i in range(5)] + [f"morgan_{i}" for i in range(5)] + ["a", "b"])
    P = model.fit(X, Y).predict(X)
    assert P.shape == (80, 2) and np.isfinite(P).all()


def test_model_section_resets_when_model_changes():
    from kcatkm import config
    mlp = config.load(ROOT / "configs" / "mlp.yaml")
    assert "alpha" not in mlp["model"]["params"] and "num_leaves" not in mlp["model"]["params"]
    huber = config.load(ROOT / "configs" / "lgbm_combo_huber.yaml")
    assert huber["model"]["params"]["num_leaves"] == 63  # same model type still inherits


def test_cross_target_model():
    import numpy as np
    from kcatkm.models import make_model
    rng = np.random.default_rng(0)
    n = 600
    X = rng.normal(size=(n, 5))
    km = X[:, 0] + 0.1 * rng.normal(size=n)
    kcat = 2 * km + 0.5 * rng.normal(size=n)  # kcat mostly via Km
    Y = np.column_stack([kcat, km])
    groups = np.arange(n) // 6
    spec = {"name": "cross", "base": {"name": "ridge", "params": {"alpha": 1.0}}, "inner_folds": 3}
    m = make_model(spec, seed=1)
    from kcatkm.models import fit_model
    fit_model(m, X, Y, groups)
    P = m.predict(X)
    assert P.shape == (n, 2) and np.isfinite(P).all()
    # Km column is stage 1's; kcat goes through stage 2
    assert np.allclose(P[:, 1], m.stage1.predict(X)[:, 1])
    assert set(m.stage2) == {"log10_kcat"}
    # inside an ensemble the groups are passed on, and the model pickles
    import pickle
    ens = make_model({"name": "ensemble", "members": [spec, {"name": "ridge"}]})
    fit_model(ens, X, Y, groups)
    assert np.allclose(pickle.loads(pickle.dumps(ens)).predict(X), ens.predict(X))


def test_cross_config_loads():
    from kcatkm import config
    cfg = config.load(ROOT / "configs" / "lgbm_lf_esm8_cross.yaml")
    assert cfg["model"]["name"] == "cross" and "params" not in cfg["model"]
    assert cfg["model"]["base"]["params"]["objective"] == "huber"


def test_sample_weight_reaches_every_model():
    import numpy as np
    import pandas as pd
    from kcatkm.metrics import testlike
    from kcatkm.pipeline import KcatKmPipeline
    rng = np.random.default_rng(0)
    seqs = np.array(["A" * 30] * 300 + [f"M{i}" * 10 for i in range(100)])  # one enzyme with 300 rows
    n = len(seqs)
    X = rng.normal(size=(n, 4))
    Y = np.column_stack([X[:, 0] + rng.normal(size=n), X[:, 1] + rng.normal(size=n)])
    meta = pd.DataFrame({"smiles": ["C"] * n, "sequence": seqs})
    for spec in [{"name": "lgbm", "params": {"n_estimators": 20}}, {"name": "ridge"}, {"name": "extratrees",
                 "params": {"n_estimators": 10}},
                 {"name": "mlp", "params": {"epochs": 2, "hidden": [8]}},
                 {"name": "ensemble", "members": [{"name": "ridge"}, {"name": "mlp", "params": {"epochs": 1}}]},
                 {"name": "cross", "base": {"name": "ridge"}}]:
        Xn = rng.normal(size=(50, 4))  # new rows: fully grown trees reproduce the training rows either way
        a = KcatKmPipeline([], spec, sample_weight={"power": 1.0}).fit_xy(X, Y, meta).predict_x(Xn)
        b = KcatKmPipeline([], spec).fit_xy(X, Y, meta).predict_x(Xn)
        assert a.shape == (50, 2) and np.isfinite(a).all()
        assert not np.allclose(a, b), spec["name"]  # the weights changed the fit
    tl = testlike(Y, Y + rng.normal(size=Y.shape), seqs)
    assert set(tl) == {"kcat_scc", "km_scc", "mean_scc"}


def test_ensemble_seed_offset_and_zero_weight_members():
    import numpy as np
    from kcatkm.models import make_model
    rng = np.random.default_rng(0)
    X, Y = rng.normal(size=(120, 6)), rng.normal(size=(120, 2))
    lg = {"name": "lgbm", "params": {"n_estimators": 10, "subsample": 0.5, "subsample_freq": 1}}
    a = make_model({"name": "ensemble", "members": [{"name": "ridge"}, {"name": "ridge"}, lg]}, seed=1).fit(X, Y)
    # the same lgbm at index 0 with seed_offset 2 must be the same model as at index 2
    b = make_model({"name": "ensemble", "members": [{**lg, "seed_offset": 2}]}, seed=1).fit(X, Y)
    assert np.allclose(a.members[2].predict(X), b.members[0].predict(X))
    c = make_model({"name": "ensemble", "members": [{"name": "ridge", "weight": 1.0},
                                                    {**lg, "weight": {"log10_kcat": 0, "log10_km": 0}}]}).fit(X, Y)
    assert c.members[1] is None and np.isfinite(c.predict(X)).all()
