"""Label-derived features: no leakage, real benefit, and the full CV/submission path."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

SPEC = {"emb_prefix": "emb_mean_", "inner_folds": 4, "smiles_te": {"smoothing": 5},
        "enzyme_knn": {"k": 5, "tau": 0.1}, "pair_nn": {"tau": 0.1}}


def world(n=2400, n_fam=60, n_smi=300, d=16, signal=True, seed=0):
    """Enzyme families (embedding = family centre + noise), substrates, labels."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(n_fam, d))
    fam = rng.integers(n_fam, size=n)
    variant = rng.integers(4, size=n)                       # a few sequences per family
    seq = np.array([f"F{f}V{v}" for f, v in zip(fam, variant)])
    emb_of = {s: centres[int(s[1:s.index('V')])] + 0.3 * rng.normal(size=d) for s in np.unique(seq)}
    E = np.stack([emb_of[s] for s in seq])
    smi = rng.integers(n_smi, size=n)
    if signal:
        Y = (rng.normal(size=(n_fam, 2))[fam] + rng.normal(size=(n_smi, 2))[smi] + 0.3 * rng.normal(size=(n, 2)))
    else:
        Y = rng.normal(size=(n, 2))
    X = np.hstack([E, rng.normal(size=(n, 4))]).astype(np.float32)
    names = [f"emb_mean_{i}" for i in range(d)] + [f"noise_{i}" for i in range(4)]
    meta = pd.DataFrame({"smiles": [f"S{s}" for s in smi], "sequence": seq, "group": fam})
    return X, Y, meta, names


def cv_score(X, Y, meta, names, lf, model="ridge"):
    from kcatkm.metrics import score
    from kcatkm.pipeline import KcatKmPipeline
    oof = np.zeros_like(Y)
    for tr, va in GroupKFold(5, shuffle=True, random_state=1).split(X, groups=meta["group"]):
        pipe = KcatKmPipeline([], {"name": model}, label_features=SPEC if lf else None)
        pipe.fit_xy(X[tr], Y[tr], meta.iloc[tr].reset_index(drop=True), names=names)
        oof[va] = pipe.predict_x(X[va], meta.iloc[va].reset_index(drop=True))
    return score(Y, oof)["mean_scc"]


def test_no_leakage_on_random_labels():
    # one world can reach ~0.05 by chance (8 worlds averaged 0.004); average 3 for a stable test
    scores = [cv_score(*world(signal=False, seed=s), lf=True) for s in (1, 2, 3)]
    assert abs(np.mean(scores)) < 0.03, scores


def test_training_rows_get_out_of_fold_features():
    from kcatkm.labelfeat import LabelFeatures
    X, Y, meta, names = world(signal=False, n_smi=800)   # ~3 rows per substrate: own label would show
    lf = LabelFeatures(SPEC, names).fit(X, Y, meta)
    F = pd.DataFrame(lf.train_features_, columns=lf.names())
    for col, j in [("lf_smi_kcat", 0), ("lf_knn_km", 1), ("lf_pair_kcat", 0)]:
        ok = F[col].notna()
        assert abs(np.corrcoef(F[col][ok], Y[ok, j])[0, 1]) < 0.06, col


def test_label_features_help_when_labels_have_structure():
    X, Y, meta, names = world(signal=True)
    without, with_lf = cv_score(X, Y, meta, names, lf=False), cv_score(X, Y, meta, names, lf=True)
    assert with_lf > without + 0.15, (without, with_lf)


def test_unseen_substrates_and_enzymes_are_handled():
    from kcatkm.labelfeat import LabelFeatures
    X, Y, meta, names = world()
    lf = LabelFeatures(SPEC, names).fit(X, Y, meta)
    q = meta.iloc[:5].copy()
    q["smiles"] = "NEVER_SEEN"
    F = pd.DataFrame(lf.transform(X[:5], q), columns=lf.names())
    assert (F["lf_smi_logn"] == 0).all() and F["lf_pair_kcat"].isna().all() and (F["lf_pair_logn"] == 0).all()
    np.testing.assert_allclose(F["lf_smi_kcat"], Y[:, 0].mean(), atol=1e-5)  # falls back to the global mean
    assert np.isfinite(F["lf_knn_kcat"]).all()


def test_cv_and_submission_with_label_features(tmp_path):
    pytest.importorskip("torch")
    import fake_data
    import test_esm
    for name in ("src", "configs", "scripts", "submission_template", "requirements-infer.txt"):
        (tmp_path / name).symlink_to(ROOT / name)
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    tr, va = fake_data.make(n_train=500, n_val=60)
    tr.to_parquet(raw / "train.parquet", index=False)
    va.to_parquet(raw / "val_inputs.parquet", index=False)
    test_esm.make_tiny_esm(tmp_path / "pretrained" / "esm_tiny")
    env = {**os.environ, "KCATKM_ROOT": str(tmp_path), "KCATKM_PRETRAINED": str(tmp_path / "pretrained")}

    def run(*a):
        r = subprocess.run([sys.executable, *a], env=env, cwd=tmp_path, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        return r.stdout + r.stderr

    out = run("scripts/run.py", "configs/lgbm_lf.yaml", "--set",
              "features=[seq_len, morgan, {esm: {model: x/esm_tiny}}]", "label_features.emb_prefix=esm_tiny_mean_",
              "model.params.n_estimators=50")
    assert "label features:" in out and "OOF mean_scc" in out
    imp = pd.read_csv(next((tmp_path / "runs").glob("*_lgbm_lf")) / "feature_importance.csv", index_col=0)
    assert any(i.startswith("lf_") for i in imp.index)
    out = run("scripts/build_submission.py", "lgbm_lf")
    assert "final predictions came from the full model" in out and "output valid" in out


def test_canonical_smiles_merges_spellings():
    pytest.importorskip("rdkit")
    from kcatkm.labelfeat import LabelFeatures
    X, Y, meta, names = world(n=400, n_smi=10)
    meta = meta.copy()
    meta["smiles"] = np.where(np.arange(len(meta)) % 2 == 0, "OCC", "C(O)C")  # two spellings of ethanol
    spec = {"inner_folds": 4, "smiles_te": {"smoothing": 5}}
    raw = LabelFeatures(spec, names).fit(X, Y, meta)
    can = LabelFeatures({**spec, "canonical_smiles": True}, names).fit(X, Y, meta)
    q = meta.iloc[:2]
    n_raw = np.expm1(raw.transform(X[:2], q)[:, -1])
    n_can = np.expm1(can.transform(X[:2], q)[:, -1])
    assert np.allclose(n_raw, len(meta) / 2) and np.allclose(n_can, len(meta))


def test_substrate_knn_uses_similar_but_not_identical_substrates():
    import numpy as np
    import pandas as pd
    from kcatkm.labelfeat import LabelFeatures
    rng = np.random.default_rng(0)
    # 20 substrates in 2 chemical families; the label follows the family
    fam = np.repeat([0, 1], 10)
    emb = np.vstack([rng.normal(size=4) * 0.05 + (5 if f else -5) for f in fam])
    rows = rng.integers(0, 20, size=400)
    names = [f"ChemBERTa-77M-MTR_{i}" for i in range(4)] + ["other"]
    X = np.hstack([emb[rows], rng.normal(size=(400, 1))])
    Y = np.column_stack([fam[rows] * 3.0 + 0.1 * rng.normal(size=400), np.zeros(400)])
    meta = pd.DataFrame({"smiles": [f"S{r}" for r in rows], "sequence": [f"E{i % 37}" for i in range(400)],
                         "group": np.arange(400) % 37})
    lf = LabelFeatures({"substrate_knn": {"k": 3, "tau": 0.1}, "inner_folds": 3}, names).fit(X, Y, meta)
    assert lf.names() == ["lf_sknn_kcat", "lf_sknn_km", "lf_sknn_maxsim", "lf_sknn_meansim"]
    F = lf.transform(X, meta)
    assert np.corrcoef(F[:, 0], fam[rows])[0, 1] > 0.95  # neighbours from the same family
    # the exact same substrate is never its own neighbour: give S0 an extreme label and check that
    # the feature of S0's own rows does not see it, while its family members do
    Y2 = Y.copy()
    Y2[rows == 0, 0] = 100.0
    F2 = LabelFeatures({"substrate_knn": {"k": 3, "tau": 0.1}, "inner_folds": 3}, names).fit(X, Y2, meta) \
        .transform(X, meta)
    assert F2[rows == 0, 0].max() < 10
    assert F2[(rows != 0) & (fam[rows] == 0), 0].max() > 10
    # a brand-new substrate close to family 1 gets family-1 labels
    Xn = np.hstack([emb[[15]] + 0.01, [[0.0]]])
    Fn = lf.transform(Xn, pd.DataFrame({"smiles": ["NEW"], "sequence": ["E0"]}))
    assert Fn[0, 0] > 2.5


def test_rows_missing_a_target_are_references_for_the_other_target():
    import numpy as np
    import pandas as pd
    from kcatkm.labelfeat import LabelFeatures
    rng = np.random.default_rng(0)
    E = np.eye(4) * 5 + rng.normal(size=(4, 4)) * 0.01  # 4 distinct enzymes
    names = [f"esm_{i}" for i in range(4)]
    X = np.repeat(E, 10, axis=0)
    Y = np.column_stack([np.repeat([0., 1., 2., 3.], 10), np.repeat([5., 6., 7., 8.], 10)])
    Y[30:, 1] = np.nan  # enzyme 3: kcat only
    meta = pd.DataFrame({"smiles": ["C"] * 40, "sequence": np.repeat(["A", "B", "C", "D"], 10),
                         "group": np.repeat([0, 1, 2, 3], 10)})
    spec = {"emb_prefix": "esm_", "inner_folds": 2, "smiles_te": {"smoothing": 1}, "enzyme_knn": {"k": 1, "tau": 0.1},
            "pair_nn": {"tau": 0.05}}
    lf = LabelFeatures(spec, names).fit(X, Y, meta)
    assert np.isfinite(lf.train_features_[:, [0, 1, 3, 4]]).all()  # smiles_te + knn values never NaN
    q = lf.transform(E[[3]], pd.DataFrame({"smiles": ["C"], "sequence": ["D2"]}))
    names_out = lf.names()
    knn_kcat, knn_km = q[0, names_out.index("lf_knn_kcat")], q[0, names_out.index("lf_knn_km")]
    assert abs(knn_kcat - 3.0) < 1e-3  # the kcat-only enzyme IS used for kcat
    assert np.isfinite(knn_km) and knn_km < 8  # ...but not for Km (it has none): falls back, no NaN
    pair_km = q[0, names_out.index("lf_pair_km")]
    assert np.isfinite(pair_km) and abs(q[0, names_out.index("lf_pair_kcat")] - 3.0) < 0.05
    te_km = q[0, names_out.index("lf_smi_km")]
    assert abs(te_km - (5 + 6 + 7) * 10 / 30) < 0.1  # Km mean over the 30 rows that have Km
