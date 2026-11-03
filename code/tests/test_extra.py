"""Extra training data: import (one split folder, competition label rules, convention check),
training-only use in CV with the original folds, and refits that include it."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))


def write_benchmark(dirpath: Path, rows: pd.DataFrame):
    """A Leak-CURBER-like folder: kcat/ and km/, two split families each re-partitioning the
    same records, raw (un-logged) values, its own column names, and a repeated measurement."""
    rng = np.random.default_rng(1)
    rep = rows.iloc[:5].copy()
    rep["log10_kcat"] -= 1.0  # a second, smaller kcat measurement of the same pair: max keeps the larger
    allr = pd.concat([rows, rep], ignore_index=True)
    for t, col in (("kcat", "kcat_value"), ("km", "km_value")):
        df = pd.DataFrame({"SMILES": allr["smiles"], "Sequence": allr["sequence"],
                           col: 10.0 ** allr[f"log10_{t}"], "Sequence_ID": np.arange(len(allr))})
        for fam in ("enzyme_sequence_splits/threshold_0.3", "enzyme_sequence_splits/threshold_0.5"):
            d = dirpath / t / fam
            d.mkdir(parents=True, exist_ok=True)
            idx = rng.permutation(len(df))
            for name, part in zip(("train", "val", "test"), np.array_split(idx, [int(.8 * len(df)), int(.9 * len(df))])):
                df.iloc[part].to_parquet(d / f"{name}.parquet", index=False)


def test_extra_data_end_to_end(tmp_path):
    import fake_data
    for name in ("src", "configs", "scripts", "submission_template", "requirements-infer.txt"):
        (tmp_path / name).symlink_to(ROOT / name)
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    everything, va = fake_data.make(n_train=700, n_val=40)
    tr, ext = everything.iloc[:500].reset_index(drop=True), everything.iloc[500:].reset_index(drop=True)
    tr.to_parquet(raw / "train.parquet", index=False)
    va.to_parquet(raw / "val_inputs.parquet", index=False)
    bench = tmp_path / "kinetic_params_dataset"
    write_benchmark(bench, pd.concat([tr.iloc[:60], ext], ignore_index=True))  # 60 pairs overlap train
    env = {**os.environ, "KCATKM_ROOT": str(tmp_path)}

    def run(*a, ok=True):
        r = subprocess.run([sys.executable, *a], env=env, cwd=tmp_path, capture_output=True, text=True)
        assert (r.returncode == 0) == ok, r.stdout + r.stderr
        return r.stdout + r.stderr

    assert "kcat_value" in run("scripts/import_extra.py", str(bench), "--inspect")
    out = run("scripts/import_extra.py", str(bench), "--seq-col", "Sequence")
    assert "2 split folders found" in out
    assert "Spearman 1.000" in out and "do NOT match" not in out
    ex = pd.read_parquet(raw / "extra_train.parquet")
    assert len(ex) == 260 and ex[["log10_kcat", "log10_km"]].notna().all().all()
    # repeated measurement: kcat keeps the max (the original, larger value)
    first = pd.concat([tr.iloc[:60], ext]).iloc[0]
    got = ex[(ex.smiles == first.smiles) & (ex.sequence == first.sequence)]["log10_kcat"].iloc[0]
    assert abs(got - first.log10_kcat) < 1e-6
    # a unit mismatch is caught
    assert "do NOT match" in run("scripts/import_extra.py", str(bench), "--seq-col", "Sequence",
                                 "--km-offset", "3")
    run("scripts/import_extra.py", str(bench), "--seq-col", "Sequence")

    run("scripts/run.py", "configs/lgbm_handcrafted.yaml", "--no-val", "--set", "model.params.n_estimators=40")
    out = run("scripts/run.py", "configs/lgbm_handcrafted.yaml", "--no-val", "--name", "with_extra",
              "--set", "model.params.n_estimators=40", "data.extra_data=true")
    assert "extra data: 200 extra rows" in out  # the 60 pairs already in train are skipped
    oofs = sorted((tmp_path / "runs").glob("*/oof.parquet"))
    a, b = (pd.read_parquet(f) for f in oofs)
    assert a["id"].equals(b["id"]) and a["fold"].equals(b["fold"])  # same rows, same folds: paired comparison
    assert "with_extra" in run("scripts/compare.py", "--vs", "lgbm_handcrafted")
    out = run("scripts/build_submission.py", "with_extra", "--skip-val")
    assert "training includes 200 extra rows" in out and "output valid" in out


def test_extra_rows_respect_the_folds(tmp_path, monkeypatch):
    from kcatkm import extra, paths
    monkeypatch.setattr(extra, "EXTRA_FILE", tmp_path / "extra.parquet")
    base = pd.DataFrame({"id": ["a", "b", "c"], "smiles": ["C", "C", "C"],
                         "sequence": ["MKTAYIAKQRQISFVKSHFSRQ", "GGGGSSSSPPPPWWWWHHHHCCCC", "DEDEDEKRKRKRNQNQNQ"],
                         "log10_kcat": [0., 1., 2.], "log10_km": [0., 1., 2.]})
    folds = pd.DataFrame({"id": base["id"], "group": [0, 1, 2], "fold": [0, 1, 2]})
    ex = pd.DataFrame({"id": ["x", "y"], "smiles": ["CC", "CC"],
                       "sequence": ["MKTAYIAKQRQISFVKSHFSRA", "LLLLIIIIVVVVFFFFYYYY"],  # near fold-0 enzyme / unlike all
                       "log10_kcat": [1., 1.], "log10_km": [1., 1.]})
    ex.to_parquet(extra.EXTRA_FILE, index=False)
    df_all, folds_all, block = extra.attach(base, folds, {"threshold": 0.3})
    assert list(df_all["id"]) == ["a", "b", "c", "ex_x", "ex_y"]
    assert block.tolist() == [[True, False, False], [False, False, False]]
    assert folds_all["fold"].tolist() == [0, 1, 2, -1, -1]
    assert folds_all["group"].iloc[3] == 0 and folds_all["group"].iloc[4] > 2


def test_missing_targets_train_only_the_other_target():
    """kcat-only extra rows: every model type skips the missing Km, nothing turns NaN."""
    import numpy as np
    import pandas as pd
    from kcatkm.pipeline import KcatKmPipeline
    rng = np.random.default_rng(0)
    n = 300
    X = rng.normal(size=(n, 6))
    Y = np.column_stack([X[:, 0] + 0.1 * rng.normal(size=n), X[:, 1] + 0.1 * rng.normal(size=n)])
    Y[200:, 1] = np.nan  # the last 100 rows have kcat only
    names = [f"esm2_t6_8M_mean_{i}" for i in range(3)] + [f"morgan_{i}" for i in range(3)]
    meta = pd.DataFrame({"smiles": [f"S{i % 7}" for i in range(n)], "sequence": [f"Q{i % 40}" for i in range(n)],
                         "group": np.arange(n) % 40})
    lf = {"emb_prefix": "esm2_t6_8M_mean_", "inner_folds": 3, "smiles_te": {"smoothing": 5},
          "enzyme_knn": {"k": 5}, "pair_nn": {}}
    for spec in [{"name": "lgbm", "params": {"n_estimators": 20}}, {"name": "ridge"},
                 {"name": "extratrees", "params": {"n_estimators": 10}},
                 {"name": "catboost", "params": {"iterations": 20}},
                 {"name": "mlp", "params": {"epochs": 40, "hidden": [16], "dropout": 0.0, "input_dropout": 0.0}},
                 {"name": "fusion", "params": {"epochs": 40, "width": 8, "other_width": 4, "hidden": [8],
                                               "dropout": 0.0, "input_dropout": 0.0}},
                 {"name": "ensemble", "members": [{"name": "ridge"}, {"name": "mlp", "params": {"epochs": 5}}]}]:
        P = KcatKmPipeline([], spec, label_features=lf, sample_weight={"power": 0.5}) \
            .fit_xy(X, Y, meta, names=names).predict_x(X, meta)
        assert np.isfinite(P).all(), spec["name"]
        # Km is learned from the complete rows about as well as by the same model without them
        full = Y.copy()
        full[200:, 1] = X[200:, 1]
        Pf = KcatKmPipeline([], spec, label_features=lf).fit_xy(X[:200], full[:200], meta[:200],
                                                                 names=names).predict_x(X, meta)
        r, rf = (np.corrcoef(q[:200, 1], Y[:200, 1])[0, 1] for q in (P, Pf))
        assert r > rf - 0.2 and r > 0.2, (spec["name"], r, rf)


def test_kcat_only_extra_end_to_end(tmp_path):
    import fake_data
    for name in ("src", "configs", "scripts", "submission_template", "requirements-infer.txt"):
        (tmp_path / name).symlink_to(ROOT / name)
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    everything, va = fake_data.make(n_train=650, n_val=30)
    tr, ext = everything.iloc[:500].reset_index(drop=True), everything.iloc[500:].reset_index(drop=True)
    tr.to_parquet(raw / "train.parquet", index=False)
    va.to_parquet(raw / "val_inputs.parquet", index=False)
    ext.to_parquet(raw / "extra_train.parquet", index=False)
    env = {**os.environ, "KCATKM_ROOT": str(tmp_path)}

    def run(*a):
        r = subprocess.run([sys.executable, *a], env=env, cwd=tmp_path, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        return r.stdout + r.stderr

    common = ["features=[seq_len, aac, morgan]", "label_features={inner_folds: 3, smiles_te: {smoothing: 5}}",
              "model.members.0.params.n_estimators=30", "model.members.1.params.n_estimators=10",
              "model.members.2.params.iterations=30"]
    run("scripts/run.py", "configs/ens3_lf_l6_trees.yaml", "--no-val", "--set", *common)
    out = run("scripts/run.py", "configs/ens3_lf_l6_trees_exk.yaml", "--no-val", "--set", *common)
    assert "extra data: 150 extra rows" in out and "OOF mean_scc" in out
    assert "ens3_lf_l6_trees_exk" in run("scripts/compare.py", "--vs", "ens3_lf_l6_trees")
    run("scripts/tune_weights.py", "ens3_lf_l6_trees_exk")
    out = run("scripts/build_submission.py", "ens3_lf_l6_trees_exk_tuned", "--skip-val")
    assert "training includes 150 extra rows" in out and "output valid" in out


def test_rows_need_only_one_of_the_chosen_targets(tmp_path, monkeypatch):
    from kcatkm import extra
    monkeypatch.setattr(extra, "EXTRA_FILE", tmp_path / "extra.parquet")
    pd.DataFrame({"id": ["k", "m", "b", "none"], "smiles": ["C", "CC", "CCC", "CCCC"], "sequence": ["A", "B", "C", "D"],
                  "log10_kcat": [1.0, np.nan, 2.0, np.nan], "log10_km": [np.nan, 0.5, 0.7, np.nan]}
                 ).to_parquet(extra.EXTRA_FILE, index=False)
    base = pd.DataFrame({"id": ["t"], "smiles": ["O"], "sequence": ["Z"], "log10_kcat": [0.0], "log10_km": [0.0]})
    both = extra.load(base, ["log10_kcat", "log10_km"])
    assert both["id"].tolist() == ["ex_k", "ex_m", "ex_b"]  # a row with no label at all is dropped
    kcat_only = extra.load(base, ["log10_kcat"])
    assert kcat_only["id"].tolist() == ["ex_k", "ex_b"] and kcat_only["log10_km"].isna().all()


def test_config_chooses_the_extra_file(tmp_path, monkeypatch):
    from kcatkm import extra, paths
    monkeypatch.setattr(paths, "RAW", tmp_path)
    assert extra.file_of({"data": {"extra_file": "extra_public.parquet"}}) == tmp_path / "extra_public.parquet"
    assert extra.file_of({"data": {}}) == extra.EXTRA_FILE
    pd.DataFrame({"id": ["p"], "smiles": ["CCN"], "sequence": ["MK"], "log10_kcat": [1.0], "log10_km": [np.nan]}
                 ).to_parquet(tmp_path / "extra_public.parquet", index=False)
    base = pd.DataFrame({"id": ["t"], "smiles": ["O"], "sequence": ["Z"], "log10_kcat": [0.0], "log10_km": [0.0]})
    got = extra.load(base, ["log10_kcat", "log10_km"], tmp_path / "extra_public.parquet")
    assert got["id"].tolist() == ["ex_p"]
    assert extra.fingerprint(tmp_path / "extra_public.parquet") != extra.fingerprint(tmp_path / "missing.parquet")
