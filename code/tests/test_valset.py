"""Labelled validation set: correct joining, leakage exclusion, and wiring into run/compare/build."""
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))


def test_labelled_val_pipeline(tmp_path):
    import fake_data
    for name in ("src", "configs", "scripts", "submission_template", "requirements-infer.txt"):
        (tmp_path / name).symlink_to(ROOT / name)
    raw, lab = tmp_path / "data" / "raw", tmp_path / "data" / "labels"
    raw.mkdir(parents=True)
    lab.mkdir(parents=True)
    everything, va = fake_data.make(n_train=620, n_val=60)
    tr, old = everything.iloc[:500].reset_index(drop=True), everything.iloc[500:].reset_index(drop=True)
    tr.to_parquet(raw / "train.parquet", index=False)
    va.to_parquet(raw / "val_inputs.parquet", index=False)  # current val: ids v_000001.. too

    # labelled set: its own rows and ids (same id style as the current val, different rows),
    # plus 7 pairs that are also in the training data and must be dropped
    leaked = tr.sample(7, random_state=0).copy()
    labelled = pd.concat([old, leaked], ignore_index=True)
    labelled["id"] = [f"v_{i + 1:06d}" for i in range(len(labelled))]
    labelled[["id", "smiles", "sequence"]].to_parquet(lab / "val_inputs.parquet", index=False)
    labelled[["id", "log10_kcat", "log10_km"]].sample(frac=1, random_state=1).to_csv(lab / "val_labels.csv", index=False)

    env = {**os.environ, "KCATKM_ROOT": str(tmp_path)}

    def run(*a):
        r = subprocess.run([sys.executable, *a], env=env, cwd=tmp_path, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        return r.stdout + r.stderr

    out = run("scripts/run.py", "configs/ridge_starter.yaml")
    assert "labelled val (120 rows; 7 dropped as already in train)" in out
    idx = pd.read_csv(tmp_path / "runs" / "index.csv")
    assert idx["val_mean_scc"].notna().all() and idx["val_mean_scc"].iloc[0] > 0.1

    # an older run without val scores gets them from val_eval.py
    run("scripts/run.py", "configs/lgbm_handcrafted.yaml", "--set", "model.params.n_estimators=50", "--no-val")
    assert pd.isna(pd.read_csv(tmp_path / "runs" / "index.csv")["val_mean_scc"].iloc[-1])
    run("scripts/val_eval.py", "lgbm_handcrafted")
    assert pd.read_csv(tmp_path / "runs" / "index.csv")["val_mean_scc"].notna().all()
    assert "val_mean_scc" in run("scripts/compare.py", "--vs", "ridge_starter")

    # a built package is scored end to end
    out = run("scripts/build_submission.py", "lgbm_handcrafted")
    assert "[labelled val] package scored end to end (stage: full" in out

    # training on the labelled val rows: CV is scored on training rows only, val out-of-fold
    out = run("scripts/run.py", "configs/lgbm_lf_esm8_xv.yaml", "--set",
              "features=[seq_len, aac, morgan]", "label_features={inner_folds: 3, smiles_te: {smoothing: 5}}",
              "model.params.n_estimators=50", "--name", "xv")
    assert "including 120 labelled-val rows" in out and "labelled val (out-of-fold, 120 rows)" in out
    idx = pd.read_csv(tmp_path / "runs" / "index.csv")
    row = idx[idx["name"] == "xv"].iloc[0]
    assert row["extra_train"] == "labelled_val" and pd.notna(row["val_mean_scc"])
    oof = pd.read_parquet(next((tmp_path / "runs").glob("*_xv")) / "oof.parquet")
    assert oof["id"].str.startswith("lv_").sum() == 120

    # an ensemble with val rows: weight tuning keeps CV and val apart
    run("scripts/run.py", "configs/ens_lf_esm8_xv.yaml", "--set", "features=[seq_len, aac, morgan]",
        "label_features={inner_folds: 3, smiles_te: {smoothing: 5}}", "model.members.0.params.n_estimators=30",
        "model.members.1.params.epochs=3", "--name", "xvens")
    out = run("scripts/tune_weights.py", "xvens")
    assert "labelled val (out-of-fold)" in out
    out = run("scripts/build_submission.py", "xvens_tuned")
    assert "training includes 120 labelled-val rows" in out and "[labelled val] skipped" in out
