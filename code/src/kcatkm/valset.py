"""Labelled validation set for local testing (provided by the course instructor).

Files (put them in data/labels/):
    val_labels.csv        id, log10_kcat, log10_km
    val_inputs.parquet    id, smiles, sequence -- the inputs file THESE ids belong to

The labels released so far belong to the 21 Oct validation set (1,502 rows); its ids
(v_000000, ...) are reused by the 23 Oct validation file for different rows, so labels
are only ever joined to the inputs file they came with.

Leakage control: pairs (SMILES, sequence) that also occur in the current training data
are dropped. Scores are also reported for the subset whose enzyme does not occur in
training at all, which is the situation of the hidden test set.
"""
import numpy as np
import pandas as pd

from . import TARGETS, data, metrics, paths

LABELS = paths.DATA / "labels"
VAL_LABELS = LABELS / "val_labels.csv"
VAL_LABEL_INPUTS = LABELS / "val_inputs.parquet"


def available() -> bool:
    return VAL_LABELS.exists() and VAL_LABEL_INPUTS.exists()


def load(train=None, verbose=False):
    """Returns (rows usable for scoring, info dict)."""
    lab = pd.read_csv(VAL_LABELS)
    inp = pd.read_parquet(VAL_LABEL_INPUTS)
    df = inp.merge(lab, on="id", how="inner", validate="one_to_one").reset_index(drop=True)
    train = data.load_train() if train is None else train
    pairs = set(zip(train["smiles"], train["sequence"]))
    in_train = np.array([p in pairs for p in zip(df["smiles"], df["sequence"])])
    df["seq_in_train"] = df["sequence"].isin(set(train["sequence"])).to_numpy()
    clean = df[~in_train].reset_index(drop=True)
    info = {"labels": len(lab), "matched": len(df), "pair_in_train_dropped": int(in_train.sum()),
            "usable": len(clean), "unseen_enzyme_rows": int((~clean["seq_in_train"]).sum())}
    if len(df) < len(lab):
        info["warning"] = f"{len(lab) - len(df)} label ids not found in {VAL_LABEL_INPUTS.name}"
    if verbose:
        print("labelled val: " + ", ".join(f"{k}={v}" for k, v in info.items()))
    return clean, info


def score(df, P) -> dict:
    """Leaderboard metrics on all usable rows, plus mean SCC on rows with an unseen enzyme."""
    Y = df[list(TARGETS)].to_numpy(float)
    P = np.asarray(P, float)
    out = metrics.score(Y, P)
    m = ~df["seq_in_train"].to_numpy()
    out["unseen_enzyme_mean_scc"] = metrics.score(Y[m], P[m])["mean_scc"] if m.sum() > 10 else float("nan")
    out["n"] = len(df)
    return out


def fmt(s: dict) -> str:
    return (f"mean_scc={s['mean_scc']:.4f} kcat={s['kcat_scc']:.4f} km={s['km_scc']:.4f} "
            f"mean_rmse={s['mean_rmse']:.4f} | unseen-enzyme rows mean_scc={s['unseen_enzyme_mean_scc']:.4f} "
            f"(n={s['n']})")


def score_package(sub_dir, workdir, env_extra=None):
    """Run a built submission's predict.py (clean environment, 2 threads, as the grader does)
    on the usable labelled-val rows. Returns (metrics, final stage, seconds)."""
    import os
    import subprocess
    import sys
    import time
    from pathlib import Path
    vdf, info = load()
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    inp, out = workdir / "labelled_val_inputs.parquet", workdir / "labelled_val_predictions.csv"
    vdf[["id", "smiles", "sequence"]].to_parquet(inp, index=False)
    keep = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "SYSTEMROOT")
    env = {**{k: v for k, v in os.environ.items() if k in keep}, **(env_extra or {})}
    t0 = time.time()
    r = subprocess.run([sys.executable, "-I", str(Path(sub_dir) / "predict.py"), "--input", str(inp),
                        "--output", str(out)], env=env, cwd="/", capture_output=True, text=True)
    dt = time.time() - t0
    if r.returncode != 0:
        raise RuntimeError(f"predict.py crashed on the labelled val inputs:\n{r.stderr[-3000:]}")
    stage = next((ln.rsplit(":", 1)[1].strip() for ln in reversed(r.stderr.splitlines())
                  if "final predictions from:" in ln), "unknown")
    pred = pd.read_csv(out).set_index("id").loc[vdf["id"]]
    return score(vdf, pred[list(TARGETS)].to_numpy()), stage, dt


PREFIX = "lv_"  # id prefix of labelled-val rows when they are used as training rows


def append_to_train(train):
    """Training data + the usable labelled-val rows (ids prefixed 'lv_'). Returns (combined, vdf)."""
    vdf, _ = load(train=train)
    rows = vdf[["id", "smiles", "sequence", *TARGETS]].copy()
    rows["id"] = PREFIX + rows["id"].astype(str)
    return pd.concat([train, rows], ignore_index=True), vdf


def is_val_row(ids):
    return pd.Series(ids).astype(str).str.startswith(PREFIX).to_numpy()


def training_data(cfg, base=None):
    """The training rows a config asks for: the train file, plus labelled val if
    data.extra_train == 'labelled_val'. Returns (df, vdf or None)."""
    base = data.load_train() if base is None else base
    extra = (cfg.get("data") or {}).get("extra_train")
    if not extra:
        return base, None
    if extra != "labelled_val":
        raise ValueError(f"unknown data.extra_train {extra!r} (only 'labelled_val')")
    if not available():
        raise FileNotFoundError(f"data.extra_train needs {VAL_LABELS} and {VAL_LABEL_INPUTS}")
    return append_to_train(base)
