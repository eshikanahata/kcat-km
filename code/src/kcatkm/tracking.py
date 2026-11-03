"""Run folders + the experiment index (runs/index.csv)."""
import json
import subprocess
from datetime import datetime

import pandas as pd
import yaml

from . import paths


def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=paths.ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "configs"], cwd=paths.ROOT,
                               capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "no-git"


def new_run_dir(name: str):
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}_{name}"
    d = paths.RUNS / run_id
    d.mkdir(parents=True, exist_ok=False)
    return run_id, d


def save_json(obj, path):
    path.write_text(json.dumps(obj, indent=2, default=str))


def save_yaml(obj, path):
    path.write_text(yaml.safe_dump(obj, sort_keys=False))


def read_index() -> pd.DataFrame:
    if paths.RUNS_INDEX.exists():
        return pd.read_csv(paths.RUNS_INDEX)
    return pd.DataFrame()


def append_index(row: dict):
    idx = read_index()
    idx = pd.concat([idx, pd.DataFrame([row])], ignore_index=True)
    paths.RUNS_INDEX.parent.mkdir(parents=True, exist_ok=True)
    idx.to_csv(paths.RUNS_INDEX, index=False)


def comparable(idx):
    """Rows on the current default split AND the current training-data version."""
    from . import config, data, splits
    split = splits.label(config.load("base.yaml")["split"])
    mask = idx["split"] == split
    if "data" in idx:
        mask &= idx["data"] == data.current_fingerprint()
    else:
        mask &= False  # runs from before data versioning cannot be matched to the current data
    return idx[mask]


def find_run(run: str):
    """Accepts a full run id, a unique prefix/suffix, or 'best' / 'last'."""
    idx = read_index()
    if idx.empty:
        raise FileNotFoundError("no runs yet")
    if run == "best":
        cur = comparable(idx)
        if cur.empty:
            raise FileNotFoundError("no runs on the current split and training data yet")
        return paths.RUNS / cur.sort_values("mean_scc", ascending=False).iloc[0]["run_id"]
    if run == "last":
        return paths.RUNS / idx.iloc[-1]["run_id"]
    hits = [r for r in idx["run_id"] if r == run or r.startswith(run) or r.endswith(run)]
    if len(hits) > 1:
        # ambiguous name: prefer runs on the current split and data, then the newest
        cur = comparable(idx)
        same = cur[cur["run_id"].isin(hits)]["run_id"].tolist()
        if same:
            if len(same) > 1:
                print(f"note: {run!r} matches {len(same)} current runs; using the newest, {same[-1]}")
            return paths.RUNS / same[-1]
    if len(hits) != 1:
        raise KeyError(f"{run!r} matched {len(hits)} runs: {hits[:5]}")
    return paths.RUNS / hits[0]
