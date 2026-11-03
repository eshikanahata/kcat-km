"""Project paths. Override the root with KCATKM_ROOT if you run from elsewhere."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("KCATKM_ROOT", Path(__file__).resolve().parents[2]))

DATA = ROOT / "data"
RAW = DATA / "raw"                 # train.parquet, val_inputs.parquet (from HF)
CACHE = DATA / "cache"             # feature + split caches (safe to delete)
FEATURE_CACHE = CACHE / "features"
SPLIT_CACHE = CACHE / "splits"

PRETRAINED = ROOT / "pretrained"   # downloaded model weights (bundled into submissions)
CONFIGS = ROOT / "configs"
RUNS = ROOT / "runs"               # one folder per experiment run
RUNS_INDEX = RUNS / "index.csv"    # one row per run: the experiment log
SUBMISSIONS = ROOT / "submissions" # built HF model repos (one folder each)
NOTES = ROOT / "notes"
LEDGER = NOTES / "submissions.csv" # every leaderboard submission + its score
TEMPLATE = ROOT / "submission_template"

TRAIN_FILE = RAW / "train.parquet"
VAL_INPUTS_FILE = RAW / "val_inputs.parquet"


def ensure_dirs():
    for p in (RAW, FEATURE_CACHE, SPLIT_CACHE, RUNS, SUBMISSIONS, NOTES):
        p.mkdir(parents=True, exist_ok=True)
