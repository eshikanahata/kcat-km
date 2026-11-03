"""Loading the public data and validating inputs/outputs against the task contract."""
import numpy as np
import pandas as pd

from . import INPUT_COLUMNS, TARGETS
from . import paths


def load_train(path=None) -> pd.DataFrame:
    df = pd.read_parquet(path or paths.TRAIN_FILE)
    missing = set(INPUT_COLUMNS + TARGETS) - set(df.columns)
    if missing:
        raise ValueError(f"train file is missing columns: {missing}")
    return df.reset_index(drop=True)


def fingerprint(df: pd.DataFrame) -> str:
    """Short id of a training-data version (ids + labels). Runs on different data are not comparable."""
    import hashlib
    h = hashlib.sha1()
    h.update("\x00".join(df["id"].astype(str)).encode())
    h.update(np.ascontiguousarray(df[list(TARGETS)].to_numpy(dtype=np.float64)).tobytes())
    return h.hexdigest()[:8]


def current_fingerprint() -> str:
    return fingerprint(load_train())


def load_val_inputs(path=None) -> pd.DataFrame:
    df = pd.read_parquet(path or paths.VAL_INPUTS_FILE)
    check_inputs(df)
    return df.reset_index(drop=True)


def check_inputs(df: pd.DataFrame):
    missing = set(INPUT_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"input frame is missing columns: {missing}")
    if df["id"].duplicated().any():
        raise ValueError("duplicate ids in input")


def check_predictions(pred: pd.DataFrame, inputs: pd.DataFrame):
    """Mirror of the leaderboard's failure conditions. Raises on any violation."""
    if list(pred.columns) != ["id", *TARGETS]:
        raise ValueError(f"columns must be exactly id,{','.join(TARGETS)}; got {list(pred.columns)}")
    if pred["id"].duplicated().any():
        raise ValueError("duplicated ids in predictions")
    got, want = set(pred["id"]), set(inputs["id"])
    if got != want:
        raise ValueError(f"id mismatch: {len(want - got)} missing, {len(got - want)} extra")
    vals = pred[list(TARGETS)].to_numpy(dtype=float)
    if not np.isfinite(vals).all():
        raise ValueError("non-finite predictions")
    if (np.abs(vals) > 1e6).any():
        raise ValueError("predictions exceed |1e6|")
