"""Leaderboard metrics: Spearman (SCC) and RMSE per target, plus their means."""
import numpy as np
from scipy.stats import spearmanr

from . import TARGETS


def scc(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    if np.ptp(p) == 0 or np.ptp(y) == 0:  # ptp is exact; std of equal floats can be 1e-17
        return 0.0  # leaderboard gives constant predictions SCC 0
    return float(spearmanr(y, p).statistic)


def rmse(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """y_true / y_pred: arrays of shape (n, 2) in TARGETS order."""
    out = {}
    for j, t in enumerate(TARGETS):
        short = t.replace("log10_", "")
        out[f"{short}_scc"] = scc(y_true[:, j], y_pred[:, j])
        out[f"{short}_rmse"] = rmse(y_true[:, j], y_pred[:, j])
    out["mean_scc"] = (out["kcat_scc"] + out["km_scc"]) / 2
    out["mean_rmse"] = (out["kcat_rmse"] + out["km_rmse"]) / 2
    return out


def testlike(y_true, y_pred, sequences, max_per=2, draws=10, seed=0) -> dict:
    """Score as if every enzyme had at most `max_per` rows, like the leaderboard's test set
    (1,714 rows for 848 enzymes), instead of letting heavily measured enzymes dominate.
    Averages `draws` random subsamples. Returns the same keys as score() (SCC only)."""
    import pandas as pd
    codes = pd.factorize(pd.Series(sequences).astype(str))[0]
    rng = np.random.default_rng(seed)
    acc = {"kcat_scc": [], "km_scc": []}
    for _ in range(draws):
        order = rng.permutation(len(codes))
        rank_in_seq = pd.Series(order).groupby(codes[order]).cumcount().to_numpy()
        keep = np.sort(order[rank_in_seq < max_per])
        for j, t in enumerate(TARGETS):
            acc[f"{t.replace('log10_', '')}_scc"].append(scc(y_true[keep, j], y_pred[keep, j]))
    out = {k: float(np.mean(v)) for k, v in acc.items()}
    out["mean_scc"] = (out["kcat_scc"] + out["km_scc"]) / 2
    return out
