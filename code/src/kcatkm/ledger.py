"""Submission ledger (notes/submissions.csv): what was submitted, its local CV
score and its leaderboard score, plus quota tracking (2/day, 20 total)."""
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from . import paths

IST = ZoneInfo("Asia/Kolkata")
# Quota (updated 27 Oct by the instructor): no daily limit, 30 submissions counted from COUNT_FROM.
DAILY, TOTAL = None, 30
COUNT_FROM = "2026-10-27 15:20"
DEADLINE = datetime(2026, 11, 7, 23, 59, tzinfo=IST)
COLUMNS = ["created", "submitted", "sub_name", "public_name", "run_id", "repo_id", "revision",
           "cv_mean_scc", "cv_kcat_scc", "cv_km_scc",
           "lb_mean_scc", "lb_kcat_scc", "lb_km_scc", "lb_mean_rmse", "status", "notes"]


def now() -> str:
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M")


def read() -> pd.DataFrame:
    if paths.LEDGER.exists():
        return pd.read_csv(paths.LEDGER, dtype={"revision": str})
    return pd.DataFrame(columns=COLUMNS)


def write(df: pd.DataFrame):
    paths.LEDGER.parent.mkdir(parents=True, exist_ok=True)
    df.reindex(columns=COLUMNS).to_csv(paths.LEDGER, index=False)  # older ledgers lack newer columns


def upsert(sub_name: str, **fields):
    df = read()
    df = df.reindex(columns=COLUMNS)
    if (df["sub_name"] == sub_name).any():
        for k, v in fields.items():
            df[k] = df[k].astype(object)
            df.loc[df["sub_name"] == sub_name, k] = v
    else:
        df = pd.concat([df, pd.DataFrame([{"sub_name": sub_name, "created": now(), **fields}])],
                       ignore_index=True)
    write(df)


def quota() -> str:
    df = read()
    done = df[df["status"].isin(["submitted", "scored"])]
    done = done[done["submitted"].astype(str) >= COUNT_FROM]
    today = datetime.now(IST).strftime("%Y-%m-%d")
    n_today = int(done["submitted"].astype(str).str.startswith(today).sum())
    days_left = (DEADLINE.date() - datetime.now(IST).date()).days + 1
    daily = f"{n_today} today (no daily limit)" if DAILY is None else f"{n_today}/{DAILY} used today (IST)"
    return (f"quota: {daily}, {len(done)}/{TOTAL} used since {COUNT_FROM}; "
            f"{max(days_left, 0)} days incl. today until the {DEADLINE:%d %b %H:%M} IST deadline")
