"""Track leaderboard submissions and how local CV relates to leaderboard score.

    python scripts/ledger.py                         # show ledger + quota + CV-vs-LB agreement
    python scripts/ledger.py submitted s03_lgbm      # mark as submitted (counts toward quota)
    python scripts/ledger.py reset s03_lgbm          # undo a 'submitted' marked by mistake
    python scripts/ledger.py score s03_lgbm --mean 0.412 --kcat 0.451 --km 0.373 --rmse 1.08
"""
import argparse

import pandas as pd

import _bootstrap  # noqa: F401
from kcatkm import ledger


def show():
    df = ledger.read()
    if df.empty:
        print("no submissions yet")
    else:
        cols = ["submitted", "sub_name", "public_name", "status", "cv_mean_scc", "lb_mean_scc", "cv_kcat_scc",
                "lb_kcat_scc", "cv_km_scc", "lb_km_scc", "revision", "notes"]
        with pd.option_context("display.width", 250, "display.max_colwidth", 40):
            print(df[cols].to_string(index=False))
        scored = df.dropna(subset=["lb_mean_scc"])
        if len(scored) >= 3:
            r = scored["cv_mean_scc"].astype(float).corr(scored["lb_mean_scc"].astype(float), method="spearman")
            gap = (scored["lb_mean_scc"].astype(float) - scored["cv_mean_scc"].astype(float))
            print(f"\nCV vs LB over {len(scored)} scored submissions: rank corr {r:.2f}, "
                  f"mean gap (LB - CV) {gap.mean():+.4f}")
            print("Low rank correlation means local CV is not predicting the leaderboard: revisit the split.")
    print("\n" + ledger.quota())


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("submitted")
    s.add_argument("name")
    s.add_argument("--notes")
    rs = sub.add_parser("reset", help="undo 'submitted' (e.g. marked by mistake): back to 'uploaded'")
    rs.add_argument("name")
    sc = sub.add_parser("score")
    sc.add_argument("name")
    sc.add_argument("--mean", type=float, required=True)
    sc.add_argument("--kcat", type=float)
    sc.add_argument("--km", type=float)
    sc.add_argument("--rmse", type=float)
    args = ap.parse_args()

    if args.cmd == "submitted":
        fields = {"status": "submitted", "submitted": ledger.now()}
        if args.notes:
            fields["notes"] = args.notes
        ledger.upsert(args.name, **fields)
    elif args.cmd == "reset":
        ledger.upsert(args.name, status="uploaded", submitted=None)
    elif args.cmd == "score":
        row = ledger.read().query("sub_name == @args.name")
        fields = {"status": "scored", "lb_mean_scc": args.mean, "lb_kcat_scc": args.kcat,
                  "lb_km_scc": args.km, "lb_mean_rmse": args.rmse}
        if row.empty or pd.isna(row.iloc[0]["submitted"]):
            fields["submitted"] = ledger.now()
        ledger.upsert(args.name, **fields)
    show()


if __name__ == "__main__":
    main()
