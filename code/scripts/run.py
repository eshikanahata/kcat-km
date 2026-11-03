"""Run a cross-validated experiment from a config.

    python scripts/run.py configs/lgbm_handcrafted.yaml
    python scripts/run.py lgbm_handcrafted.yaml --set model.params.learning_rate=0.05 --name lgbm_lr05
    python scripts/run.py configs/a.yaml configs/b.yaml          # several in a row

With several configs, a failing one is reported and the rest still run.
"""
import argparse
import sys
import traceback

import _bootstrap  # noqa: F401
from kcatkm import config, experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    ap.add_argument("--name", help="run name (defaults to config name)")
    ap.add_argument("--notes", default="", help="free text stored in runs/index.csv")
    ap.add_argument("--force", action="store_true", help="rerun even if an identical config exists")
    ap.add_argument("--no-val", action="store_true",
                    help="skip scoring the labelled validation set (data/labels/), which costs one extra full fit")
    args = ap.parse_args()

    failed = []
    for path in args.configs:
        try:
            cfg = config.apply_overrides(config.load(path), args.set)
            if args.name:
                cfg["name"] = args.name
            print(f"\n=== {cfg['name']} ===")
            experiment.run_cv(cfg, force=args.force, notes=args.notes, val=not args.no_val)
        except KeyboardInterrupt:
            raise
        except Exception:
            traceback.print_exc()
            failed.append(path)
            print(f"!!! {path} failed; continuing with the next config", file=sys.stderr)
    if failed:
        print(f"\n{len(failed)} of {len(args.configs)} configs failed: {', '.join(failed)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
