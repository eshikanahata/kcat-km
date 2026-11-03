"""Benchmark a config's features under grader conditions, then fill the cache.

    python scripts/precompute.py configs/lgbm_esm.yaml                    # benchmark + cache val and train
    python scripts/precompute.py configs/lgbm_esm.yaml --benchmark-only   # just: will it fit in 10 minutes?

1. Benchmark: computes every feature for the public val inputs from scratch
   with 2 CPU threads (what the grader gives you), including model loading.
   The hidden test set is a similar size, so this is your inference estimate.
2. Cache: computes features for val and all training rows with every core
   (or the GPU) and stores them, so `run.py` starts instantly afterwards.
   Interrupt any time; slow extractors resume where they stopped.
"""
import argparse
import os
import time

import _bootstrap  # noqa: F401
from kcatkm import config, data
from kcatkm.features import make_extractors

LIMIT = 600


def set_threads(n):
    try:
        import torch
        torch.set_num_threads(n)
    except ImportError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--benchmark-only", action="store_true")
    ap.add_argument("--skip-benchmark", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()

    cfg = config.apply_overrides(config.load(args.config), args.set)
    val, train = data.load_val_inputs(), data.load_train()

    if not args.skip_benchmark:
        print(f"benchmark on {len(val)} val rows with 2 threads (grader conditions):")
        default_threads = os.cpu_count()
        set_threads(2)
        total = 0.0
        per_input = {"sequence": [], "smiles": []}  # seconds per unique input, per extractor
        for ex in make_extractors(cfg["features"]):  # fresh objects: model loading is included
            t0 = time.time()
            ex.transform(val[ex.entity].tolist(), use_cache=False)
            dt = time.time() - t0
            total += dt
            n_uniq = val[ex.entity].nunique()
            print(f"  {str(ex):60s} {dt:7.1f}s  ({n_uniq} unique inputs)")
            per_input[ex.entity].append(dt / n_uniq)
        verdict = "OK" if total < LIMIT * 0.5 else ("RISKY" if total < LIMIT * 0.8 else "TOO SLOW")
        print(f"  total feature time {total:.0f}s of the {LIMIT}s limit -> {verdict}"
              " (your laptop CPU may differ from the grader's; keep a 2x margin)")
        # Hidden test set (dataset manifest, 23 Oct): 1,714 rows, 848 unique sequences.
        # Unique SMILES are not published; estimate them from the val ratio.
        s_seq, s_smi = sum(per_input["sequence"]), sum(per_input["smiles"])
        n_smi_test = int(round(val["smiles"].nunique() * 1714 / len(val)))
        print("  projected feature time (your machine, 2 threads):")
        for label, n_seq, n_smi in [("val", val["sequence"].nunique(), val["smiles"].nunique()),
                                    ("TEST (848 sequences)", 848, n_smi_test),
                                    ("worst case (all 1,714 unique)", 1714, 1714)]:
            est = n_seq * s_seq + n_smi * s_smi
            flag = "OK" if est < 330 else ("tight" if est < 450 else "the ESM time guard would shorten sequences")
            print(f"    {label:32s} {n_seq:5d} seqs, {n_smi:5d} SMILES -> ~{est:4.0f}s  {flag}")
        print("  (predict.py stops the full model at ~465 s; the grader may be slower than your laptop)")
        set_threads(default_threads)
        if args.benchmark_only:
            return

    print(f"\ncaching features for val ({len(val)}) and train ({len(train)}) rows:")
    for ex in make_extractors(cfg["features"]):
        t0 = time.time()
        ex.transform(val[ex.entity].tolist())
        ex.transform(train[ex.entity].tolist())
        print(f"  {str(ex):60s} done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
