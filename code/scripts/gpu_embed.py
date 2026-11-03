"""Compute ONLY the protein-embedding (ESM) features of some configs, e.g. on a free GPU,
and pack them into one zip to unzip into the project on your laptop.

    python scripts/gpu_embed.py configs/lgbm_lf_l8avg.yaml configs/lgbm_lf_l6.yaml --zip esm_cache.zip

On the laptop, in the project folder:   unzip -n esm_cache.zip
(-n never overwrites embeddings you already have). After that, run.py finds the
embeddings in data/cache/features/ and starts training straight away.

The embeddings are computed in full precision (fp32) on the GPU; the grader recomputes
them on its CPU at inference, and the two agree to ~1e-5, far below anything that
changes a prediction. Never use quantize: int8 for this (int8 is CPU-only).
"""
import argparse
import time
import zipfile

import _bootstrap  # noqa: F401
import pandas as pd
from kcatkm import config, data, paths
from kcatkm.features import make_extractors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--zip", default="esm_cache.zip")
    ap.add_argument("--allow-cpu", action="store_true", help="run even without a GPU (hours instead of minutes)")
    args = ap.parse_args()

    parts = [data.load_val_inputs()["sequence"], data.load_train()["sequence"]]
    from kcatkm import extra
    from kcatkm import paths as _paths
    for f in sorted(_paths.RAW.glob("extra*.parquet")):  # imported extra training data (all variants)
        parts.append(pd.read_parquet(f, columns=["sequence"])["sequence"])
    seqs = pd.concat(parts).dropna().unique().tolist()
    print(f"{len(seqs)} unique sequences (val + train" + (" + extra" if len(parts) > 2 else "") + ")")
    try:
        import torch
        if not torch.cuda.is_available() and not args.allow_cpu:
            raise SystemExit("No GPU found. In Colab: Runtime -> Change runtime type -> T4 GPU, then rerun "
                             "(on a CPU this takes hours; pass --allow-cpu to do it anyway).")
        print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (--allow-cpu)")
    except ImportError:
        raise SystemExit("torch is not installed")

    keys = []
    for path in args.configs:
        for ex in make_extractors(config.load(path)["features"]):
            if ex.name != "esm" or ex.key in keys:
                continue
            if ex.params.get("quantize"):
                raise SystemExit(f"{path}: int8 ESM runs on CPU only; compute it on the laptop")
            t0 = time.time()
            print(f"{path}: {ex}", flush=True)
            ex.transform(seqs)
            keys.append(ex.key)
            print(f"  done in {time.time() - t0:.0f}s", flush=True)

    with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as z:
        for k in keys:
            for f in (paths.FEATURE_CACHE / k).rglob("*"):
                if f.is_file():
                    z.write(f, f.relative_to(paths.ROOT))
    from pathlib import Path
    size = Path(args.zip).stat().st_size
    print(f"wrote {args.zip} ({size / 2**20:.0f} MiB) with {len(keys)} embedding sets -> download it, then "
          "in the project folder on your laptop:  unzip -n " + args.zip.split("/")[-1])


if __name__ == "__main__":
    main()
