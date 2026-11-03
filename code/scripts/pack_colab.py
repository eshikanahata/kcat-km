"""Pack what a GPU machine (e.g. Google Colab) needs to compute embeddings: code, configs and
the two data files (no caches, no model weights; those are downloaded there, which is fast).

    python scripts/pack_colab.py          # -> kcat-km-colab.zip in the project folder

Then in a Colab notebook with a GPU (Runtime -> Change runtime type -> T4 GPU), upload the zip and run:

    !unzip -q kcat-km-colab.zip -d kcat-km
    %cd kcat-km
    !pip install -q transformers==5.20.2 rdkit pyyaml pyarrow
    !python scripts/get_pretrained.py facebook/esm2_t12_35M_UR50D facebook/esm2_t6_8M_UR50D
    !python scripts/gpu_embed.py configs/lgbm_lf_l8avg.yaml configs/lgbm_lf_l6.yaml configs/lgbm_lf_l8_mm.yaml configs/lgbm_lf_esm8_mm.yaml
    from google.colab import files; files.download("esm_cache.zip")
"""
import zipfile

import _bootstrap  # noqa: F401
from kcatkm import paths


def main():
    out = paths.ROOT / "kcat-km-colab.zip"
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for top in ("src", "scripts", "configs"):
            for f in (paths.ROOT / top).rglob("*"):
                if f.is_file() and "__pycache__" not in f.parts:
                    z.write(f, f.relative_to(paths.ROOT))
                    n += 1
        for f in (paths.TRAIN_FILE, paths.VAL_INPUTS_FILE, paths.ROOT / "requirements-infer.txt",
                  *sorted(paths.RAW.glob("extra*.parquet"))):
            if f.exists():
                z.write(f, f.relative_to(paths.ROOT))
                n += 1
    print(f"wrote {out.name} ({out.stat().st_size / 2**20:.1f} MiB, {n} files). Upload it to Colab; "
          "the commands to run there are at the top of scripts/pack_colab.py")


if __name__ == "__main__":
    main()
