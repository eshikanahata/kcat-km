"""Download a pretrained model into pretrained/<basename>/ for offline use.

    python scripts/get_pretrained.py facebook/esm2_t12_35M_UR50D
    python scripts/get_pretrained.py facebook/esm2_t6_8M_UR50D facebook/esm2_t30_150M_UR50D

ESM-2 sizes (hidden dim): t6_8M (320), t12_35M (480), t30_150M (640), t33_650M (1280).
Only config, tokenizer files and one set of weights are fetched.
"""
import sys

import _bootstrap  # noqa: F401
from huggingface_hub import snapshot_download
from kcatkm import paths


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for repo in sys.argv[1:]:
        dest = paths.PRETRAINED / repo.split("/")[-1]
        snapshot_download(repo, local_dir=dest, allow_patterns=["*.json", "*.txt", "*.safetensors"])
        if not list(dest.glob("*.safetensors")):
            snapshot_download(repo, local_dir=dest, allow_patterns=["pytorch_model.bin"])
        size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file() and ".cache" not in f.parts)
        print(f"{repo} -> {dest} ({size / 2**20:.0f} MiB)")


if __name__ == "__main__":
    main()
