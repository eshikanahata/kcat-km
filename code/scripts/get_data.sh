#!/usr/bin/env bash
# Download the public data and the organisers' starter (for reference) from Hugging Face.
set -euo pipefail
cd "$(dirname "$0")/.."
hf download starkAhmed43/leak-curber-kcat-km train.parquet val_inputs.parquet \
  --repo-type dataset --local-dir data/raw
hf download starkAhmed43/leak-curber-kcat-km-starter --local-dir reference/starter
echo "data in data/raw, starter in reference/starter"
