#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
conda run --no-capture-output -n myenv3.13 python -m pip install \
  --target .cache/deepseek-ocr2-packages -r requirements-deepseek.txt
conda run --no-capture-output -n myenv3.13 python scripts/run_deepseek_ocr.py --setup-only
