#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
conda run --no-capture-output -n myenv3.13 python -m pip install \
  --target .cache/unlimited-ocr-packages -r requirements-unlimited.txt
conda run --no-capture-output -n myenv3.13 python scripts/run_unlimited_ocr.py --setup-only
