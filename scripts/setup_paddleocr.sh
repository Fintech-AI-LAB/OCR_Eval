#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
conda run --no-capture-output -n myenv3.13 python -m pip install --only-binary=:all: --upgrade --target .cache/paddleocr-packages -r requirements-paddleocr.txt
conda run --no-capture-output -n myenv3.13 python scripts/run_paddleocr_v4.py --smoke-test
