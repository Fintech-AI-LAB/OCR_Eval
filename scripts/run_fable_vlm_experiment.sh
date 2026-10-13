#!/usr/bin/env bash
# claude-fable-5-1 VLM OCR experiment: install SDK, offline tests, preflight, inference, evaluation.
# Requires ANTHROPIC_API_KEY in the environment (never printed or saved). Safe to rerun: resumes.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${ANTHROPIC_API_KEY:?Set ANTHROPIC_API_KEY first, e.g.: read -s ANTHROPIC_API_KEY && export ANTHROPIC_API_KEY}"
PY=(conda run --no-capture-output -n myenv3.13 python)
"${PY[@]}" -m pip install -r requirements-anthropic.txt
"${PY[@]}" -m unittest tests.test_fable_vlm_ocr
"${PY[@]}" scripts/run_fable_vlm_ocr.py --dry-run
"${PY[@]}" scripts/run_fable_vlm_ocr.py "$@"
"${PY[@]}" scripts/evaluate_fable_vlm.py
