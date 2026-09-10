#!/usr/bin/env bash
# Install into the existing Conda environment myenv3.13; run from any directory.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == "--help" ]]; then
  echo 'Usage: bash scripts/setup_mineru.sh [--skip-smoke-test]'
  echo 'Downloads the checkpoint and tests MPS inference by default.'
  exit 0
fi
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--skip-smoke-test" ) ]]; then
  echo 'Unknown argument; use --help.' >&2; exit 2
fi
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo 'Run this script on macOS with native Apple Silicon (not Rosetta).' >&2
  exit 1
fi
CONDA="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "$CONDA" && -x /opt/miniconda3/bin/conda ]]; then
  CONDA=/opt/miniconda3/bin/conda
fi
if [[ -z "$CONDA" ]]; then
  echo 'Conda was not found. Initialize Conda or set CONDA_EXE.' >&2; exit 1
fi
ENV_NAME=myenv3.13
RUN=("$CONDA" run --no-capture-output -n "$ENV_NAME")
"${RUN[@]}" python -c '
import platform, sys
if not (3, 10) <= sys.version_info[:2] < (3, 14):
    raise SystemExit("MinerU requires Python 3.10–3.13; myenv3.13 has " + platform.python_version() +
                     ". Change this environment to a supported Python version before setup.")
if platform.machine() != "arm64":
    raise SystemExit("The Conda environment must use native arm64 Python for MPS.")
'
export HF_HOME="$ROOT/.cache/huggingface"
export PYTORCH_ENABLE_MPS_FALLBACK=1
"${RUN[@]}" python -m pip install -r "$ROOT/requirements-mineru.txt"
"${RUN[@]}" python -c "
import torch
from huggingface_hub import snapshot_download
if not torch.backends.mps.is_available():
    raise SystemExit('PyTorch MPS is unavailable. Check native arm64 Python and macOS support.')
x = torch.ones((16, 16), device='mps')
assert (x @ x).cpu()[0, 0].item() == 16
print('MPS tensor check passed; downloading the requested model.')
snapshot_download('opendatalab/MinerU2.5-Pro-2604-1.2B')
"
"${RUN[@]}" python -m pip freeze > "$ROOT/requirements-mineru-resolved.txt"
if [[ "${1:-}" != --skip-smoke-test ]]; then
  "${RUN[@]}" python "$ROOT/scripts/run_mineru.py" --smoke-test
fi
echo "Setup complete. Run: conda run -n $ENV_NAME python $ROOT/scripts/run_mineru.py"
