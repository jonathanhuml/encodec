#!/usr/bin/env bash

# Source this file from the repo root:
#   source start.sh
#
# Optional:
#   START_SKIP_INSTALL=1 source start.sh

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Run this with: source start.sh"
  exit 1
fi

set -e

START_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${START_SH_DIR}/.venv"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Creating virtualenv at ${VENV_DIR}"
  python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

export PYTHONPATH="${START_SH_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${START_SH_DIR}/.matplotlib"
mkdir -p "${MPLCONFIGDIR}"

if [[ "${START_SKIP_INSTALL:-0}" != "1" ]]; then
  if ! python - <<'PY' >/dev/null 2>&1
import encodec
import einops
import matplotlib
import numpy
import torch
import torchaudio
PY
  then
    echo "Installing project dependencies into ${VENV_DIR}"
    python -m pip install --upgrade pip
    python -m pip install -e .
    python -m pip install matplotlib
  fi
fi

echo "Activated ${VENV_DIR}"
echo "Python: $(python --version)"
python - <<'PY'
try:
    import torch
except ModuleNotFoundError:
    print("PyTorch: not installed yet")
else:
    print(f"PyTorch: {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
PY
