#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MOCHI_REPO="${REPO_ROOT}"
MOCHI_VENV="${MOCHI_VENV:-${MOCHI_REPO}/.venv}"
PYTHON_SPEC="${PYTHON_SPEC:-3.11}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

if [ ! -d "${MOCHI_REPO}" ]; then
    echo "MoCHI repository not found at ${MOCHI_REPO}" >&2
    exit 1
fi

cd "${MOCHI_REPO}"

echo "Creating virtual environment at ${MOCHI_VENV}"
uv venv "${MOCHI_VENV}" --python "${PYTHON_SPEC}"

echo "Installing MoCHI runtime dependencies with uv"
uv pip install --python "${MOCHI_VENV}/bin/python" \
    loguru \
    numpy \
    pandas \
    scipy \
    scikit-learn \
    matplotlib \
    seaborn \
    pyreadr \
    setuptools \
    wheel

echo "Installing PyTorch from ${TORCH_INDEX_URL}"
uv pip install --python "${MOCHI_VENV}/bin/python" --index-url "${TORCH_INDEX_URL}" torch

echo "Installing MoCHI package in editable mode"
uv pip install --python "${MOCHI_VENV}/bin/python" -e .

echo "Verifying environment"
"${MOCHI_VENV}/bin/python" - <<'PY'
import importlib
import sys

modules = [
    "loguru",
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "matplotlib",
    "seaborn",
    "pyreadr",
    "torch",
    "pymochi",
]

for name in modules:
    importlib.import_module(name)

import torch

print(f"python={sys.executable}")
print(f"torch={torch.__version__}")
print(f"torch_cuda_build={torch.version.cuda}")
PY
