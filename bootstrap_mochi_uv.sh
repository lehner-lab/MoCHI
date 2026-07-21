#!/usr/bin/env bash
set -euo pipefail

MOCHI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOCHI_REPO="${MOCHI_REPO:-${MOCHI_ROOT}}"
MOCHI_VENV="${MOCHI_VENV:-${MOCHI_REPO}/.venv}"
PYTHON_SPEC="${PYTHON_SPEC:-3.11}"
UV_INSTALL_DIR="${UV_INSTALL_DIR:-${HOME}/.local/bin}"

ensure_uv() {
    if command -v uv >/dev/null 2>&1; then
        return
    fi

    echo "uv not found; installing it into ${UV_INSTALL_DIR}"

    if ! command -v curl >/dev/null 2>&1; then
        echo "curl is required to install uv automatically" >&2
        exit 1
    fi

    installer="$(mktemp)"
    trap 'rm -f "${installer}"' EXIT

    curl -LsSf https://astral.sh/uv/install.sh -o "${installer}"
    env UV_INSTALL_DIR="${UV_INSTALL_DIR}" sh "${installer}"
    export PATH="${UV_INSTALL_DIR}:${PATH}"

    if ! command -v uv >/dev/null 2>&1; then
        echo "uv installation completed but uv is still not on PATH" >&2
        exit 1
    fi
}

if [ ! -f "${MOCHI_REPO}/pyproject.toml" ]; then
    echo "MoCHI pyproject.toml not found at ${MOCHI_REPO}/pyproject.toml" >&2
    exit 1
fi

if [ ! -f "${MOCHI_REPO}/uv.lock" ]; then
    echo "MoCHI uv.lock not found at ${MOCHI_REPO}/uv.lock" >&2
    exit 1
fi

ensure_uv

echo "Syncing MoCHI runtime dependencies"
echo "  project: ${MOCHI_REPO}"
echo "  venv:    ${MOCHI_VENV}"
UV_PROJECT_ENVIRONMENT="${MOCHI_VENV}" uv sync --project "${MOCHI_REPO}" --frozen --python "${PYTHON_SPEC}"

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
