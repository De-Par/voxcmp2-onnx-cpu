#!/usr/bin/env bash
#
# Source this file from the repository root:
#
#   source setup.sh base
#   source setup.sh dev
#
# The script creates/activates .venv, restores the VoxCPM submodule when it is
# missing, and installs this project in editable mode. It is intentionally
# sourced so the virtual environment remains active in the caller's shell.
#
# Modes:
#   base = core runtime dependencies + export/parity dependencies
#   dev  = base + developer tools

IS_SOURCED=0
(return 0 2>/dev/null) && IS_SOURCED=1

if [ "$IS_SOURCED" -ne 1 ]; then
    echo "[ERROR] Run this script via: source setup.sh [base|dev]"
    exit 1
fi

if [ -n "${ZSH_VERSION:-}" ]; then
    SCRIPT_SOURCE="${(%):-%N}"
elif [ -n "${BASH_SOURCE[0]:-}" ]; then
    SCRIPT_SOURCE="${BASH_SOURCE[0]}"
else
    SCRIPT_SOURCE="$0"
fi

SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_SOURCE")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}" && pwd -P)"

if [ ! -f "${PROJECT_ROOT}/pyproject.toml" ]; then
    echo "[ERROR] Could not resolve project root from: ${SCRIPT_SOURCE}" >&2
    echo "[ERROR] Resolved PROJECT_ROOT='${PROJECT_ROOT}', but pyproject.toml was not found there." >&2
    return 1
fi

cd "${PROJECT_ROOT}" || return 1

_VOXCPM2_MODE="${1:-base}"
case "$_VOXCPM2_MODE" in
  base)
        _VOXCPM2_EXTRAS="export"
        ;;
  dev)
        _VOXCPM2_EXTRAS="export,dev"
        ;;
  *)
        echo "Usage: source setup.sh <base|dev>" >&2
        echo "  base: install core dependencies plus export/parity dependencies" >&2
        echo "  dev:  install base plus developer tools" >&2
        return 1
        ;;
esac

python3 -m venv .venv || return 1

. .venv/bin/activate || return 1

python -m pip install --upgrade "pip>=24,<26" "setuptools>=70,<81" "wheel>=0.43,<1" || return 1

python -m pip install -e ".[${_VOXCPM2_EXTRAS}]" || return 1

if [ ! -d "third_party/VoxCPM" ] && [ -f ".gitmodules" ]; then
    if command -v git >/dev/null 2>&1; then
        echo "[INFO] third_party/VoxCPM is missing; initializing git submodules."
        git submodule update --init --recursive || return 1
    else
        echo "[ERROR] third_party/VoxCPM is missing and git is not available." >&2
        echo "[ERROR] Install git, then run: git submodule update --init --recursive" >&2
        return 1
    fi
fi

if [ -d "third_party/VoxCPM" ]; then
    python -m pip install -e "third_party/VoxCPM" --no-deps || return 1
else
    echo "[ERROR] third_party/VoxCPM is still missing after setup." >&2
    return 1
fi

rm -rf ./*.egg-info

echo "[INFO] VoxCPM2 ONNX CPU environment is active."
echo "[INFO] Repository: $PROJECT_ROOT"
echo "[INFO] Python: $(python -c 'import sys; print(sys.executable)')"
echo "[INFO] Mode: ${_VOXCPM2_MODE}"
echo "[INFO] Installed project extras: ${_VOXCPM2_EXTRAS}"
