#!/usr/bin/env zsh
# Auralux — Start local API server on macOS Apple Silicon
#
# This starts our thin adapter server (server.py) which imports the
# ACE-Step 1.5 Python package for real inference.  Models are auto-
# downloaded from HuggingFace on first generation request.
#
# Requirements: macOS with Apple Silicon (arm64)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ACE_STEP_DIR="$SCRIPT_DIR/ACE-Step-1.5"
VENV_DIR="$ACE_STEP_DIR/.venv"
PORT="${AURALUX_SERVER_PORT:-8765}"

# ── Verify macOS ARM64 ───────────────────────────────────────────────────

if [[ "$(uname)" != "Darwin" ]]; then
    echo "ERROR: This script is for macOS only."
    exit 1
fi

ARCH="$(uname -m)"
if [[ "$ARCH" != "arm64" ]]; then
    echo "WARNING: Apple Silicon (arm64) recommended. Detected: $ARCH"
fi

# ── Setup environment if needed ──────────────────────────────────────────

if [ ! -d "$VENV_DIR" ]; then
    echo "[Auralux] First run — setting up environment …"
    "$SCRIPT_DIR/setup_env.sh"
fi

# ── Environment variables for macOS Apple Silicon ────────────────────────

# PyTorch MPS fallback — prevents Metal shader assertion crashes
# (e.g. masked_fill_scalar_strided_32bit read-only binding error)
# by transparently running unsupported ops on CPU instead.
export PYTORCH_ENABLE_MPS_FALLBACK=1

export ACESTEP_LM_BACKEND="${ACESTEP_LM_BACKEND:-mlx}"
export TOKENIZERS_PARALLELISM="false"
export ACESTEP_CONFIG_PATH="${ACESTEP_CONFIG_PATH:-acestep-v15-turbo}"
export ACESTEP_LM_MODEL_PATH="${ACESTEP_LM_MODEL_PATH:-acestep-5Hz-lm-0.6B}"
export ACESTEP_DEVICE="${ACESTEP_DEVICE:-auto}"
export ACESTEP_INIT_LLM="${ACESTEP_INIT_LLM:-auto}"

echo "============================================"
echo "  Auralux API — ACE-Step v1.5 (macOS)"
echo "============================================"
echo
echo "  Port:       $PORT"
echo "  DiT model:  $ACESTEP_CONFIG_PATH"
echo "  LM model:   $ACESTEP_LM_MODEL_PATH"
echo "  LM backend: $ACESTEP_LM_BACKEND"
echo "  Device:     $ACESTEP_DEVICE"
echo

# ── Activate venv and run server ─────────────────────────────────────────

source "$VENV_DIR/bin/activate"
exec python "$SCRIPT_DIR/server.py" --port "$PORT"
