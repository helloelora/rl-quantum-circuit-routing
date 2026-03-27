#!/bin/bash
# =============================================================================
# setup_ruche.sh — One-time setup for La Ruche HPC
# Run this from the login node (or an interactive cpu_short session).
# Usage:  bash ruche/setup_ruche.sh
# =============================================================================
set -e

echo "=== Setting up RL Quantum Circuit Routing on La Ruche ==="

# ---------- 1. Find a working Python 3 ----------------------------------------
# Try system python3, then search for a module
if command -v python3 &>/dev/null; then
    PY=$(command -v python3)
    echo "Found system python3: $PY ($(python3 --version))"
else
    echo "No python3 in PATH. Trying to find a python module..."
    # Try common module names on Ruche
    for mod in python/3.10 python/3.9 python/3.8 python-nospack/3.6.8/gcc-9.2.0; do
        if module load "$mod" 2>/dev/null; then
            echo "Loaded module: $mod"
            PY=$(command -v python3)
            break
        fi
    done
    if [ -z "$PY" ]; then
        echo "ERROR: Cannot find Python 3. Run 'module spider python' and update this script."
        exit 1
    fi
fi

# ---------- 2. Create venv in $WORKDIR (avoid $HOME 50GB quota) ---------------
VENV_DIR="$WORKDIR/venvs/rl_qrouting"

if [ -d "$VENV_DIR" ]; then
    echo "Venv already exists at $VENV_DIR, skipping creation."
else
    echo "Creating venv at $VENV_DIR ..."
    $PY -m venv "$VENV_DIR"
fi

# ---------- 3. Activate and install dependencies ------------------------------
source "$VENV_DIR/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing PyTorch (CPU)..."
pip install torch --index-url https://download.pytorch.org/whl/cpu

echo "Installing project dependencies..."
pip install qiskit==1.0.2 gymnasium==0.29.1 networkx==3.2.1 numpy==1.26.4 matplotlib==3.8.3 "shimmy[gymnasium]"

echo ""
echo "=== Setup complete ==="
echo "Venv:        $VENV_DIR"
echo "To activate: source $VENV_DIR/bin/activate"
echo "Python:      $(python --version)"
echo "Torch:       $(python -c 'import torch; print(torch.__version__)')"
