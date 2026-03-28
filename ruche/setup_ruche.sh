#!/bin/bash
# =============================================================================
# setup_ruche.sh — One-time setup for La Ruche HPC
# Run this from the login node (or an interactive cpu_short session).
# Usage:  bash ruche/setup_ruche.sh
# =============================================================================
set -e

echo "=== Setting up RL Quantum Circuit Routing on La Ruche ==="

# ---------- 1. Load Python module ---------------------------------------------
module purge
module load python/3.14.0/gcc-15.1.0
PY=$(command -v python3)
echo "Using: $PY ($(python3 --version))"

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

# Skip installation if packages are already present
if python -c "import torch, qiskit, gymnasium" 2>/dev/null; then
    echo "Dependencies already installed, skipping pip install."
else
    echo "Upgrading pip..."
    pip install --upgrade pip

    echo "Installing PyTorch (CUDA)..."
    pip install torch --index-url https://download.pytorch.org/whl/cu124

    echo "Installing project dependencies..."
    pip install qiskit gymnasium networkx numpy matplotlib pandas "shimmy[gymnasium]"
fi

echo ""
echo "=== Setup complete ==="
echo "Venv:        $VENV_DIR"
echo "To activate: source $VENV_DIR/bin/activate"
echo "Python:      $(python --version)"
echo "Torch:       $(python -c 'import torch; print(torch.__version__)')"
