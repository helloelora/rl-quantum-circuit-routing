#!/bin/bash
# =============================================================================
# setup_ruche.sh — One-time setup for La Ruche HPC (GPU with conda)
# Run this from the login node.
# Usage:  bash ruche/setup_ruche.sh
# =============================================================================
set -e

echo "=== Setting up RL Quantum Circuit Routing on La Ruche ==="

CONDA_DIR="$WORKDIR/miniconda3"
ENV_NAME="rl_qrouting"
ENV_DIR="$CONDA_DIR/envs/$ENV_NAME"

# ---------- 1. Install miniconda if not present --------------------------------
if [ ! -d "$CONDA_DIR" ]; then
    echo "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
    rm /tmp/miniconda.sh
fi

# Make conda available
eval "$($CONDA_DIR/bin/conda shell.bash hook)"

# ---------- 2. Create conda env with Python 3.12 + CUDA PyTorch ---------------
if [ -d "$ENV_DIR" ]; then
    echo "Conda env '$ENV_NAME' already exists, activating..."
    conda activate "$ENV_NAME"
else
    echo "Creating conda env with Python 3.12..."
    conda create -y -n "$ENV_NAME" python=3.12
    conda activate "$ENV_NAME"

    echo "Installing PyTorch (CUDA 12.4)..."
    pip install torch --index-url https://download.pytorch.org/whl/cu124

    echo "Installing project dependencies..."
    pip install qiskit gymnasium networkx numpy matplotlib pandas "shimmy[gymnasium]"
fi

echo ""
echo "=== Setup complete ==="
echo "Conda env:   $ENV_DIR"
echo "To activate: eval \"\$($CONDA_DIR/bin/conda shell.bash hook)\" && conda activate $ENV_NAME"
echo "Python:      $(python --version)"
echo "Torch:       $(python -c 'import torch; print(torch.__version__, \"CUDA:\", torch.cuda.is_available())')"
