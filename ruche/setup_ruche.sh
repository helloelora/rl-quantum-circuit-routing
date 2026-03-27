#!/bin/bash
# =============================================================================
# setup_ruche.sh — One-time setup for La Ruche HPC
# Run this ONCE from the login node after cloning the repo.
# Usage:  bash setup_ruche.sh
# =============================================================================
set -e

echo "=== Setting up RL Quantum Circuit Routing on La Ruche ==="

# ---------- 1. Move conda cache to $WORKDIR (avoid $HOME 50GB quota) ---------
if [ ! -L "$HOME/.conda" ] && [ -d "$HOME/.conda" ]; then
    echo "Moving .conda to \$WORKDIR to save \$HOME quota..."
    mv "$HOME/.conda" "$WORKDIR/.conda"
    ln -s "$WORKDIR/.conda" "$HOME/.conda"
elif [ ! -e "$HOME/.conda" ]; then
    mkdir -p "$WORKDIR/.conda"
    ln -s "$WORKDIR/.conda" "$HOME/.conda"
fi

# ---------- 2. Load conda module ---------------------------------------------
module purge
module load anaconda3/2020.02/gcc-9.2.0

# ---------- 3. Create conda environment --------------------------------------
ENV_NAME="rl_qrouting"

if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' already exists, skipping creation."
else
    echo "Creating conda environment '$ENV_NAME' with Python 3.10..."
    conda create -n "$ENV_NAME" python=3.10 -y
fi

# ---------- 4. Activate and install dependencies ------------------------------
source activate "$ENV_NAME"

echo "Installing PyTorch (CPU — this project is CPU-bound RL)..."
pip install torch --index-url https://download.pytorch.org/whl/cpu

echo "Installing project dependencies..."
pip install qiskit==1.0.2 gymnasium==0.29.1 networkx==3.2.1 numpy==1.26.4 matplotlib==3.8.3 shimmy[gymnasium]

echo ""
echo "=== Setup complete ==="
echo "Project dir:  $(pwd)"
echo "Conda env:    $ENV_NAME"
echo "To activate:  module load anaconda3/2020.02/gcc-9.2.0 && source activate $ENV_NAME"
