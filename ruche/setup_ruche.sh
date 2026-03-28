#!/bin/bash
# =============================================================================
# setup_ruche.sh — One-time setup: build Apptainer image
# Run from login node:  bash ruche/setup_ruche.sh
# =============================================================================
set -e

echo "=== Building Apptainer image for RL Quantum Routing ==="

module purge
module load apptainer/1.4.4/gcc-15.1.0

export APPTAINER_CACHEDIR=$WORKDIR/.apptainer_cache
export APPTAINER_TMPDIR=$WORKDIR/apptainer_tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

SIF_PATH="$WORKDIR/rl_qrouting.sif"
DEF_PATH="$WORKDIR/rl-quantum-circuit-routing/ruche/rl_qrouting.def"

if [ -f "$SIF_PATH" ]; then
    echo "Image already exists at $SIF_PATH"
    echo "Delete it and re-run to rebuild: rm $SIF_PATH"
else
    apptainer build "$SIF_PATH" "$DEF_PATH"
    echo "Image built: $SIF_PATH"
fi

echo ""
echo "=== Setup complete ==="
echo "Test with: apptainer exec --nv $SIF_PATH python -c \"import torch; print(torch.cuda.is_available())\""
