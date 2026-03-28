#!/bin/bash
#SBATCH --job-name=quantum_routing
#SBATCH --output=outputs/slurm_%j.out
#SBATCH --error=outputs/slurm_%j.err
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --partition=gpu_prod_long
#SBATCH --exclude=sh07,sh[10-19]
#SBATCH --propagate=NONE
#SBATCH --mail-user=ali.dor@student-cs.fr
#SBATCH --mail-type=END,FAIL

# ============================================================
# Quantum Circuit Routing — D3QN+PER Training + Evaluation
#
# Usage:
#   sbatch experiment.slurm <PRESET> [extra args...]
#
# Presets:
#   linear5    — 2k episodes, ~5 min (sanity check)
#   heavy_hex  — 20k episodes, ~6 hours (primary)
#   multi      — 30k episodes, ~8-10 hours (generalization)
#
# Examples:
#   sbatch experiment.slurm linear5
#   sbatch experiment.slurm heavy_hex --seed 123
#   sbatch experiment.slurm multi --episodes 50000
#
# Output:
#   outputs/run_NNN/  (config, logs, checkpoints, figures, eval)
# ============================================================

PRESET="${1:?Usage: sbatch experiment.slurm <PRESET> [args...]}"
shift

PROJECT_DIR="/usr/users/rl_for_wsi/dor_ali/projects/rl-quantum-circuit-routing"
cd "$PROJECT_DIR"

mkdir -p outputs

echo "============================================================"
echo "QUANTUM CIRCUIT ROUTING EXPERIMENT"
echo "Preset:     ${PRESET}"
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       ${SLURM_NODELIST}"
echo "Extra args: $@"
echo "Start time: $(date)"
echo "============================================================"

# ======================== SETUP VENV ========================
if [ ! -f ".venv/setup_done" ]; then
    echo ""
    echo "========== SETTING UP ENVIRONMENT =========="
    echo ""

    # Create venv if needed
    if [ ! -d ".venv" ]; then
        ~/.local/bin/virtualenv .venv
    fi

    source .venv/bin/activate

    pip install --upgrade pip
    pip install torch --index-url https://download.pytorch.org/whl/cu118
    pip install qiskit==1.0.2 gymnasium==0.29.1 networkx==3.2.1 \
                numpy==1.26.4 matplotlib==3.8.3 imageio tqdm

    touch .venv/setup_done
    echo "Environment setup complete."
else
    source .venv/bin/activate
    echo "Using existing environment."
fi

export PYTHONUNBUFFERED=1

# ======================== PHASE 1: TRAINING ========================
echo ""
echo "========== PHASE 1: TRAINING (${PRESET}) =========="
echo ""

# If preset is a path to a .json file, use --config; otherwise use --preset
if [[ "${PRESET}" == *.json ]]; then
    # Auto-resolve: if file not found, check configs/ directory
    if [[ ! -f "${PRESET}" && -f "configs/${PRESET}" ]]; then
        PRESET="configs/${PRESET}"
    fi
    echo "Using config file: ${PRESET}"
    python3 -u main.py train \
        --config "${PRESET}" \
        --output-dir outputs \
        --device cuda \
        "$@"
else
    python3 -u main.py train \
        --preset "${PRESET}" \
        --output-dir outputs \
        --device cuda \
        "$@"
fi

TRAIN_EXIT=$?

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "ERROR: Training failed with exit code ${TRAIN_EXIT}"
    echo "End time: $(date)"
    exit $TRAIN_EXIT
fi

echo ""
echo "Training complete at $(date)"

# ======================== PHASE 2: EVALUATION ========================
# Find THIS job's run directory from its own stdout (avoids race with concurrent jobs)
RUN_DIR=$(grep -m1 "Run directory:" "outputs/slurm_${SLURM_JOB_ID}.out" 2>/dev/null | awk '{print $NF}')
if [ -z "$RUN_DIR" ]; then
    # Fallback if grep fails
    RUN_DIR=$(ls -td outputs/run_* 2>/dev/null | head -1)
fi
LATEST_RUN="$RUN_DIR"

# Move SLURM logs into run dir (mv on same FS keeps SLURM's FD writing to the inode)
if [ -n "$LATEST_RUN" ]; then
    mv "outputs/slurm_${SLURM_JOB_ID}.out" "${LATEST_RUN}/slurm_${SLURM_JOB_ID}.out" 2>/dev/null || true
    mv "outputs/slurm_${SLURM_JOB_ID}.err" "${LATEST_RUN}/slurm_${SLURM_JOB_ID}.err" 2>/dev/null || true
fi

if [ -z "$LATEST_RUN" ]; then
    echo "WARNING: No run directory found."
    echo "End time: $(date)"
    exit 0
fi

# Find checkpoint
CHECKPOINT="${LATEST_RUN}/checkpoints/checkpoint_final.pt"

if [ ! -f "${CHECKPOINT}" ]; then
    echo "WARNING: No final checkpoint at ${CHECKPOINT}"
    echo "End time: $(date)"
    exit 0
fi

echo ""
echo "========== PHASE 2: EVALUATION =========="
echo ""

python3 -u main.py evaluate \
    --checkpoint "${CHECKPOINT}" \
    --episodes 100 \
    --save-trajectories \
    --output-dir "${LATEST_RUN}/eval"

EVAL_EXIT=$?

if [ $EVAL_EXIT -ne 0 ]; then
    echo "WARNING: Evaluation failed with exit code ${EVAL_EXIT}"
fi

# ======================== PHASE 3: VISUALIZATIONS ========================
echo ""
echo "========== PHASE 3: VISUALIZATIONS =========="
echo ""

python3 -u main.py visualize \
    --run-dir "${LATEST_RUN}" \
    --gif

echo ""
echo "============================================================"
echo "EXPERIMENT COMPLETE: ${PRESET}"
echo "  Run dir:  ${LATEST_RUN}"
echo "  End time: $(date)"
echo "============================================================"
