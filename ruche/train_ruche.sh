#!/bin/bash
# =============================================================================
# train_ruche.sh — SLURM batch script for PPO training on La Ruche
# Submit with:  sbatch ruche/train_ruche.sh
# =============================================================================

#SBATCH --job-name=rl_qrouting
#SBATCH --output=%x.o%j
#SBATCH --error=%x.e%j
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --time=23:30:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE

# =============================================================================
# CONFIGURATION — edit these to change the training run
# =============================================================================
SEED=42
RUN_NAME="ppo_ruche_3stage_$(date +%Y%m%d_%H%M%S)_s${SEED}"

# Stage budgets (timesteps)
STAGE1_STEPS=100000
STAGE2_STEPS=400000
STAGE3_STEPS=600000

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================
module purge
module load apptainer/1.4.4/gcc-15.1.0

SIF_PATH="$WORKDIR/rl_qrouting.sif"
PROJECT_DIR="$WORKDIR/rl-quantum-circuit-routing"
RUN_DIR="$WORKDIR/rl_qrouting_runs"
mkdir -p "$RUN_DIR"

export APPTAINER_CACHEDIR=$WORKDIR/.apptainer_cache
mkdir -p "$APPTAINER_CACHEDIR"

echo "========================================"
echo "  Job ID:       $SLURM_JOB_ID"
echo "  Node:         $(hostname)"
echo "  Run name:     $RUN_NAME"
echo "  Project dir:  $PROJECT_DIR"
echo "  Output dir:   $RUN_DIR"
echo "  CPUs:         $SLURM_CPUS_PER_TASK"
echo "  Started:      $(date)"
echo "========================================"

# =============================================================================
# TRAINING — 3-stage curriculum
# =============================================================================
apptainer exec \
    --nv \
    --writable-tmpfs \
    --bind "$WORKDIR:$WORKDIR:rw" \
    --pwd "$PROJECT_DIR" \
    "$SIF_PATH" \
    python -m src.main \
    --curriculum \
    --seed "$SEED" \
    --project-root "$RUN_DIR" \
    --run-name "$RUN_NAME" \
    \
    --topologies "grid_3x3,heavy_hex_19" \
    --stage1-topologies "linear_5" \
    --stage2-topologies "linear_5,grid_3x3" \
    --stage1-steps "$STAGE1_STEPS" \
    --stage2-steps "$STAGE2_STEPS" \
    --stage3-steps "$STAGE3_STEPS" \
    --stage1-depth 10 \
    --stage2-depth 12 \
    --stage3-depth 16 \
    \
    --matrix-size 27 \
    --max-steps 500 \
    --max-steps-per-two-qubit-gate 10 \
    --max-steps-min 60 \
    --max-steps-max 450 \
    \
    --rollout-steps 4096 \
    --learning-rate 3e-4 \
    --gamma 0.995 \
    --gae-lambda 0.97 \
    --clip-range 0.15 \
    --update-epochs 8 \
    --minibatch-size 256 \
    --entropy-coef-start 0.003 \
    --entropy-coef-end 0.0001 \
    --value-coef 0.5 \
    --max-grad-norm 0.5 \
    --target-kl 0.015 \
    \
    --gate-reward-coeff 1.0 \
    --step-penalty -0.05 \
    --reverse-swap-penalty -0.2 \
    --repeat-swap-penalty-coeff -0.2 \
    --no-progress-penalty-coeff -0.03 \
    --no-progress-penalty-cap -1.5 \
    --completion-bonus 15.0 \
    --timeout-penalty -8.0 \
    --distance-reward-coeff-start 0.03 \
    --distance-reward-coeff-end 0.015 \
    \
    --action-repeat-logit-penalty 0.20 \
    --no-progress-terminate-streak 30 \
    \
    --linear-topology-weight 0.25 \
    --grid-topology-weight 1.5 \
    --heavy-hex-topology-weight 2.0 \
    \
    --min-two-qubit-gates 6 \
    --eval-interval-updates 10 \
    --eval-circuits-per-topology 16 \
    --trace-interval-updates 10 \
    --trace-cases-per-topology 1 \
    --trace-alert-dom-threshold 0.60 \
    --trace-alert-backtrack-threshold 0.50 \
    --trace-alert-patience 2 \
    \
    --device cuda

# =============================================================================
# PLOTS — generate training_curves.png + eval_comparison.png
# =============================================================================
echo "Generating plots..."
for PHASE_DIR in "$RUN_DIR/runs/$RUN_NAME"/*/; do
    if [ -f "$PHASE_DIR/metrics.csv" ]; then
        apptainer exec \
            --nv \
            --bind "$WORKDIR:$WORKDIR:rw" \
            --pwd "$PROJECT_DIR" \
            "$SIF_PATH" \
            python -m src.visualize "$PHASE_DIR" || echo "Warning: plot failed for $PHASE_DIR"
    fi
done

echo "========================================"
echo "  Finished:  $(date)"
echo "  Results:   $RUN_DIR/runs/$RUN_NAME"
echo "========================================"
