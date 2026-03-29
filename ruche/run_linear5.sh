#!/bin/bash
#SBATCH --job-name=ppo_linear5
#SBATCH --output=%x.o%j
#SBATCH --error=%x.e%j
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --time=23:59:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE

SEED=42
RUN_NAME="ppo_linear5_$(date +%Y%m%d_%H%M%S)_s${SEED}"

module purge
module load apptainer/1.4.4/gcc-15.1.0

SIF_PATH="$WORKDIR/rl_qrouting.sif"
PROJECT_DIR="$WORKDIR/rl-quantum-circuit-routing"
RUN_DIR="$WORKDIR/rl_qrouting_runs"
mkdir -p "$RUN_DIR"
export APPTAINER_CACHEDIR=$WORKDIR/.apptainer_cache
mkdir -p "$APPTAINER_CACHEDIR"

echo "========================================"
echo "  Job: $SLURM_JOB_NAME | ID: $SLURM_JOB_ID"
echo "  Run: $RUN_NAME"
echo "  Started: $(date)"
echo "========================================"

apptainer exec --nv --writable-tmpfs \
    --bind "$WORKDIR:$WORKDIR:rw" --pwd "$PROJECT_DIR" "$SIF_PATH" \
    python -m src.main \
    --seed "$SEED" \
    --project-root "$RUN_DIR" \
    --run-name "$RUN_NAME" \
    --topologies "linear_5" \
    --total-timesteps 2000000 \
    \
    --circuit-depth 10 \
    --matrix-size 27 \
    --max-steps 500 \
    --max-steps-per-two-qubit-gate 10 \
    --max-steps-min 100 \
    --max-steps-max 300 \
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
    --completion-bonus 25.0 \
    --timeout-penalty -8.0 \
    --distance-reward-coeff-start 0.03 \
    --distance-reward-coeff-end 0.015 \
    \
    --action-repeat-logit-penalty 0.20 \
    --no-progress-terminate-streak 50 \
    \
    --min-two-qubit-gates 6 \
    --eval-interval-updates 20 \
    --eval-circuits-per-topology 16 \
    --trace-interval-updates 20 \
    --trace-cases-per-topology 1 \
    --trace-alert-dom-threshold 0.60 \
    --trace-alert-backtrack-threshold 0.50 \
    --trace-alert-patience 2 \
    \
    --device cuda

# Plots
apptainer exec --bind "$WORKDIR:$WORKDIR:rw" --pwd "$PROJECT_DIR" "$SIF_PATH" \
    python -m src.visualize "$RUN_DIR/runs/$RUN_NAME/single_stage" || echo "Plot failed"

echo "========================================"
echo "  Finished: $(date)"
echo "  Results:  $RUN_DIR/runs/$RUN_NAME"
echo "========================================"
