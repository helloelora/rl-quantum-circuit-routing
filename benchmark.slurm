#!/bin/bash
#SBATCH --job-name=qr_benchmark
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --partition=gpu_prod_long
#SBATCH --exclude=sh07,sh[10-19]
#SBATCH --propagate=NONE
#SBATCH --mail-user=ali.dor@student-cs.fr
#SBATCH --mail-type=END,FAIL

# ============================================================
# Full Benchmark — Agent vs SABRE (Generated + QASMBench)
#
# Usage:
#   sbatch benchmark.slurm <CHECKPOINT> [extra args...]
#
# Runs both suites by default (generated + QASMBench).
# To skip QASMBench, pass --no-qasmbench.
#
# Examples:
#   sbatch benchmark.slurm outputs/run_030/checkpoints/checkpoint_best.pt
#   sbatch benchmark.slurm outputs/run_030/checkpoints/checkpoint_best.pt --topology heavy_hex_19
#
# Output:
#   outputs/tests/benchmark_NNN/
#     ├── benchmark_results.{json,md}
#     ├── category_stats.json
#     ├── test_config.json
#     ├── combined_summary.png
#     ├── combined_ratio_comparison.png
#     ├── combined_agent_vs_sabre.png
#     ├── generated/   (6 per-suite figures)
#     ├── qasmbench/   (6 per-suite figures)
#     └── slurm_XXXXX.{out,err}
# ============================================================

CHECKPOINT="${1:?Usage: sbatch benchmark.slurm <CHECKPOINT> [extra args...]}"
shift

PROJECT_DIR="/usr/users/rl_for_wsi/dor_ali/projects/rl-quantum-circuit-routing"
cd "$PROJECT_DIR"

# Temporary logs (moved to output dir after test)
mkdir -p outputs/.pending_logs
LOG_OUT="outputs/.pending_logs/bench_${SLURM_JOB_ID}.out"
LOG_ERR="outputs/.pending_logs/bench_${SLURM_JOB_ID}.err"
exec > "$LOG_OUT" 2> "$LOG_ERR"

echo "============================================================"
echo "BENCHMARK TEST"
echo "Checkpoint: ${CHECKPOINT}"
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       ${SLURM_NODELIST}"
echo "Extra args: $@"
echo "Start time: $(date)"
echo "============================================================"

source .venv/bin/activate
export PYTHONUNBUFFERED=1

# Check for --no-qasmbench flag
QASMBENCH_FLAG="--qasmbench QASMBench"
for arg in "$@"; do
    if [ "$arg" = "--no-qasmbench" ]; then
        QASMBENCH_FLAG=""
    fi
done
# Remove --no-qasmbench from args passed to python
EXTRA_ARGS=()
for arg in "$@"; do
    if [ "$arg" != "--no-qasmbench" ]; then
        EXTRA_ARGS+=("$arg")
    fi
done

python3 -u scripts/benchmark_test.py \
    --checkpoint "${CHECKPOINT}" \
    --device cuda \
    ${QASMBENCH_FLAG} \
    "${EXTRA_ARGS[@]}"

EXIT_CODE=$?

# Move logs to output dir
OUTPUT_DIR=$(ls -td outputs/tests/benchmark_* 2>/dev/null | head -1)
if [ -n "$OUTPUT_DIR" ] && [ -d "$OUTPUT_DIR" ]; then
    mv "$LOG_OUT" "${OUTPUT_DIR}/slurm_${SLURM_JOB_ID}.out" 2>/dev/null
    mv "$LOG_ERR" "${OUTPUT_DIR}/slurm_${SLURM_JOB_ID}.err" 2>/dev/null
    exec >> "${OUTPUT_DIR}/slurm_${SLURM_JOB_ID}.out" 2>> "${OUTPUT_DIR}/slurm_${SLURM_JOB_ID}.err"
fi

echo ""
echo "============================================================"
echo "BENCHMARK COMPLETE"
echo "  Exit code: ${EXIT_CODE}"
echo "  Output:    ${OUTPUT_DIR}"
echo "  End time:  $(date)"
echo "============================================================"
exit $EXIT_CODE
