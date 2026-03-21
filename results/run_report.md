# PPO Run Report (from `results/metrics.csv`)

## 1) What was tried
- Algorithm: PPO (symmetry-aware CNN policy).
- Training budget: `200,704` env steps (`98` updates with `rollout_steps=2048`).
- Topologies: `heavy_hex_19`, `grid_3x3`, `linear_5` (multi-topology training).
- Initial mapping strategy: mixed (`80% random`, `20% SABRE`).
- Strategy masking: disabled (topology-validity mask only).
- Gate-demand decay: `gamma_decay=0.5`.
- Periodic validation: every 10 updates, PPO vs SABRE on fixed holdout circuits.

## 2) Results obtained
### Training metrics (start vs end)
- Mean step reward (first 10 updates): `-0.8735`
- Mean step reward (last 10 updates): `-0.8420`
- Mean episode return (first 10 updates): `-245.20`
- Mean episode return (last 10 updates): `-210.66`
- Done rate (first 10 updates): `0.0039`
- Done rate (last 10 updates): `0.0045`

Interpretation: slight improvement in training signals, but still weak overall.

### Validation vs SABRE (critical metric)
At updates `10, 20, ..., 90`:
- `eval_mean_ppo_swaps = 500.0` (constant)
- `eval_mean_sabre_swaps = 73.42` (constant)
- `eval_improvement_pct = -1705.56%` (constant)
- `eval_win_rate = 0.00` (constant)

Interpretation: validation episodes are almost certainly hitting the episode cap (`max_steps=500`) with no solved routing on the holdout set.

## 3) Diagnostic
- The run is executing correctly, but learning is not reaching useful routing behavior.
- The current setup is likely too hard from scratch:
  - multi-topology from step 1,
  - no strategy masking at all,
  - relatively hard circuit depth,
  - sparse success signal.
- Validation stagnation is severe (flat at `-1705.56%`).

## 4) Recommended changes for next run
## A) Training curriculum (most important)
1. Stage 1 (easy): train on `linear_5` only, lower depth (e.g. 8-12).
2. Stage 2 (intermediate): add `grid_3x3`.
3. Stage 3 (target): include `heavy_hex_19` (final multi-topology training).

## B) Hyperparameters to try next
1. Lower exploration pressure:
   - `entropy_coef_start=0.01`
   - `entropy_coef_end=0.001`
2. Slightly stronger reward shaping early:
   - `distance_reward_coeff=0.03` (then anneal back to `0.01`)
3. Keep PPO LR moderate:
   - `learning_rate=3e-4` (or keep `2.5e-4` if unstable)
4. Keep evaluation every 10 updates.

## C) Bias control (important)
1. Keep training on random circuits only.
2. Keep periodic validation on a fixed holdout random set (already done).
3. Keep final QASMBench evaluation strictly for final reporting.
4. Do not tune hyperparameters on final QASMBench test results.

## D) Practical monitoring rule
- If `eval_improvement_pct` stays flat for several checkpoints, keep the full planned run and then adjust the next run configuration.

## 5) Bottom line
- Current run did not approach the project goal (improvement vs SABRE).
- Next run should prioritize curriculum + reduced entropy + stronger early shaping.

## 6) Update After Run2 (2026-03-21)
### Run2 files reviewed
- `results/Run2/metrics.csv`
- `results/Run2/metrics (1).csv`
- `results/Run2/metrics (2).csv`

### Observed result
- Evaluation is still dominated by timeouts:
  - `eval_mean_ppo_swaps` reaches `500.0`
  - `eval_timeout_rate` reaches `1.0`
  - `eval_win_rate` remains `0.0` at the end of each file
- One partial signal appeared in `metrics.csv` around update 20 (`eval_win_rate=0.25`), but did not persist.

### Changes implemented for next run
1. Curriculum stage-aware evaluation depth:
   - In curriculum mode, eval depth now follows stage difficulty with `eval_circuit_depth = min(user_eval_depth, stage_depth)`.
   - This avoids evaluating stage1/stage2 on harder depth than what they train on.
2. Stronger anti-timeout reward shaping defaults:
   - `completion_bonus`: `5.0 -> 8.0`
   - `timeout_penalty`: `-10.0 -> -25.0`
   - `distance_reward_coeff_end`: `0.01 -> 0.015`

### Code locations updated
- `src/main.py`
- `src/environment.py`

## 7) Update (Early-Stop Removed Completely)
- Automatic early-stop logic has been fully removed from training.
- Removed from:
  - CLI args in `src/main.py`
  - PPO config and training loop logic in `src/agent.py`
  - Notebook command/help references in `notebooks/train_ppo_colab.ipynb`

## 8) Update (Run3 Long Curriculum + Better Data Quality)
- Target run budget prepared: `1.0M` timesteps curriculum
  - Stage1: `200k`
  - Stage2: `200k`
  - Stage3: `600k`
- Evaluation noise reduction:
  - `eval_circuits_per_topology` increased from `4` to `12`.
  - Notebook launch uses `eval_interval_updates=20` to keep wall-time manageable.
- Random-circuit quality constraint added:
  - New training/eval controls to enforce a minimum number of 2-qubit gates
    in generated random circuits.
  - Defaults now use `min_two_qubit_gates=6` in the training entrypoint.
- Training logs now show:
  - elapsed time, ETA, and eval mean 2-qubit gate count (`eval_2q`) to verify
    holdout difficulty quality.
