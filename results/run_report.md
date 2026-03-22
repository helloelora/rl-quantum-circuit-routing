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

## 9) Long Run Analysis (results/Long run)
### Files mapped to stages
- `metrics (2).csv` -> stage1 (150k, 74 updates)
- `metrics (1).csv` -> stage2 (250k, 123 updates)
- `metrics.csv` -> stage3 (600k, 293 updates)

### Key observations
- Training progresses in stage1:
  - done_rate improves (~0.0147 -> ~0.0167)
  - mean episode return improves (~-39 -> ~-31)
- Stage2 and stage3 do not show stable success growth:
  - stage2 done_rate stays ~0.004
  - stage3 done_rate stays ~0.004
  - stage3 returns remain very negative (~-252 first10, ~-239 last10)

### Validation vs SABRE (critical)
- For all three stages, all eval checkpoints are flat and failing:
  - `eval_mean_ppo_swaps = 500.0`
  - `eval_timeout_rate = 1.0`
  - `eval_win_rate = 0.0`
- This means the policy still times out on every holdout circuit, despite the
  longer training budget and improved circuit quality constraint.

### Interpretation
- The pipeline runs correctly (metrics update, stage progression, eval quality tracked).
- The current PPO policy does not yet learn a strong deterministic routing
  behavior under holdout evaluation (greedy policy remains too weak/random).

## 10) Parameter Log: Last Long Run vs Planned Run4
### Last Long Run (completed)
- Curriculum schedule:
  - stage1 topologies: `linear_5,grid_3x3`
  - stage2 topologies: `linear_5,grid_3x3,heavy_hex_19`
  - stage3 topologies: `heavy_hex_19,grid_3x3,linear_5`
  - stage steps: `150k / 250k / 600k` (total `1.0M`)
  - stage depths: `10 / 14 / 20`
- PPO:
  - `rollout_steps=2048`
  - `learning_rate=2.5e-4`
  - `gamma=0.99`
  - `gae_lambda=0.95`
  - `clip_range=0.2`
  - `update_epochs=4`
  - `entropy_coef: 0.01 -> 0.001`
- Reward/env:
  - reward form: `(gates_executed - 1) + distance_reward_coeff * delta_dist`
  - `distance_reward_coeff: 0.03 -> 0.015`
  - `completion_bonus=8`
  - `timeout_penalty=-25`
  - no anti-backtracking mask
- Eval:
  - `eval_interval_updates=20`
  - `eval_circuits_per_topology=12`
  - `min_two_qubit_gates(train/eval)=6`

### Run4 (planned now)
- Curriculum schedule kept:
  - stage1 topologies: `linear_5,grid_3x3`
  - stage2 topologies: `linear_5,grid_3x3,heavy_hex_19`
  - stage3 topologies: `heavy_hex_19,grid_3x3,linear_5`
  - stage steps: `150k / 250k / 600k` (total `1.0M`)
  - stage depths: `10 / 14 / 20`
- PPO changes:
  - `rollout_steps=4096`
  - `learning_rate=3e-4`
  - `gamma=0.995`
  - `gae_lambda=0.97`
  - `clip_range=0.15`
  - `update_epochs=8`
  - `entropy_coef: 0.003 -> 0.0001`
- Reward/env changes:
  - reward form changed to:
    - `gate_reward_coeff * gates_executed + distance_reward_coeff * delta_dist + step_penalty`
    - plus `reverse_swap_penalty` for immediate edge backtracking
    - plus `completion_bonus` on done
    - plus `timeout_penalty` on truncation
  - `gate_reward_coeff=1.0`
  - `step_penalty=-0.05`
  - `reverse_swap_penalty=-0.2`
  - `completion_bonus=15`
  - `timeout_penalty=-8`
  - no anti-backtracking mask (topology-validity mask only)
  - `distance_reward_coeff: 0.03 -> 0.015`
- Eval/data quality:
  - `eval_interval_updates=20`
  - `eval_circuits_per_topology=12`
  - `min_two_qubit_gates(train/eval)=8`

## 11) Update (Masking Decision Confirmed)
- Decision validated: remove light anti-backtracking action masking.
- Keep only a small `reverse_swap_penalty` for immediate backtracking.
- Rationale: allows useful return moves when globally beneficial, while still
  discouraging trivial oscillation loops.

## 12) Run4 Deep Analysis (results/Run4)
### Stage mapping
- `metrics.csv` -> stage1 (`~150k`, 37 updates)
- `metrics (1).csv` -> stage2 (`~250k`, 62 updates)
- `metrics (2).csv` -> stage3 (`~600k`, 147 updates)

### Training dynamics
- Stage1:
  - `mean_step_reward` improves (`~0.484 -> ~0.583` on first10/last10 windows)
  - `done_rate` improves (`~0.0155 -> ~0.0187`)
  - `mean_ep_return` stays strongly positive (`~31`)
- Stage2:
  - positive rewards and returns (`mean_ep_return ~17.5 -> ~21.4`)
  - `done_rate` slightly up (`~0.0046 -> ~0.0051`)
- Stage3:
  - positive rewards and returns (`mean_ep_return ~22.1 -> ~22.3`)
  - `done_rate` stays low (`~0.004`)

### Evaluation vs SABRE
- Still far from target absolute performance, but not flat-failing anymore:
  - stage1 last eval: `ppo=458.79`, `timeout=0.917`, `win=0.083`
  - stage2 last eval: `ppo=459.25`, `timeout=0.917`, `win=0.028`
  - stage3 last eval: `ppo=433.14`, `timeout=0.861`, `win=0.083`

### Delta vs previous Long run (same stage mapping)
- stage1:
  - `ppo_swaps 500 -> 458.79`
  - `timeout 1.00 -> 0.917`
  - `win_rate 0.00 -> 0.083`
- stage2:
  - `ppo_swaps 500 -> 459.25`
  - `timeout 1.00 -> 0.917`
  - `win_rate 0.00 -> 0.028`
- stage3:
  - `ppo_swaps 500 -> 433.14`
  - `timeout 1.00 -> 0.861`
  - `win_rate 0.00 -> 0.083`

### Conclusion
- Run4 is a clear step in the right direction (timeouts reduced, first wins vs SABRE).
- Main blocker remains: success rate is still too low, and swap counts remain far
  above SABRE on holdout circuits.

## 13) Policy Trace Diagnostic (Run4, heavy_hex_19)
### Files
- `results/Run4/trace_hh19.csv`
- `results/Run4/trace_hh19_summary.json`

### What happened in this traced episode
- Episode reached max horizon:
  - `steps=500`, `truncated=true`, `done=false`
- Final comparison:
  - `ppo_swaps=500`, `sabre_swaps=184`, `improvement=-171.74%`
- Routing progress stalled:
  - `remaining_gates: 124 -> 124` (no reduction during steps)
  - `total_gates_executed_end=7` (all from initial auto-exec only)
  - `steps_with_gate_execution=0`
- Behavioral collapse:
  - one action dominates almost entirely: `action_index=3` used `494/500` steps
  - `backtrack_rate=0.984` (`492/500` immediate repeats)

### Interpretation
- Even without explicit backtracking masking, the greedy policy collapses to a
  near-deterministic oscillation around one SWAP edge and no longer progresses
  on front-layer executability.

## 14) Run5 Setup (Integrated In-Training Traces)
### What changed in code for Run5
- Periodic policy traces are now integrated directly in PPO training.
- New CLI args:
  - `--trace-interval-updates`
  - `--trace-cases-per-topology`
  - `--trace-max-steps`
- Trace outputs are saved during training under:
  - `runs/<run_name>/<stage_name>/traces/update_XXXXX/`
  - Per case: `*_trace.csv` and `*_summary.json`.
- `metrics.csv` now also logs aggregated trace indicators:
  - `trace_timeout_rate`
  - `trace_backtrack_rate`
  - `trace_action_dom_ratio`
  - `trace_improvement_pct`
  - plus trace swap and completion stats.

### Run5 baseline launch parameters (kept comparable to Run4)
- Curriculum: `150k / 250k / 600k` steps (`1.0M` total)
- Depths: `10 / 14 / 20`
- Topologies:
  - stage1: `linear_5,grid_3x3`
  - stage2: `linear_5,grid_3x3,heavy_hex_19`
  - stage3: `heavy_hex_19,grid_3x3,linear_5`
- PPO:
  - `rollout_steps=4096`
  - `lr=3e-4`
  - `gamma=0.995`
  - `gae_lambda=0.97`
  - `clip_range=0.15`
  - `update_epochs=8`
  - `entropy 0.003 -> 0.0001`
- Reward/data:
  - `completion_bonus=15`
  - `timeout_penalty=-8`
  - `gate_reward_coeff=1.0`
  - `step_penalty=-0.05`
  - `reverse_swap_penalty=-0.2`
  - `distance_reward_coeff 0.03 -> 0.015`
  - `min_two_qubit_gates(train/eval)=8`
- Eval/trace:
  - `eval_interval_updates=20`
  - `eval_circuits_per_topology=12`
  - `trace_interval_updates=20`
  - `trace_cases_per_topology=1`
  - `trace_max_steps=500`

### Run5 monitoring rules (during training)
- If `trace_action_dom_ratio > 0.60` and `trace_backtrack_rate > 0.50` for several trace checkpoints, policy is collapsing to repetitive behavior.
- If `trace_timeout_rate` stays close to `1.0`, routing is still mostly not solved in held-out traces.
- If `trace_improvement_pct` remains very negative with flat `eval_win_rate`, prioritize reward/architecture changes before spending more compute.
