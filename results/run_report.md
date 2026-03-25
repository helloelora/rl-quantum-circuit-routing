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
  - `trace_cases_per_topology=2`
  - `trace_max_steps=500`

### Run5 monitoring rules (during training)
- If `trace_action_dom_ratio > 0.60` and `trace_backtrack_rate > 0.50` for several trace checkpoints, policy is collapsing to repetitive behavior.
- If `trace_timeout_rate` stays close to `1.0`, routing is still mostly not solved in held-out traces.
- If `trace_improvement_pct` remains very negative with flat `eval_win_rate`, prioritize reward/architecture changes before spending more compute.

## 15) Run5 Analysis (results/Run5/ppo_run5_20260322_163629_analysis_only)
### Stage-level training behavior
- Stage1 (`37` updates, `151,552` steps):
  - `mean_step_reward`: `0.491 -> 0.528` (first10/last10)
  - `done_rate`: `0.0156 -> 0.0170`
  - `mean_ep_return`: `31.37 -> 31.01` (flat)
- Stage2 (`62` updates, `253,952` steps):
  - `mean_step_reward`: `0.0838 -> 0.0930`
  - `done_rate`: `0.0045 -> 0.0045` (flat, low)
  - `mean_ep_return`: `17.80 -> 17.86` (flat)
- Stage3 (`147` updates, `602,112` steps):
  - `mean_step_reward`: `0.0973 -> 0.0970` (flat)
  - `done_rate`: `0.0040 -> 0.0039` (flat, very low)
  - `mean_ep_return`: `22.33 -> 23.02` (slight increase)

### Evaluation vs SABRE (critical)
- Stage1 (only update 20 eval):
  - `eval_improve=-5618.52%`, `win=0.00`, `timeout=1.00`, `ppo=500`, `sabre=9.5`
- Stage2 (updates 20/40/60):
  - best eval at update 60: `eval_improve=-2026.05%`, `win=0.028`, `timeout=0.889`
- Stage3 (updates 20..140):
  - best eval at update 120: `eval_improve=-1298.24%`, `win=0.028`, `timeout=0.889`
  - final eval at update 140 regresses to:
    - `eval_improve=-1537.63%`, `win=0.028`, `timeout=0.972`
    - `ppo=486.39`, `sabre=77.92`

### Trace diagnostics (integrated in training)
- Stage1 trace checkpoint (update 20):
  - `trace_timeout=1.0`, `trace_backtrack=0.995`, `trace_dom=0.997`
  - strong immediate collapse.
- Stage2 traces (3 checkpoints):
  - averages: `trace_timeout=0.944`, `trace_backtrack=0.953`, `trace_dom=0.970`
  - still severe action collapse.
- Stage3 traces (7 checkpoints):
  - averages: `trace_timeout=0.833`, `trace_backtrack=0.877`, `trace_dom=0.929`
  - little/no evolution across checkpoints (`trace_improve` nearly constant).

### Topology-specific trace signal (stage3, 42 summaries)
- `grid_3x3`:
  - timeout `14/14`, mean improvement `-2172.73%`
  - mean backtrack `0.994`, dominant action ratio `0.996`
- `heavy_hex_19`:
  - timeout `14/14`, mean improvement `-155.10%`
  - mean backtrack `0.990`, dominant action ratio `0.993`
- `linear_5`:
  - done `7/14`, timeout `7/14`, mean improvement `-1338.89%`
  - two fixed trace cases split:
    - one solved repeatedly (`ppo=10`, `sabre=10`, done)
    - one collapsed repeatedly (`ppo=500`, `sabre=18`, timeout)

### Comparison vs Run4 (stage3 final checkpoint)
- Run4 final stage3:
  - `eval_improve=-1256.88%`, `win=0.083`, `timeout=0.861`, `ppo=433.14`
- Run5 final stage3:
  - `eval_improve=-1537.63%`, `win=0.028`, `timeout=0.972`, `ppo=486.39`
- Conclusion: Run5 underperforms Run4 on final stage3 holdout metrics.

### Conclusion
- Run5 confirms the core blocker: policy collapse to repetitive swap behavior
  on non-trivial topologies.
- The integrated traces were useful: they clearly reveal near-deterministic
  action loops (`trace_dom` high) with very high immediate backtracking.

## 16) Run6 Planned Changes (Implemented in Code)
### Goal alignment
- Primary objective remains unchanged:
  - beat SABRE on held-out evaluation circuits,
  - and generalize across different topologies.

### A) Best-model selection aligned with SABRE objective
- `best_model.pt` is now selected by periodic eval metrics (not train return).
- Selection priority (lexicographic):
  1. higher `eval_improvement_pct`
  2. higher `eval_win_rate`
  3. lower `eval_timeout_rate`
- Training-return checkpoint is still saved separately as `best_train_model.pt`.
- `best_eval_metrics.json` is written for reproducibility.

### B) Progressive anti-loop penalty (no hard masking)
- New environment reward term:
  - `repeat_swap_penalty_coeff * (same_edge_streak - 1)`
- This penalizes consecutive reuse of the same undirected physical edge,
  progressively, while still allowing backtracking when globally useful.
- Existing `reverse_swap_penalty` is kept.

### C) Topology rebalancing toward harder generalization targets
- Added weighted topology sampling at reset.
- Default training weights now emphasize harder topologies:
  - `linear_topology_weight=0.5`
  - `grid_topology_weight=1.5`
  - `heavy_hex_topology_weight=1.5`
  - `other_topology_weight=1.0`
- Weights are normalized per phase and printed in logs.

### D) Trace alerts as primary collapse signal
- Traces remain integrated during training.
- Added configurable alert logic based on:
  - `trace_action_dom_ratio >= 0.60`
  - `trace_backtrack_rate >= 0.50`
  - for `trace_alert_patience=2` consecutive trace checkpoints.
- Alert state is now logged in `metrics.csv`:
  - `trace_alert_flag`, `trace_alert_streak`.

### E) Run6 notebook launch settings
- Added explicit flags:
  - `--repeat-swap-penalty-coeff -0.15`
  - `--linear-topology-weight 0.5`
  - `--grid-topology-weight 1.5`
  - `--heavy-hex-topology-weight 1.5`
  - `--trace-alert-dom-threshold 0.6`
  - `--trace-alert-backtrack-threshold 0.5`
  - `--trace-alert-patience 2`
- `RUN_NAME` switched to `ppo_run6_<timestamp>`.

## 17) Run6 Deep Analysis (results/Run6/ppo_run6_20260322_232212_analysis_only)
### Data reviewed
- `stage1_easy/metrics.csv`
- `stage2_mid/metrics.csv`
- `stage3_full/metrics.csv`
- all trace summaries under `stage*/traces/update_*/ *_summary.json`
- run config (`config.json`)

### Stage-level training behavior
- Stage1 (`37` updates, `151,552` steps):
  - `mean_step_reward`: `0.4377 -> 0.5039`
  - `done_rate`: `0.0137 -> 0.0156`
  - `mean_ep_return`: `31.80 -> 32.39`
- Stage2 (`62` updates, `253,952` steps):
  - `mean_step_reward`: `0.0486 -> 0.0621`
  - `done_rate`: `0.0036 -> 0.0039`
  - `mean_ep_return`: `10.87 -> 13.20`
- Stage3 (`147` updates, `602,112` steps):
  - `mean_step_reward`: `0.0684 -> 0.0924`
  - `done_rate`: `0.0034 -> 0.0038`
  - `mean_ep_return`: `17.32 -> 24.22`

Interpretation: train rewards and returns improve, but completion rate remains very low.

### Eval vs SABRE (critical)
- Stage1 (update 20):
  - `eval_improve=-4775.46%`, `win=0.125`, `timeout=0.875`, `ppo=438.33`, `sabre=9.54`
- Stage2 (updates 20/40/60):
  - best at update 60: `eval_improve=-2276.97%`, `win=0.00`, `timeout=0.944`, `ppo=473.06`
- Stage3 (updates 20..140):
  - best improvement at update 120: `-1263.98%`, `win=0.028`, `timeout=0.889`
  - best win appears at updates 100 and 140: `win=0.083`
  - final checkpoint (140):
    - `eval_improve=-1372.80%`, `win=0.083`, `timeout=0.917`, `ppo=459.53`, `sabre=76.89`

### Trace diagnostics (Run6)
- Stage1 traces:
  - all 4 trace cases timeout; very high loop pattern (`dom=0.993`, `backtrack=0.992`)
- Stage2 traces:
  - all trace cases timeout; strong collapse remains (`dom~0.995`, `backtrack~0.992`)
- Stage3 traces:
  - averages across 42 summaries:
    - `trace_dom=0.890`, `trace_backtrack=0.852`, `trace_timeout=0.833`
    - `trace_improve=-1180.22%` (still far from SABRE)
  - topology breakdown:
    - `grid_3x3`: timeout `14/14`, mean improve `-2036.36%`
    - `heavy_hex_19`: timeout `14/14`, mean improve `-170.40%`
    - `linear_5`: done `7/14`, timeout `7/14`, one case solved repeatedly and one case repeatedly collapsed

### Comparison with Run5 and Run4
- Versus Run5 (stage3 final):
  - improvement: `-1537.63% -> -1372.80%` (better)
  - timeout: `0.972 -> 0.917` (better)
  - win rate: `0.028 -> 0.083` (better)
  - ppo swaps: `486.39 -> 459.53` (better)
- Versus Run4 (stage3 final):
  - Run4 still better: `-1256.88%`, `timeout=0.861`, `ppo=433.14` (vs Run6 `-1372.80%`, `0.917`, `459.53`)

### Important instrumentation note
- `trace_alert_flag` stayed `0` at all checkpoints.
- Root cause: current streak logic resets on non-trace updates, so it never exceeds `1` with `trace_interval_updates > 1`.
- Consequence: alert signal is currently under-reporting collapse risk.

### Conclusion
- Run6 is a meaningful improvement over Run5, especially on stage3 final metrics.
- However, the core objective (beating SABRE robustly and generalizing on hard topologies) is still not reached.
- Main blocker remains persistent loop behavior on `grid_3x3` and `heavy_hex_19`.

## 18) Run7 Planned Changes (Implemented)
### A) Trace alert bug fix
- Fixed alert streak behavior:
  - streak is no longer reset on non-trace updates,
  - alerts can now trigger correctly with `trace_interval_updates > 1`.
- This restores the intended monitoring signal for loop-collapse detection.

### B) Stronger anti-stagnation reward (still no hard masking)
- Added progressive no-progress penalty:
  - `no_progress_penalty_coeff * no_progress_streak`
  - capped by `no_progress_penalty_cap` (negative floor)
- Motivation:
  - penalize long sequences with zero executed gates,
  - reduce attractors where policy keeps swapping without routing progress.

### C) Harder curriculum setup in launch notebook
- Stage setup shifted to reduce early overfitting to easy topology:
  - stage1 topologies: `linear_5`
  - stage2 topologies: `linear_5,grid_3x3`
  - stage3 topologies unchanged (`heavy_hex_19,grid_3x3,linear_5`)
- Topology weights strengthened toward difficult targets:
  - `linear=0.25`, `grid=1.5`, `heavy_hex=2.0`

### D) Run7 notebook launch flags
- `--repeat-swap-penalty-coeff -0.2`
- `--no-progress-penalty-coeff -0.03`
- `--no-progress-penalty-cap -1.5`
- `--linear-topology-weight 0.25`
- `--grid-topology-weight 1.5`
- `--heavy-hex-topology-weight 2.0`
- Trace alert thresholds unchanged:
  - `dom>=0.6`, `backtrack>=0.5`, `patience=2`

## 19) Run7 Analysis (results/Run7)
### Files reviewed
- `results/Run7/metrics.csv` (stage1)
- `results/Run7/metrics (1).csv` (stage2)
- `results/Run7/metrics (2).csv` (stage3, interrupted by compute budget)
- `results/Run7/best_eval_metrics*.json`
- all trace summaries under:
  - `results/Run7/update_00020-20260323T101416Z-1-001/...`
  - `results/Run7/traces-20260323T101450Z-1-001/...`
  - `results/Run7/traces-20260323T101511Z-1-001/...`

### Stage1 (linear_5): learns
- Train improved (`mean_step_reward 0.45 -> 2.30`, `done_rate 0.026 -> 0.084`).
- Eval at update 20 still far from SABRE:
  - `eval_improve=-4071.38%`, `win=0.25`, `timeout=0.75`, `ppo=376.25`, `sabre=8.67`.
- Trace at update 20:
  - `done=0.50`, `timeout=0.50`, `backtrack=0.498`, `dom=0.666`.

### Stage2 (linear_5 + grid_3x3): mixed
- Eval improved from update 20 to 60:
  - `-1500.97% -> -955.85%`
  - `win=0.25 -> 0.292`
  - `timeout=0.50 -> 0.333`.
- Best Run7 checkpoint is stage2 update 60 (`best_eval_metrics (1).json`).
- Traces show non-monotonic behavior:
  - update 40 trace looked healthy (`done=1.00`, `timeout=0.00`, `dom=0.277`, `backtrack=0.101`),
  - but update 60 degraded again (`done=0.75`, `timeout=0.25`).

### Stage3 (linear_5 + grid_3x3 + heavy_hex_19): collapse
- Final logged point before interruption:
  - `mean_step_reward=-47.50`, `mean_ep_return=-23070.8`,
  - `value_loss=1,293,161`, `entropy=0.045`, `done_rate=0.002`.
- Eval stayed poor and worsened:
  - update 20: `eval_improve=-1440.71%`, `win=0.028`, `timeout=0.944`
  - update 60: `eval_improve=-1544.35%`, `win=0.028`, `timeout=0.972`
- Trace collapse signal is explicit:
  - update 20: `dom=0.922`, `backtrack=0.895`
  - update 40: `dom=0.926`, `backtrack=0.852`, `alert_streak=2`
  - update 60: `dom=0.949`, `backtrack=0.899`, `alert_streak=3`
- Case-level traces (stage3):
  - `grid_3x3`: `done=0/6`, timeout `6/6`, mean `ppo=500` vs `sabre=24`
  - `heavy_hex_19`: `done=0/6`, timeout `6/6`, mean `ppo=500` vs `sabre=190`
  - `linear_5`: `done=3/6`, timeout `3/6`, mean `ppo=253.5` vs `sabre=14`

### Diagnosis
- Stage3 policy collapses into repeated same-edge swapping (very high `dom` and `backtrack`).
- Progressive repeated-edge penalty is currently unbounded, which can explode negative returns and destabilize value learning on collapse trajectories.

## 20) Next Run Stabilization Changes (Implemented in code)
### A) Cap repeated-edge penalty
- Added `repeat_swap_penalty_cap` in environment reward:
  - repeated-edge term is now:
    - `max(repeat_swap_penalty_cap, repeat_swap_penalty_coeff * (same_edge_streak - 1))`
- New CLI flag:
  - `--repeat-swap-penalty-cap` (must be `<= 0`), default `-2.0`.

### B) Dynamic per-episode max steps (optional)
- Added dynamic episode cap based on circuit size:
  - `episode_max_steps = ceil(num_2q_gates * max_steps_per_two_qubit_gate)`
  - clamped by `max_steps_min`/`max_steps_max` (if set).
- New CLI flags:
  - `--max-steps-per-two-qubit-gate` (default `0.0`, disabled)
  - `--max-steps-min` (default `0`)
  - `--max-steps-max` (default `0`)
- `info` now logs `episode_max_steps`.

### C) Rationale for Run8
- Keep no hard masking.
- Preserve trace diagnostics.
- Reduce collapse severity and wasted 500-step failures on hard topologies.
- Make training signal denser and more stable when policy starts looping.

## 21) Latest Update (2026-03-24): Shortlist PPO Attempt
### What was tested
- New shortlist-based PPO variant:
  - action set reduced to frontier/shortest-path candidate swaps,
  - same `27x27x3` CNN input and RL loop,
  - multi-topology training (`heavy_hex_19, grid_3x3, linear_5`).
- Run:
  - `shortlist_ppo_20260324_120618`
  - best eval checkpoint at update `20`, final logged update `80`.

### Key metrics
- Best eval:
  - `eval_improvement_pct = -1101.47%`
  - `eval_win_rate = 0.00`
  - `eval_timeout_rate = 1.00`
  - `eval_mean_ppo_swaps = 250.56`
  - `eval_mean_sabre_swaps = 50.58`
- Final trace indicators:
  - `trace_timeout_rate = 1.00`
  - `trace_backtrack_rate = 0.989`
  - `trace_action_dom_ratio = 0.995`
  - `trace_alert_streak = 4`

### Verdict
- This shortlist attempt still collapsed into repetitive swaps and full timeout.
- No improvement over the best Run8 checkpoint.
- Practical decision: keep Run8 as PPO reference, and improve diagnostics + run control before launching further architecture variants.

## 22) Latest Update (2026-03-24): DQN Warm-Start and Stability Checks
### A) `grid_3x3` warm-start DQN (3 seeds, 500k steps each)
- Runs:
  - `dqn_warm_grid_3x3_seed42`
  - `dqn_warm_grid_3x3_seed123`
  - `dqn_warm_grid_3x3_seed999`
- Aggregate outcome:
  - `best_eval_improve mean = -1689.84% (std 90.52)`
  - `best_eval_win mean = 0.00`
  - `best_eval_timeout mean = 0.972`
  - latest traces: `dom mean = 0.971`, `backtrack mean = 0.960`, timeout `= 1.00`
- Interpretation:
  - Strong loop-collapse pattern; warm-start from supervised model did not transfer to rollout success on `grid_3x3`.

### B) `heavy_hex_19` probe (seed 42, 250k steps)
- Run:
  - `dqn_probe_heavy_heavy_hex_19_seed42`
- Best/last eval:
  - `improvement = -271.62%`
  - `win_rate = 0.00`
  - `timeout_rate = 1.00`
  - `model_swaps/sabre_swaps = 450.0 / 121.6`
- Trace:
  - `dom = 0.989`, `backtrack = 0.984`, `trace_timeout = 1.00`
- Interpretation:
  - Full-timeout failure mode remains on heavy-hex despite warm-start initialization.

### C) Mixed stability run (`linear_5 + grid_3x3`)
- Run:
  - `dqn_stab_lingrid_20260324_091230`
- Best checkpoint:
  - `improve = -1342.98%`, `win = 0.021`, `timeout = 0.896`
- Last checkpoint:
  - `improve = -1502.86%`, `win = 0.00`, `timeout = 1.00`
  - trace: `dom = 0.977`, `backtrack = 0.971`
- Interpretation:
  - Brief non-zero wins appeared, then regressed to looped timeout behavior.

### Decision
- Keep `Run8` PPO as the best current RL reference.
- Add stronger visualization/diagnostic tooling to detect collapse earlier and compare runs consistently before the next training cycle.

## 23) Visualization/Diagnostic Layer (Implemented on `version-27`)
### Added tool
- New script: `scripts/visualize_run_diagnostics.py`

### What it generates for one run directory
- Per-stage dashboard figure:
  - reward/completion trends,
  - episode return trend,
  - eval vs SABRE (`improvement`, `win_rate`, `timeout`),
  - optimization signals (`policy/value loss`, entropy, KL),
  - collapse signals from traces (`dom`, `backtrack`, `trace_timeout`, alert streak).
- Per-stage latest-trace topology figure:
  - mean agent SWAPs vs SABRE SWAPs by topology,
  - annotations for `done`, `dom`, `backtrack`.
- Run overview figure:
  - best vs last eval improvement per stage.

### Expected usage
- Input:
  - `--run-dir runs/<run_name>`
- Output (default):
  - `runs/<run_name>/figures_diag/*.png`

## 24) Best Checkpoint Selection Upgrade (Implemented on `version-27`)
### Why
- Previous `best_model.pt` selection used only eval SABRE metrics, so a checkpoint with loop-collapse behavior could still be selected.

### What changed
- `best_model.pt` is now selected with a lexicographic key that prioritizes:
  1) exploitable checkpoint flag,
  2) eval win rate,
  3) eval improvement vs SABRE,
  4) lower eval timeout,
  5) lower trace timeout,
  6) lower trace dominant-action ratio,
  7) lower trace backtrack rate.
- Exploitability flag uses configurable thresholds:
  - `--best-model-max-eval-timeout-rate`
  - `--best-model-max-trace-timeout-rate`
  - `--best-model-max-trace-dom-ratio`
  - `--best-model-max-trace-backtrack-rate`
  - `--best-model-reject-trace-alert` / `--no-best-model-reject-trace-alert`
  - `--best-model-require-trace`
  - `--best-model-use-latest-trace` / `--no-best-model-use-latest-trace`
- `best_eval_metrics.json` now contains:
  - `selection_key_labels`
  - `selection_diagnostics`
  - optional `trace_metrics_for_selection`

### Expected effect
- Reduce accidental selection of checkpoints that look good on one noisy eval but are clearly unstable in traces.

## 25) Curriculum Transition and Per-Topology Eval Logging (Implemented)
### A) Curriculum handoff changed
- In curriculum mode, each next phase now initializes from previous phase `best_model.pt` when available (instead of always using the last update state).
- Goal:
  - reduce regression when entering harder stages (`linear -> linear+grid -> full`),
  - preserve the most stable checkpoint behavior across phase transitions.

### B) Eval now tracked by topology
- Periodic eval now computes and stores per-topology metrics:
  - `eval_improvement_pct`,
  - `eval_win_rate`,
  - `eval_timeout_rate`,
  - mean SWAP counts (agent vs SABRE),
  - mean 2-qubit gate count.
- Saved snapshots:
  - `runs/<run_name>/<stage>/eval/update_XXXXX.json`
- Console output now prints compact per-topology eval summary at eval checkpoints.

### Why this matters
- Avoids hidden averaging effects where one topology (often `grid_3x3`) silently dominates and masks progress on another (`linear_5`).

## 26) Next 2-Stage PPO Config (Applied in Code + Notebook)
### Goal
- Keep strong stage1 behavior while reducing stage2 collapse risk.

### Applied settings
- Stage1 unchanged.
- Stage2 softened:
  - `stage2_depth=12`
  - `stage2_steps=200000`
- Anti-loop strengthened:
  - `repeat_swap_penalty_coeff=-0.3`
  - `no_progress_penalty_coeff=-0.05`
  - `no_progress_penalty_cap=-2.0`
- Dynamic episode cap enabled:
  - `max_steps_per_two_qubit_gate=7`
  - `max_steps_min=40`
  - `max_steps_max=320`
- More frequent eval:
  - `eval_interval_updates=10`

### Where updated
- `src/main.py` (defaults)
- `notebooks/train_ppo_colab.ipynb` (2-stage run cell)

## 27) Follow-up Adjustment After Early 2-Stage Collapse (Applied)
### Observation
- With aggressive anti-loop + tight dynamic step cap, collapse appeared early:
  - stage1 already unstable at first eval checkpoints,
  - stage2 timeout reached 1.0 quickly.

### Updated strategy
- Keep:
  - `stage2_depth=12`
  - `stage2_steps=200000`
- Relax anti-loop penalties back to stable values:
  - `repeat_swap_penalty_coeff=-0.2`
  - `no_progress_penalty_coeff=-0.03`
  - `no_progress_penalty_cap=-1.5`
- Loosen dynamic max steps:
  - `max_steps_per_two_qubit_gate=10`
  - `max_steps_min=60`
  - `max_steps_max=450`
- Align eval and trace frequency:
  - `eval_interval_updates=10`
  - `trace_interval_updates=10`

### Where applied
- `src/main.py` defaults
- `notebooks/train_ppo_colab.ipynb` run cell (stable v3, subprocess-based)
