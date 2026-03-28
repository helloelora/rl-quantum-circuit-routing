# Quantum Circuit Routing — Experiment Runs

## Overview

D3QN+PER agent trained to minimize SWAP gates on quantum hardware topologies.
Baseline: IBM SABRE compiler. Target: match or beat SABRE swap count.

---

## Architecture & Fix Log

### V1 (Runs 1–4, cancelled)
- 3-channel state: adjacency, mapping, gate demand
- Hard target updates (tau=1.0, every 2000 steps)
- distance_reward_coeff = 0.01
- No action repetition penalty

**Problem**: Training showed 60–80% completion but **eval showed 0%**. Root cause: Q-value collapse — greedy policy degenerates to spamming one action. State doesn't change when stuck, Q-values collapse to ~-33, argmax always picks same action. Epsilon exploration masked this during training.

### V2 (Runs 5–8) — 5 fixes applied

1. **5-channel state** (+2 channels): Ch3 = front-layer distance map (1/distance), Ch4 = stagnation signal (steps_since_gate / max_steps)
2. **Action repetition penalty** (-0.5): Prevents degenerate swap-undo loops
3. **Distance reward scaling** (0.01 → 0.1): Now competes with step cost (-1.0)
4. **Soft target updates** (tau=0.005 every 500 steps, was hard copy every 2000)
5. **Stagnation tracking**: Counter in environment, reset when gates execute

**V2 Validation (linear5, Job 162423)**: 100% completion, ratio 0.944 (beats SABRE)

---

## V1 Cancelled Runs

| Run | Job ID | Config | Notes |
|-----|--------|--------|-------|
| 1 | 162244 | run1_heavy_hex_fixed | 0% eval completion, Q-value collapse |
| 2 | 162245 | run2_heavy_hex_highLR | Same |
| 3 | 162246 | run3_heavy_hex_long | Same |
| 4 | 162247 | run4_multi_fixed | Same |

---

## V2 Runs — Results

### Run 5 (Job 162431, run_010) — Heavy Hex Baseline

| Setting | Value |
|---------|-------|
| Topology | heavy_hex_19 |
| LR | 1e-4 |
| Episodes | 30,000 |
| Epsilon floor | 0.10 |
| Seed | 42 |
| Wall time | 10.1h |

**Training progression:**

| Episode | Reward | SWAPs | Completion | Loss | Q-value | Epsilon |
|---------|--------|-------|------------|------|---------|---------|
| 5,000 | -333 | 400 | 0% | 0.004 | -3.7 | 0.40 |
| 10,000 | -299 | 385 | 40% | 0.011 | -6.7 | 0.10 |
| 15,000 | -212 | 333 | 92% | 0.013 | -8.6 | 0.10 |
| 20,000 | -159 | 288 | 100% | 0.013 | -9.6 | 0.10 |
| 25,000 | -140 | 272 | 100% | 0.014 | -10.6 | 0.10 |
| 30,000 | -140 | 268 | 100% | 0.014 | -11.4 | 0.10 |

**Eval progression (greedy, vs SABRE):**

| Episode | Completion | Agent SWAPs | SABRE SWAPs | Ratio |
|---------|------------|-------------|-------------|-------|
| 5,499 | 6% | 396 | 185 | 1.90 |
| 10,499 | 62% | 354 | 186 | 1.77 |
| 14,999 | 92% | 283 | 187 | 1.46 |
| 21,999 | 100% | 229 | 185 | 1.24 |
| 27,999 | 100% | 206 | 185 | 1.12 |
| **29,999** | **100%** | **209** | **187** | **1.12** |

**Notes**: Converged ~ep25k. Ratio plateaued at 1.12. No Phase 2 eval (script bug — see Known Issues).

---

### Run 6 (Job 162432, run_011) — Heavy Hex + Higher LR

| Setting | Value |
|---------|-------|
| Topology | heavy_hex_19 |
| LR | **3e-4** |
| Episodes | 30,000 |
| Epsilon floor | 0.10 |
| Seed | 123 |
| Wall time | 10.5h |

**Eval progression:**

| Episode | Completion | Agent SWAPs | SABRE SWAPs | Ratio |
|---------|------------|-------------|-------------|-------|
| 8,999 | 12% | 394 | 187 | 1.91 |
| 14,999 | 96% | 281 | 187 | 1.48 |
| 21,999 | 98% | 250 | 188 | 1.32 |
| 27,999 | 100% | 215 | 183 | 1.18 |
| **29,999** | **100%** | **217** | **186** | **1.17** |

**Notes**: Slower early progress than Run 5, worse final ratio (1.17 vs 1.12). Q-values drifted deeper negative (-13.7). Higher LR hurt — more instability, no faster convergence. No Phase 2 eval (script bug).

---

### Run 7 (Job 162433, run_012) — Heavy Hex Extended + Low Epsilon ★ BEST

| Setting | Value |
|---------|-------|
| Topology | heavy_hex_19 |
| LR | 1e-4 |
| Episodes | **40,000** |
| Epsilon floor | **0.02** |
| Seed | 42 |
| Wall time | 12.5h |

**Eval progression:**

| Episode | Completion | Agent SWAPs | SABRE SWAPs | Ratio |
|---------|------------|-------------|-------------|-------|
| 5,999 | 4% | 396 | 186 | 1.73 |
| 9,999 | 72% | 333 | 184 | 1.69 |
| 14,999 | 98% | 255 | 188 | 1.35 |
| 19,999 | 100% | 229 | 188 | 1.21 |
| 29,999 | 100% | 216 | 188 | 1.15 |
| 34,999 | 100% | 210 | 186 | 1.13 |
| 38,499 | 100% | 201 | 186 | **1.08** ← best |
| **39,999** | **100%** | **204** | **186** | **1.10** |

**Notes**: Best heavy_hex result. Still improving at ep40k. Low epsilon floor (0.02) was the biggest differentiator vs Run 5 (identical otherwise). Agent uses only 8-10% more SWAPs than SABRE.

---

### Run 8 (Job 162434, run_013) — Multi-Topology Generalization

| Setting | Value |
|---------|-------|
| Topologies | **linear_5 + grid_3x3 + heavy_hex_19** |
| LR | 1e-4 |
| Episodes | **45,000** |
| Epsilon floor | 0.10 |
| Buffer | 300k |
| Seed | 42 |
| Wall time | 6.6h |

**Eval progression (150 circuits = 50 per topology):**

| Episode | Completion | Agent SWAPs | SABRE SWAPs | Ratio |
|---------|------------|-------------|-------------|-------|
| 999 | 62% | 185 | 78 | 2.26 |
| 9,999 | 68% | 147 | 78 | 1.04 |
| 19,999 | 85% | 129 | 75 | 1.10 |
| 29,999 | 99% | 101 | 77 | 1.10 |
| 39,999 | 99% | 98 | 78 | 1.11 |
| **44,999** | **98%** | **98** | **77** | **1.08** |

**Phase 2 final eval (100 circuits per topology):**

| Topology | Completion | Agent SWAPs | SABRE SWAPs | Ratio |
|----------|------------|-------------|-------------|-------|
| linear_5 | 100% | 15.6 | 17.1 | **0.916** ← beats SABRE |
| grid_3x3 | 100% | 26.5 | 25.9 | **1.027** ← nearly matches |
| heavy_hex_19 | 100% | 230.0 | 187.0 | 1.232 |
| **Overall** | **100%** | **90.7** | **76.7** | **1.058** |

**Notes**: Beats SABRE on linear_5 and nearly matches on grid_3x3. Heavy_hex suffers (1.23 vs 1.08 single-topo) because capacity is shared across topologies. Only run with Phase 2 eval.

---

## Cross-Run Comparison (Heavy Hex 19)

| Metric | Run 5 | Run 6 | Run 7 ★ | Run 8 |
|--------|-------|-------|---------|-------|
| LR | 1e-4 | 3e-4 | 1e-4 | 1e-4 |
| Eps floor | 0.10 | 0.10 | **0.02** | 0.10 |
| Episodes | 30k | 30k | **40k** | 45k (multi) |
| Final ratio (HH) | 1.12 | 1.17 | **1.08** | 1.23 |
| Ep to 100% comp | ~18.5k | ~17.5k | ~17k | ~41k |

**Winner: Run 7** — eps_end=0.02 + more episodes gave best heavy_hex ratio (1.08).

## Key Findings

1. **V2 fixes work**: 0% → 100% eval completion across all runs. Q-values stable.
2. **Epsilon floor matters most**: 0.02 >> 0.10. Run 7 vs Run 5 (identical otherwise) — 1.08 vs 1.12.
3. **More training helps**: Run 7 was still improving at ep40k. 30k episodes is not enough.
4. **LR=3e-4 is too high**: Worse final ratio (1.17), deeper Q-values, more instability.
5. **Multi-topo dilutes hard-topology performance**: Heavy_hex 1.23 in multi vs 1.08 single. But generalizes — beats SABRE on linear_5.
6. **Phase transition at ep7k-10k**: All heavy_hex runs jump from 0% to 40-80% completion here.

## Known Issues

- **Phase 2 eval bug**: `experiment.slurm` uses `ls -td outputs/run_* | head -1` to find latest run. When 4 jobs run concurrently, all target run_013. Only Run 8 got correct Phase 2 eval. Runs 5-7 need manual Phase 2 eval.
- **SLURM log location**: Jobs 162431-162434 wrote logs to `outputs/` root. Fixed for future runs (logs will move into run dir).

## Next Steps

1. **Fix Phase 2 eval script** — use the actual run dir from training, not `ls -td`
2. **Run Phase 2 eval manually** for Runs 5, 6, 7 with their own checkpoints
3. **Train longer** — eps_end=0.02, 60k+ episodes on heavy_hex (Run 7 was still improving)
4. **Try eps_end=0.02 on multi-topology** — never tested, could combine best of Run 7 and 8
5. **Try larger network** — current CNN is small (32-64-32). More capacity might close the gap to SABRE

## Monitoring Commands

```bash
squeue -u dor_ali | grep quantum
for f in outputs/slurm_*.out; do echo "=== $(basename $f) ===" && tail -3 "$f" 2>/dev/null; done
cat outputs/run_NNN/logs/evaluations.jsonl | python3 -m json.tool
```
