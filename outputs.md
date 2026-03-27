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

**Problem**: Training showed 60–80% completion but **eval showed 0%**. Root cause investigation revealed Q-value collapse — agent's greedy policy degenerates to spamming a single action because:
1. State doesn't change when agent is stuck (no gate executes → same observation)
2. All Q-values collapse to the same deeply negative value (~-33)
3. `argmax` always picks the same action → infinite loop
4. Epsilon exploration masked the broken policy during training

### V2 (Runs 5–8, current) — 5 fixes applied

1. **5-channel state** (+2 new channels):
   - Channel 3: Front-layer distance map — for each front-layer gate, encodes `1/distance` between its two qubits' physical positions. Tells the network _which positions need to become adjacent_.
   - Channel 4: Stagnation signal — uniform value = `steps_since_last_gate / max_steps`. Gives network awareness of being stuck.

2. **Action repetition penalty** (`-0.5`): Repeating the same SWAP twice undoes it (wastes 2 steps). Penalty discourages this degenerate pattern.

3. **Distance reward scaling** (`0.01 → 0.1`): Old coefficient was 10× smaller than step cost (-1.0), making distance shaping negligible. Now properly competes.

4. **Soft target updates** (`tau=0.005` every 500 steps instead of hard copy every 2000): Smoother target tracking prevents Q-value oscillation.

5. **Stagnation tracking**: `_steps_since_gate` counter in environment, reset when gates execute.

### Files modified for V2
- `environment.py`: 5-channel state, repetition penalty, stagnation tracking
- `networks.py`: `NUM_STATE_CHANNELS = 5`
- `dqn_agent.py`: State shape from `NUM_STATE_CHANNELS`
- `config.py`: Updated defaults (distance_reward_coeff=0.1, repetition_penalty=-0.5, tau=0.005, target_update_freq=500)
- `train.py`: Pass `repetition_penalty` to env constructor
- `main.py`: Pass `repetition_penalty` to eval env constructor
- `configs/*.json`: All 4 configs updated with new fields
- `experiment.slurm`: Hard-link SLURM logs into run directory

### V2 Validation — linear5 sanity test (Job 162423)
- **100% completion** throughout training and eval
- **Final eval: agent/SABRE ratio = 0.944** (agent beats SABRE on linear5)
- SWAPs decreased from ~26 → 5 over 2000 episodes
- All 3 phases (train → eval → visualize) completed cleanly

---

## Cancelled Runs (V1 — pre-fix)

| Run | Job ID | Config | Status | Notes |
|-----|--------|--------|--------|-------|
| 1 | 162244 | run1_heavy_hex_fixed | Cancelled | 0% eval completion, Q-value collapse |
| 2 | 162245 | run2_heavy_hex_highLR | Cancelled | Same issue |
| 3 | 162246 | run3_heavy_hex_long | Cancelled | Same issue |
| 4 | 162247 | run4_multi_fixed | Cancelled | Same issue |

---

## V2 Runs (pending — configs TBD)

Runs 5–8 will use the fixed V2 codebase. Configs to be determined.

---

## Monitoring Commands

```bash
# Quick status for all runs
squeue -u dor_ali | grep quantum

# Tail latest output
for f in outputs/slurm_*.out; do echo "=== $(basename $f) ===" && tail -3 "$f" 2>/dev/null; done

# Training curves (once run finishes)
python3 main.py visualize --run-dir outputs/run_NNN

# Full evaluation logs
cat outputs/run_NNN/logs/evaluations.jsonl | python3 -m json.tool
```
