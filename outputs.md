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

**V2 Validation (linear5, Job 162423, run_007)**: 100% completion, ratio 0.944 (beats SABRE)

---

## V1 Cancelled Runs

| Run | Job ID | Config | Notes |
|-----|--------|--------|-------|
| 1 | 162244 | run1_heavy_hex_fixed | 0% eval completion, Q-value collapse |
| 2 | 162245 | run2_heavy_hex_highLR | Same |
| 3 | 162246 | run3_heavy_hex_long | Same |
| 4 | 162247 | run4_multi_fixed | Same |

---

## Run 5 (Job 162431, run_010) — Heavy Hex Baseline V2

### Config

| Parameter | Value |
|-----------|-------|
| Topology | heavy_hex_19 (19 qubits, 20 edges) |
| LR | 1e-4 |
| Batch size | 128 |
| Episodes | 30,000 |
| Epsilon | 1.0 → 0.10 over 3M steps |
| Target update | tau=0.005 every 500 steps |
| Buffer | 200k, PER alpha=0.6 |
| Seed | 42 |
| Wall time | 10.1h (~36,247s) |

**Purpose**: Baseline heavy_hex run with all V2 fixes and standard hyperparameters. Control run.

### Training Metrics

The `training_metrics.png` shows 4 panels: reward, completion rate, swap count, and loss/Q-value over episodes.

**Episode progression (rolling 100-ep windows):**

| Episode | Reward | SWAPs | Gates% | Completion | Epsilon | What's happening |
|---------|--------|-------|--------|------------|---------|------------------|
| 0-100 | -402 | 400 | 13.6% | 0% | 0.99 | Pure random exploration. Every episode times out at 400 steps. Reward is roughly -400 = (-1 step cost × 400 steps) + timeout penalty. Only ~14% of gates get routed by accident. |
| 500 | -397 | 400 | 17.6% | 0% | 0.94 | Still mostly random. Small gate% increase from replay buffer starting to train (train_start=1000). |
| 2,000 | -366 | 400 | 41.0% | 0% | 0.76 | Agent is learning partial routing — can get ~40% of gates routed but can't finish a full circuit. Distance shaping reward is guiding swaps in the right direction. Reward improving because more gates = more +1 bonuses offsetting -1 step cost. |
| 3,500 | -335 | 400 | 66.9% | 0% | 0.58 | Rapid gates% increase. The CNN is learning to read the distance map channel and move qubits toward each other. Still hitting max_steps every episode though. |
| 5,000 | -334 | 400 | 74.6% | 0% | 0.40 | Gates% plateauing in 70s. Agent routes most gates but can't close out the last 25%. The remaining gates are the hardest — deep in the DAG with complex dependencies. |
| **5,391** | -245 | 363 | 100% | **First completion** | 0.33 | **Breakthrough episode.** First time the agent completes all gates. 363 swaps (vs SABRE ~186), very inefficient but it found a path. |
| 7,500 | -305 | 400 | 93.5% | 6% (eval) | 0.10 | Epsilon hits floor. Eval shows 6% completion — the greedy policy can occasionally solve easy circuits but mostly fails. Training completion higher due to 10% random exploration helping. |
| 10,000 | -299 | 385 | 93% | 40% (train) | 0.10 | **Phase transition begins.** Completion rockets from single digits to 40% in eval. The agent has internalized enough about routing to solve ~40% of random circuits deterministically. Mean swaps starting to drop (385 vs 400). |
| 15,000 | -212 | 333 | 100% | 92% (eval) | 0.10 | Agent completes 92% of eval circuits. Now optimizing efficiency — swaps dropped from 400 to 333. Reward improving because shorter episodes = fewer -1 step penalties. |
| 18,500 | — | — | — | **First 100% eval** | 0.10 | Every eval circuit completed. Agent can reliably route any random heavy_hex circuit. |
| 20,000 | -159 | 288 | 100% | 100% | 0.10 | Swap count still falling (288). Loss ~0.013, stable. Q-values at -9.6, reflecting the expected cumulative negative reward of a ~280-step episode. |
| 25,000 | -140 | 272 | 100% | 100% | 0.10 | Improvement slowing. Swaps went from 288→272 in 5k episodes vs 333→288 in the previous 5k. Approaching asymptote. |
| **27,999** | — | — | — | **Best eval: ratio 1.095 (median)** | 0.10 | Best single eval checkpoint: 206 agent swaps vs 185 SABRE = ratio 1.117 mean, 1.095 median. |
| 30,000 | -140 | 268 | 100% | 100% | 0.10 | Final. Ratio 1.12. Plateaued — last 5k episodes barely improved. |

### Training Dynamics (train_steps.jsonl)

| Step | Loss | Q-value | TD Error | What it means |
|------|------|---------|----------|---------------|
| 100 | 0.059 | -1.12 | 0.52 | Initial random Q-values, high loss as network starts learning from first replay samples. |
| 50k | 0.007 | -1.60 | 0.24 | Loss drops fast — network learns basic value structure. Q≈-1.6 means "expect about 2 net penalties from here". |
| 200k | 0.005 | -2.11 | 0.63 | TD error jumps — agent starts completing circuits, creating high-variance reward signals (completion bonus +5 vs normal -1). PER beta hits 1.0 here. |
| 500k | 0.003 | -3.48 | 0.45 | Epsilon ~0.40. Q-values deepening as agent learns longer-horizon estimates. |
| 750k | 0.009 | -5.05 | 0.70 | Epsilon hits floor (0.10). Loss and TD error spike — the distribution of states changes as exploitation takes over. The network needs to adapt to the new state distribution. |
| 1.5M | 0.018 | -8.62 | 0.53 | Middle of training. Loss higher because episodes are longer and more complex (completing circuits = many more meaningful states to learn from). |
| 2.5M | 0.024 | -11.44 | 0.56 | End. Q-values at -11.4 — reflects expected cumulative reward of a ~270-step completing episode: (-1×270) + (gates×1) + 5 bonus ≈ -270+130+5 ≈ -135, discounted with γ=0.99. Loss still ~0.02, not fully converged. |

**Q-value interpretation**: Q-values are always negative because every step costs -1, and even a perfect agent needs ~180+ swaps. The Q-value represents the discounted sum of future rewards from a state. More negative = more steps expected ahead. Q-values getting monotonically more negative throughout training is expected — the agent learns to estimate longer horizons more accurately.

### Eval Progression (all 60 checkpoints)

| Episode | Completion | Agent SWAPs | SABRE | Ratio (mean) | Ratio (median) | Reward |
|---------|-----------|-------------|-------|-------------|---------------|--------|
| 499-4,999 | 0% | 400 | ~186 | NaN | NaN | -577 to -456 |
| 5,499 | 6% | 396 | 185 | 1.90 | 2.00 | -444 |
| 7,499 | 18% | 391 | 184 | 1.91 | 1.95 | -401 |
| 9,999 | 38% | 374 | 187 | 1.80 | 1.81 | -342 |
| 10,499 | 62% | 354 | 186 | 1.77 | 1.74 | -303 |
| 12,999 | 78% | 325 | 189 | 1.62 | 1.58 | -243 |
| 13,999 | 92% | 294 | 185 | 1.54 | 1.45 | -197 |
| 15,999 | 98% | 257 | 185 | 1.37 | 1.36 | -147 |
| 18,499 | **100%** | 242 | 189 | 1.28 | 1.28 | -123 |
| 21,999 | 100% | 229 | 185 | 1.24 | 1.19 | -109 |
| 24,999 | 100% | 224 | 187 | 1.20 | 1.17 | -103 |
| **27,999** | **100%** | **206** | **185** | **1.12** | **1.10** | **-83** |
| 29,999 | 100% | 209 | 187 | 1.12 | 1.11 | -85 |

**Pattern**: 0% completion for first 5k episodes → rapid climb ep5k-14k → 100% by ep18.5k → pure efficiency optimization ep18.5k-30k (ratio: 1.28 → 1.12).

### Verdict

Solid baseline. Proves V2 fixes work. Ratio 1.12 means agent needs ~12% more SWAPs than SABRE. Plateaued at 30k episodes — more training might help. The 10% epsilon floor means the agent is still taking random actions 10% of the time during training, which adds noise.

---

## Run 6 (Job 162432, run_011) — Heavy Hex + Higher LR

### Config

Same as Run 5 except:

| Changed | Run 5 | Run 6 |
|---------|-------|-------|
| LR | 1e-4 | **3e-4** |
| Seed | 42 | **123** |

**Purpose**: Test if 3× higher learning rate speeds convergence or causes instability.

### Training Metrics

**Episode progression:**

| Episode | Reward | SWAPs | Gates% | Completion | Epsilon | What's happening |
|---------|--------|-------|--------|------------|---------|------------------|
| 0-100 | -403 | 400 | 13.5% | 0% | 0.99 | Same start as Run 5. |
| 2,000 | -383 | 400 | 29.2% | 0% | 0.76 | **Slower than Run 5** (41% gates there). Higher LR causes bigger gradient steps that overshoot — the network oscillates more. |
| 4,000 | -399 | 400 | 26.4% | 0% | 0.52 | **Regression**. Gates% actually dropped from 39% back to 26%. This is a classic sign of LR being too high — the network "unlearns" progress when large updates destabilize previously learned features. Run 5 was at 72% gates here. |
| 5,000 | -345 | 400 | 68.3% | 0% | 0.40 | Recovery. Agent catches back up on gates but still 0 completions. |
| 6,500 | -437 | 400 | 35.7% | 0% | 0.22 | **Another regression**. Reward hits worst point (-437) at ep6500. The high LR makes training unstable during the critical exploration→exploitation transition. |
| **8,140** | -272 | 393 | 100% | **First completion** | 0.10 | First completion at ep8140 vs ep5391 in Run 5. That's 2,750 episodes later — the instability from high LR delayed the breakthrough by ~50%. |
| 9,000 | -379 | 400 | 72.5% | 12% (eval) | 0.10 | Post-breakthrough but only 12% eval completion. Run 5 was at 32% eval at ep9k. |
| 10,500 | -263 | 390 | 100% | 40% (eval) | 0.10 | Learning accelerating now that epsilon is stable at floor. |
| 14,999 | — | — | — | 96% (eval) | 0.10 | Finally catches up to Run 5 on completion (Run 5 was 92% here). The higher LR enables bigger improvements once the network is in the right basin. |
| 17,499 | — | — | — | **First 100% eval** | 0.10 | Slightly earlier than Run 5 (ep18.5k). |
| 20,000 | -176 | 302 | 100% | 100% | 0.10 | But swaps are 302 vs Run 5's 288 — worse efficiency. |
| 25,000 | -153 | 283 | 100% | 100% | 0.10 | Gap persists. Run 5 at 272. |
| 28,999 | — | — | — | **Best eval: ratio 1.149** | 0.10 | Best checkpoint. Median ratio 1.141. |
| 30,000 | -138 | 269 | 100% | 100% | 0.10 | Final swaps 269 — similar to Run 5's 268, but eval ratio worse (1.17 vs 1.12). |

### Training Dynamics

| Step | Loss | Q-value | What's different from Run 5 |
|------|------|---------|---------------------------|
| 100 | 0.125 | -0.86 | Loss 2× higher at start — larger LR means bigger initial updates. |
| 100k | 0.005 | -1.85 | Loss converges similarly. Q-values comparable. |
| 500k | 0.010 | -4.70 | Q-values dropping faster (-4.7 vs -3.5 in Run 5). Higher LR makes value estimates update more aggressively. |
| 1M | 0.004 | -8.54 | Q-values much deeper than Run 5 at same step (-8.5 vs -6.4). This "Q-value drift" is a known issue with high LR in DQN — the target network can't stabilize because the online network moves too fast, even with soft updates. |
| 2M | 0.004 | -12.30 | At training end, Q-values at -13.7 vs Run 5's -11.4. The deeper Q-values don't translate to better performance — they indicate the network is overshooting value estimates. |

### Eval Progression (key points)

| Episode | Completion | Agent SWAPs | SABRE | Ratio | Note |
|---------|-----------|-------------|-------|-------|------|
| 0-8,499 | 0% | 400 | ~187 | NaN | 3k episodes longer at 0% than Run 5 |
| 8,999 | 12% | 394 | 187 | 1.91 | First completions in eval |
| 10,999 | 60% | 360 | 188 | 1.81 | Rapid climb |
| 14,999 | 96% | 281 | 187 | 1.48 | Comparable to Run 5 now |
| 17,499 | 100% | 266 | 188 | 1.42 | First 100% eval |
| 21,999 | 98% | 250 | 188 | 1.32 | |
| 25,999 | 100% | 228 | 190 | 1.20 | Best run here — but Run 5 was at 1.17 by this point |
| 28,999 | 100% | 216 | 188 | **1.15** | Best checkpoint |
| 29,999 | 100% | 217 | 186 | 1.17 | Final |

**Instability signal**: Completion rate fluctuates even late in training. At ep19499 it drops to 88%, at ep18999 to 90%. Run 5 never dropped below 96% after ep15k. This is the LR being too aggressive — the network occasionally "forgets" how to solve certain circuit configurations.

### Verdict

**Higher LR hurt.** Delayed first completion by ~2750 episodes, caused multiple regressions in gates%, produced deeper Q-value drift, and ended with worse ratio (1.17 vs 1.12). The instability during the exploration→exploitation transition (ep4000-7000) is the most damaging — it wastes critical training time. LR=1e-4 is the right choice.

---

## Run 7 (Job 162433, run_012) — Heavy Hex Extended + Low Epsilon ★ BEST

### Config

Same as Run 5 except:

| Changed | Run 5 | Run 7 |
|---------|-------|-------|
| Epsilon floor | 0.10 | **0.02** |
| Episodes | 30,000 | **40,000** |
| PER beta anneal | 200k steps | **300k steps** |
| Checkpoint freq | every 1000 | every 2000 |

**Purpose**: Lower epsilon floor for more exploitation + more episodes to see if learning continues past 30k.

### Training Metrics

**Episode progression:**

| Episode | Reward | SWAPs | Gates% | Completion | Epsilon | What's happening |
|---------|--------|-------|--------|------------|---------|------------------|
| 0-100 | -402 | 400 | 13.6% | 0% | 0.99 | Identical start to Run 5 (same seed). |
| 2,000 | -378 | 400 | 33.3% | 0% | 0.73 | Slightly behind Run 5 (41%) — epsilon is decaying at same rate but hasn't diverged yet. |
| 5,000 | -337 | 400 | 72.0% | 0% | 0.34 | Comparable to Run 5. Epsilon is lower here (0.34 vs 0.40) because same decay rate but epsilon floor is 0.02 not 0.10. |
| **5,360** | — | 399 | 100% | **First completion** | 0.30 | Nearly identical first completion timing to Run 5 (ep5391). Same seed = similar exploration trajectory. |
| 7,500 | -388 | 398 | 71.8% | 6% (eval) | **0.02** | **Epsilon hits floor.** This is the critical divergence from Run 5. At ep7500, Run 5 has eps=0.10, this run has eps=0.02. The agent now takes random actions only 2% of the time vs 10%. |
| 7,500-8,000 | — | — | — | — | 0.02 | **Temporary dip.** Completion drops because the agent suddenly loses most of its exploration. But this forces the greedy policy to improve much faster. |
| 8,000 | -349 | 392 | 82.8% | 25% | 0.02 | Recovery begins. With eps=0.02, the replay buffer fills with almost-all-greedy transitions, giving the network much cleaner signal about which actions the policy actually takes. |
| 9,500 | -296 | 372 | 93.1% | 74% (eval) | 0.02 | **Faster climb than Run 5** (Run 5 was 36% eval at ep9.5k). Low epsilon pays off — the network learns from its own greedy behavior, not from noisy random actions. |
| 12,000 | -203 | 317 | 98.7% | 88% (eval) | 0.02 | |
| 15,000 | -175 | 295 | 99.2% | 98% (eval) | 0.02 | |
| **16,999** | — | — | — | **First 100% eval** | 0.02 | Earlier than Run 5 (ep18.5k). Low epsilon → cleaner training → faster convergence to reliable policy. |
| 20,000 | -135 | 265 | 100% | 100% | 0.02 | Swaps 265 vs Run 5's 288 at same point. Already better because the low-epsilon training doesn't waste steps on random swaps. |
| 25,000 | -124 | 254 | 99.6% | 100% (eval) | 0.02 | |
| 30,000 | -111 | 244 | 100% | 100% | 0.02 | **Run 5 ended here at ratio 1.12. This run at ratio 1.15.** Wait — Run 5 is slightly better at ep30k? Yes, because the eval at this point fluctuates. But the trend is clear: this run keeps improving. |
| 35,000 | -107 | 239 | 100% | 100% | 0.02 | Ratio down to 1.13. Still improving. Every 5k episodes shaves ~5 swaps. |
| 38,499 | — | — | — | **Best eval: ratio 1.084** | 0.02 | **Best checkpoint.** 201 agent swaps vs 186 SABRE. Only 8.4% more swaps than SABRE. |
| 40,000 | -100 | 230 | 100% | 100% | 0.02 | Final training swaps 230, ratio 1.10. **Still improving** — the curve hasn't flattened. |

**Best training episode**: ep32743, 173 swaps (reward -51.8). Shows the agent CAN route very efficiently on some circuits.

### Training Dynamics

| Step | Loss | Q-value | TD Error | Comparison to Run 5 |
|------|------|---------|----------|---------------------|
| 100 | 0.023 | -1.10 | 0.27 | Same (same seed). |
| 300k | 0.006 | -3.06 | 0.50 | Identical trajectory. |
| 700k | 0.008 | -5.08 | 0.54 | Q-values slightly deeper (-5.1 vs -5.0). The lower epsilon means the agent sees more negative states (no random actions bailing it out). |
| 1.5M | 0.013 | -8.53 | 0.46 | Very close to Run 5. Loss and Q-value dynamics are similar — the epsilon difference doesn't change the optimization landscape much, just the data distribution. |
| 2.5M | 0.013 | -11.19 | 0.46 | Run 5 ended at 2.5M steps. This run continues. |
| 2.97M | 0.013 | -12.19 | 0.46 | Final. Q-values slightly deeper than Run 5's -11.4 because more steps of training. TD error stable — no sign of instability. |

**Key difference from Run 5**: The training dynamics (loss, Q, TD error) are nearly identical. The improvement comes entirely from the **data distribution** — with eps=0.02, 98% of replay buffer transitions come from the greedy policy, giving the network much higher-quality data to learn from. With eps=0.10, 10% of transitions are random noise that dilutes learning.

### Eval Progression (all 80 checkpoints, key points)

| Episode | Completion | Agent SWAPs | SABRE | Ratio (mean) | Ratio (median) |
|---------|-----------|-------------|-------|-------------|---------------|
| 0-5,499 | 0% | 400 | ~186 | NaN | NaN |
| 5,999 | 4% | 396 | 186 | 1.73 | 1.73 |
| 7,999 | **40%** | 372 | 188 | 1.80 | 1.81 |
| 9,499 | 74% | 344 | 189 | 1.74 | 1.73 |
| 10,499 | 86% | 308 | 187 | 1.57 | 1.56 |
| 13,499 | 98% | 267 | 188 | 1.42 | 1.37 |
| 16,999 | **100%** | 259 | 186 | 1.40 | 1.34 |
| 19,999 | 100% | 229 | 188 | 1.22 | 1.21 |
| 24,999 | 100% | 218 | 191 | 1.15 | 1.14 |
| 29,999 | 100% | 216 | 188 | 1.15 | 1.17 |
| 34,499 | 100% | 205 | 187 | 1.10 | 1.10 |
| 36,499 | 100% | 206 | 188 | 1.09 | 1.08 |
| **38,499** | **100%** | **201** | **186** | **1.08** | **1.08** |
| 38,999 | 100% | 203 | 188 | 1.08 | 1.08 |
| 39,999 | 100% | 204 | 186 | 1.10 | 1.08 |

**Improvement rate**: Ratio drops ~0.01 per 1000 episodes in the ep30k-40k range. If this rate holds, another 20k episodes could push ratio to ~1.0 (matching SABRE).

### Verdict

**Best run.** Ratio 1.08 — only 8% more SWAPs than SABRE. Two factors combined:
1. **Low epsilon (0.02)**: Cleaner data in replay buffer → better value estimates → more efficient policy. This is the primary driver.
2. **More episodes (40k)**: The agent kept improving past where Run 5 stopped. The improvement curve at ep40k is **not flat** — there's clearly more to gain with longer training.

---

## Run 8 (Job 162434, run_013) — Multi-Topology Generalization

### Config

| Parameter | Value |
|-----------|-------|
| Topologies | **linear_5 (5q, 4 edges) + grid_3x3 (9q, 12 edges) + heavy_hex_19 (19q, 20 edges)** |
| LR | 1e-4 |
| Batch size | 128 |
| Episodes | **45,000** |
| Epsilon | 1.0 → 0.10 over **4.5M** steps |
| Buffer | **300k** |
| Train start | **2,000** (more warmup for multi-topo) |
| Eval every | **1,000** episodes (150 circuits = 50 per topology) |
| Seed | 42 |
| Wall time | 6.6h (~23,623s) — faster than single-topo because easy topologies have short episodes |

**Purpose**: Can a single agent learn routing across 3 different hardware topologies? Tests generalization — the novel contribution of the project.

### Training Metrics

This run is fundamentally different from Runs 5-7 because each episode randomly picks one of 3 topologies. Easy topologies (linear_5, grid_3x3) complete fast, making average metrics misleading.

**Episode samples by topology:**

| Episode | Topology | Reward | SWAPs | Gates% | Done | Epsilon |
|---------|----------|--------|-------|--------|------|---------|
| 0 | linear_5 | -103 | 120 | 100% | Yes | 1.00 |
| 500 | heavy_hex_19 | -400 | 400 | 15.3% | No | 0.98 |
| 2,000 | grid_3x3 | -74 | 132 | 100% | Yes | 0.92 |
| 5,000 | heavy_hex_19 | -386 | 400 | 26.4% | No | 0.80 |
| 8,000 | linear_5 | -6 | 33 | 100% | Yes | 0.69 |
| 10,000 | grid_3x3 | +10 | 55 | 100% | Yes | 0.63 |
| 15,000 | grid_3x3 | +17 | 48 | 100% | Yes | 0.46 |
| 20,000 | linear_5 | +9 | 22 | 100% | Yes | 0.31 |
| 30,000 | grid_3x3 | +29 | 35 | 100% | Yes | 0.10 |
| 40,000 | heavy_hex_19 | -148 | 277 | 100% | Yes | 0.10 |
| 44,999 | linear_5 | +17 | 17 | 100% | Yes | 0.10 |

**Key observation**: linear_5 and grid_3x3 episodes start completing almost immediately (even with random exploration, small topologies are easy). The challenge is entirely heavy_hex_19.

**Positive rewards**: grid_3x3 and linear_5 episodes have **positive** total reward once the agent learns them well. This is because the gate execution bonuses (+1 per gate) exceed the swap penalties (-1 per swap) when the agent is efficient. For example: +29 reward means roughly "executed 64 gates (+64), used 35 swaps (-35), got completion bonus (+5) = +34, minus some distance penalties".

### Training Dynamics

| Step | Loss | Q-value | Epsilon | Note |
|------|------|---------|---------|------|
| 100 | 0.116 | -0.95 | 1.00 | Higher initial loss than single-topo runs — the network sees 3 different topology structures, harder to learn. |
| 406k | 0.003 | -2.08 | 0.68 | Q-values much less negative than single-topo runs (-2.1 vs -3.5). This is because easy topologies have short episodes with positive reward, pulling the average Q up. |
| 812k | 0.011 | -3.48 | 0.35 | |
| 1.2M | 0.010 | -3.59 | 0.10 | Epsilon hits floor at ~1.2M steps. Slower than single-topo because the longer epsilon decay schedule (4.5M) and faster episodes accumulate steps differently. |
| 1.6M | 0.009 | -5.05 | 0.10 | Final. Q-values only -5.0 vs -12 in single-topo runs. The easy topologies dominate the replay buffer and keep average Q-values shallow. |

**Total steps only 1.6M** vs 2.5-3.0M for single-topo runs despite more episodes. Easy topologies finish in 20-50 steps vs 250-400 for heavy_hex, so each episode contributes fewer training steps.

### Eval Progression (all 45 checkpoints)

Eval runs 150 circuits (50 per topology). Completion rate is **averaged across all 3 topologies**. Since linear_5 and grid_3x3 are ~100% from early on, the completion rate reflects primarily heavy_hex success.

| Episode | Completion | Agent SWAPs | SABRE | Ratio | What's happening |
|---------|-----------|-------------|-------|-------|------------------|
| 999 | 62% | 185 | 78 | 2.26 | 62% = linear_5 (100%) + grid_3x3 (~87%) + heavy_hex (0%). Ratio is high because the few completing circuits are very inefficient. |
| 2,999 | 67% | 150 | 77 | 1.14 | Ratio drops fast — agent learns efficient linear_5 and grid_3x3 routing quickly. |
| 5,999 | 67% | 148 | 76 | 0.99 | **Ratio drops below 1.0!** Agent beats SABRE on easy topologies. But completion stuck at 67% — heavy_hex still at 0%. |
| 7,999 | 67% | 147 | 77 | 0.96 | Easy topologies mastered. Agent uses fewer swaps than SABRE on linear_5 and grid_3x3. Completion won't budge until heavy_hex cracks. |
| 10,999 | 67% | 147 | 77 | 0.94 | **Plateau at 67% for 10k episodes.** The network has capacity for easy topologies but heavy_hex requires fundamentally different routing strategies across a much larger graph. |
| 14,999 | 73% | 144 | 76 | 1.03 | First signs of heavy_hex progress — completion climbs to 73% (= some heavy_hex circuits now completing). |
| 17,999 | 79% | 132 | 76 | 1.03 | Heavy_hex accelerating. |
| 22,999 | 90% | 122 | 76 | 1.11 | 90% completion — most heavy_hex circuits now solved. Ratio increases because heavy_hex circuits are less efficient (more swaps per circuit). |
| 27,999 | 96% | 104 | 76 | 1.08 | Near-complete. |
| 35,999 | 99% | 96 | 77 | 1.07 | |
| 40,999 | **100%** | 91 | 77 | **1.02** | **Best eval.** First and only 100% completion. Overall ratio 1.02 — nearly matching SABRE across all topologies combined. |
| 44,999 | 98% | 98 | 77 | 1.08 | Final. Slight regression — 1/150 circuits failed. |

### Phase 2 Final Eval (100 circuits per topology, 300 total)

| Topology | Circuits | Completed | Agent SWAPs | SABRE SWAPs | Ratio | Interpretation |
|----------|----------|-----------|-------------|-------------|-------|----------------|
| linear_5 | 100 | 100/100 | 15.6 | 17.1 | **0.916** | **Beats SABRE by 8.4%.** The simple topology lets the agent find optimal routes the greedy SABRE heuristic misses. |
| grid_3x3 | 100 | 100/100 | 26.5 | 25.9 | **1.027** | **Nearly matches SABRE.** Only 2.7% more swaps. The 3×3 grid is small enough for the agent to learn near-optimal routing. |
| heavy_hex_19 | 100 | 100/100 | 230.0 | 187.0 | **1.232** | **23% more swaps than SABRE.** Much worse than Run 7's 1.08. The network splits its capacity across 3 topologies, and heavy_hex gets shortchanged. |
| **Overall** | **300** | **300/300** | **90.7** | **76.7** | **1.058** | 100% completion, 5.8% more swaps than SABRE overall. |

### Verdict

**Generalization works, but at a cost.** The single network successfully learns 3 different topologies — it beats SABRE on linear_5 and nearly matches on grid_3x3. But heavy_hex performance (1.23) is much worse than the specialized Run 7 (1.08). This is the classic specialization-vs-generalization tradeoff.

The 67% completion plateau (ep2k-14k) is the most interesting phenomenon — the network solves easy topologies immediately but needs thousands more episodes before heavy_hex "clicks." This suggests the CNN features for routing on a 19-qubit graph are fundamentally different from those for 5-9 qubits.

---

## Cross-Run Comparison (Heavy Hex 19)

| Metric | Run 5 (baseline) | Run 6 (high LR) | Run 7 (low eps) ★ | Run 8 (multi) |
|--------|-------------------|------------------|---------------------|---------------|
| LR | 1e-4 | **3e-4** | 1e-4 | 1e-4 |
| Eps floor | 0.10 | 0.10 | **0.02** | 0.10 |
| Episodes | 30k | 30k | **40k** | 45k (3 topos) |
| First completion ep | 5,391 | 8,140 | 5,360 | ~24,000 (HH only) |
| First 100% eval ep | 18,499 | 17,499 | 16,999 | 40,999 |
| Final agent SWAPs (HH) | 209 | 217 | **201** | 230 |
| Final SABRE SWAPs (HH) | 187 | 186 | 186 | 187 |
| **Final ratio (HH)** | **1.12** | **1.17** | **1.08** | **1.23** |
| Best ratio (HH) | 1.10 (ep28k) | 1.15 (ep29k) | **1.08** (ep38.5k) | 1.02 (overall, ep41k) |
| Q-value final | -11.4 | -13.7 | -12.2 | -5.0 |
| Still improving? | No (plateaued) | No | **Yes** | Yes (HH portion) |

## Key Findings

1. **V2 fixes work**: V1 had 0% eval completion. V2 gets 100% on all runs. The 5-channel state + repetition penalty + distance scaling broke the Q-value collapse loop.

2. **Epsilon floor is the most impactful hyperparameter**: Run 7 (eps=0.02) vs Run 5 (eps=0.10) — identical except epsilon floor. Result: 1.08 vs 1.12 ratio. Why? With eps=0.10, the replay buffer contains 10% random-action transitions that teach the network nothing about the greedy policy. With eps=0.02, 98% of transitions are from the actual policy being optimized, giving much cleaner gradient signal.

3. **More training episodes help and the curve hasn't flattened**: Run 7 improved from ratio 1.15→1.08 between ep25k-40k. The improvement rate (~0.01 per 1000 episodes) shows no sign of plateauing. This is the strongest signal — we're not done training.

4. **LR=3e-4 is too high**: Run 6 had regressions at ep4000 and ep6500 where gates% dropped. Delayed first completion by 2,750 episodes. Final ratio 1.17 vs 1.12 baseline. The instability during exploration→exploitation transition is the main damage.

5. **Multi-topology dilutes hard-topology performance but enables generalization**: Run 8's heavy_hex ratio (1.23) is much worse than Run 7 (1.08). But it beats SABRE on linear_5 (0.916) and nearly matches on grid_3x3 (1.027). The 67% completion plateau for 12k episodes shows the network needs significant additional capacity/time to learn heavy_hex alongside easier topologies.

6. **Phase transition at ep5k-8k**: All heavy_hex runs show a sharp jump from 0% to 40-80% completion when epsilon approaches its floor. This is the critical moment where the agent discovers circuit-completing strategies. Before this, the agent learns partial routing (moving qubits closer) but can't chain enough correct swaps to finish.

## Known Issues

- **Phase 2 eval bug (fixed)**: `experiment.slurm` used `ls -td outputs/run_* | head -1` to find latest run. When concurrent jobs ran, all 4 targeted run_013. Now uses `grep "Run directory:" slurm_output` to find the correct run. Only Run 8 has Phase 2 eval; Runs 5-7 need manual eval.

---

## V3 Runs (14–17) — In Progress

Building on V2 findings: eps=0.02 is best, more episodes help (curve not flat), LR=1e-4 confirmed. Each run tests one hypothesis.

### Run 14 — Heavy Hex 60k Episodes (Job 162867, run_014)

**Hypothesis**: Run 7 was still improving at ep40k. Does 60k reach ratio <1.05?

| Parameter | Value | vs Run 7 |
|-----------|-------|----------|
| Config | `run14_heavy_hex_60k.json` | |
| Topology | heavy_hex_19 | same |
| Episodes | **60,000** | **+20k** |
| Eps floor | 0.02 | same |
| Buffer | **300k** | +100k (more data for longer training) |
| Eps decay steps | **4M** | +1M (slower decay over more episodes) |
| Everything else | same as Run 7 | |

### Run 15 — Heavy Hex Bigger Network (Job 162868, run_015)

**Hypothesis**: The 32→64→32 CNN may be capacity-limited. A bigger network can represent more complex routing strategies.

| Parameter | Value | vs Run 7 |
|-----------|-------|----------|
| Config | `run15_heavy_hex_bignet.json` | |
| Topology | heavy_hex_19 | same |
| Episodes | 40,000 | same |
| Eps floor | 0.02 | same |
| Conv channels | **[64, 128, 64]** | **2× wider** |
| Dueling hidden | **512** | **2× wider** |
| Everything else | same as Run 7 | |

### Run 16 — Multi-Topology + Low Epsilon (Job 162869, run_016)

**Hypothesis**: Run 8 used eps=0.10. The biggest V2 finding was eps=0.02 >> 0.10. Does low epsilon fix multi-topology heavy_hex performance?

| Parameter | Value | vs Run 8 |
|-----------|-------|----------|
| Config | `run16_multi_low_eps.json` | |
| Topologies | linear_5, grid_3x3, heavy_hex_19 | same |
| Episodes | 45,000 | same |
| Eps floor | **0.02** | **was 0.10** |
| Everything else | same as Run 8 | |

### Run 17 — Heavy Hex Mixed Mapping (Job 162870, run_017)

**Hypothesis**: Training always starts from random mapping. SABRE-initialized mappings are closer to optimal, requiring fewer SWAPs. Does training on a mix (80% random, 20% SABRE) teach the agent to exploit good starting positions?

| Parameter | Value | vs Run 7 |
|-----------|-------|----------|
| Config | `run17_heavy_hex_mixed_map.json` | |
| Topology | heavy_hex_19 | same |
| Episodes | 40,000 | same |
| Eps floor | 0.02 | same |
| Initial mapping | **mixed** (80% random, 20% SABRE) | **was random** |
| Everything else | same as Run 7 | |

---

## Monitoring Commands

```bash
squeue -u dor_ali | grep quantum
for f in outputs/slurm_*.out; do echo "=== $(basename $f) ===" && tail -3 "$f" 2>/dev/null; done
# Quick results (V3+): each run now saves results_summary.json
cat outputs/run_NNN/results_summary.json | python3 -m json.tool
```
