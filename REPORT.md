# Reinforcement learning for quantum circuit routing

**Authors:** elora drouilhet, ali dor
**Project:** msc reinforcement learning, centralesupélec, 2026
**Repository:** https://github.com/helloelora/rl-quantum-circuit-routing
**D3QN branch:** https://github.com/helloelora/rl-quantum-circuit-routing/tree/ali-dev
**PPO branch:** https://github.com/helloelora/rl-quantum-circuit-routing/tree/elora-ruche

---

## 1. Motivation

Quantum computers can only execute two-qubit gates between physically adjacent qubits. Real circuits contain gates between arbitrary qubit pairs. The compiler must therefore insert SWAP gates to bring qubits next to each other before each two-qubit gate can run. Each SWAP costs 3 CNOTs. More CNOTs means more decoherence and lower fidelity, so the goal is to minimise the number of inserted SWAPs.

This is the **qubit routing problem**. It is np-hard in the general case (siraichi et al., 2018). The industry baseline is **SABRE** (li et al., 2019), a bidirectional greedy heuristic shipped with ibm qiskit. SABRE is fast but greedy, so it can miss globally optimal routes.

We chose this problem for three reasons. First, it is a real bottleneck in current nisq hardware. Second, it has a natural mdp structure: each state is a partial routing, each action is a swap. Third, it is a hard combinatorial problem where rl can plausibly beat hand-crafted heuristics, which is one of the few clean wins for rl over classical algorithms.

We implemented two agents to compare on-policy and off-policy methods on the same environment. ali built a **d3qn + per** (off-policy, value-based) agent on the `ali-dev` branch. elora built a **ppo** (on-policy, actor-critic) agent on the `elora-ruche` branch. Both share the same gymnasium environment.

---

## 2. Environment

The environment is a custom gymnasium env: `QubitRoutingEnv`. It is shared between both agents.

### State space

The state is a `5 x 27 x 27` float tensor. The matrix is padded to 27 to support all topologies with the same network. The five channels are:

| Channel | Encodes | Update frequency |
|---------|---------|------------------|
| 0 | hardware adjacency (binary) | constant per episode |
| 1 | qubit mapping (permutation matrix, 1 at `[q, p]` if logical qubit `q` is at physical position `p`) | every swap |
| 2 | depth-decayed gate demand (`gamma^depth` accumulated, gamma = 0.5) | when gates execute |
| 3 | front-layer distance map (`1/distance` for each ready gate's two qubits) | every swap |
| 4 | stagnation signal (`steps_since_last_gate / max_steps`, uniform value) | every swap |

Channels 0-2 are the original 3-channel state. Channels 3-4 were added later because the 3-channel version caused q-value collapse — the network gave the same value to every state and the policy degenerated. Channel 3 gives a direct signal on which gates are close to executable. Channel 4 tells the agent it is stuck and should change strategy.

### Action space

The action space is `Discrete(max_edges)`, where `max_edges` is the largest edge count across all supported topologies. Each action selects one hardware edge on which to perform a swap. Actions beyond the current topology's edge count are masked: their logits / q-values are set to `-inf` before the softmax / argmax. Two qubits cannot be swapped if they do not share a hardware edge.

### Reward

The per-step reward combines a sparse progress signal with several shaping terms:

```
r = gates_executed * gate_reward            # +1 per gate routed
  + distance_coeff * delta_distance         # pozzi-style distance shaping
  + step_penalty                            # -0.05 per step
  + reverse_swap_penalty                    # -0.2 if same swap as previous
  + repeat_swap_penalty                     # -0.2 * streak (capped at -2.0)
  + no_progress_penalty                     # -0.03 * streak (capped at -1.5)
  + completion_bonus (if all gates routed)  # +15 (ppo) / +5 (d3qn)
  + timeout_penalty (if truncated)          # -8 (ppo) / -10 (d3qn)
```

The repetition and stagnation penalties are critical. Without them the agent learns to repeat one swap forever or oscillate between two swaps, since both give a small distance shaping reward.

### Episode flow

1. **reset:** pick a topology, generate a random circuit (depth 5-20 depending on the curriculum stage), set an initial mapping (80% random, 20% sabre).
2. **step:** the agent picks an edge. The swap is applied. All front-layer gates whose two qubits are now adjacent execute automatically (with cascading if a gate execution unlocks the next layer).
3. **terminate:** when all gates are routed (success), or after `max_steps` (timeout), or after a no-progress streak (early truncation).

The maximum number of steps per episode is dynamic: `max_steps = 10 * num_two_qubit_gates`, clamped between 60 and 450. Easy circuits get tighter budgets, hard circuits get more room.

---

## 3. Agents

### 3.1. D3QN + PER (ali, off-policy)

The d3qn agent is a **double dueling deep q-network** with **prioritized experience replay**. Full code, configs, and results live on the `ali-dev` branch: https://github.com/helloelora/rl-quantum-circuit-routing/tree/ali-dev.

**Network.** The 5-channel state goes through the shared cnn backbone (`Conv2d(5,32) -> Conv2d(32,64) -> Conv2d(64,32)`, all 3x3 with relu). Then a flatten feeds two heads:

- **value stream:** linear(23328, 256) -> relu -> linear(256, 1) → V(s)
- **advantage stream:** linear(23328, 256) -> relu -> linear(256, num_edges) → A(s, a)

The dueling combination is `Q(s, a) = V(s) + A(s, a) - mean(A)`. The mean subtraction is what makes the decomposition identifiable (otherwise V and A could shift by an arbitrary constant). The advantage stream learns "which swap is best given the state", and the value stream learns "how good is this state in general". This separation converges much faster than learning q-values directly, especially when many actions have similar values.

**Training loop.**

1. collect a transition `(s, a, r, s')` via epsilon-greedy exploration. epsilon decays from 1.0 to 0.02 over 4-6m steps.
2. store it in a prioritized replay buffer (sumtree, capacity 300-500k). New transitions get the maximum priority so they are sampled at least once.
3. every 4 environment steps, sample a minibatch of 128 transitions weighted by td error.
4. compute the **double dqn target**: pick the action with the online network, evaluate it with the target network. This eliminates the q-value overestimation bias of vanilla dqn.
5. minimise huber loss weighted by the per importance-sampling correction (beta annealed from 0.4 to 1.0).
6. soft-update the target network with polyak averaging (`tau = 0.005` every 500 steps).

**Why off-policy works here.** The replay buffer reuses each transition ~50 times. Per focuses sampling on transitions with high td error, which tend to be the hard, rare cases (gates near the end of a circuit, awkward mappings). This is exactly where the value function needs the most updates.

### 3.2. PPO with symmetric score map (elora, on-policy)

The ppo agent is a **proximal policy optimisation** actor-critic. Code lives on the `elora-ruche` branch.

**Network.** Same shared cnn backbone as d3qn. Then two heads:

- **policy head:** `Conv2d(32, 1, 1x1)` produces a 27x27 spatial score map. The map is symmetrised: `S = (S + S^T) / 2`. This bakes in the constraint `swap(i, j) = swap(j, i)` at the architecture level. Edge logits are gathered from the symmetric score at the position of each hardware edge. Invalid edges get `-inf`. A softmax gives the action distribution.
- **value head:** `Linear(23328, 256) -> relu -> Linear(256, 1)` → V(s).

**Training loop.**

1. collect a rollout of 4096 environment steps with the current policy. The policy is frozen during collection.
2. compute **gae advantages** (lambda = 0.97) for every step in the rollout, using V(s) as the baseline.
3. for 8 epochs, shuffle the rollout into minibatches of 256, compute the clipped ppo objective, and gradient-step on `policy_loss + 0.5 * value_loss - entropy_coef * entropy`.
4. throw the rollout away. Collect a new one with the updated policy. ppo is on-policy: old data is no longer valid because the policy generated it has changed.

**Anti-collapse engineering.** Vanilla ppo collapses on this combinatorial action space. The policy converges to spamming a single swap, since the entropy bonus is too weak to prevent it on a 22-action space and the action distribution can become very peaked. We added several fixes:

- **reverse-swap penalty** (-0.2) for choosing the same edge as the previous step.
- **progressive same-edge penalty** (-0.2 × streak, capped at -2.0).
- **no-progress termination**: kill the episode after 30-50 steps without any gate executed.
- **action-repeat logit penalty** (-0.20) subtracted from the logit of the previous action during the forward pass. This is a soft anti-loop bias at the policy level.
- **symmetric score map** in the policy head, as described above.

Without these, ppo on `linear_5` had a 75% timeout rate. With them, it converged to a stable 0% timeout in under 2m steps.

**Hyperparameters used:**
- learning rate: 3e-4 (annealed to 3e-5 with linear schedule on most runs)
- gamma: 0.995, gae lambda: 0.97
- clip range: 0.15
- update epochs: 8, minibatch size: 256
- entropy coef: 0.003 → 0.0001
- distance reward coef: 0.03 → 0.015 (annealed to reduce shaping bias late in training)

---

## 4. Results

### 4.1. PPO results (elora-ruche branch)

| Topology | Steps | Best ratio | Win rate | Completion |
|----------|-------|-----------|----------|------------|
| `linear_5` | 2m | **0.871** | 94% | 100% |
| `grid_3x3` (lr annealing) | 10m | 1.096 | 38% | 100% |
| `heavy_hex_19` (curriculum) | 5m (killed at 24h) | not converged | — | — |

**Linear_5 is solved.** ppo uses 14.1 swaps on average vs sabre's 16.2. That is a 12.9% improvement, with 94% win rate and 0% timeout. The agent converged in under 250k steps and stayed stable for the rest of the training.

**Grid_3x3 plateaus.** the first run (5m steps, constant lr) reached -24% improvement at update 1100 then regressed to -29% by 5m. The lr annealing run (10m steps, 3e-4 → 3e-5) eliminated the regression: it converged to a stable plateau at -10% improvement (ratio 1.096) for the last 6m steps. The agent completes 100% of grid circuits but uses ~3 extra swaps per circuit compared to sabre.

**Heavy_hex_19 is unsolved.** several attempts:
- 8m steps single-stage: policy stayed near-uniform (entropy 2.99/3.0), 100% timeout, no learning.
- 10m steps curriculum (depth 5 → 10 → 16): stage 1 and 2 made some progress (entropy dropped to ~2.0), stage 3 timed out at 24h before convergence.
- 10m steps curriculum (depth 5 → 10 → 20, aligned with ali's config): same pattern. Killed at 24h after ~5m steps.

The fundamental issue is **sample efficiency**. ppo throws away each rollout after 8 epochs, while d3qn reuses transitions ~50 times via the replay buffer. To match ali's 24m environment steps, we would need either a much longer wall-clock budget or vectorised environments to batch the inference.

### 4.2. D3QN results (ali-dev branch)

Best single-topology run on `heavy_hex_19`: **ratio 0.991** (run 019, 80k episodes ≈ 24m env steps). The agent uses 183.3 swaps on average vs sabre's 185.3.

The fine-tuning approach (run 029) reloaded run 019's best checkpoint and continued training with `lr = 1e-5` and `epsilon = 0.05 → 0.01`. It reached **ratio 0.969** in 20k additional episodes — the best result of the entire project.

| Run | Topology | Ratio vs sabre | Notes |
|-----|----------|---------------|-------|
| 019 | `heavy_hex_19` | **0.991** | 80k episodes, standard d3qn |
| 023 | `heavy_hex_19` | 0.994 | curriculum + cosine lr, 60k episodes (25% less time) |
| 018 | multi-topo (3 topologies) | 0.999 overall (linear 0.890, grid 1.008, heavy_hex 1.107) | one network for all three |
| 029 | `heavy_hex_19` (finetune) | **0.969** | best result, 2-stage training |

Full training curves, eval comparison plots, routing gifs, and per-run reports are on the [`ali-dev` branch](https://github.com/helloelora/rl-quantum-circuit-routing/tree/ali-dev).

### 4.3. Side-by-side comparison

| Topology | PPO best ratio | D3QN best ratio | Winner |
|----------|---------------|----------------|--------|
| `linear_5` | **0.871** | 0.890 | ppo |
| `grid_3x3` | 1.096 | **1.008** | d3qn |
| `heavy_hex_19` | not converged | **0.969** (finetune) | d3qn |

ppo wins on the smallest topology where the 4-action space is easy to explore on-policy. d3qn dominates on the larger topologies thanks to its sample-efficient replay buffer. This matches the conventional wisdom: on-policy methods are better when you have abundant fresh data and a small action space; off-policy methods are better when sample efficiency matters and the action space is large.

### 4.4. Discussion

**What worked.**
- the 5-channel state was the single biggest contribution. Both agents collapsed without channels 3-4, and both converged with them.
- ppo's symmetric score map enforces the swap symmetry at the architecture level instead of relying on the network to learn it.
- d3qn's per is a major sample efficiency win on hard topologies.
- lr annealing prevents late-training regression on grid_3x3 for ppo.
- curriculum learning (depth ramp) gives a 25% training time saving on d3qn for heavy_hex.
- fine-tuning a converged checkpoint with a much lower lr is the highest-impact technique we found (0.991 → 0.969 for d3qn).

**What didn't work.**
- the original 3-channel state caused q-value collapse on both agents.
- vanilla ppo without anti-collapse engineering spammed one action.
- replacing the stagnation channel with a per-position action history channel hurt grid_3x3 performance, so we reverted.
- n-step returns for d3qn caused catastrophic divergence (run 25, q-values dropped to -123).
- doubling the gate reward to 2.0 degraded d3qn ratio from 1.014 to 1.377 — the agent optimised for gate throughput instead of routing efficiency.

**Limitations.**
- ppo on heavy_hex remains an open problem with our current step budget. Vectorised environments would likely close the gap.
- both agents struggle with the largest available topology (`heavy_hex_27`, 27 qubits) because the action space grows.
- training is sensitive to the reward magnitude. Doubling any single component breaks the policy.
- the "win rate" metric is misleading when the agent times out — it can show 75% wins while never completing a circuit (we hit this several times).

**Future work.**
- vectorised environments for ppo to match d3qn's effective batch size.
- transfer learning: warm-start ppo on heavy_hex from a converged grid_3x3 model.
- larger topologies (`heavy_hex_27`) with the same architecture.
- hybrid rl + mcts at inference time for an extra optimisation pass.

---

## 5. Code structure

The code is split across two branches:

**`elora-ruche`** (ppo):
- `src/environment.py` — gymnasium environment shared by both agents
- `src/agent.py` — ppo agent with symmetric cnn actor-critic
- `src/main.py` — cli entry point and curriculum runner
- `src/visualize.py` — training curves and eval comparison plots
- `src/circuit_utils.py` — dag, front layer, coupling maps, sabre baseline
- `ruche/` — slurm scripts and apptainer image definition for the la ruche cluster
- `docs/architecture_explained.md` — notes on ppo vs d3qn, channels, training loop
- `docs/poster.tex` — final poster source

**`ali-dev`** (d3qn + per): full d3qn implementation, configs, results, and `ARCHITECTURE.md`. See https://github.com/helloelora/rl-quantum-circuit-routing/tree/ali-dev.

---

## 6. References

- li, g., ding, y., & xie, y. (2019). *tackling the qubit mapping problem for nisq-era quantum devices.* asplos.
- siraichi, m. y., et al. (2018). *qubit allocation.* cgo.
- schulman, j., et al. (2017). *proximal policy optimization algorithms.* arxiv:1707.06347.
- wang, z., et al. (2016). *dueling network architectures for deep reinforcement learning.* icml.
- van hasselt, h., guez, a., & silver, d. (2016). *deep reinforcement learning with double q-learning.* aaai.
- schaul, t., et al. (2016). *prioritized experience replay.* iclr.
- pozzi, m. g., et al. (2022). *using reinforcement learning to perform qubit routing in quantum compilers.* acm computing surveys.
- bengio, y., et al. (2009). *curriculum learning.* icml.
