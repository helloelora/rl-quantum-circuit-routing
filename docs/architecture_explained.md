# Architecture & RL Concepts — Poster Preparation Notes

## The 5-Channel State Representation

Think of it like an RGB image (3 channels) but with 5 channels encoding the routing problem state. Each channel is a 27x27 matrix (padded to support all topologies).

| Channel | Name | What it encodes | Changes when? |
|---------|------|----------------|---------------|
| Ch 0 | Hardware adjacency | Which physical qubits are neighbours (binary) | Never (fixed per topology) |
| Ch 1 | Qubit mapping | Permutation matrix: logical qubit q is at physical position p | Every SWAP |
| Ch 2 | Gate demand | Depth-decayed urgency of remaining gates (gamma=0.5^depth) | When gates are executed |
| Ch 3 | Front-layer distance | 1/distance for front-layer gate pairs (1.0 = adjacent) | Every SWAP (positions change) |
| Ch 4 | Action history | Decayed trail of recent SWAPs (0.8^k decay) | Every SWAP |

**Why 5 channels?** The original 3-channel design (Ch 0-2) caused Q-value collapse: the agent couldn't distinguish good from bad states and degenerated into repeating one action. Ch 3 gives a direct signal on which gates are close to being executable. Ch 4 shows the agent its own recent actions so it can avoid loops.

## The CNN Backbone (shared by both agents)

3-layer convolutional network: Conv2d(5->32, 3x3) -> Conv2d(32->64, 3x3) -> Conv2d(64->32, 3x3), all with padding=1 and ReLU.

**Why 3 layers?**
- Layer 1: detects local patterns (5x5 receptive field) — "there's a gate demand at this edge"
- Layer 2: combines patterns (5x5 receptive field) — "a routable gate is 2 hops away via this path"
- Layer 3: synthesises over 7x7 — "this region has high gate demand and qubits are close"

Heavy_hex_19 has diameter ~8, so 7x7 receptive field covers most routing decisions. Ali confirmed that a bigger network [64,128,64] performed worse on single-topology (ratio 1.160 vs 1.014) — more capacity doesn't help, 3 layers suffice.

## PPO Architecture (Actor-Critic)

```
Input: 5 x 27 x 27
       |
  CNN backbone [32, 64, 32]  (shared)
       |
       +-- Policy head (actor) --> "what action to take?"
       |   Conv 1x1 -> 27x27 score map -> symmetrise (S+S^T)/2
       |   -> extract scores for valid edges -> softmax -> probabilities
       |   -> SAMPLE from distribution -> action
       |
       +-- Value head (critic) --> "how good is this state?"
           Flatten -> Linear(256) -> Linear(1) -> V(s)
           (used to compute advantage, not to choose actions)
```

**Why two heads?**
- Policy head decides the action
- Value head provides a baseline for the advantage calculation
- Without value head, the policy gradient would be too noisy

**The advantage:**
```
Advantage = (reward + V(s')) - V(s)
If advantage > 0: action was better than expected -> increase its probability
If advantage < 0: action was worse than expected -> decrease its probability
```

The value head is the "critic" that judges actions. The policy head is the "actor" that adjusts. This is why it's called actor-critic.

**Symmetric score map:** The policy head produces a 27x27 matrix, then symmetrises it: S = (S + S^T) / 2. This encodes the fact that SWAP(i,j) = SWAP(j,i). Scores for each hardware edge are extracted from this symmetric matrix.

## D3QN Architecture (Dueling Double DQN + PER)

```
Input: 5 x 27 x 27
       |
  CNN backbone [32, 64, 32]  (shared)
       |
       +-- Value stream:     Flatten -> Linear(256) -> Linear(1) = V(s)
       |
       +-- Advantage stream: Flatten -> Linear(256) -> Linear(n_edges) = A(s,a)
       |
  Q(s,a) = V(s) + A(s,a) - mean(A)
       |
  Action masking: Q = -inf for invalid edges
       |
  Select max-Q edge (deterministic)
  Epsilon-greedy: with prob epsilon, pick random valid edge instead
```

**Dueling (Wang et al., 2016):** Separates Q into V(s) (how good is the state) + A(s,a) (how much better is this action than average). V is updated by every transition since it's shared across all Q-values. This converges faster than learning Q directly.

**Double DQN:** Uses a separate target network (soft-updated with tau=0.005) to avoid Q-value overestimation.

**PER (Prioritized Experience Replay):** Surprising transitions (high TD error) are replayed more often, focusing training on difficult cases.

## Action Masking (both agents)

The matrix is padded to 27x27 to support all topologies, but linear_5 has only 4 edges, grid_3x3 has 12, heavy_hex_19 has 22. Invalid edges must never be chosen.

- **DQN:** set Q = -infinity on invalid edges -> argmax never selects them
- **PPO:** set logits = -infinity on invalid edges -> softmax gives probability 0

Same concept, both topology-aware. A single network works on all three topologies.

## On-Policy (PPO) vs Off-Policy (DQN)

**PPO (on-policy):**
1. Collect 4096 steps using the current policy
2. Compute advantages using the value head
3. Do 8 epochs of gradient updates on this data
4. Throw away the data and start over

Data can only be used once because the policy gradient requires `log pi(a|s)` from the *current* policy. Old data was collected by a different policy, so the gradient would be wrong.

**DQN (off-policy):**
1. Collect 1 step, store transition (s, a, r, s') in replay buffer (400k capacity)
2. Every 4 steps, sample a batch of 128 from the buffer
3. Compute TD target and update Q-network
4. Transitions stay in the buffer and are reused ~50 times

Q-learning asks "what is the VALUE of this action?" — the answer doesn't depend on who took the action, so old data is perfectly valid.

**Sample efficiency comparison (heavy_hex runs):**

| | Ali (DQN) | Us (PPO) |
|---|---|---|
| Env steps | ~24M | 8M |
| Each transition reused | ~50 times | 8 times |
| Total gradient samples | ~768M | ~64M |
| Training time | ~20h (RTX 3090) | ~16h estimated (A100, with cache fix) |

PPO needs more env steps to match DQN's learning because it throws away data. But PPO's exploration is more structured (stochastic policy vs random epsilon-greedy).

## Key Differences Summary

| | PPO | D3QN |
|---|---|---|
| Output | Probability per edge | Q-value per edge |
| Action selection | Sample from distribution | Take the max (+ epsilon-greedy) |
| Exploration | Natural (policy entropy) | Forced (epsilon random) |
| Data reuse | 8 times then discarded | ~50 times via replay buffer |
| Architecture trick | Symmetric score map | Dueling V/A streams |
| Strength | Stable, structured exploration | Sample-efficient (PER) |
| Best result | linear_5: 0.871 | heavy_hex: 0.991 |

## Training Loop

**Each step in the environment:**
1. Agent observes state (5 x 27 x 27)
2. CNN produces scores for each valid edge
3. Agent picks one edge (sample for PPO, max for DQN)
4. Environment performs SWAP on that edge
5. Auto-execute any front-layer gates that are now routable (qubits adjacent)
6. Environment returns: new state + reward
7. Episode ends when all gates are routed (success) or max_steps reached (timeout)

**Reward per step:**
- +1 per gate executed
- +0.1 * distance reduction (front-layer qubits moved closer)
- -0.05 step cost
- -0.2 if same SWAP as previous step (reverse)
- -0.2 * streak if same edge repeated
- -0.03 * streak if no gates executed for consecutive steps
- +15 completion bonus / -8 timeout penalty

## Performance Optimisation: Front-Layer Cache

The front layer (gates ready to execute) was recomputed 5-6 times per step. Since it only changes when gates are executed (~30% of steps), we cache it and recompute only when needed. Similarly, DAG depths (BFS traversal) are cached. This roughly halves the per-step compute time on heavy_hex_19.
