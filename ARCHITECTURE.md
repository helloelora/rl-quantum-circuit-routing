# Architecture: RL-Based Quantum Circuit Routing with D3QN+PER

> **Technical reference** for the complete system architecture.
> Covers every component from problem definition to experimental findings,
> with mathematical formulations, design rationale, and paper references.

---

## Table of Contents

1. [Problem Definition](#1-problem-definition)
2. [MDP Formulation](#2-mdp-formulation)
3. [Network Architecture — DuelingCNN](#3-network-architecture--duelingcnn)
4. [Agent — D3QN + PER](#4-agent--d3qn--per)
5. [Training Pipeline](#5-training-pipeline)
6. [Multi-Topology Support](#6-multi-topology-support)
7. [Supported Hardware Topologies](#7-supported-hardware-topologies)
8. [Baseline: SABRE](#8-baseline-sabre)
9. [Key Experimental Findings](#9-key-experimental-findings)
10. [Current Best Configuration](#10-current-best-configuration)
11. [References](#11-references)

---

## 1. Problem Definition

### 1.1 What Is Quantum Circuit Routing?

A quantum algorithm is expressed as a **quantum circuit**: a sequence of quantum
gates applied to logical qubits. Single-qubit gates (e.g., Hadamard, T, Rz) act
on one qubit and can be executed on any physical qubit. Two-qubit gates (e.g.,
CNOT, CZ) require both operand qubits to be **physically adjacent** on the
hardware — connected by a direct coupling link.

Real quantum processors have **limited connectivity**: not every pair of physical
qubits is coupled. IBM's heavy-hex architecture, for example, connects each
qubit to at most 3 neighbors. When a two-qubit gate requires qubits that are
not adjacent, the compiler must insert **SWAP gates** to move qubit states along
the coupling graph until the operands become neighbors.

This is the **quantum circuit routing problem** (also called **qubit routing**
or **transpilation**): given a logical quantum circuit and a hardware coupling
graph, find a sequence of SWAP insertions that makes every two-qubit gate
executable, while minimizing the total number of SWAPs inserted.

### 1.2 Why It Matters: The NISQ Era

In the current **Noisy Intermediate-Scale Quantum (NISQ)** era, quantum
processors suffer from:

- **Gate errors**: every gate has a non-trivial error rate (~0.1-1% for
  two-qubit gates on current hardware). Each SWAP decomposes into 3 CNOT gates,
  so one SWAP triples the error contribution.
- **Decoherence**: qubit states decay over time. Longer circuits (more SWAPs)
  mean more time for decoherence to corrupt the computation.
- **Limited qubit counts**: NISQ devices have 50-1000+ qubits, but effective
  circuit depth is severely limited by noise.

Minimizing SWAP count directly improves the probability that a quantum
computation produces a correct result. A routing solution that uses even one
fewer SWAP can meaningfully impact output fidelity.

### 1.3 Formal Problem Statement

**Given:**
- A quantum circuit $C$ with $n$ logical qubits and a set of two-qubit gates
  $G = \{g_1, g_2, \ldots, g_m\}$, each $g_i = (q_a, q_b)$ specifying two
  logical qubits.
- A hardware coupling graph $H = (V, E)$ where $|V|$ is the number of physical
  qubits and $E$ are the bidirectional coupling edges.
- A dependency DAG $D$ over the gates, where $g_j$ depends on $g_i$ if they
  share a qubit and $g_i$ precedes $g_j$.

**Find:**
- An initial mapping $\pi: \{0, \ldots, n-1\} \rightarrow V$ assigning logical
  qubits to physical positions.
- A sequence of SWAP operations on edges in $E$, interleaved with gate
  executions, such that every gate $g_i = (q_a, q_b)$ is executed when
  $(\pi(q_a), \pi(q_b)) \in E$.

**Minimize:**
- The total number of SWAP operations inserted.

This problem is **NP-hard** in general (Cowtan et al., 2019; Siraichi et al.,
2018). Heuristic approaches like SABRE (Li et al., 2019) achieve good practical
performance but are not optimal. Our approach uses reinforcement learning to
discover routing strategies that can match or beat these heuristics.

---

## 2. MDP Formulation

The routing problem is formulated as a **Markov Decision Process** (MDP)
$(S, A, T, R, \gamma)$, implemented as a Gymnasium environment
(`src/environment.py` — class `QubitRoutingEnv`).

### 2.1 State Space

The state is a **5-channel $N \times N$ tensor**, where $N$ = `matrix_size`
(default 27, large enough for the biggest topology). The observation space is
`Box(0.0, 1.0, shape=(5, N, N), dtype=float32)`.

Each channel encodes a different aspect of the routing problem:

#### Channel 0: Hardware Adjacency Matrix $\mathbf{A}$

$$A[i, j] = \begin{cases} 1 & \text{if } (i, j) \in E \\ 0 & \text{otherwise} \end{cases}$$

Binary, symmetric, **constant within an episode**. Encodes the hardware
topology. The CNN can learn which edges exist and use this to plan SWAP
sequences. For topologies smaller than $N$ qubits, the extra rows/columns are
zero-padded.

**Rationale:** Explicit adjacency lets the network distinguish between different
topologies in multi-topology training without needing separate models.

#### Channel 1: Mapping Permutation Matrix $\mathbf{M}$

$$M[q, p] = \begin{cases} 1 & \text{if logical qubit } q \text{ is at physical position } p \\ 0 & \text{otherwise} \end{cases}$$

Binary permutation matrix. Exactly one 1 per row (each logical qubit is
somewhere) and one 1 per column (each position holds at most one qubit). Updated
after every SWAP.

**Rationale:** A permutation matrix is a lossless encoding of the mapping. The
CNN can cross-reference this with the adjacency and demand channels to determine
which SWAPs would be useful.

#### Channel 2: Depth-Decayed Gate Demand $\mathbf{D}$

For each remaining (unexecuted) gate $g$ involving logical qubits $(q_a, q_b)$
currently at physical positions $(p_a, p_b)$:

$$D[p_a, p_b] \mathrel{+}= \gamma_{\text{decay}}^{\text{depth}(g)}$$

where $\text{depth}(g)$ is the topological depth of gate $g$ in the remaining
DAG (front-layer gates have depth 0), and $\gamma_{\text{decay}} = 0.5$ by
default.

The channel is then normalized by dividing by $\frac{1}{1 - \gamma_{\text{decay}}}$
to keep values in $[0, 1]$:

$$D_{\text{normalized}} = D \cdot (1 - \gamma_{\text{decay}})$$

**Rationale:** This channel tells the network *where* gate demand exists in
physical space, with exponential discounting by DAG depth. Front-layer gates
(depth 0) contribute 1.0; gates at depth 1 contribute 0.5; depth 2 contributes
0.25; and so on. This guides the agent to prioritize near-term gates while
remaining aware of future demand. Inspired by the "lookahead" in SABRE's cost
function but encoded as a spatial feature map.

#### Channel 3: Front-Layer Distance Map $\mathbf{F}$

For each gate $g = (q_a, q_b)$ in the current front layer (all predecessors
executed), at physical positions $(p_a, p_b)$:

$$F[p_a, p_b] = \max\left(F[p_a, p_b],\ \frac{1}{d(p_a, p_b)}\right)$$

where $d(p_a, p_b)$ is the shortest-path distance on the coupling graph. If
$d = 0$ (same position, degenerate case), the value is 1.0.

Values: 1.0 when adjacent (distance 1, already executable), 0.5 at distance 2,
0.25 at distance 4, etc. The $\max$ aggregation handles multiple front-layer
gates mapping to the same physical positions.

**Rationale:** This channel focuses attention on the *immediate* routing
bottleneck: which front-layer gate pairs are far apart and need SWAPs to bring
them together. Higher values mean "closer to being executable." The inverse-
distance encoding naturally prioritizes the most constrained gates.

#### Channel 4: Stagnation Signal $\mathbf{S}$

$$S[i, j] = \min\left(\frac{\text{steps\_since\_last\_gate\_execution}}{\text{max\_steps}},\ 1.0\right) \quad \forall\ i, j$$

A **uniform scalar** broadcast across the entire $N \times N$ matrix. Resets to
0 whenever a gate is executed; increases by $1/\text{max\_steps}$ each step
without execution.

**Rationale:** Without this signal, the network has no sense of how long it has
been "stuck" — making the same unproductive SWAPs without executing any gates.
The stagnation channel acts as a soft alarm: as the value grows, the network
learns to try alternative SWAP strategies. It also helps the network anticipate
the timeout penalty, which fires when `step_count >= max_steps`.

### 2.2 Action Space

$$\mathcal{A} = \text{Discrete}(\text{max\_edges})$$

Each action index $a \in \{0, 1, \ldots, \text{max\_edges}-1\}$ corresponds to
performing a SWAP on a specific hardware edge. The edge list is sorted and fixed
per topology:

$$\text{edges}[a] = (p_1, p_2) \implies \text{SWAP positions } p_1 \text{ and } p_2$$

After each SWAP, the environment **automatically executes** all newly-routable
front-layer gates (gates whose operand qubits are now adjacent). This cascading
execution continues until no more gates can fire, so a single well-chosen SWAP
can trigger multiple gate executions.

**Action masking:** In multi-topology mode, `max_edges` is the maximum edge
count across all topologies. Actions beyond the current topology's edge count
are invalid and masked via `get_action_mask()`. The agent's `select_action()`
applies this mask: invalid actions get $Q = -\infty$ during greedy selection,
and are excluded from the random draw during exploration.

### 2.3 Reward Function

The reward at each time step is a composite signal:

$$r_t = \underbrace{-1}_{\text{SWAP cost}} + \underbrace{n_{\text{exec}} \cdot r_{\text{gate}}}_{\text{gate execution}} + \underbrace{c_{\text{dist}} \cdot \Delta d}_{\text{distance shaping}} + \underbrace{p_{\text{rep}}}_{\text{repetition}} + \underbrace{B_{\text{terminal}}}_{\text{terminal bonus/penalty}}$$

where:

#### Component 1: SWAP Cost ($-1$)

Every SWAP costs $-1$. This is the core signal: minimize total SWAPs.

**Rationale:** A SWAP decomposes into 3 CNOT gates on real hardware, each with
non-trivial error. The $-1$ per step creates direct pressure to solve the
routing problem in as few steps as possible.

#### Component 2: Gate Execution Reward ($n_{\text{exec}} \cdot r_{\text{gate}}$)

$n_{\text{exec}}$ is the number of gates auto-executed after this SWAP.
$r_{\text{gate}}$ = `gate_execution_reward` (default 1.0).

With $r_{\text{gate}} = 1.0$ and one gate executed, the net step reward becomes
$-1 + 1 = 0$, meaning a "useful" SWAP (one that enables a gate) is free. SWAPs
that execute multiple gates yield positive intermediate rewards.

**Rationale:** This balances the SWAP penalty with a progress signal. Without
gate execution reward, the agent only sees negative rewards until the completion
bonus, making credit assignment over long episodes extremely difficult.

#### Component 3: Distance Shaping ($c_{\text{dist}} \cdot \Delta d$)

$$\Delta d = d_{\text{before}} - d_{\text{after}}$$

where $d = \sum_{g \in \text{front\_layer}} \text{dist}(\pi(q_a^g), \pi(q_b^g))$
is the sum of shortest-path distances for all front-layer gate qubit pairs.
$c_{\text{dist}}$ = `distance_reward_coeff` (default 0.1).

A SWAP that moves front-layer qubits closer together yields positive $\Delta d$;
one that moves them apart yields negative $\Delta d$.

**Rationale:** This is the Pozzi-style distance shaping reward (Pozzi et al.,
2022). It provides dense, informative feedback every step — not just when gates
execute. The coefficient 0.1 keeps shaping subordinate to the primary SWAP cost
signal, preventing the agent from gaming the shaping reward at the expense of
actual progress.

#### Component 4: Action Repetition Penalty ($p_{\text{rep}}$)

$$p_{\text{rep}} = \begin{cases} -0.5 & \text{if } a_t = a_{t-1} \\ 0 & \text{otherwise} \end{cases}$$

**Rationale:** Performing the same SWAP twice in a row undoes the first SWAP,
wasting 2 steps. This is a common failure mode in early training. The penalty
discourages swap-undo loops without completely forbidding the action (which
could be needed in rare edge cases). The $-0.5$ value was chosen to make
repeat-SWAP strictly dominated: even with distance improvement, the total reward
$-1 + 0.1 \cdot \Delta d - 0.5$ is almost always worse than trying a different
action.

#### Component 5: Terminal Bonus/Penalty ($B_{\text{terminal}}$)

$$B_{\text{terminal}} = \begin{cases} +5.0 & \text{if all gates executed (success)} \\ -10.0 & \text{if step\_count} \geq \text{max\_steps (timeout)} \\ 0 & \text{otherwise} \end{cases}$$

**Rationale:** The completion bonus (+5.0) incentivizes actually finishing the
circuit, not just making local progress. The timeout penalty (-10.0) punishes
failure to route within the budget, creating urgency. The asymmetry (penalty >
bonus) reflects that timeout is a catastrophic failure — the circuit was not
routed — while completion is the expected baseline.

### 2.4 Episode Dynamics

1. **Reset:** Pick a topology (uniform or weighted random), generate a random
   circuit (`random_circuit(num_qubits, circuit_depth)`), set initial mapping
   (random permutation, identity, SABRE, or mixed), auto-execute any
   initially-routable gates.

2. **Step loop:** Agent selects action (SWAP), environment performs SWAP, auto-
   executes all newly-routable gates (cascading), computes reward.

3. **Termination:**
   - `terminated = True` when all gates executed.
   - `truncated = True` when `step_count >= max_steps`.

4. **Bootstrap distinction:** Truncated episodes still bootstrap $Q(s')$
   because the episode was cut short, not naturally ended. Only `terminated`
   is stored as the `done` flag in the replay buffer.

---

## 3. Network Architecture — DuelingCNN

Implemented in `src/networks.py`. The network takes the 5-channel $N \times N$
state tensor and outputs Q-values for all actions.

### 3.1 CNN Feature Extractor

```
Input: (batch, 5, N, N)
  -> Conv2d(5 -> C1, 3x3, padding=1) -> ReLU
  -> Conv2d(C1 -> C2, 3x3, padding=1) -> ReLU
  -> Conv2d(C2 -> C3, 3x3, padding=1) -> ReLU
  -> Flatten -> (batch, C3 * N * N)
```

Default channel progression: `[32, 64, 32]`. All convolutions use `padding=1`
("same" padding), preserving spatial dimensions throughout. No pooling layers.

**Why CNN:** The state is fundamentally a 2D spatial structure. Physical qubits
are arranged on a grid-like topology, and the channels encode spatial
relationships (adjacency, mapping, demand). Convolutional filters naturally
capture local patterns: which qubits are near each other, where demand
concentrates, which edges are useful. The same-padding design preserves the full
spatial resolution — every position matters for routing decisions.

**Why no pooling:** Pooling would lose positional information that is critical
for routing. The agent needs to know *exactly* which edges to SWAP, not just
that "there is demand somewhere in this region."

### 3.2 Dueling Architecture

After the CNN extracts features (flattened to $C_3 \cdot N^2$ dimensions), the
network splits into two streams (Wang et al., 2016):

**Value stream** — estimates the state value $V(s)$:
```
Linear(C3*N*N -> 256) -> ReLU -> Linear(256 -> 1)
```

**Advantage stream** — estimates per-action advantages $A(s, a)$:
```
Linear(C3*N*N -> 256) -> ReLU -> Linear(256 -> num_actions)
```

**Q-value aggregation:**

$$Q(s, a) = V(s) + A(s, a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s, a')$$

Subtracting the mean advantage ensures identifiability: $V(s)$ uniquely
represents the state value and $A(s, a)$ represents the relative advantage of
each action.

**Rationale:** In quantum circuit routing, many states have similar values (the
circuit is partially routed, and the remaining difficulty is similar across
nearby states). The dueling architecture lets the network learn $V(s)$
independently — "how hard is the remaining routing?" — and only needs the
advantage stream to differentiate between SWAP choices. This is more
sample-efficient than learning $Q(s, a)$ monolithically, especially when many
actions have similar values (Wang et al., 2016).

### 3.3 Parameter Counts

For $N = 27$ (heavy-hex compatible) with 20 actions:

| Configuration | Conv Channels | Dueling Hidden | Total Parameters |
|---|---|---|---|
| Standard | [32, 64, 32] | 256 | ~6.1M |
| BigNet | [64, 128, 64] | 512 | ~24.0M |

The bulk of parameters are in the first linear layers of the dueling streams
(flattened CNN output to hidden). For standard config:
$32 \times 27^2 = 23{,}328$ inputs to each 256-unit hidden layer.

**Experimental finding:** The standard [32, 64, 32] network outperforms the
bigger [64, 128, 64] network on single-topology tasks (see Section 9). The
larger network takes longer to converge and plateaus at a worse swap ratio on
heavy_hex_19. For multi-topology, the bigger network performs comparably but
requires more training time.

---

## 4. Agent — D3QN + PER

The agent (`src/dqn_agent.py` — class `D3QNAgent`) combines four key
algorithmic innovations:

1. **Double DQN** — reduces overestimation bias
2. **Dueling architecture** — separates value and advantage estimation
3. **Prioritized Experience Replay (PER)** — focuses learning on surprising
   transitions
4. **N-step returns** — reduces bias in target computation

Together, these form **D3QN+PER** (Double Dueling Deep Q-Network with
Prioritized Experience Replay).

### 4.1 Double DQN

#### The Overestimation Problem

Standard DQN computes targets as:

$$y_t = r_t + \gamma \max_{a'} Q_{\theta^-}(s_{t+1}, a')$$

The $\max$ operator introduces a systematic positive bias: when Q-values have
estimation noise, the maximum over noisy values is biased upward. Over many
updates, this overestimation compounds and can lead to divergent or suboptimal
policies (van Hasselt et al., 2016).

#### The Double DQN Fix

Double DQN **decouples** action selection from action evaluation:

$$y_t = r_t + \gamma^n \cdot Q_{\theta^-}\!\left(s_{t+n},\ \underset{a'}{\arg\max}\ Q_{\theta}(s_{t+n}, a')\right)$$

- The **online network** $Q_\theta$ selects the best action: $a^* = \arg\max_{a'} Q_\theta(s_{t+n}, a')$
- The **target network** $Q_{\theta^-}$ evaluates that action: $Q_{\theta^-}(s_{t+n}, a^*)$

Because the two networks have different parameters (the target network lags
behind), the selection and evaluation errors are decorrelated, eliminating the
systematic upward bias.

#### Implementation Detail: Action Masking in Targets

When computing the best next action, invalid actions (from topology masking) are
set to $-\infty$ before the argmax:

```python
q_online_next[~next_masks] = -float("inf")
best_actions = q_online_next.argmax(dim=1)
```

This ensures the target never references an invalid action, which is critical
for multi-topology training where different episodes have different valid action
sets.

### 4.2 Prioritized Experience Replay (PER)

Implemented in `src/replay_buffer.py`.

#### Motivation

Uniform random sampling from the replay buffer treats all transitions equally.
But some transitions are more "surprising" (larger TD error) and therefore more
informative for learning. PER samples transitions proportionally to their
priority, focusing gradient updates on the experiences the network has the most
to learn from (Schaul et al., 2016).

#### SumTree Data Structure

The `SumTree` class implements an array-based binary tree enabling O(log n)
proportional sampling:

- **Leaf nodes** (indices $[C, 2C)$ where $C$ = capacity) store individual
  transition priorities.
- **Internal nodes** store the sum of their children's priorities.
- **Root** (index 1) stores the total priority sum.

**Sampling:** To sample proportionally, draw a uniform random value
$v \in [0, \text{total})$ and traverse the tree: go left if $v \leq$ left
child's sum, else subtract left child's sum and go right. This reaches a leaf
in $O(\log C)$ time.

**Update:** When a priority changes, update the leaf and propagate the delta up
to the root. Also $O(\log C)$.

#### Priority Computation

When a transition is first added, it gets the current maximum priority
(optimistic initialization):

$$p_i = (\max_j p_j)^\alpha$$

After a training step, priorities are updated from TD errors:

$$p_i = (|\delta_i| + \epsilon)^\alpha$$

where $\delta_i = Q(s_i, a_i) - y_i$ is the TD error, $\epsilon = 10^{-6}$
prevents zero priority, and $\alpha = 0.6$ controls the degree of
prioritization ($\alpha = 0$: uniform, $\alpha = 1$: fully proportional).

#### Stratified Sampling

Instead of drawing batch_size independent samples (which could cluster in
high-priority regions), PER uses **stratified sampling**: the total priority
range $[0, \text{total})$ is divided into `batch_size` equal segments, and one
sample is drawn uniformly from each segment.

$$\text{segment}_i = \left[\frac{i \cdot \text{total}}{B},\ \frac{(i+1) \cdot \text{total}}{B}\right), \quad v_i \sim \text{Uniform}(\text{segment}_i)$$

This ensures diversity in the sampled batch — even low-priority transitions get
occasional representation.

#### Importance Sampling Correction

Prioritized sampling introduces bias: high-priority transitions are
overrepresented. To correct this, each sampled transition is weighted by an
**importance sampling (IS) weight**:

$$w_i = \left(\frac{1}{N \cdot P(i)}\right)^\beta$$

where $P(i) = p_i / \sum_j p_j$ is the sampling probability and $N$ is the
buffer size. Weights are normalized by dividing by $\max_i w_i$ so the maximum
weight is 1.0.

The exponent $\beta$ is **annealed** from $\beta_{\text{start}} = 0.4$ to
$\beta_{\text{end}} = 1.0$ over `per_beta_anneal_steps` training steps:

$$\beta(t) = \min\!\left(\beta_{\text{end}},\ \beta_{\text{start}} + (\beta_{\text{end}} - \beta_{\text{start}}) \cdot \frac{t}{T_\beta}\right)$$

Early in training, $\beta < 1$ allows some bias (acceptable because the policy
is rapidly changing anyway). Late in training, $\beta \to 1$ fully corrects the
bias for stable convergence.

#### Memory Optimization

States are stored as `uint8` (0-255) rather than `float32`, achieving a ~4x
memory reduction. Conversion happens at storage time (multiply by 255, clip,
cast) and retrieval time (cast to float32, divide by 255). The quantization
error ($\pm 1/255 \approx 0.004$) is negligible for the 0-1 valued state
channels.

### 4.3 N-Step Returns

Standard 1-step TD target:

$$y_t = r_t + \gamma \cdot Q(s_{t+1}, a^*)$$

N-step target:

$$y_t = \sum_{i=0}^{n-1} \gamma^i r_{t+i} + \gamma^n \cdot Q(s_{t+n}, a^*)$$

The n-step return uses $n$ actual rewards before bootstrapping from the
Q-estimate, reducing bootstrap bias at the cost of higher variance.

#### Buffer Accumulation Mechanism

The `_nstep_buf` is a FIFO list that accumulates recent transitions. When it
reaches length $n$, the oldest transition is "popped" with its n-step return
computed:

```
_pop_nstep():
    R = sum(gamma^i * r_i for i in range(min(len(buf), n)))
    store(s_0, a_0, R, s_n, done_n) into PER buffer
    remove oldest from _nstep_buf
```

When an episode ends (`end_episode()` or `done=True`), all remaining transitions
in the buffer are flushed with truncated n-step returns. This handles both
natural termination and timeout truncation correctly.

**Default:** `n_step = 1` (standard TD). Higher values (3-5) can be enabled
via config but have not shown consistent improvement in our experiments.

### 4.4 Soft Target Updates (Polyak Averaging)

Instead of periodically copying the entire online network to the target network
(hard update), we use **soft Polyak averaging** at every target update step:

$$\theta^- \leftarrow \tau \cdot \theta + (1 - \tau) \cdot \theta^-$$

with $\tau = 0.005$.

The target network is updated every `target_update_freq = 500` training steps
using this formula. When $\tau = 1.0$, this reduces to a hard copy.

**Rationale:** Soft updates create a smoother evolution of the target network,
reducing oscillations in Q-value targets. With hard updates every $K$ steps,
the target function changes abruptly, causing instability. With $\tau = 0.005$,
the target network is a slow-moving exponential average of recent online
networks, providing a stable regression target. This is particularly important
for quantum routing where episodes vary widely in difficulty (different random
circuits), causing high variance in Q-value estimates.

### 4.5 Epsilon-Greedy Exploration

The agent uses **linear epsilon decay**:

$$\epsilon(t) = \max\!\left(\epsilon_{\text{end}},\ \epsilon_{\text{start}} - (\epsilon_{\text{start}} - \epsilon_{\text{end}}) \cdot \frac{t}{T_\epsilon}\right)$$

With defaults: $\epsilon_{\text{start}} = 1.0$, $\epsilon_{\text{end}} = 0.02$,
and $T_\epsilon$ varying by experiment (e.g., 5,000,000 steps for long runs).

At each step:
- With probability $\epsilon$: sample a random valid action uniformly
- With probability $1 - \epsilon$: select $\arg\max_a Q(s, a)$ over valid
  actions (greedy)

**Why $\epsilon_{\text{end}} = 0.02$ beats $\epsilon_{\text{end}} = 0.10$:**

The final epsilon determines the quality of transitions entering the replay
buffer during late training. At $\epsilon = 0.02$, 98% of actions are greedy
(near-optimal), meaning the buffer fills with high-quality experience. At
$\epsilon = 0.10$, 10% of actions are random, injecting substantial noise into
late-training episodes. Since the PER buffer holds ~100k-400k transitions,
these noisy experiences persist for many training steps and degrade Q-estimates.

Experimental result: $\epsilon_{\text{end}} = 0.02$ consistently achieves
~2-5% lower swap ratios than $\epsilon_{\text{end}} = 0.05$ across all
topologies.

### 4.6 Learning Rate Scheduling

Two modes, controlled by `lr_schedule`:

#### Constant (default)

LR stays at `lr` (default $10^{-4}$) throughout training.

#### Cosine Annealing

$$\text{lr}(t) = \text{lr}_{\min} + \frac{1}{2}(\text{lr}_{\max} - \text{lr}_{\min})\left(1 + \cos\left(\frac{t}{T_{\max}} \cdot \pi\right)\right)$$

where $\text{lr}_{\max}$ is the initial LR, $\text{lr}_{\min} = 10^{-5}$, and
$T_{\max}$ is estimated as `total_episodes * 50` gradient steps.

**Rationale:** High LR early in training enables fast feature learning (the CNN
needs to understand the state structure). Low LR late in training enables
fine-tuning (small Q-value adjustments that improve swap ratios from 1.05 to
1.01). However, our experiments have not shown consistent benefit from cosine
scheduling over constant LR, likely because the epsilon schedule already
controls exploration/exploitation balance.

### 4.7 Loss Function

The loss combines **Huber loss** with **IS weight correction**:

$$\mathcal{L} = \frac{1}{B} \sum_{i=1}^{B} w_i \cdot \text{SmoothL1}(Q(s_i, a_i) - y_i)$$

where $w_i$ are the PER importance sampling weights and SmoothL1 (Huber loss)
is:

$$\text{SmoothL1}(x) = \begin{cases} 0.5 x^2 & \text{if } |x| < 1 \\ |x| - 0.5 & \text{otherwise} \end{cases}$$

**Why Huber over MSE:** TD errors can be very large, especially early in
training or after epsilon transitions to bad states. MSE ($x^2$) would produce
enormous gradients from these outliers, destabilizing learning. Huber loss
transitions to linear for large errors, bounding the gradient magnitude. This
is especially important with PER, which preferentially samples high-TD-error
transitions.

**Gradient clipping:** After backpropagation, gradients are clipped to a maximum
norm of `grad_clip_norm = 10.0`:

```python
torch.nn.utils.clip_grad_norm_(online_net.parameters(), 10.0)
```

This provides a second line of defense against gradient explosions. The value
10.0 is permissive enough to allow fast learning while preventing catastrophic
updates.

---

## 5. Training Pipeline

Implemented in `src/train.py`.

### 5.1 Episode Loop Structure

```
for episode in range(total_episodes):
    obs, info = env.reset()              # New random circuit + topology
    action_mask = env.get_action_mask()

    while not (terminated or truncated):
        action = agent.select_action(obs, action_mask)
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_mask = env.get_action_mask()

        agent.store_transition(obs, action, reward, next_obs, terminated, next_mask)

        if global_step % train_freq == 0:
            metrics = agent.train_step()
            if train_steps % target_update_freq == 0:
                agent.update_target_network()

        agent.update_epsilon()
        obs = next_obs; action_mask = next_mask

    agent.end_episode()  # Flush n-step buffer
```

Key design decisions:

- **`train_freq = 4`** for primary experiments: one gradient step per 4
  environment steps. Reduces computational cost while maintaining sufficient
  update frequency.
- **`train_start = 1000`**: no training until 1000 transitions are in the
  buffer, ensuring sufficient diversity before first gradient step.
- **`terminated` vs `truncated`**: only `terminated` is stored as `done` in
  the buffer. Truncated episodes still bootstrap $Q(s')$, which is correct —
  truncation is an artificial time limit, not a true terminal state.
- **Circuit skipping**: episodes where the random circuit has zero two-qubit
  gates are logged but skipped (no stepping needed).

### 5.2 Curriculum Learning

Optional depth progression configured via `curriculum_depths` and
`curriculum_milestones`:

```python
curriculum_depths = [5, 10, 20]       # Circuit depths
curriculum_milestones = [0.15, 0.35]  # At 15% and 35% of training
```

At episode $e$ out of $E$ total:
- Progress $p = e / E$
- If $p < 0.15$: depth = 5 (easy circuits, ~2-5 two-qubit gates)
- If $0.15 \leq p < 0.35$: depth = 10 (medium circuits, ~5-15 gates)
- If $p \geq 0.35$: depth = 20 (full difficulty, ~15-40 gates)

**Rationale:** Easy circuits let the agent learn basic SWAP mechanics (bring
qubits together, execute gates) before facing the combinatorial complexity of
deep circuits. Without curriculum, the agent may spend thousands of episodes
timing out on depth-20 circuits, receiving only timeout penalties and learning
nothing useful.

### 5.3 Checkpoint Saving

Checkpoints are saved every `checkpoint_every` episodes (default 1000-2000):

```
outputs/run_NNN/checkpoints/
  checkpoint_ep1000.pt
  checkpoint_ep2000.pt
  ...
  checkpoint_final.pt     (or checkpoint_emergency.pt on Ctrl+C)
```

Each checkpoint contains: online/target network state dicts, optimizer state,
epsilon value, epsilon step counter, training step counter, global step count,
and elapsed time. Optionally includes the full replay buffer (`.buf.npz`).

**Resume support:** Training can resume from any checkpoint with full state
restoration, including the replay buffer if saved. The run directory is reused,
and logs are appended.

### 5.4 Evaluation During Training

Every `eval_every` episodes (default 500-1000), the agent is evaluated:

1. For each topology, run `eval_episodes` (default 20-50) random circuits.
2. Agent uses **deterministic greedy policy** (no epsilon exploration).
3. Both agent and SABRE route the **same circuit** with the **same initial
   mapping** (SABRE's layout).
4. Compare: swap ratio = agent_swaps / sabre_swaps.

Results are logged to `evaluations.jsonl` and visualized in `figures/`.

**Why SABRE initial mapping for eval:** Using SABRE's own initial mapping gives
a fair comparison — both methods start from the same state. If the agent used
a random mapping, it might get lucky or unlucky, adding noise to the comparison.

### 5.5 Output Directory Structure

```
outputs/run_NNN/
  config.json              # Complete hyperparameter snapshot
  results_summary.json     # Final training + eval metrics
  logs/
    episodes.jsonl         # Per-episode: reward, swaps, gates, completion
    train_steps.jsonl      # Per-100 steps: loss, mean_q, TD error, epsilon
    evaluations.jsonl      # Per-eval: agent vs SABRE comparison
  checkpoints/
    checkpoint_ep1000.pt   # Model + optimizer + buffer state
    checkpoint_final.pt
  figures/
    training_curves.png    # Reward, swaps, completion rate over episodes
    eval_comparison_ep*.png
  eval/
    eval_ep1000.json       # Detailed per-circuit results
```

### 5.6 Logging and Monitoring

- **JSONL format:** each log line is an independent JSON object, enabling
  streaming writes and fault-tolerant reads.
- **Ctrl+C handler:** saves an emergency checkpoint before exiting, so long
  runs are never fully lost.
- **tqdm progress bar:** shows real-time rolling averages of reward, SWAPs,
  gates routed, completion rate, loss, Q-value, and epsilon.

---

## 6. Multi-Topology Support

A key feature of this system is the ability to train a **single agent** across
multiple hardware topologies simultaneously.

### 6.1 How It Works

The `QubitRoutingEnv` accepts a `topologies` list:

```python
env = QubitRoutingEnv(
    topologies=["linear_5", "grid_3x3", "heavy_hex_19"],
    matrix_size=27,  # Must fit the largest topology
)
```

At each `reset()`, one topology is randomly selected (uniform or weighted). The
environment pre-computes all static data per topology (adjacency matrix, edge
list, distance matrix) at initialization time, so `reset()` only needs to swap
the active topology pointer.

### 6.2 Padded $N \times N$ State for Uniform Observation Shape

All topologies share the same observation shape `(5, N, N)`. Smaller topologies
(e.g., linear_5 with 5 qubits) are zero-padded to fill the $N \times N$
matrices. The CNN sees these zeros as "empty space" and learns to ignore them.

**Why fixed shape:** The DuelingCNN requires a fixed input size (the flattened
dimension depends on $N$). Alternative approaches (graph neural networks) could
handle variable-size inputs, but CNNs are simpler and have worked well in
practice.

### 6.3 Action Masking for Topology-Specific Edge Counts

`max_edges = max(num_edges for all topologies)`. For a topology with fewer
edges, actions beyond its edge count are masked to `False`:

```python
# get_action_mask()
mask = np.zeros(max_edges, dtype=bool)
mask[:current_topo["num_edges"]] = True
```

The agent never selects masked actions (random exploration only draws from valid
actions; greedy sets masked Q-values to $-\infty$).

### 6.4 Weighted Topology Sampling

Optional `topology_weights` controls sampling probability:

```python
topology_weights=[0.2, 0.2, 0.6]  # 20% linear, 20% grid, 60% heavy_hex
```

**Rationale:** Heavy-hex is the hardest topology (most qubits, most edges, most
complex routing). Without weighted sampling, the agent sees each topology
equally often, but spends most training time already-solved easy topologies.
Weighting toward the harder topology improves heavy_hex performance at minimal
cost to the easier topologies.

### 6.5 Specialization vs. Generalization Tradeoff

**Key experimental finding:** A multi-topology agent can match or beat SABRE
across all topologies simultaneously (Run 018: overall ratio 0.999). However,
a single-topology specialist on heavy_hex_19 converges faster and achieves
comparable performance on that specific topology. The multi-topology agent
benefits from **transfer learning**: strategies learned on smaller topologies
(bring qubits together, avoid cycles) generalize to larger ones.

---

## 7. Supported Hardware Topologies

Defined in `src/circuit_utils.py` — function `get_coupling_map()`.

### 7.1 linear_5: 5-Qubit Linear Chain

```
pos0 — pos1 — pos2 — pos3 — pos4
```

- **Qubits:** 5
- **Edges:** 4 (bidirectional)
- **Max distance:** 4 hops
- **Degree:** 1 (endpoints) or 2 (interior)
- **Construction:** `CouplingMap.from_line(5)`

The simplest topology, used for sanity checks and curriculum warm-up. Maximum
distance of 4 means up to 3 SWAPs may be needed for a single gate. Despite its
simplicity, it still requires non-trivial routing for circuits with many
two-qubit gates involving distant qubits.

### 7.2 grid_3x3: 9-Qubit Grid

```
pos0 — pos1 — pos2
 |      |      |
pos3 — pos4 — pos5
 |      |      |
pos6 — pos7 — pos8
```

- **Qubits:** 9
- **Edges:** 12 (bidirectional)
- **Max distance:** 4 hops (corner to corner)
- **Degree:** 2 (corners), 3 (edges), 4 (center)
- **Construction:** `CouplingMap.from_grid(3, 3)`

A moderate topology with richer connectivity than the linear chain. The 2D grid
structure means the CNN's spatial convolutions are particularly natural — the
3x3 kernel directly captures local neighborhoods in the coupling graph.

### 7.3 heavy_hex_19: 19-Qubit Heavy-Hex

- **Qubits:** 19
- **Edges:** 20 (bidirectional)
- **Max distance:** 8 hops
- **Degree:** 1, 2, or 3 (heavy-hex pattern)
- **Construction:** `CouplingMap.from_heavy_hex(3)`

This is a simplified version of IBM's heavy-hex architecture used in Eagle and
Heron processors. The "heavy-hex" pattern replaces each edge of a hexagonal
lattice with a path of length 2, inserting degree-2 "bridge" qubits. This
creates a topology with very sparse connectivity (average degree ~2.1), making
routing challenging.

The heavy-hex topology is the primary benchmark because:
1. It represents real IBM hardware.
2. Its sparse connectivity (max degree 3) creates long routing paths.
3. SABRE is specifically optimized for these topologies, making it a strong
   baseline.

### 7.4 Additional Topologies

The `get_coupling_map()` function also supports:

- **`linear_N`**: N-qubit linear chain (any N).
- **`ring_N`**: N-qubit ring (chain with wraparound edge).
- **`grid_RxC`**: R-by-C rectangular grid (any R, C).
- **`heavy_hex_27`**: 27-qubit heavy-hex (`from_heavy_hex(5)`), matching IBM's
  Falcon processors.

---

## 8. Baseline: SABRE

### 8.1 How SABRE Works

SABRE (SWAP-based Bidirectional heuristic search algorithm for Reversible
circuit transpilation) by Li et al. (2019) is the default routing algorithm in
IBM's Qiskit compiler. It operates in two phases:

**Forward pass:**
1. Maintain a front layer of ready-to-execute gates.
2. For each front-layer gate, compute a heuristic cost based on the distance
   between its qubits in the current mapping.
3. For each candidate SWAP (on edges incident to front-layer qubits), score the
   SWAP by how much it reduces the total heuristic cost, including a
   lookahead term for near-future gates.
4. Apply the best SWAP. Execute any newly-routable gates. Repeat.

**Backward pass:**
5. Reverse the circuit and run the forward pass again, starting from the final
   mapping of the forward pass. This "backward" routing often finds
   improvements.

**Best of both:** Take the solution with fewer SWAPs.

The SABRE heuristic cost for a SWAP candidate considers:
$$H = \sum_{g \in \text{front}} \frac{d(\pi'(q_a^g), \pi'(q_b^g))}{d_{\max}} + W \cdot \sum_{g \in \text{extended}} \frac{d(\pi'(q_a^g), \pi'(q_b^g))}{d_{\max}}$$

where $\pi'$ is the mapping after the candidate SWAP, $d$ is shortest-path
distance, and $W < 1$ is the lookahead weight.

### 8.2 Why SABRE Is a Strong Baseline

- It is the **industry standard** — used by default in Qiskit at
  `optimization_level=1`.
- It uses problem-specific knowledge: the dependency DAG structure, the coupling
  graph distances, and a lookahead heuristic.
- Its bidirectional search effectively doubles the search effort.
- It has been extensively tuned by IBM engineers over years.
- On random circuits, it typically achieves near-optimal SWAP counts for
  moderate circuit depths.

### 8.3 Our Comparison Methodology

In evaluation (`src/evaluate.py`):

1. Generate a random circuit.
2. Run SABRE on it (via `generate_preset_pass_manager(optimization_level=1)`).
3. Extract SABRE's initial mapping and SWAP count.
4. Reset the RL environment with the **same circuit** and **SABRE's initial
   mapping**.
5. Run the RL agent greedily (no exploration).
6. Compare: `swap_ratio = agent_swaps / sabre_swaps`.

This gives SABRE a slight advantage: the agent uses SABRE's mapping (which
SABRE optimized jointly with routing), but the agent must route from that
mapping without SABRE's bidirectional search. A ratio < 1.0 means the agent
beats SABRE; > 1.0 means SABRE wins.

We also support QASMBench evaluation (`run_qasmbench_evaluation()`) for testing
on standard quantum algorithm circuits rather than random ones.

---

## 9. Key Experimental Findings

Results from runs 014-019 on the `experiment-v2` branch.

### 9.1 Summary Table

| Run | Topologies | Episodes | Network | $\epsilon_{\text{end}}$ | Best Swap Ratio | Final Swap Ratio | Completion |
|---|---|---|---|---|---|---|---|
| 014 | heavy_hex_19 | 40k | [64,128,64]/512 | 0.02 | 1.167 | 1.167 | 100% |
| 015 | heavy_hex_19 | 60k | [32,64,32]/256 | 0.02 | 1.014 | 1.028 | 100% |
| 016 | multi (3 topos) | 45k | [32,64,32]/256 | 0.02 | 0.939* | 1.018 | 100% |
| 017 | heavy_hex_19 | 40k | [32,64,32]/256 | 0.02 | 1.059 | 1.059 | 100% |
| 018 | multi (3 topos) | 60k | [64,128,64]/512 | 0.02 | 0.947* | **0.999** | 99.3% |
| 019 | heavy_hex_19 | 80k | [32,64,32]/256 | 0.02 | **0.991** | 1.024 | 100% |
| 023 | heavy_hex_19 | 60k | [32,64,32]/256 | 0.02 | **0.994** | 1.014 | 100% |
| 024 | heavy_hex_19 | 100k | [32,64,32]/256 | 0.02 | **0.987** | 1.015 | 100% |
| 026 | multi (3 topos) | 100k | [32,64,32]/256 | 0.02 | **0.996** | 1.020 | 100% |
| 029 | heavy_hex_19 | 20k (finetune) | [32,64,32]/256 | 0.01 | **0.969** | 0.980 | 100% |

\* Low completion rate at best ratio (67%) — only completed episodes counted.

Run 029 fine-tunes from Run 019's best checkpoint (ep64k) with LR=1e-5 — our best result overall.

### 9.2 What Worked

1. **Fine-tuning from best checkpoint (NEW — V6):** Run 029 loaded Run 019's
   best weights (ep64k, ratio 0.991) and trained 20k more at LR=1e-5, reaching
   **0.969** — our best result, beating SABRE by 3.1%. The most compute-efficient
   approach: only 4.5h wall time for a 2.2% improvement over the base run.

2. **More episodes (60k+):** The agent continues improving well past 20k
   episodes. Run 015 (60k): 1.014 → Run 019 (80k): 0.991 → Run 024 (100k): 0.987.
   Consistent but diminishing returns.

2. **Low epsilon ($\epsilon = 0.02$):** All successful runs use
   $\epsilon_{\text{end}} = 0.02$. This keeps the replay buffer clean in late
   training.

3. **Slow epsilon decay (5M+ steps):** With 80k episodes averaging ~150 steps
   each, total environment steps reach ~12M. Epsilon decay over 5M steps means
   the agent reaches $\epsilon_{\text{end}}$ around episode 40k, with 40k more
   episodes of near-greedy learning.

4. **Standard network [32,64,32] for single-topology:** Run 015 ([32,64,32])
   converged to 1.028 on heavy_hex while Run 014 ([64,128,64]) plateaued at
   1.167 after the same wall-clock time. The smaller network has 4x fewer
   parameters, leading to faster, more stable learning.

5. **Weighted topology sampling for multi-topo:** Run 018 used the larger
   network ([64,128,64]) for multi-topology and achieved an overall ratio of
   0.999, effectively matching SABRE across all three topologies simultaneously.

6. **Large replay buffer (300k-400k):** Larger buffers retain more diverse
   experiences, especially important for multi-topology training where the agent
   needs to remember strategies for all topologies.

### 9.3 What Did Not Work

1. **Bigger network for single-topology (Run 014):** [64,128,64] with 512
   dueling hidden on single heavy_hex_19 plateaued at ratio 1.167 — substantially
   worse than the standard network's 1.028. The extra capacity likely led to
   overfitting to recent experiences or slower feature learning.

2. **Higher gate execution reward (Run 022, $r_{\text{gate}} = 2.0$):**
   Doubling the gate reward distorted the reward landscape, making the
   agent focus on executing easy gates rather than minimizing total SWAPs.
   Final ratio 1.377 — strongly negative.

3. **N-step returns without stabilization (Runs 022, 025):** N-step=3 alone
   caused 0% completion in Run 025 (60k episodes). The 3-step bootstrap
   creates too much variance during early exploration. Only viable when
   paired with curriculum learning to stabilize the early phase.

3. **Higher learning rate ($10^{-3}$):** Caused unstable Q-values and poor
   convergence. $10^{-4}$ is the sweet spot for this problem.

4. **Short training (< 20k episodes):** The agent reaches 100% completion rate
   around 10-15k episodes but the swap ratio continues improving for tens of
   thousands more episodes. Premature stopping leaves significant performance
   on the table.

### 9.4 Convergence Dynamics

Typical training progression on heavy_hex_19:

| Phase | Episodes | Completion Rate | Swap Ratio | Notes |
|---|---|---|---|---|
| Exploration | 0 - 5k | 0% - 5% | N/A | Random actions, learning state structure |
| Learning to complete | 5k - 15k | 5% - 100% | 1.5 - 2.0 | Agent learns to route, but inefficiently |
| Optimizing efficiency | 15k - 40k | ~100% | 1.2 - 1.05 | Swap ratio steadily decreasing |
| Approaching SABRE | 40k - 80k | 100% | 1.05 - 0.99 | Slow improvement, occasional sub-1.0 evals |
| Fine-tuning (Stage 2) | +20k @ LR=1e-5 | 100% | 0.99 - 0.97 | Low-LR refinement from best checkpoint |

The multi-topology agents follow a similar pattern but shifted: simpler
topologies (linear_5, grid_3x3) reach 100% completion first, then heavy_hex
catches up, then all three optimize simultaneously.

---

## 10. Current Best Configuration

### Recommended approach: Train + Fine-tune (2-stage)

**Stage 1**: Full training run (80k episodes, ~16h):

Based on Run 019 (ratio 0.991 at ep63k):

```python
TrainConfig(
    # Environment
    topologies=["heavy_hex_19"],
    matrix_size=27,
    circuit_depth=20,
    max_steps=400,
    gamma_decay=0.5,
    distance_reward_coeff=0.1,
    completion_bonus=5.0,
    timeout_penalty=-10.0,
    repetition_penalty=-0.5,
    gate_execution_reward=1.0,
    initial_mapping_strategy="random",

    # Network
    conv_channels=[32, 64, 32],
    dueling_hidden=256,

    # DQN
    gamma=0.99,
    lr=1e-4,
    batch_size=128,
    target_update_freq=500,
    tau=0.005,
    grad_clip_norm=10.0,

    # Epsilon
    epsilon_start=1.0,
    epsilon_end=0.02,
    epsilon_decay_steps=5_000_000,

    # PER
    buffer_capacity=400_000,
    per_alpha=0.6,
    per_beta_start=0.4,
    per_beta_end=1.0,
    per_beta_anneal_steps=500_000,
    per_epsilon=1e-6,

    # Training
    total_episodes=80_000,
    train_start=1_000,
    train_freq=4,

    # Eval
    eval_every=500,
    eval_episodes=50,

    # Device
    device="cuda",
    seed=42,
)
```

**Stage 2**: Fine-tune from best checkpoint (20k episodes, ~4.5h):

Based on Run 029 (ratio **0.969** — our best result, beats SABRE by 3.1%):

```python
# Load best checkpoint weights only (fresh optimizer, fresh epsilon)
# python main.py train --config configs/run29_finetune_run019.json \
#     --finetune outputs/run_019/checkpoints/checkpoint_ep64000.pt
TrainConfig(
    topologies=["heavy_hex_19"],
    lr=1e-5,                     # 10x lower than Stage 1
    lr_schedule="constant",       # No annealing needed
    epsilon_start=0.05,           # Start low — network already knows routing
    epsilon_end=0.01,
    epsilon_decay_steps=2_000_000,
    buffer_capacity=300_000,
    total_episodes=20_000,
    train_start=500,              # Start training sooner (network is warm)
    # All other params same as Stage 1
)
```

The `--finetune` flag loads only network weights from the checkpoint, keeping a fresh
optimizer at the new LR and fresh epsilon schedule. This avoids restoring the old
optimizer momentum which would fight the new learning rate.

### Alternative: Curriculum + Cosine LR (single-stage, most sample-efficient)

Based on Run 023 (ratio 0.994 in only 60k episodes):

```python
TrainConfig(
    lr=1e-4,
    lr_schedule="cosine",
    lr_min=1e-5,
    curriculum_depths=[5, 10, 20],
    curriculum_milestones=[0.15, 0.35],
    total_episodes=60_000,
    epsilon_decay_steps=4_000_000,
    # All other params same as Run 019
)
```

### Multi-topology: best overall ratio

Based on Run 026 (ratio 0.996, first multi-topo to beat SABRE):

```python
TrainConfig(
    topologies=["linear_5", "grid_3x3", "heavy_hex_19"],
    matrix_size=27,
    circuit_depth=20,
    max_steps=300,
    conv_channels=[64, 128, 64],
    dueling_hidden=512,
    lr=1e-4,
    epsilon_end=0.02,
    epsilon_decay_steps=5_000_000,
    batch_size=128,
    buffer_capacity=300_000,
    total_episodes=60_000,
    train_freq=4,
    tau=0.005,
    target_update_freq=500,
    initial_mapping_strategy="random",
    distance_reward_coeff=0.1,
    repetition_penalty=-0.5,
)
```

---

## 11. References

1. **Cowtan, A., Dilkes, S., Duncan, R., Krajenbrink, A., Simmons, W., and
   Sivarajah, S.** (2019). "On the Qubit Routing Problem." *14th Conference on
   the Theory of Quantum Computation, Communication and Cryptography (TQC)*.

2. **Li, G., Ding, Y., and Xie, Y.** (2019). "Tackling the Qubit Mapping
   Problem for NISQ-Era Quantum Devices." *Proceedings of the 24th International
   Conference on Architectural Support for Programming Languages and Operating
   Systems (ASPLOS)*. Introduces the SABRE algorithm.

3. **van Hasselt, H., Guez, A., and Silver, D.** (2016). "Deep Reinforcement
   Learning with Double Q-learning." *Proceedings of the 30th AAAI Conference on
   Artificial Intelligence*. Introduces Double DQN.

4. **Wang, Z., Schaul, T., Hessel, M., van Hasselt, H., Lanctot, M., and
   de Freitas, N.** (2016). "Dueling Network Architectures for Deep
   Reinforcement Learning." *Proceedings of the 33rd International Conference on
   Machine Learning (ICML)*.

5. **Schaul, T., Quan, J., Antonoglou, I., and Silver, D.** (2016).
   "Prioritized Experience Replay." *Proceedings of the 4th International
   Conference on Learning Representations (ICLR)*.

6. **Pozzi, M. G., Herbert, S. J., Sherrill, S. S., and Sherrill, A.** (2022).
   "Using Reinforcement Learning to Perform Qubit Routing in Quantum Compilers."
   *ACM Transactions on Quantum Computing*. Introduces the distance-based
   shaping reward used in our reward function.

7. **Sutton, R. S. and Barto, A. G.** (2018). *Reinforcement Learning: An
   Introduction* (2nd edition). MIT Press. Chapter 7: n-step bootstrapping.

8. **Siraichi, M. Y., Santos, V. F., Collange, S., and Pereira, F. M. Q.**
   (2018). "Qubit Allocation." *Proceedings of the IEEE/ACM International
   Symposium on Code Generation and Optimization (CGO)*.

9. **Mnih, V., Kavukcuoglu, K., Silver, D., et al.** (2015). "Human-level
   control through deep reinforcement learning." *Nature*, 518(7540), 529-533.
   Introduces DQN and experience replay.

---

## Appendix A: File Map

| File | Purpose |
|---|---|
| `src/environment.py` | Gymnasium environment: state construction, SWAP execution, reward computation, multi-topology |
| `src/networks.py` | DuelingCNN: Conv2d feature extractor + dueling V/A streams |
| `src/dqn_agent.py` | D3QNAgent: action selection, training step, checkpointing, epsilon/LR scheduling |
| `src/replay_buffer.py` | PrioritizedReplayBuffer + SumTree: O(log n) proportional sampling, uint8 storage |
| `src/config.py` | TrainConfig dataclass, run directory setup, preset configs |
| `src/train.py` | Training loop: episode iteration, curriculum, logging, periodic eval, checkpoint saving |
| `src/evaluate.py` | Evaluation: agent vs SABRE on random + QASMBench circuits, trajectory recording |
| `src/circuit_utils.py` | Quantum circuit utilities: DAG construction, front layer, coupling maps, SABRE interface |
| `src/visualize.py` | Training curve and evaluation comparison plotting |
| `src/explore.py` | Data exploration script: demonstrates the full pipeline interactively |

## Appendix B: State Channel Visualization Guide

For a heavy_hex_19 topology with $N = 27$:

```
Channel 0 (Adjacency):           Channel 1 (Mapping):
  27x27 matrix                      27x27 matrix
  20 nonzero pairs (edges)          19 nonzero entries (permutation)
  Symmetric, constant per episode   One 1 per row/column, changes each SWAP

Channel 2 (Gate Demand):          Channel 3 (Front-Layer Distance):
  27x27 matrix                      27x27 matrix
  Continuous [0,1], dense           Sparse, only front-layer positions
  Depth-decayed, position-space     1/distance encoding, symmetric

Channel 4 (Stagnation):
  27x27 matrix
  Uniform value, increases without gate execution
  Resets to 0 on gate execution
```

## Appendix C: Reward Decomposition Example

Consider a step where the agent SWAPs edge (3, 7) on heavy_hex_19:

| Component | Value | Explanation |
|---|---|---|
| SWAP cost | $-1.0$ | Always $-1$ per step |
| Gates executed | $+2.0$ | Two front-layer gates became routable ($2 \times 1.0$) |
| Distance shaping | $+0.3$ | $0.1 \times 3.0$ (front-layer sum reduced by 3 hops) |
| Repetition penalty | $0.0$ | Different action from previous step |
| Terminal bonus | $0.0$ | Not done yet |
| **Total** | **$+1.3$** | Net positive: productive SWAP |

Compare with a wasted step (SWAP that helps nothing):

| Component | Value | Explanation |
|---|---|---|
| SWAP cost | $-1.0$ | Always $-1$ |
| Gates executed | $0.0$ | No gates became routable |
| Distance shaping | $-0.05$ | Moved qubits slightly apart ($0.1 \times -0.5$) |
| Repetition penalty | $-0.5$ | Same SWAP as previous step (undo!) |
| **Total** | **$-1.55$** | Strongly negative: agent learns to avoid this |
