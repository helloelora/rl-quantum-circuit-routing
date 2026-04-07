# D3QN+PER Architecture for Quantum Circuit Routing

A complete technical reference for the Double Dueling Deep Q-Network with Prioritized Experience Replay (D3QN+PER) system that learns to route quantum circuits on hardware topologies, competing against IBM's SABRE heuristic.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [System Overview](#2-system-overview)
3. [Environment (`environment.py`)](#3-environment)
4. [Circuit Utilities (`circuit_utils.py`)](#4-circuit-utilities)
5. [Neural Network (`networks.py`)](#5-neural-network)
6. [Replay Buffer (`replay_buffer.py`)](#6-replay-buffer)
7. [Agent (`dqn_agent.py`)](#7-agent)
8. [Training Loop (`train.py`)](#8-training-loop)
9. [Evaluation (`evaluate.py`)](#9-evaluation)
10. [Visualization (`visualize.py`)](#10-visualization)
11. [Configuration (`config.py`)](#11-configuration)
12. [CLI Entry Point (`main.py`)](#12-cli-entry-point)
13. [Data Flow: End-to-End](#13-data-flow-end-to-end)
14. [Training Phases](#14-training-phases)
15. [Key Design Decisions](#15-key-design-decisions)
16. [Output Directory Structure](#16-output-directory-structure)

---

## 1. Problem Statement

**Quantum circuit routing** (also called qubit mapping/routing) is the problem of making a logical quantum circuit executable on physical hardware where only certain qubit pairs are connected.

Given:
- A **logical quantum circuit** containing two-qubit gates (e.g., CNOT) between arbitrary qubit pairs
- A **hardware topology** (coupling graph) defining which physical qubit pairs can directly interact

The task: insert **SWAP gates** to move logical qubits around the hardware until every two-qubit gate operates on adjacent physical qubits. Fewer SWAPs = better, since each SWAP adds noise and gate overhead (1 SWAP = 3 CNOTs).

**Baseline**: IBM's SABRE algorithm (Stochastic Architecture-agnostic Balancing of Estimated Routing), a hand-crafted heuristic from Qiskit that uses lookahead scoring. Our RL agent aims to match or beat SABRE.

---

## 2. System Overview

```
┌──────────────────────────────────────────────────────────┐
│                    main.py (CLI)                         │
│          train / evaluate / visualize                     │
└───────────┬──────────────┬──────────────┬────────────────┘
            │              │              │
    ┌───────▼──────┐  ┌───▼────────┐  ┌──▼──────────────┐
    │  train.py    │  │ evaluate.py│  │  visualize.py    │
    │  TrainLoop   │  │ Agent+SABRE│  │  Curves/GIFs     │
    └──┬───────┬───┘  └────┬───────┘  └─────────────────┘
       │       │           │
  ┌────▼───┐ ┌─▼──────────┐
  │ agent  │ │ environment │
  │ D3QN   │ │ Gymnasium   │
  └──┬──┬──┘ └──────┬──────┘
     │  │           │
┌────▼┐ ▼───────┐ ┌─▼───────────────┐
│net- │ │replay_ │ │ circuit_utils.py│
│works│ │buffer  │ │ Qiskit/DAG/SABRE│
└─────┘ └───────┘ └─────────────────┘
```

### File Map

| File | Lines | Purpose |
|------|-------|---------|
| `src/environment.py` | ~504 | Gymnasium environment: state, actions, rewards, multi-topology |
| `src/circuit_utils.py` | ~250 | Circuit loading, gate extraction, DAG, coupling maps, SABRE |
| `src/networks.py` | ~67 | Dueling CNN architecture (Q-value prediction) |
| `src/replay_buffer.py` | ~164 | SumTree + Prioritized Experience Replay buffer |
| `src/dqn_agent.py` | ~261 | D3QN agent: action selection, training, checkpoints |
| `src/train.py` | ~276 | Training loop with logging, periodic eval, Ctrl+C handling |
| `src/evaluate.py` | ~307 | Agent vs SABRE evaluation on random + QASMBench circuits |
| `src/visualize.py` | ~726 | Training dashboard, bar charts, step-by-step routing GIFs |
| `src/config.py` | ~191 | TrainConfig dataclass, presets, run directory setup |
| `main.py` | ~239 | CLI entry point with train/evaluate/visualize subcommands |

---

## 3. Environment

**File**: `src/environment.py` — `QubitRoutingEnv(gym.Env)`

### State Space: `(3, N, N)` float32

The observation is a 3-channel N×N matrix (N=27 by default) that encodes everything the agent needs:

| Channel | Content | Type | What it encodes |
|---------|---------|------|-----------------|
| 0 | Adjacency matrix | Binary (0/1) | Hardware topology — which physical qubits are connected. Constant within an episode. |
| 1 | Mapping permutation matrix | Binary (0/1) | Current qubit assignment — `state[1, q, p] = 1` means logical qubit `q` is at physical position `p`. Exactly one 1 per row and column. |
| 2 | Depth-decayed gate demand | Continuous [0,1] | Which gates need to be executed and how urgent they are. `state[2, pa, pb] += γ^depth` for each remaining gate between qubits currently at positions pa, pb. Front-layer gates (depth=0) contribute 1.0, deeper gates contribute exponentially less. |

**Why 3 channels?** This mirrors how image CNNs process RGB — channel 0 is the "map", channel 1 is "where things are", channel 2 is "what needs to happen". The CNN can learn spatial patterns like "these qubits need to move toward each other".

**Padding**: All topologies are padded to N×N. A linear_5 topology uses only rows/columns 0–4; the rest are zeros. This lets the same CNN process any topology up to N qubits.

### Action Space: `Discrete(max_edges)`

Each action corresponds to performing a SWAP on one hardware edge. For heavy_hex_19, there are 20 edges, so actions are integers 0–19.

**Multi-topology masking**: When training on multiple topologies (e.g., linear_5 has 4 edges, heavy_hex_19 has 20), the action space is always `Discrete(20)` (the max). For a linear_5 episode, actions 4–19 are masked as invalid via `get_action_mask()`.

### Reward Function

```
r_t = (gates_auto_executed - 1) + 0.01 * delta_distance + bonus/penalty
```

| Component | Value | Purpose |
|-----------|-------|---------|
| `gates_auto_executed - 1` | Usually -1, sometimes 0 or positive | Each SWAP costs -1. If the SWAP makes gates routable (they auto-execute), you earn +1 per gate, offsetting the -1 cost. |
| `0.01 * delta_distance` | Small positive/negative | Distance shaping: reward for moving front-layer qubits closer together (sum of shortest-path distances before vs after SWAP). |
| Completion bonus | +5.0 | When all gates are routed. |
| Timeout penalty | -10.0 | When `max_steps` (500) is reached without completing. |

### Auto-Execute Mechanism

After every SWAP, the environment automatically checks which front-layer gates have their qubits now adjacent and executes them. This repeats until no more gates can execute (cascading execution). The agent doesn't choose to execute gates — it only chooses SWAPs. This is because executing a routable gate is always optimal; there's no reason to delay it.

### Episode Flow

```
1. reset():
   a. Pick a topology (random if multi-topology)
   b. Generate a random circuit OR accept a provided one
   c. Extract two-qubit gates and build dependency DAG
   d. Set initial mapping (random/identity/SABRE)
   e. Auto-execute any initially routable gates
   f. Return initial observation + info

2. step(action):
   a. Validate action is within current topology's edges
   b. Record distance before SWAP
   c. Perform SWAP: update mapping arrays
   d. Auto-execute routable gates (may cascade)
   e. Compute reward
   f. Check done (all gates executed) or truncated (max_steps)
   g. Return (observation, reward, terminated, truncated, info)
```

---

## 4. Circuit Utilities

**File**: `src/circuit_utils.py`

### Gate Extraction

`extract_two_qubit_gates(circuit)` converts a Qiskit circuit into a flat list of `(logical_qubit_a, logical_qubit_b)` tuples in topological (dependency) order. Single-qubit gates are ignored since they don't require routing.

### Dependency DAG

`build_dependency_graph(gates)` creates a directed acyclic graph (DAG):
- **Predecessor**: gate A is a predecessor of gate B if they share a qubit and A comes first
- **Successor**: inverse of predecessor
- Only tracks *direct* (nearest) dependencies per qubit, not transitive ones

Example: gates `[(0,1), (1,2), (0,3)]`
- Gate 1 depends on gate 0 (shared qubit 1)
- Gate 2 depends on gate 0 (shared qubit 0)
- Gates 1 and 2 are independent of each other

### Front Layer

`compute_front_layer(gates, executed, predecessors)` returns gates that are ready to execute — all their predecessors are already done. This is the set of gates the agent should be working toward making routable.

### DAG Depths

`compute_dag_depths(...)` assigns a depth level to each remaining gate via BFS from the front layer. Front-layer gates have depth 0, their dependents have depth 1, etc. Used to weight the gate demand channel (channel 2) — closer gates matter more.

### Coupling Maps

`get_coupling_map(topology_name)` maps names to Qiskit CouplingMap objects:

| Name | Qubits | Edges | Description |
|------|--------|-------|-------------|
| `linear_5` | 5 | 4 | Simple line: 0-1-2-3-4 |
| `grid_3x3` | 9 | 12 | 3×3 grid |
| `heavy_hex_19` | 19 | 20 | IBM heavy-hex lattice (3 heavy-hex cells) |
| `ring_N` | N | N | Ring topology |
| `grid_RxC` | R×C | varies | Arbitrary grid |

### SABRE Interface

- `get_sabre_initial_mapping(circuit, coupling_map)`: Runs Qiskit's SABRE layout pass to get an initial qubit placement
- `get_sabre_swap_count(circuit, coupling_map)`: Full SABRE transpilation, returns number of SWAPs inserted

---

## 5. Neural Network

**File**: `src/networks.py` — `DuelingCNN`

### Architecture

```
Input: (batch, 3, 27, 27)   ← 3-channel state
                │
    ┌───────────▼────────────┐
    │  Conv2d(3→32, 3×3, p=1)│ ← same-padding preserves 27×27
    │  ReLU                   │
    │  Conv2d(32→64, 3×3, p=1)│
    │  ReLU                   │
    │  Conv2d(64→32, 3×3, p=1)│
    │  ReLU                   │
    └───────────┬────────────┘
                │
        Flatten: 32 × 27 × 27 = 23,328
                │
        ┌───────┴───────┐
        │               │
   ┌────▼────┐    ┌─────▼─────┐
   │  Value  │    │ Advantage │
   │ Stream  │    │  Stream   │
   │ 23328→  │    │ 23328→    │
   │  256→1  │    │  256→20   │
   └────┬────┘    └─────┬─────┘
        │               │
        └───────┬───────┘
                │
        Q = V + A - mean(A)
                │
    Output: (batch, 20)  ← Q-value per action
```

**Total parameters**: ~12 million

### Why Dueling?

The **dueling architecture** (Wang et al., 2016) separates Q(s,a) into:
- **V(s)**: How good is this state? (independent of which action)
- **A(s,a)**: How much better is action a compared to the average?

This helps the network generalize: it can learn that some states are bad regardless of action (V is low) without needing to evaluate every action separately. The mean-subtraction `Q = V + A - mean(A)` ensures identifiability.

### Why CNN?

The state is a 27×27 grid, much like a small image. CNNs naturally capture:
- **Local adjacency patterns** (which qubits are neighbors — from channel 0)
- **Spatial relationships** between current positions and gate demands
- **Translation-like equivariance** across the grid

---

## 6. Replay Buffer

**File**: `src/replay_buffer.py`

### SumTree

An array-based binary tree that supports O(log n) operations for proportional sampling:

```
           [root: sum of all]
          /                  \
    [sum left]          [sum right]
    /        \          /          \
  [p1]     [p2]     [p3]        [p4]     ← leaf = priority of transition
```

- **Tree array**: `tree[1]` = root (total sum), leaves at indices `[capacity, 2*capacity)`
- **add(priority)**: Set leaf, propagate sum up to root — O(log n)
- **sample(value)**: Walk down from root. At each node, go left if `value <= tree[left]`, else go right with `value -= tree[left]` — O(log n)
- **update(idx, priority)**: Change a leaf and fix the sums — O(log n)

### PrioritizedReplayBuffer

Wraps SumTree with transition storage:

**Storage** (pre-allocated numpy arrays):
- `states[capacity, 3, 27, 27]` — **uint8** (not float32!)
- `next_states[capacity, 3, 27, 27]` — **uint8**
- `actions[capacity]` — int64
- `rewards[capacity]` — float32
- `dones[capacity]` — bool
- `next_action_masks[capacity, max_edges]` — bool

**Memory**: With float32 states, 100K transitions would need ~1.75 GB. With uint8, it's ~440 MB (4x savings). States are stored as `(value * 255).astype(uint8)` and recovered as `states.astype(float32) / 255.0`. The precision loss (1/255 ≈ 0.004) is negligible since our channel 2 values differ by at least 0.25.

**Priority management**:
- New transitions are added with `max_priority^alpha` (optimistic: assume they're important until proven otherwise)
- After training, priorities are updated to `(|TD_error| + epsilon)^alpha`
- **alpha** (0.6): Controls how much prioritization. 0 = uniform sampling, 1 = fully prioritized.
- **beta** (0.4 → 1.0): Importance-sampling correction. Starts low (biased but low variance), anneals to 1.0 (fully corrected) over training.

**Stratified sampling**: The priority range is divided into `batch_size` equal segments. One sample is drawn from each segment. This reduces variance compared to pure proportional sampling.

**Why store `next_action_mask`?**: In multi-topology training, a transition from linear_5 has only 4 valid actions. When computing Double DQN targets, we need to know which actions were valid in the *next state* to correctly mask the argmax. Without storing this, we'd incorrectly consider all 20 actions.

---

## 7. Agent

**File**: `src/dqn_agent.py` — `D3QNAgent`

### Components

```
D3QNAgent
├── online_net    (DuelingCNN — trained via backprop)
├── target_net    (DuelingCNN — periodically copied from online)
├── optimizer     (Adam, lr=1e-4)
├── buffer        (PrioritizedReplayBuffer)
├── epsilon       (exploration rate, linearly decayed)
└── device        (auto-detected: cuda > mps > cpu)
```

### Action Selection: `select_action(state, action_mask, deterministic)`

```python
if not deterministic and random() < epsilon:
    # Random action among valid ones
    return random_choice(where(action_mask))
else:
    # Greedy: pick highest Q among valid actions
    q = online_net(state)   # (num_actions,)
    q[~action_mask] = -inf  # mask invalid
    return argmax(q)
```

**Key**: Invalid actions always get Q = -inf, so they are never selected, even greedily. This is critical for multi-topology correctness.

### Training Step: `train_step()`

This is the core learning algorithm. Called every environment step after the buffer has enough samples.

```
1. Sample batch from PER (with importance-sampling weights)

2. Compute current Q-values:
   q_current = online_net(states).gather(actions)
   → Q(s, a) for the actions actually taken

3. Compute Double DQN targets:
   a. Online net selects best next action (MASKED):
      q_online_next = online_net(next_states)
      q_online_next[~next_masks] = -inf
      best_actions = argmax(q_online_next)

   b. Target net evaluates that action:
      q_target_next = target_net(next_states)
      q_next = q_target_next.gather(best_actions)

   c. Bellman target:
      targets = rewards + gamma * q_next * (1 - dones)

4. Compute loss:
   td_errors = q_current - targets
   element_loss = HuberLoss(q_current, targets)  # smooth_l1
   loss = mean(IS_weights * element_loss)

5. Backprop:
   loss.backward()
   clip_grad_norm(parameters, max_norm=10.0)
   optimizer.step()

6. Update priorities in buffer:
   new_priority = (|td_error| + epsilon)^alpha
```

### Why Double DQN?

Standard DQN uses `max(Q_target(s'))` which overestimates Q-values because the same network both selects and evaluates the best action. **Double DQN** (van Hasselt et al., 2016) decouples this:
- **Online net** picks which action is best: `a* = argmax Q_online(s')`
- **Target net** evaluates it: `Q_target(s', a*)`

This reduces overestimation bias significantly.

### Why Huber Loss?

PER can produce large TD errors (high-priority transitions). MSE loss would amplify these, causing unstable gradients. **Huber loss** (smooth L1) is linear for large errors instead of quadratic, making training more robust.

### Why Importance-Sampling Weights?

PER samples transitions non-uniformly (high TD-error transitions appear more often). This biases the gradient. IS weights correct for this:
```
w_i = (N * P(i))^(-beta)  / max(w)
```
where `P(i)` is the sampling probability. Early in training, beta is low (allowing bias for faster learning). It anneals to 1.0 (fully corrected) for convergence guarantees.

### Target Network Updates

Every 1,000 gradient steps, the target network is synchronized:
- **Hard copy** (tau=1.0): `target_net = copy(online_net)` — simple, works well
- **Soft Polyak** (tau<1.0): `target = tau*online + (1-tau)*target` — smoother but slower

We use hard copy (tau=1.0) by default.

### Epsilon Schedule

Linear decay from 1.0 to 0.05 over 50,000 steps:
```
epsilon = max(0.05, 1.0 - (1.0 - 0.05) * step / 50000)
```

- **Early training**: epsilon ≈ 1.0 → almost all random actions → explore the state space
- **Late training**: epsilon ≈ 0.05 → mostly greedy → exploit learned Q-values, with 5% random to keep exploring

### Checkpointing

**`.pt` file** contains:
- online_net and target_net state_dicts
- optimizer state_dict
- epsilon, epsilon_step, train_steps
- episode number

**`.buf.npz` file** (optional, via `--save-buffer`):
- All buffer arrays (states, actions, rewards, next_states, dones, masks)
- SumTree data (priorities)
- Buffer pointer and size

The buffer file is large (~400MB for 100K transitions) so it's optional. The model checkpoint is ~10MB.

---

## 8. Training Loop

**File**: `src/train.py`

### Episode Loop (Pseudocode)

```
for episode in range(total_episodes):
    obs, info = env.reset()

    if info["done"]:  # No two-qubit gates in this circuit
        log and skip

    while True:
        action = agent.select_action(obs, mask)          # epsilon-greedy
        next_obs, reward, terminated, truncated, info = env.step(action)

        agent.store_transition(obs, action, reward, next_obs, terminated, next_mask)
        # Note: stores `terminated` not `truncated` as done flag
        # so truncated episodes still bootstrap Q(s')

        if global_step % train_freq == 0:
            metrics = agent.train_step()                  # sample + backprop
            if train_steps % target_update_freq == 0:
                agent.update_target_network()             # copy online → target

        agent.update_epsilon()                            # linear decay

        if terminated or truncated:
            break

    # Periodic: log, evaluate, checkpoint
```

### Why `terminated` not `truncated` as done flag?

When an episode times out (truncated=True), the environment isn't actually in a terminal state — there are still gates to route. If we stored `done=True`, the Q-target would be just `reward` (no bootstrapping). But the true value of that state is `reward + gamma * Q(s')`. Storing `done=False` for truncated episodes preserves the bootstrap, preventing the agent from learning that "timeout states have zero future value".

### Logging

Three JSONL files are written continuously:

1. **`episodes.jsonl`**: One line per episode
   ```json
   {"episode": 100, "reward": -3.2, "steps": 15, "swaps": 15, "gates": 8, "completed": true, "topology": "heavy_hex_19", "epsilon": 0.812}
   ```

2. **`train_steps.jsonl`**: Every 100 gradient steps
   ```json
   {"step": 500, "loss": 0.234, "mean_q": -1.45, "mean_td_error": 0.89, "epsilon": 0.76, "beta": 0.42}
   ```

3. **`evaluations.jsonl`**: At each periodic evaluation
   ```json
   {"episode": 500, "mean_agent_swaps": 12.3, "mean_sabre_swaps": 10.1, "mean_swap_ratio": 1.22, "completion_rate": 0.95}
   ```

### Periodic Evaluation

Every `eval_every` episodes (default 500):
1. Run `run_evaluation()` with `eval_episodes` (20) random circuits per topology
2. Log summary to evaluations.jsonl
3. Save full eval results to `eval/eval_epN.json`
4. Update `figures/training_curves.png` (overwrites)
5. Save `figures/eval_comparison_epN.png`

### Signal Handling

Ctrl+C triggers a graceful shutdown:
1. First Ctrl+C: sets `interrupted=True`, saves emergency checkpoint
2. Second Ctrl+C: forces exit

### Final Actions

When training ends (naturally or interrupted):
1. Save final/emergency checkpoint
2. Generate final training curves figure
3. Close logger files

---

## 9. Evaluation

**File**: `src/evaluate.py`

### Fair Comparison Protocol

For each evaluation circuit, both the agent and SABRE route the **exact same circuit** with the **exact same initial mapping**:

```
1. Generate random circuit
2. Run SABRE → get (initial_mapping, swap_count)
3. Reset env with same circuit + SABRE's initial_mapping
4. Run agent greedily (epsilon=0) → get agent's swap_count
5. Compare: agent_swaps / sabre_swaps
```

**Why use SABRE's initial mapping for the agent?** This is the fairest comparison. SABRE chooses an optimized initial placement; giving the agent a random one would be unfair. By using the same starting point, we isolate the *routing* quality from the *initial placement* quality.

### Trajectory Recording

When `log_trajectories=True`, each step records:
- Action taken (which edge was SWAPped)
- Reward received
- Current mapping after the SWAP
- Set of executed gates after auto-execution

Plus initial state: topology edges, all gates, dependency graph, initial mapping, initially auto-executed gates. This is everything needed to reconstruct and animate the routing process.

### QASMBench Evaluation

`run_qasmbench_evaluation()` loads real quantum circuits from `.qasm` files (QASMBench benchmark suite), filters by qubit count, and runs the same agent-vs-SABRE comparison. This tests generalization to real-world circuits rather than random ones.

---

## 10. Visualization

**File**: `src/visualize.py`

### Training Dashboard (2×3 grid)

`plot_training_curves()` generates a 6-panel overview:

| Panel | Content | What to look for |
|-------|---------|-----------------|
| (0,0) Reward | Per-episode + smoothed | Should trend upward over training |
| (0,1) SWAP count | Agent SWAPs + SABRE eval line | Agent SWAPs should decrease toward SABRE |
| (0,2) Completion | % of episodes completed | Should approach 100% |
| (1,0) Loss | Huber training loss | Should decrease then stabilize |
| (1,1) Epsilon | Exploration rate decay | Linear decay from 1.0 to 0.05 |
| (1,2) Q-value + TD error | Mean Q and |TD error| (dual axis) | Q should stabilize, TD error should decrease |

All lines have labeled legends. SABRE baseline appears as red dashed markers from periodic evaluations.

### Eval Comparison Bar Chart

`plot_eval_comparison()`: Grouped bar chart showing agent (blue) vs SABRE (coral) SWAP counts for each evaluated circuit. Quick visual check of which circuits the agent wins/loses on.

### Swap Ratio Distribution

`plot_swap_ratio_distribution()`: Histogram of `agent_swaps / sabre_swaps` across all eval circuits. Red line at 1.0 (SABRE parity), green line at mean ratio. Values <1.0 mean the agent beat SABRE.

### Step-by-Step Routing GIF

`create_routing_gif()`: Animated GIF showing every SWAP decision:

**Left panel** — Hardware graph:
- Nodes colored by which logical qubit occupies each position (HSV palette)
- Node labels: `q0`, `q1`, ... (logical qubit at that physical position)
- **Red edge**: SWAP being performed this step
- **Green edge**: Gate(s) that just executed (CNOT became routable)
- **Blue dashed**: Front-layer demands (gates that need these qubits to become adjacent)
- **Gray**: Background hardware edges
- Legend explains all colors

**Right panel** — Text info:
- **ACTION**: What SWAP was performed and which logical qubits swapped, plus which gates executed
- Reward this step and cumulative reward
- SABRE's total SWAP count for reference
- **Gate checklist**: Every gate in the circuit with status:
  - `✓` done (executed)
  - `▸` READY (in front layer, dependencies met)
  - ` ` waiting (dependencies not yet met)
  - Each gate shows logical qubits and their current physical positions

**Speed**: 1 fps (1 second per frame) so you can read the action descriptions.

### Side-by-Side Comparison GIF

`create_side_by_side_gif()`: Three-panel layout:
- **Left**: Same graph visualization as above
- **Center**: Horizontal bar chart — Agent SWAPs so far vs SABRE total SWAPs (with numbers on bars)
- **Right**: Gate checklist

---

## 11. Configuration

**File**: `src/config.py`

### TrainConfig Dataclass

All hyperparameters in one place, serializable to/from JSON.

#### Environment Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `topologies` | `["heavy_hex_19"]` | List of topology names to train on |
| `matrix_size` | 27 | Observation matrix side length (must fit largest topology) |
| `circuit_depth` | 20 | Depth of random training circuits |
| `max_steps` | 500 | Maximum SWAPs per episode |
| `gamma_decay` | 0.5 | Decay for gate demand channel |
| `distance_reward_coeff` | 0.01 | Weight of distance shaping reward |
| `completion_bonus` | 5.0 | Reward when all gates routed |
| `timeout_penalty` | -10.0 | Penalty when max_steps reached |
| `initial_mapping_strategy` | `"random"` | How to initialize qubit positions |

#### Network Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `conv_channels` | `[32, 64, 32]` | Channels in each conv layer |
| `dueling_hidden` | 256 | Hidden units in V/A streams |

#### DQN Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gamma` | 0.99 | Discount factor |
| `lr` | 1e-4 | Learning rate (Adam) |
| `batch_size` | 64 | Training batch size |
| `target_update_freq` | 1000 | Steps between target net sync |
| `tau` | 1.0 | Polyak averaging (1.0 = hard copy) |
| `grad_clip_norm` | 10.0 | Max gradient norm |

#### PER Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `buffer_capacity` | 100000 | Max transitions in buffer |
| `per_alpha` | 0.6 | Priority exponent (0=uniform, 1=full priority) |
| `per_beta_start` | 0.4 | IS correction start (low = biased, fast) |
| `per_beta_end` | 1.0 | IS correction end (high = unbiased) |
| `per_beta_anneal_steps` | 100000 | Steps to anneal beta |
| `per_epsilon` | 1e-6 | Small constant added to priorities |

#### Epsilon Schedule

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epsilon_start` | 1.0 | Initial exploration rate |
| `epsilon_end` | 0.05 | Minimum exploration rate |
| `epsilon_decay_steps` | 50000 | Steps for linear decay |

### Presets

| Preset | Topology | Episodes | Purpose |
|--------|----------|----------|---------|
| `linear5_sanity_config()` | linear_5 | 2,000 | Quick test: 4 edges, simple circuits |
| `heavy_hex_config()` | heavy_hex_19 | 20,000 | Primary target: real IBM topology |
| `multi_topology_config()` | linear_5 + grid_3x3 + heavy_hex_19 | 30,000 | Generalization across topologies |

---

## 12. CLI Entry Point

**File**: `main.py`

```bash
# Train with preset
python main.py train --preset heavy_hex
python main.py train --preset linear5 --episodes 1000 --device mps

# Train with custom config
python main.py train --config path/to/config.json

# Resume training
python main.py train --preset heavy_hex --resume outputs/run_001/checkpoints/checkpoint_ep5000.pt

# Evaluate
python main.py evaluate --checkpoint outputs/run_001/checkpoints/checkpoint_final.pt --episodes 50

# Evaluate with QASMBench
python main.py evaluate --checkpoint outputs/run_001/checkpoints/checkpoint_final.pt --qasmbench path/to/qasmbench/

# Visualize
python main.py visualize --run-dir outputs/run_001
python main.py visualize --run-dir outputs/run_001 --gif --fps 1
```

---

## 13. Data Flow: End-to-End

### A Single Training Step

```
┌─────────────┐    state (3,27,27)     ┌──────────┐
│ Environment │ ──────────────────────► │  Agent   │
│             │ ◄────── action (int) ── │          │
│ - topology  │                         │ - CNN    │
│ - circuit   │   reward, next_state    │ - ε-grdy │
│ - mapping   │ ──────────────────────► │          │
│ - gate DAG  │                         │          │
└─────────────┘                         └────┬─────┘
                                             │
                                    store transition
                                             │
                                    ┌────────▼────────┐
                                    │  Replay Buffer  │
                                    │  (PER+SumTree)  │
                                    │                 │
                                    │  sample batch   │
                                    └────────┬────────┘
                                             │
                                    ┌────────▼────────┐
                                    │  Train Step     │
                                    │                 │
                                    │  Q_current =    │
                                    │    online(s)[a] │
                                    │                 │
                                    │  a* = argmax    │
                                    │    online(s')   │ ← masked
                                    │  Q_target =     │
                                    │    target(s')[a*]│
                                    │                 │
                                    │  loss = Huber   │
                                    │    × IS weights │
                                    │                 │
                                    │  backprop +     │
                                    │  update PER     │
                                    └─────────────────┘
```

### A Single Evaluation Episode

```
Generate circuit ──► SABRE transpile ──► get initial_mapping + swap_count
                         │
                         ▼
        env.reset(circuit=same, mapping=SABRE's)
                         │
                    ┌────▼────┐
                    │  Agent  │ (deterministic, no ε)
                    │  route  │
                    │  loop   │
                    └────┬────┘
                         │
                    agent_swaps
                         │
            Compare: agent_swaps / sabre_swaps
```

---

## 14. Training Phases

Training proceeds in three phases with increasing difficulty:

### Phase 1: Sanity Check (linear_5)

```bash
python main.py train --preset linear5
```

- **Topology**: linear_5 (5 qubits, 4 edges)
- **Episodes**: 2,000
- **Circuit depth**: 5
- **Expected time**: ~2 minutes on MPS
- **Success criteria**: 100% completion, SWAP count decreasing

This validates the entire pipeline works — the environment, agent, training loop, logging, evaluation, and visualization all function correctly. With only 4 possible actions, the agent should learn quickly.

### Phase 2: Scale Up (heavy_hex_19)

```bash
python main.py train --preset heavy_hex
```

- **Topology**: heavy_hex_19 (19 qubits, 20 edges)
- **Episodes**: 20,000
- **Circuit depth**: 20
- **Expected time**: ~1–2 hours on MPS, ~30 min on A100
- **Success criteria**: Completion rate >90%, SWAP ratio approaching 1.0 (SABRE parity)

This is the primary training target — a realistic IBM hardware topology with depth-20 circuits. The agent needs to learn which SWAPs move qubits toward where they need to be, considering both the current front layer and future gates (via the depth-weighted demand channel).

### Phase 3: Multi-Topology Generalization

```bash
python main.py train --preset multi
```

- **Topologies**: linear_5 + grid_3x3 + heavy_hex_19
- **Episodes**: 30,000
- **Circuit depth**: 20
- **Expected time**: ~2–3 hours on MPS
- **Success criteria**: Good performance on all three topologies

This is the novel contribution — training a single agent that generalizes across different hardware architectures. The fixed observation shape (3×27×27) and action masking make this possible without any architecture changes. Each episode randomly picks a topology, so the agent learns topology-agnostic routing strategies.

**Important**: Do NOT transfer checkpoints between phases. Each phase starts fresh because the topologies (and thus the Q-value landscape) are different.

---

## 15. Key Design Decisions

### 1. Off-Policy (DQN) vs On-Policy (PPO)

DQN is off-policy: it stores transitions in a buffer and can reuse each transition many times. This is much more sample-efficient for our setting because:
- Environment steps are CPU-bound (circuit DAG operations)
- We can train on every step, not just at batch boundaries
- PER lets us focus on the most informative transitions

PPO would need to collect fresh on-policy rollouts, discarding them after each update — wasteful for expensive environment steps.

### 2. uint8 State Storage

States are continuous [0,1] float32, but stored as uint8 (×255) in the buffer. This saves 4× memory (440MB vs 1.75GB for 100K transitions) with negligible precision loss. The key insight is that our channel 2 values are quantized enough (gamma_decay=0.5 means values are 1.0, 0.5, 0.25, ...) that 8-bit precision is more than sufficient.

### 3. Action Masking

Invalid actions (edges beyond current topology's count) get Q = -inf in both action selection AND Double DQN target computation. This is essential for multi-topology training — without it, the agent might select a heavy_hex edge during a linear_5 episode, or compute targets using invalid next-actions.

### 4. Terminated vs Truncated

We store `terminated` (not `truncated`) as the done flag in the buffer. When an episode times out (truncated), the environment state isn't truly terminal — there are still gates to route. Using `done=False` preserves Q-value bootstrapping: `target = r + gamma * Q(s')` rather than `target = r`. This prevents the agent from learning that "being near the step limit means zero future reward."

### 5. SABRE Initial Mapping for Evaluation

During evaluation, the agent uses SABRE's initial qubit placement (not a random one). This ensures the comparison is fair — both methods start from the same position. SABRE's initial placement is already optimized, so the agent must demonstrate routing quality, not just luck in initial placement.

### 6. Auto-Execution of Gates

After every SWAP, routable gates execute automatically. The agent never "chooses" to execute a gate — only SWAPs are actions. This simplifies the action space and is optimal: there's never a reason to delay executing a routable gate, since it only frees up dependencies and never blocks anything.

---

## 16. Output Directory Structure

Each training run creates an auto-numbered directory:

```
outputs/
├── run_001/
│   ├── config.json                    ← Full hyperparameters
│   ├── logs/
│   │   ├── episodes.jsonl             ← One line per episode
│   │   ├── train_steps.jsonl          ← Every 100 gradient steps
│   │   └── evaluations.jsonl          ← Each periodic eval summary
│   ├── checkpoints/
│   │   ├── checkpoint_ep1000.pt       ← Periodic checkpoints
│   │   ├── checkpoint_ep2000.pt
│   │   ├── checkpoint_final.pt        ← End of training
│   │   └── checkpoint_emergency.pt    ← If Ctrl+C'd
│   ├── figures/
│   │   ├── training_curves.png        ← Updated each eval (2×3 dashboard)
│   │   ├── eval_comparison_ep500.png  ← Agent vs SABRE at ep 500
│   │   ├── eval_comparison_ep1000.png
│   │   └── ...
│   └── eval/
│       ├── eval_ep500.json            ← Full eval results + trajectories
│       ├── eval_ep1000.json
│       └── ...
├── run_002/
│   └── ...
└── ...
```

Run numbers auto-increment. Everything for a run is self-contained in its directory. The `config.json` records the exact hyperparameters used, making runs reproducible.
