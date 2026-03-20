# RL-Based Quantum Circuit Routing with CNN State Representation

## Complete Project Specification for Implementation

---

## 1. Problem Statement

### What We're Solving

Quantum computers have physical connectivity constraints — each qubit can only directly interact with its physically adjacent neighbors on the chip. When a quantum algorithm requires two non-adjacent qubits to interact (via a CNOT gate), the compiler must insert SWAP gates to move qubit states across the chip until they become neighbors. Each SWAP decomposes into 3 CNOT gates, which are the noisiest operations on the hardware. Minimizing inserted SWAPs directly improves circuit fidelity and execution success probability.

This is the **qubit routing problem**: given a quantum circuit (a DAG of two-qubit gates) and a hardware connectivity graph (which physical positions are connected), find a sequence of SWAP insertions that makes every gate executable while minimizing total SWAPs.

### Why RL Is the Right Approach

The qubit routing problem is NP-hard. Current industrial compilers (IBM Qiskit's SABRE) use greedy heuristics with limited lookahead. RL is suited because:

1. **Sequential decisions with delayed consequences**: A SWAP that helps the current gate may force additional SWAPs later. RL agents learn long-horizon planning.
2. **Combinatorial search space**: Exact methods are intractable beyond small circuits. RL learns to prune the search space through experience.
3. **No analytical solution exists**: The optimal policy depends on the specific circuit and topology in complex ways that can't be captured by simple heuristics.
4. **Correctness by construction**: SWAPs only permute qubit positions — they cannot change circuit logic. No equivalence verification needed.

### What We're Building

An RL agent (DQN and PPO, coded from scratch) that learns to route quantum circuits on real hardware topologies. Our novel contribution is a **CNN-based state representation** that encodes the hardware topology, current qubit mapping, and gate demand as a 3-channel matrix — treating the routing state like a tiny image. This approach is relatively unexplored in the qubit routing literature (most papers use flat vectors or GNNs) and naturally supports multi-topology training.

---

## 2. Literature and Prior Work

### The Baseline We're Beating: SABRE

**Paper**: Li, Ding, Xie. "Tackling the Qubit Mapping Problem for NISQ-Era Quantum Devices." ASPLOS 2019. arXiv:1809.02573

SABRE is IBM Qiskit's default routing algorithm. How it works:
1. Start with a random initial mapping (or use reverse traversal to find a good one)
2. Maintain a "front layer" — gates whose dependencies are satisfied
3. For each front-layer gate that isn't routable (qubits not adjacent), evaluate all candidate SWAPs using a heuristic cost function
4. The cost function: sum of distances between qubit pairs in the front layer, plus a weighted lookahead term considering the next layer of gates
5. Insert the lowest-cost SWAP, update mapping, repeat
6. Run bidirectionally (forward pass, backward pass, forward again) and keep the best result

SABRE's weakness: it's **greedy with limited one-step lookahead**. It cannot sacrifice short-term cost for long-term gain. It also uses a hand-designed cost function with arbitrary weights. RL can potentially learn a better cost evaluation from experience.

SABRE serves as our **primary baseline**. We use Qiskit's built-in SABRE implementation to get baseline SWAP counts for every circuit we test on.

### RL for Qubit Routing: Pozzi et al.

**Paper**: Pozzi, Herbert, Sengupta, Mullins. "Using Reinforcement Learning to Perform Qubit Routing in Quantum Compilers." ACM Transactions on Quantum Computing, 2022. arXiv:2007.15957
**Code**: https://github.com/Macro206/qubit-routing-with-rl

Key details of their approach:
- **State representation**: A compressed distance histogram — d[i] = how many qubits are i hops from their target. Plus an edge availability histogram. This is very compact (~20 numbers) but loses individual qubit identity.
- **Network**: Standard MLP (fully connected layers)
- **Action selection**: They evaluate Q(s_t, s_{t+1}) — quality of state transitions, not individual actions. To find the best action, they use **simulated annealing** to search over possible sets of parallel SWAPs. The network doesn't directly output action scores.
- **Key innovation**: Evaluating state transitions instead of state-action pairs avoids the combinatorial action space problem. The annealing search finds good SWAP sets.
- **Results**: Outperforms Qiskit and TKET routing on random and realistic circuits up to 50 qubits, reducing circuit depth overhead.
- **Reward**: Fixed reward per gate executed + distance reduction reward when qubits move closer to targets + circuit completion reward.

What we take from Pozzi: the distance reduction reward (helps with sparse reward), the general MDP formulation. What we change: richer state representation (CNN matrix instead of compressed histogram), direct action scoring (instead of state-transition evaluation + annealing).

### State of the Art: AlphaRouter

**Paper**: Tang, Duan, Kharkov, Shi. "AlphaRouter: Quantum Circuit Routing with Reinforcement Learning and Tree Search." arXiv:2410.05115, 2024 (published at IEEE/ACM ICCAD).

Key details:
- **Approach**: Combines RL with Monte Carlo Tree Search (MCTS), inspired by AlphaZero.
- **State representation**: Encoded state comprising the remaining logical circuit and current qubit mapping, processed by a **transformer-based agent**.
- **Training**: MCTS generates training data by exploring the action space deeply. The RL agent learns from MCTS-generated trajectories.
- **Inference**: The trained agent routes circuits WITHOUT MCTS (just direct inference), achieving fast runtime while maintaining quality.
- **Results**: Up to 20% less routing overhead than SABRE. Demonstrates generalization to unseen benchmarks.
- **Key insight**: MCTS during training enables deeper exploration than pure RL, but can be dropped at inference time for speed.

This is the current state of the art. We don't aim to beat AlphaRouter (it uses transformers + MCTS, significantly more complex). We aim to demonstrate that a CNN-based state representation with standard DQN/PPO can beat SABRE and approach AlphaRouter-level performance with a simpler architecture.

### Other Relevant Work

**Maskable PPO for Initial Mapping**: "On the Optimality of Quantum Circuit Initial Mapping Using Reinforcement Learning." EPJ Quantum Technology, 2024.
- Uses Maskable PPO to find optimal initial qubit placements on 20-qubit hardware
- Achieves 2.2% improvement over best available RL approaches and 15% over heuristics
- Uses a fixed-length feature vector state, trains per topology

**DPSO + DRL for Circuit Mapping**: "Quantum Circuit Mapping based on Discrete Particle Swarm Optimization and Deep Reinforcement Learning." Swarm and Evolutionary Computation, 2025.
- Combines particle swarm optimization for initial placement with Double DQN for SWAP routing
- Reduces SWAP gates by 14.73% and 16.55% vs TKET and Qiskit respectively on IBM Q20
- Uses structured state representations

**LightSABRE**: Zou, Treinish, Hartman, Ivrii, Lishman. "LightSABRE: A Lightweight and Enhanced SABRE Algorithm." arXiv:2409.08368, 2024.
- Optimized implementation of SABRE now default in Qiskit
- 200x faster than original SABRE
- Important to note: our SABRE baseline uses this optimized version via Qiskit

---

## 3. Data Sources

### Training Data: Random Circuits

Generated on-the-fly using Qiskit during training. Each episode uses a fresh random circuit.

```python
from qiskit.circuit.random import random_circuit
circuit = random_circuit(num_qubits=N, depth=D, max_operands=2, seed=random_seed)
```

Parameters to vary during training:
- num_qubits: match the hardware topology size (e.g., 5 for dev, 19 for Heavy-Hex)
- depth: vary between 5-30 to expose the agent to different circuit complexities
- Random seed: different each episode

From each circuit, we extract only the two-qubit gates (single-qubit gates are irrelevant for routing) and their dependency ordering (the DAG).

### Evaluation Data: QASMBench

**Source**: https://github.com/pnnl/QASMBench
**Paper**: Li, Stein, Krishnamoorthy, Ang. "QASMBench: A Low-Level Quantum Benchmark Suite for NISQ Evaluation and Simulation." ACM Transactions on Quantum Computing, 2023.

QASMBench contains standard quantum algorithms as .qasm files:
- **Small scale (2-10 qubits)**: QFT, Grover, Deutsch, Bernstein-Vazirani, variational circuits, basis change, teleportation, error correction
- **Medium scale (11-27 qubits)**: Larger QFT, VQE, QAOA, quantum arithmetic, machine learning circuits
- **Large scale (28+ qubits)**: For scalability testing if time permits

We load these using Qiskit: `QuantumCircuit.from_qasm_file(path)`

These circuits are **never used during training** — only for evaluation. This tests generalization from random circuits to real algorithms.

### Hardware Topologies

Defined using Qiskit's CouplingMap. The environment supports **multi-topology training** — training on multiple topologies simultaneously with a single model (enabled by padding all states to a fixed N×N matrix size).

Supported topology types (via `get_coupling_map()`):
- **Heavy-Hex**: `heavy_hex_19` (from_heavy_hex(3), 19q, 20 edges), `heavy_hex_27` (from_heavy_hex(5), 27q). IBM's standard processor layout (Falcon, Eagle, Heron all use Heavy-Hex).
- **Grid**: `grid_RxC` (e.g., `grid_3x3`, `grid_4x4`, `grid_5x5`). Rectangular 2D lattice.
- **Linear**: `linear_N` (e.g., `linear_5`, `linear_10`). Simple chain, useful for development.
- **Ring**: `ring_N` (e.g., `ring_5`, `ring_10`). Circular chain.

New topologies can be added to the training list at any time without code changes — just add the name to the `topologies` list parameter. The CNN input shape stays fixed at (3, N, N) regardless of topology.

Real IBM processor sizes for reference: Falcon (27q), Eagle (127q), Heron (133-156q).

### Initial Mapping

The environment supports four initial mapping strategies, configurable via `initial_mapping_strategy`:

- **`"random"`**: Random permutation each episode. Maximum training variety.
- **`"identity"`**: Logical qubit q → physical position q. Simple baseline.
- **`"sabre"`**: Uses Qiskit's SABRE layout pass to compute a realistic initial placement. This is what a real compiler pipeline would provide — isolates routing quality from placement quality. Slower (runs SABRE each reset).
- **`"mixed"` (recommended for training)**: 80% random, 20% SABRE. Gives the agent diverse training scenarios while also learning to work with realistic SABRE placements.

```python
# In the environment:
env = QubitRoutingEnv(
    topologies=["heavy_hex_19", "grid_3x3", "linear_5"],
    initial_mapping_strategy="mixed",  # 80% random, 20% SABRE
)
```

For evaluation/benchmarking, use `"sabre"` to match real deployment conditions.

---

## 4. Environment Specification

### MDP Formulation

The qubit routing environment is a fully deterministic, episodic MDP.

**Episode**: Route one quantum circuit on one hardware topology with one initial mapping. In multi-topology mode, the topology is randomly selected each episode; the circuit is always freshly generated.

**State**: A 3-channel N×N matrix (where N = matrix_size, padded to a fixed size for all topologies). Default N=27.

**Channel 0 — Adjacency (topology)**:
Binary N×N matrix. Cell [i][j] = 1 if physical positions i and j are directly connected on the hardware, 0 otherwise. Symmetric (undirected graph). This channel is **constant within an episode** but changes if you switch topologies.

Source: directly from the hardware's edge list.

**Channel 1 — Qubit Assignment (current mapping)**:
Binary N×N matrix. Cell [q][p] = 1 if logical qubit q is currently at physical position p. This is a permutation matrix — exactly one 1 per row and per column. This channel **updates after every SWAP**.

Source: the current mapping list. If mapping[q] = p, then cell [q][p] = 1.

**Channel 2 — Gate Demand (depth-decayed full circuit visibility)**:
Continuous N×N matrix encoding ALL remaining gates, weighted by their depth in the remaining DAG. This gives the agent full visibility of every gate that still needs to be executed, with urgency encoded as intensity.

For each remaining gate g at depth d in the remaining DAG:
- Look up physical positions p_a = mapping[q_a], p_b = mapping[q_b] under the current mapping
- Add γ^d to cells [p_a][p_b] and [p_b][p_a], where γ = 0.5 (decay factor)
- After accumulating all gates, normalize by dividing by 1/(1-γ) = 2.0 to keep values in [0, 1]

**Depth definition**: A gate's depth in the remaining DAG is its topological level:
- Depth 0 (front layer): gates with no unexecuted predecessors — ready to execute now
- For any other gate: depth = max(depth of all its predecessors) + 1
- A gate depends on another if they share a qubit AND the predecessor appears earlier in the circuit with no other gate on that shared qubit between them

This channel is symmetric and **updates after every SWAP** (because the mapping changes, and gates may be auto-executed, changing the remaining DAG and all depths).

Example with 4 gates, γ=0.5, normalized by 2.0:
```
depth 0: g0: CNOT(q0,q1), g1: CNOT(q2,q3)     → contribute (0.5)^0 / 2.0 = 0.5
depth 1: g2: CNOT(q1,q2), g3: CNOT(q0,q3)     → contribute (0.5)^1 / 2.0 = 0.25
```
The agent sees bright cells (0.5) for urgent gates, dimmer cells (0.25) for upcoming ones. Values accumulate when the same physical positions have multiple interactions across depths.

Source implementation:
```python
def compute_gate_demand_channel(remaining_gates, dag_depths, mapping, n_positions, gamma=0.5):
    channel = np.zeros((n_positions, n_positions))
    for gate in remaining_gates:
        q_a, q_b = gate.qubits
        p_a, p_b = mapping[q_a], mapping[q_b]
        channel[p_a][p_b] += gamma ** dag_depths[gate]
        channel[p_b][p_a] += gamma ** dag_depths[gate]
    channel = channel / (1.0 / (1.0 - gamma))  # normalize to [0, 1]
    return channel
```

**State shape**: (3, N, N) where N = matrix_size (default 27). All topologies are padded to the same N with zeros for non-existent positions. This enables multi-topology training with a fixed CNN input shape.

**Actions**: Discrete(max_edges). One action per hardware edge (possible SWAP). No explicit EXECUTE action — gate execution is automatic. In multi-topology mode, max_edges is the largest edge count across all topologies.

- Action i (for i = 0 to num_edges - 1): Insert SWAP at hardware edge i. This swaps the quantum states of the two qubits at the endpoints of that edge, updating the mapping.
- After every SWAP, all currently routable front-layer gates are automatically executed (gates whose qubits are adjacent under the updated mapping). This matches how all papers handle execution (Pozzi Section 4.1, AlphaRouter Section IV-B) — gates are "mandatory" and execute as soon as their qubits become adjacent.

Action masking (two levels):
1. **Topology mask** (built into environment): In multi-topology mode, actions beyond the current topology's edge count are masked as invalid. E.g., if Heavy-Hex has 20 edges but linear_5 has 4, actions 4-19 are masked when on linear_5.
2. **Strategy mask** (agent-side, optional): Mask SWAPs that cannot possibly help any front-layer gate. See Section 12 for Level 1/2/3 options.

**Reward function** (AlphaRouter-inspired with Pozzi shaping):

After each SWAP action:
```
r_t = (gates_auto_executed) - 1 + 0.01 × Δd  [+ completion_bonus if done]
```

- **Base penalty**: -1 for each SWAP inserted (every action costs)
- **Gate execution bonus**: +1 for each gate that was auto-executed after this SWAP (AlphaRouter-style: r_t = |G_t| - |G_{t+1}| - 1)
- **Distance reduction shaping**: +0.01 × (total_distance_before - total_distance_after), measuring whether front-layer qubits moved closer (inspired by Pozzi et al., helps early training convergence)
- **Circuit completion bonus**: +5 when all gates are routed

This reward is not "all negative" — a SWAP that enables 2 gates yields reward = 2 - 1 + shaping = +1 + shaping. Good SWAPs are rewarded, bad SWAPs are penalized. The distance shaping can optionally be annealed (reduced) over training once the agent learns to find gates directly.

**Termination**: Episode ends when all gates in the circuit have been executed.

**Front layer computation**: At each step, compute the set of gates whose dependencies are satisfied. A gate CNOT(q_a, q_b) is in the front layer if all gates that precede it in the DAG (i.e., earlier gates involving q_a or q_b) have already been executed. Multiple independent gates can be in the front layer simultaneously.

**Step logic** (every step is a SWAP):
1. Agent selects SWAP(edge_i): identify the two physical positions (p1, p2) of the edge, find which logical qubits are at those positions, swap their mapping entries.
2. Auto-execute: iterate through all front-layer gates, execute every one whose qubits are now adjacent. Remove executed gates from the DAG. Update the front layer (new gates may become ready). Recompute DAG depths for remaining gates.
3. Compute reward: r_t = (number of gates just executed) - 1 + distance_shaping + (completion bonus if all gates done).
4. Update state: Channel 1 (mapping) and Channel 2 (gate demand) are recomputed with the new mapping and new remaining DAG.
5. Check termination: if all gates have been executed, episode ends.

---

## 5. Neural Network Architecture

### CNN Policy/Value Network

The CNN processes the (3, N, N) state matrix and outputs action scores (for DQN) or action probabilities + value estimate (for PPO).

```
Input: (3, N, N) state matrix

Conv2d(3, 32, kernel_size=3, padding=1) → ReLU → (32, N, N)
Conv2d(32, 64, kernel_size=3, padding=1) → ReLU → (64, N, N)
Conv2d(64, 32, kernel_size=3, padding=1) → ReLU → (32, N, N)

Flatten → (32 * N * N)

For DQN:
  Linear(32*N*N, 256) → ReLU
  Linear(256, num_actions) → Q-values

For PPO:
  Policy head: Linear(32*N*N, 256) → ReLU → Linear(256, num_actions) → action logits
  Value head: Linear(32*N*N, 256) → ReLU → Linear(256, 1) → state value
```

Symmetry enforcement: before selecting an action, average the N×N output with its transpose (for the SWAP score interpretation), or simply use the flattened approach with one output per edge.

Action masking: set logits/Q-values of invalid actions to -infinity before softmax (PPO) or argmax (DQN).

### For 19-qubit Heavy-Hex

- Input: (3, 19, 19) = 1,083 values
- After 3 conv layers: 32 × 19 × 19 = 11,552 features (flattened)
- Hidden layer: 256 neurons
- Output: 20 actions (20 edges, no execute action)
- Total parameters: approximately 3M (very manageable)

---

## 6. RL Algorithms (Implemented From Scratch)

### DQN (Deep Q-Network)

Components to implement:
- **Q-Network**: CNN architecture above, outputs Q(s, a) for each action
- **Target Network**: Copy of Q-network, updated every C steps (target_update_freq)
- **Replay Buffer**: Stores (state, action, reward, next_state, done) transitions, capacity ~100K
- **Epsilon-Greedy Exploration**: Start at ε=1.0, decay to ε=0.05 over training
- **Action Masking**: Before argmax, set Q-values of invalid actions to -∞

Training loop:
1. Get state from environment
2. With probability ε: random valid action. Otherwise: argmax Q(s, a) over valid actions.
3. Execute action, observe (next_state, reward, done)
4. Store transition in replay buffer
5. Sample mini-batch from replay buffer
6. Compute target: y = r + γ * max_a' Q_target(s', a') (or 0 if done)
7. Loss = MSE(Q(s, a), y)
8. Update Q-network weights via Adam optimizer
9. Every C steps: copy Q-network weights to target network

Hyperparameters:
- Learning rate: 1e-4
- Discount factor (γ): 0.99
- Replay buffer size: 100,000
- Batch size: 64
- Target update frequency: 1,000 steps
- Epsilon decay: linear from 1.0 to 0.05 over 50,000 steps

### PPO (Proximal Policy Optimization)

Components to implement:
- **Policy Network**: CNN → action logits (with masking)
- **Value Network**: CNN → scalar value estimate (shared backbone or separate)
- **GAE (Generalized Advantage Estimation)**: For computing advantage estimates
- **Clipped Surrogate Objective**: Prevents destructive policy updates

Training loop:
1. Collect a batch of trajectories (e.g., 2048 steps across multiple environments)
2. Compute returns and advantages using GAE (λ=0.95, γ=0.99)
3. For K epochs (e.g., 4):
   a. Compute ratio: π_new(a|s) / π_old(a|s)
   b. Clipped objective: min(ratio * advantage, clip(ratio, 1-ε, 1+ε) * advantage)
   c. Value loss: MSE(V(s), returns)
   d. Total loss = -policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus
   e. Update network weights via Adam

Hyperparameters:
- Learning rate: 3e-4
- Discount factor (γ): 0.99
- GAE lambda (λ): 0.95
- Clip epsilon: 0.2
- Epochs per update: 4
- Batch size: 64
- Entropy coefficient: 0.01
- Value loss coefficient: 0.5

---

## 7. Evaluation Plan

### Metrics

1. **SWAP Count**: Total SWAPs inserted by the agent vs. SABRE. Primary metric.
2. **CNOT Overhead**: Total additional CNOTs = SWAPs × 3. Directly corresponds to noise added.
3. **Circuit Depth Overhead**: Additional layers in the routed circuit vs. the original.
4. **Routing Time**: Wall-clock time for the agent to route a circuit (inference speed).
5. **Generalization**: Performance on QASMBench circuits never seen during training.

### Experiments

**Experiment 1: Main comparison on Heavy-Hex**
- Train DQN and PPO on random circuits for the 19-qubit Heavy-Hex topology
- Evaluate on QASMBench small and medium circuits
- Compare SWAP counts against SABRE baseline
- Report: bar chart of SWAP counts per benchmark circuit across all methods

**Experiment 2: Learning curves**
- Plot episode reward and SWAP count vs. training steps
- Compare DQN vs PPO convergence speed and final performance
- Show training stability (variance across random seeds)

**Experiment 3: Ablation studies**
- Effect of gate demand decay factor γ (0.3 vs. 0.5 vs. 0.7) — how much lookahead helps
- Effect of distance reduction reward shaping (with vs. without)
- Effect of action masking (with vs. without)
- Effect of CNN depth (2 vs. 3 vs. 4 conv layers)
- Effect of training circuit depth (shallow vs. deep random circuits)

**Experiment 4: Scalability**
- Performance vs. circuit size (number of two-qubit gates)
- Performance vs. topology size (5-qubit vs. 19-qubit)

**Experiment 5 (if time permits): Multi-topology generalization**
- Train on Heavy-Hex, test on grid and linear topologies (zero-shot transfer)
- Or: train on multiple topologies with padded state, test on each

### Visualizations

1. Training curves (reward, SWAP count, loss vs. steps)
2. Per-benchmark SWAP comparison bar charts
3. Step-by-step routing visualization on the hardware graph (for poster)
4. Scalability plots (SWAP count vs. circuit size)
5. Heatmaps of qubit interaction frequency on the hardware graph
6. Ablation result tables/charts

---

## 8. Technical Stack

### Libraries and Their Purpose

```
qiskit              — Load QASMBench circuits, generate random circuits, define hardware topologies,
                      run SABRE baseline, extract circuit DAGs. NOT used for RL.
torch (PyTorch)     — Build and train the CNN, DQN, and PPO networks. Core ML framework.
gymnasium           — Standard RL environment API. Our environment inherits from gym.Env.
networkx            — Build hardware graph, compute all-pairs shortest paths for reward shaping.
numpy               — Array operations for state construction, reward computation.
matplotlib          — Plot training curves, benchmark comparisons, visualizations.
```

### Installation

```bash
pip install qiskit qiskit-aer networkx matplotlib numpy torch gymnasium
git clone https://github.com/pnnl/QASMBench.git
```

### Data Pipeline

```
QASMBench .qasm files  ──→  Qiskit QuantumCircuit  ──→  DAG (gate list + dependencies)
                                                              │
Random circuit generator ──→  Qiskit QuantumCircuit  ──→  DAG (gate list + dependencies)
                                                              │
                                                              ▼
Hardware topology (CouplingMap)  ──→  Edge list + Distance matrix (NetworkX)
                                                              │
                                                              ▼
                                              QubitRoutingEnv (Gymnasium)
                                                    │           │
                                              state (3,N,N)   reward (float)
                                                    │           │
                                                    ▼           │
                                              CNN + DQN/PPO    │
                                                    │           │
                                              action (int)  ◄──┘
                                                    │
                                                    ▼
                                              SWAP count (evaluation metric)
                                              vs. SABRE baseline
```

---

## 9. File Structure

```
project/
├── environments/
│   ├── routing_env.py          # Gymnasium environment (state, actions, reward, step logic)
│   └── circuit_utils.py        # Load circuits, extract gates, compute front layer, DAG operations
├── agents/
│   ├── dqn.py                  # DQN agent (replay buffer, target network, training loop)
│   ├── ppo.py                  # PPO agent (GAE, clipped objective, training loop)
│   └── networks.py             # CNN architecture (shared by DQN and PPO)
├── baselines/
│   └── sabre_baseline.py       # Run SABRE on circuits and record SWAP counts
├── training/
│   ├── train_dqn.py            # DQN training script
│   └── train_ppo.py            # PPO training script
├── evaluation/
│   ├── evaluate.py             # Run trained agents on QASMBench, compare vs SABRE
│   └── visualize.py            # Generate plots, routing animations, comparison charts
├── configs/
│   └── config.yaml             # Hyperparameters, topology choice, training settings
├── QASMBench/                  # Cloned benchmark circuits (git submodule)
└── notebooks/
    ├── data_exploration.ipynb  # Explore circuits, topologies, distances
    └── results_analysis.ipynb  # Analyze and visualize results
```

---

## 10. Implementation Order

### Phase 1: Environment (Priority: Critical)

Build and thoroughly test the Gymnasium environment. This is the foundation — if the environment has bugs, nothing else works.

1. Implement `circuit_utils.py`: load .qasm files, extract two-qubit gates, compute DAG dependencies, compute front layer
2. Implement `routing_env.py`: the full Gymnasium Env with 3-channel matrix state, action handling, reward computation, termination
3. Test with a trivial policy (random actions) to verify: states are valid, rewards are correct, episodes terminate, SWAP count is tracked
4. Test that SABRE's SWAP count on the same circuit is reproducible as a comparison point

### Phase 2: DQN Agent (Priority: High)

5. Implement `networks.py`: CNN architecture with forward pass
6. Implement `dqn.py`: replay buffer, epsilon-greedy, target network, training loop
7. Train on 5-qubit linear chain with simple random circuits — verify learning happens (reward increases, SWAP count decreases)
8. Scale to 19-qubit Heavy-Hex

### Phase 3: PPO Agent (Priority: High)

9. Implement `ppo.py`: trajectory collection, GAE, clipped surrogate loss
10. Train on same benchmarks as DQN, compare

### Phase 4: Evaluation (Priority: High)

11. Implement `sabre_baseline.py`: run SABRE on all QASMBench circuits, record results
12. Implement `evaluate.py`: run trained DQN and PPO on same circuits, compare
13. Generate all plots and tables for the report

### Phase 5: Extensions (Priority: Medium)

14. Ablation studies (reward shaping, masking, network depth)
15. Multi-topology experiments (if time permits)
16. Noise-aware routing with weighted edges (if time permits)

---

## 11. Key Implementation Details

### Front Layer Computation

A gate CNOT(q_a, q_b) is in the front layer if and only if there is no earlier unexecuted gate in the circuit that involves q_a or q_b. "Earlier" means appearing before this gate in the topological ordering of the circuit DAG.

Efficient implementation: maintain a pointer per qubit tracking the next unexecuted gate involving that qubit. A gate is in the front layer if both its qubits' pointers point to it.

### Automatic Gate Execution

After every SWAP, the environment automatically executes all routable front-layer gates (those whose qubits are adjacent under the updated mapping). Multiple gates can execute simultaneously if they involve different qubits. After execution, the front layer is updated — new gates may become available, and DAG depths are recomputed for the remaining gates.

There is no explicit EXECUTE action. This design matches Pozzi et al. (Section 4.1: gates are "mandatory") and AlphaRouter (Section IV-B: "Any logical gates that now satisfy the topology are greedily scheduled"). In a real compiler, there is never a reason to delay executing a gate whose qubits are already adjacent.

### Symmetry in the Output

Since the hardware graph is undirected, SWAP(pos_i, pos_j) is the same action as SWAP(pos_j, pos_i). We handle this by defining actions based on the edge list (each undirected edge appears once). The output has one neuron per edge, not per directed pair.

For the 3-channel state matrix, channels 0 and 2 are symmetric (adjacency and demand are both symmetric for undirected graphs). Channel 1 (mapping) is a permutation matrix, which is not symmetric but has structure (one 1 per row and column). Channel 2 values are continuous in [0, 1] after normalization, while channels 0 and 1 are binary — all three channels share the same [0, 1] value range.

### Episode Length Limits

Set a maximum step limit per episode (e.g., 500 steps) to prevent infinite loops during early training when the agent hasn't learned anything useful. If the limit is reached, terminate with a large penalty (-10) to discourage non-terminating behavior.

### Reward Normalization

For PPO, normalize advantages to have zero mean and unit variance within each batch. This stabilizes training. For DQN, consider reward clipping to [-5, 5] if training is unstable.

### DAG Depth Computation for Channel 2

The depth of each remaining gate must be recomputed after every step (since executed gates change the remaining DAG structure). Use a BFS/topological-level computation:

```python
def compute_dag_depths(remaining_gates, get_predecessors):
    depths = {}
    front_layer = [g for g in remaining_gates if not get_predecessors(g)]
    current_level = front_layer
    d = 0
    while current_level:
        for gate in current_level:
            depths[gate] = d
        next_level = []
        for gate in current_level:
            for successor in get_successors(gate):
                if all(pred in depths for pred in get_predecessors(successor)):
                    if successor not in depths:
                        next_level.append(successor)
        current_level = next_level
        d += 1
    return depths
```

Complexity: O(gates) per step — a full BFS over the remaining DAG. For circuits with hundreds of gates this is negligible compared to the neural network forward pass.

---

## 12. Design Decisions and Rationale

This section documents the key design decisions made during the project planning phase, the reasoning behind each one, and considerations for implementation. These decisions were reached after careful analysis of the Pozzi, SABRE, and AlphaRouter papers.

### Decision 1: Remove the EXECUTE action — gate execution is automatic

**What changed**: The original design had an explicit EXECUTE action in the action space. The agent would choose when to execute routable gates. This has been removed. Now, after every SWAP, all routable front-layer gates execute automatically.

**Why**: In all three reference papers, gate execution is automatic/mandatory:
- Pozzi (Section 4.1): "gates are still mandatory, i.e. gates are performed as soon as their two qubits land next to each other"
- AlphaRouter (Section IV-B): "Any logical gates that now satisfy the topology are greedily scheduled"
- SABRE: executes routable gates immediately before searching for the next SWAP

There is never a reason to delay executing a gate whose qubits are already adjacent in a real compiler. The EXECUTE action added meaningless decisions to the MDP — the agent could waste steps doing SWAP → SWAP → SWAP → EXECUTE instead of executing gates immediately after each SWAP. This inflated episode length, made credit assignment harder, and didn't model anything physically meaningful.

**Impact on action space**: For 19-qubit Heavy-Hex, actions go from 21 (20 edges + 1 execute) to 20 (20 edges only). Simpler action space = faster learning.

**Impact on reward**: No stagnation penalty needed (agent can't choose EXECUTE when nothing is routable). No separate gate execution bonus — gates execute as a natural consequence of good SWAPs.

### Decision 2: Redesign Channel 2 — depth-decayed full circuit visibility instead of front-layer-only

**What changed**: The original Channel 2 was a binary matrix showing only the current front-layer gate demands. The new Channel 2 shows ALL remaining gates, weighted by their depth in the remaining DAG using exponential decay (γ = 0.5), normalized to [0, 1].

**Why — the information gap problem**: The original design gave the agent strictly less information than SABRE at each decision point. SABRE uses an "extended set" (Section IV-D) that considers gates beyond the front layer with a weighted lookahead. AlphaRouter uses up to 48 gates of lookahead. With only front-layer visibility, our agent would be blind to upcoming demands — it might pick a SWAP that solves the current gate but pushes qubits for the next gate far apart.

**Why depth-decay instead of multiple channels**: We considered adding separate channels for each depth layer (Channel 2 = front layer, Channel 3 = next layer, etc.). This would require a variable number of channels depending on circuit depth and add complexity. The depth-decay approach encodes all the same information in a single channel:
- Bright values (~0.5 after normalization) = front-layer gates (execute now)
- Medium values (~0.25) = next-layer gates (coming soon)
- Dim values (~0.125) = deeper gates (coming eventually)
- Values accumulate when the same qubit pair interacts at multiple depths, signaling "keep these positions close"

The CNN naturally learns "brighter = more urgent" — the same intensity-based reasoning it uses in image processing.

**Why γ = 0.5**: This gives a clear 2:1 ratio between adjacent depths, making urgency differences easy for the CNN to learn. The theoretical maximum cell value is 1/(1-γ) = 2.0 (geometric series bound), which stays well-behaved. Smaller γ (0.3) would essentially collapse back to front-layer-only. Larger γ (0.7+) would make all depths look similar, losing urgency signal. γ = 0.5 is the default; we ablate over {0.3, 0.5, 0.7} in experiments.

**Why normalize by 1/(1-γ)**: With γ=0.5, raw cell values can reach up to 2.0 (when a qubit pair interacts at every depth). Dividing by 2.0 guarantees all values fall in [0, 1], matching the range of channels 0 and 1 (both binary). This is consistent with standard CNN practice where all input channels share the same scale. The normalization constant is fixed (not data-dependent), so the agent sees consistent scales across all circuits. After normalization, differences between depths (e.g., 0.5 vs 0.25 = difference of 0.25) are still far larger than what CNNs routinely distinguish in image processing (1/255 ≈ 0.004).

**How it updates**: After every SWAP: (1) mapping changes, (2) routable gates auto-execute and are removed, (3) DAG depths are recomputed from the new front layer via BFS, (4) Channel 2 is rebuilt with updated positions and depths. The agent always sees a fresh, accurate picture. Gates that were dim become bright as their predecessors get executed.

**Depth definition (precise)**: A gate's depth = max(depth of all its direct predecessors in the remaining DAG) + 1. Gates with no remaining predecessors have depth 0. A gate g_b directly depends on gate g_a if they share a qubit AND g_a is the most recent earlier gate on that shared qubit (not any earlier gate — only the immediately preceding one on each qubit).

### Decision 3: AlphaRouter-inspired reward with Pozzi shaping

**What changed**: The original reward had separate SWAP penalty (-1), gate execution bonus (+0.1), distance shaping (+0.01 × Δd), completion bonus (+5), and stagnation penalty (-0.5). The new reward is:

```
r_t = (gates_auto_executed) - 1 + 0.01 × Δd  [+ 5 if circuit complete]
```

**Why**: With automatic gate execution, the reward naturally follows AlphaRouter's formulation (r_t = |G_t| - |G_{t+1}| - 1). A SWAP that enables 0 gates: reward ≈ -1. A SWAP that enables 1 gate: reward ≈ 0. A SWAP that enables 2+ gates: reward is positive. The agent learns to pick SWAPs that make gates executable — the objective emerges directly from the reward structure without needing separate bonuses for different events.

The Pozzi-style distance shaping (+0.01 × Δd) is kept as an auxiliary signal to help early training when the agent rarely finds gates. It can be annealed (reduced) over training once the agent starts succeeding more often.

The stagnation penalty is no longer needed (there's no EXECUTE action to misuse).

### Decision 4: Padded N×N matrices for multi-topology support

**What kept**: The original design of padding state matrices to a maximum N was retained.

**Why it works**: Channel 1 (qubit assignment) is an N×N permutation matrix — sparse by nature (exactly N ones). For a 19-qubit topology padded to a 27-qubit max, most cells are 0. This is intentional: the padding allows a single trained model to work on multiple topologies without retraining. The CNN learns that zero-padded regions are inactive. This is one of our novel contributions — most papers (Pozzi, AlphaRouter) train separate models per topology.

### Considerations and Open Questions

**γ sensitivity**: The optimal γ likely depends on circuit structure. Shallow circuits (few layers) may benefit from lower γ (focus on immediate gates). Deep circuits with long dependency chains may benefit from higher γ (more lookahead). The ablation study over {0.3, 0.5, 0.7} will clarify this.

**Channel 2 position updates**: Future gates' physical positions are computed under the current mapping, which will change as SWAPs happen. This is the standard approach — SABRE evaluates its extended set under the current mapping too. The channel updates every step, so positions stay fresh. The agent implicitly learns that dim cells' positions are tentative.

**Accumulation behavior**: When the same qubit pair interacts at multiple depths, their contributions add up. With γ=0.5 and normalization, a pair interacting at depths 0 and 1 produces a cell value of (1.0 + 0.5)/2.0 = 0.75. This is informative — it tells the agent "these positions need to interact both now AND soon, prioritize keeping them close." In practice, most qubit pairs interact 1-3 times across a circuit, so accumulation is modest. QFT circuits have each pair interacting exactly once. VQE/QAOA with repeated ansatz layers may have 3-5 interactions per pair spread across depths, but the exponential decay ensures deep contributions are negligible.

**CNN inductive bias for permutation matrices**: Channel 1 is a permutation matrix where row/column indices are arbitrary labels, not spatial positions. A CNN's spatial convolution kernels may not be the ideal architecture for this structure. However, the padding design (Decision 4) depends on this matrix format, and the 3×3 kernels with padding=1 can still learn useful local patterns. If training struggles, adding a BatchNorm2d as the first CNN layer could help the network adapt to the different statistical properties of each channel. This is a discussion point for the report, not a correctness issue.

### Open Decision: Action Masking Strategy

Action masking controls which SWAPs the agent is allowed to consider at each step. This decision does NOT affect the environment implementation — masking is applied on the agent/policy side. The environment should expose the necessary data (front layer, distance matrix, mapping), and the agent computes the mask.

**Three levels under consideration:**

**Level 1 — Neighbor masking**: Only allow SWAPs on edges where at least one endpoint holds a qubit involved in a front-layer gate. Removes SWAPs in completely unrelated parts of the topology.

**Level 2 — Shortest-path masking**: Only allow SWAPs on edges that lie along a shortest path between any front-layer gate's qubit pair. More restrictive — every valid action provably moves a gate qubit closer to its partner.

**Level 3 — SABRE-style masking**: Only allow SWAPs where one endpoint holds a front-layer gate qubit AND the swap moves it toward its partner. Most restrictive — used by SABRE itself.

**Optional relaxation**: Level 2 can be relaxed to include edges within 1 hop of a shortest path (replace `== distance` with `<= distance + 1`), allowing near-optimal alternatives that might avoid congestion.

**Trade-off**: Tighter masking dramatically speeds up exploration (random actions are more likely to be useful) but could theoretically block non-obvious strategic moves — e.g., moving an uninvolved qubit out of the way preemptively. In practice this is rare: SABRE uses Level 3 (the most restrictive) and is the industry standard. The agent can also react to displacement on subsequent steps rather than preemptively avoiding it, since Channel 2's depth-decay encoding shows upcoming demands.

**Implementation note**: This is purely an agent-side concern. The environment accepts any edge index as a valid action. Masking is computed by the agent before action selection:
```python
# Agent-side masking (any level)
q_values = network(state)
q_values[~action_mask] = -float('inf')  # mask invalid actions
action = q_values.argmax()
```

The environment only needs to provide the data used to compute masks (coupling map edges, distance matrix, front layer, mapping) — all of which it already exposes for state construction. The masking level can be changed or ablated without modifying the environment.

**Decision**: To be finalized during agent implementation. Recommended approach: implement all levels, default to Level 2, and ablate masking levels as an experiment.

### Decision 5: Multi-topology training with random topology selection

**What**: The environment supports training on multiple topologies simultaneously. Pass a list of topology names (e.g., `["heavy_hex_19", "grid_3x3", "linear_5"]`), and each `reset()` randomly picks one. The CNN input shape is always (3, N, N) where N = matrix_size (default 27), with smaller topologies zero-padded.

**Why**: This is one of our novel contributions — most papers (Pozzi, AlphaRouter, SABRE) train and evaluate on a single fixed topology. Multi-topology training teaches the agent general routing strategies that transfer across chip layouts. The CNN naturally handles this because Channel 0 (adjacency) tells it the chip layout, and the padding makes the input shape uniform.

**Action space in multi-topology mode**: `Discrete(max_edges)` where max_edges = largest edge count across all topologies. The environment's `get_action_mask()` returns a mask that blocks actions corresponding to edges that don't exist on the current episode's topology. Agents MUST apply this mask before action selection.

**How to apply**: Add topologies to the list at any time. Ensure `matrix_size >= max(qubit_count)` across all topologies. Supported topology formats: `heavy_hex_19`, `heavy_hex_27`, `grid_RxC`, `linear_N`, `ring_N`.

### Decision 6: Mixed initial mapping strategy for training

**What**: The environment supports four initial mapping strategies: `"random"`, `"identity"`, `"sabre"`, and `"mixed"`. The recommended training strategy is `"mixed"` (80% random permutations, 20% SABRE initial placements).

**Why**:
- **Random mappings** give the agent maximum training diversity — it sees many different starting configurations and learns robust routing strategies.
- **SABRE placements** provide realistic starting points matching what a real compiler pipeline would produce. Training only on random would leave the agent unprepared for the realistic placements it will encounter at deployment.
- **80/20 split** balances variety with realism. The agent mostly explores diverse scenarios but also gets regular exposure to SABRE-quality placements.

**For evaluation**: Use `"sabre"` exclusively, since this matches the real deployment scenario and gives a fair comparison against SABRE's own routing.

### Decision 7: Random circuit generation for training data

**What**: Training episodes use freshly generated random circuits (via `qiskit.circuit.random.random_circuit`), not pre-existing circuit datasets. Each `reset()` creates a new circuit with the current topology's qubit count.

**Why**:
- Infinite training data with no dataset management needed.
- Each episode is unique — prevents overfitting to specific circuit patterns.
- QASMBench circuits are reserved exclusively for evaluation — tests true generalization from random circuits to real quantum algorithms.
- Circuit depth is configurable (`circuit_depth` parameter, default 20) and can be varied during training to expose the agent to different complexity levels.
