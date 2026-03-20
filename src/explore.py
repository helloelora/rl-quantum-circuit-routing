"""
=============================================================================
QUBIT ROUTING: COMPLETE DATA EXPLORATION
=============================================================================
This single file shows you EVERYTHING from raw data to RL-ready input.
Run it and read the output top to bottom.
=============================================================================
"""

import numpy as np
import networkx as nx
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap
from qiskit.converters import circuit_to_dag
from qiskit.circuit.random import random_circuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

print("=" * 70)
print("PART 1: THE RAW CIRCUIT DATA (from QASMBench)")
print("=" * 70)

# Load a real quantum algorithm from QASMBench
qasm_path = "/home/claude/QASMBench/small/qft_n4/qft_n4.qasm"
circuit = QuantumCircuit.from_qasm_file(qasm_path)

print(f"\nLoaded: Quantum Fourier Transform (4 qubits)")
print(f"Source: QASMBench/small/qft_n4/qft_n4.qasm")
print(f"\nThis is what the algorithm looks like as code:")
print("-" * 50)
with open(qasm_path) as f:
    print(f.read())

print("-" * 50)
print(f"\nCircuit stats:")
print(f"  Number of qubits: {circuit.num_qubits}")
print(f"  Total gates: {circuit.size()}")
print(f"  Circuit depth: {circuit.depth()}")
print(f"  Gate breakdown: {dict(circuit.count_ops())}")

# Extract the two-qubit gates (these are what matter for routing)
dag = circuit_to_dag(circuit)
two_qubit_gates = []
for node in dag.topological_op_nodes():
    if len(node.qargs) == 2:
        q1 = circuit.qubits.index(node.qargs[0])
        q2 = circuit.qubits.index(node.qargs[1])
        two_qubit_gates.append((node.name, q1, q2))

print(f"\n  Two-qubit gates (the ones that need routing): {len(two_qubit_gates)}")
print(f"  Single-qubit gates (don't need routing): {circuit.size() - len(two_qubit_gates)}")
print(f"\n  The two-qubit gates are:")
for i, (name, q1, q2) in enumerate(two_qubit_gates):
    print(f"    Gate {i}: {name}(q{q1}, q{q2})")

print(f"\n  These are the interactions that MUST happen.")
print(f"  Each one requires q{two_qubit_gates[0][1]} and q{two_qubit_gates[0][2]} to be ADJACENT on the chip.")
print(f"  If they're not adjacent, we need SWAPs to bring them together.")


print("\n" + "=" * 70)
print("PART 2: THE HARDWARE TOPOLOGY (the chip)")
print("=" * 70)

# Define a small topology for clarity
edges_5q = [(0,1), (1,2), (2,3), (3,4)]
coupling_5q = CouplingMap(edges_5q)

print(f"\n--- 5-Qubit Linear Chain (for development) ---")
print(f"  Positions: 0, 1, 2, 3, 4")
print(f"  Connections: {edges_5q}")
print(f"  Layout:  pos0 — pos1 — pos2 — pos3 — pos4")
print(f"  Number of possible SWAP actions: {len(edges_5q)}")

# Build graph and compute distances
G_5q = nx.Graph()
G_5q.add_edges_from(edges_5q)
dist_5q = dict(nx.all_pairs_shortest_path_length(G_5q))

print(f"\n  Distance matrix (hops between every pair of positions):")
print(f"  {'':>6}", end="")
for j in range(5):
    print(f"pos{j:>2}", end="  ")
print()
for i in range(5):
    print(f"  pos{i}: ", end="")
    for j in range(5):
        d = dist_5q[i][j]
        print(f"  {d}  ", end=" ")
    print()

print(f"\n  Notice: pos0 to pos4 = 4 hops = need 3 SWAPs minimum")
print(f"  Notice: pos1 to pos2 = 1 hop = can execute directly, no SWAP needed")

# Now show a real 27-qubit topology
print(f"\n--- 27-Qubit IBM Heavy-Hex (real hardware) ---")
coupling_27q = CouplingMap.from_heavy_hex(3)
edges_27q = list(set(tuple(sorted(e)) for e in coupling_27q.get_edges()))
edges_27q.sort()

G_27q = nx.Graph()
G_27q.add_edges_from(edges_27q)
dist_27q = dict(nx.all_pairs_shortest_path_length(G_27q))

print(f"  Number of qubits: {coupling_27q.size()}")
print(f"  Number of connections (edges): {len(edges_27q)}")
print(f"  Number of possible SWAP actions: {len(edges_27q)}")

# Show degree distribution
degrees = [G_27q.degree(n) for n in G_27q.nodes()]
print(f"\n  Connectivity per qubit:")
for node in sorted(G_27q.nodes()):
    neighbors = list(G_27q.neighbors(node))
    print(f"    pos{node:>2}: {G_27q.degree(node)} connections → neighbors: {neighbors}")

# Show some distance examples
print(f"\n  Example distances on the 27-qubit chip:")
max_node = max(G_27q.nodes())
example_pairs = [(0, 1), (0, 6), (0, max_node), (1, 8), (5, 17)]
for p1, p2 in example_pairs:
    d = dist_27q[p1][p2]
    path = nx.shortest_path(G_27q, p1, p2)
    print(f"    pos{p1} → pos{p2}: {d} hops, path: {' → '.join(str(p) for p in path)}")

max_dist = max(dist_27q[i][j] for i in G_27q.nodes() for j in G_27q.nodes())
print(f"\n  Maximum distance on chip: {max_dist} hops")
print(f"  Worst case: need {max_dist - 1} SWAPs = {(max_dist-1)*3} extra CNOTs just for ONE gate")


print("\n" + "=" * 70)
print("PART 3: THE ROUTING PROBLEM (circuit + topology together)")
print("=" * 70)

# Use a 5-qubit example for clarity
print(f"\n  Circuit: QFT on 4 qubits (loaded from QASMBench)")
print(f"  Hardware: 5-qubit linear chain")
print(f"\n  The circuit has these two-qubit gates:")
for i, (name, q1, q2) in enumerate(two_qubit_gates):
    pos_q1 = q1  # trivial initial mapping
    pos_q2 = q2
    dist = dist_5q[pos_q1][pos_q2] if pos_q1 < 5 and pos_q2 < 5 else "N/A"
    adjacent = "YES ✓" if dist == 1 else f"NO ✗ (distance={dist})"
    print(f"    Gate {i}: {name}(q{q1}, q{q2}) → adjacent? {adjacent}")

routable = sum(1 for _, q1, q2 in two_qubit_gates 
               if q1 < 5 and q2 < 5 and dist_5q[q1][q2] == 1)
not_routable = len(two_qubit_gates) - routable
print(f"\n  With trivial mapping (q_i → pos_i):")
print(f"    {routable} gates can execute directly (qubits already adjacent)")
print(f"    {not_routable} gates are BLOCKED (need SWAPs)")


print("\n" + "=" * 70)
print("PART 4: SABRE BASELINE (what we're trying to beat)")
print("=" * 70)

# Run SABRE on the circuit — use higher optimization to handle gate decomposition
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Decompose

pm = generate_preset_pass_manager(
    optimization_level=2, 
    coupling_map=coupling_5q,
    basis_gates=['cx', 'u1', 'u2', 'u3', 'id', 'swap']
)
transpiled = pm.run(circuit)

ops = transpiled.count_ops()
swap_count = ops.get('swap', 0)
cx_count = ops.get('cx', 0)

print(f"\n  SABRE routed the QFT circuit on 5-qubit linear chain:")
print(f"  Original circuit: {circuit.size()} gates, depth {circuit.depth()}")
print(f"  After SABRE:      {transpiled.size()} gates, depth {transpiled.depth()}")
print(f"  SWAPs inserted:   {swap_count}")
print(f"  Total CX gates:   {cx_count}")
print(f"  Gate breakdown:   {dict(ops)}")
print(f"\n  THIS IS THE NUMBER TO BEAT. Your RL agent should insert ≤ {swap_count} SWAPs.")

# Run on a few more circuits to show variation
print(f"\n  SABRE results on other QASMBench circuits (5-qubit linear chain):")
print(f"  {'Circuit':<30} {'Qubits':<8} {'2Q Gates':<10} {'SABRE SWAPs':<12}")
print(f"  {'-'*60}")

qasm_files = [
    ("QASMBench/small/qft_n4/qft_n4.qasm", "QFT (4q)"),
    ("QASMBench/small/basis_change_n3/basis_change_n3.qasm", "Basis Change (3q)"),
    ("QASMBench/small/deutsch_n2/deutsch_n2.qasm", "Deutsch (2q)"),
    ("QASMBench/small/variational_n4/variational_n4.qasm", "Variational (4q)"),
]

for path, name in qasm_files:
    full_path = f"/home/claude/{path}"
    if os.path.exists(full_path):
        try:
            c = QuantumCircuit.from_qasm_file(full_path)
            if c.num_qubits <= 5:
                pm2 = generate_preset_pass_manager(
                    optimization_level=2, coupling_map=coupling_5q,
                    basis_gates=['cx', 'u1', 'u2', 'u3', 'id', 'swap']
                )
                dag_temp = circuit_to_dag(c)
                two_q = sum(1 for n in dag_temp.topological_op_nodes() if len(n.qargs) == 2)
                t = pm2.run(c)
                s = t.count_ops().get('swap', 0)
                print(f"  {name:<30} {c.num_qubits:<8} {two_q:<10} {s:<12}")
        except Exception as e:
            pass


print("\n" + "=" * 70)
print("PART 5: GENERATING RANDOM CIRCUITS (training data)")
print("=" * 70)

print(f"\n  For training, we generate random circuits on the fly.")
print(f"  Each episode = one random circuit to route.")
print(f"\n  Here are 5 random circuits and their properties:")
print(f"  {'Circuit':<15} {'Qubits':<8} {'Depth':<8} {'2Q Gates':<10} {'SABRE SWAPs':<12}")
print(f"  {'-'*53}")

for i in range(5):
    rc = random_circuit(num_qubits=5, depth=10, max_operands=2, seed=42+i)
    dag_temp = circuit_to_dag(rc)
    two_q = sum(1 for n in dag_temp.topological_op_nodes() if len(n.qargs) == 2)
    pm3 = generate_preset_pass_manager(
        optimization_level=2, coupling_map=coupling_5q,
        basis_gates=['cx', 'u1', 'u2', 'u3', 'id', 'swap']
    )
    t = pm3.run(rc)
    s = t.count_ops().get('swap', 0)
    print(f"  Random #{i+1:<9} {rc.num_qubits:<8} {rc.depth():<8} {two_q:<10} {s:<12}")

print(f"\n  Each random circuit gives the agent a different routing challenge.")
print(f"  Over thousands of episodes, the agent learns general routing strategies.")


print("\n" + "=" * 70)
print("PART 6: BUILDING THE RL STATE (what the agent actually sees)")
print("=" * 70)

print(f"\n  Let's build the actual state vector for one specific moment.")
print(f"\n  --- Setup ---")
print(f"  Hardware: 5-qubit linear chain")
print(f"  Circuit gates (two-qubit only, in dependency order):")

# Simple example circuit
example_gates = [(0, 2), (1, 3), (0, 3), (1, 4)]
for i, (q1, q2) in enumerate(example_gates):
    print(f"    Gate {i}: CNOT(q{q1}, q{q2})")

mapping = [1, 0, 2, 3, 4]  # q0→pos1, q1→pos0 (one SWAP already happened)
executed = [True, False, False, False]  # Gate 0 is done

print(f"\n  Current mapping (after one SWAP already happened):")
for q, p in enumerate(mapping):
    print(f"    Logical qubit q{q} → Physical position {p}")

print(f"\n  Gates executed: Gate 0 ✓")
print(f"  Gates remaining: Gate 1, Gate 2, Gate 3")

# Compute front layer
print(f"\n  --- Front Layer Computation ---")
print(f"  Gate 1: CNOT(q1, q3) — no dependency on remaining gates → READY")
print(f"  Gate 2: CNOT(q0, q3) — depends on Gate 0 (done) → READY")
print(f"  Gate 3: CNOT(q1, q4) — depends on Gate 1 (not done, shares q1) → BLOCKED")
print(f"  Front layer = {{Gate 1, Gate 2}}")

# Check routability
print(f"\n  --- Routability Check ---")
front_layer = [1, 2]
for gi in front_layer:
    q1, q2 = example_gates[gi]
    p1, p2 = mapping[q1], mapping[q2]
    dist = dist_5q[p1][p2]
    adjacent = dist == 1
    print(f"  Gate {gi}: CNOT(q{q1}, q{q2})")
    print(f"    q{q1} at pos{p1}, q{q2} at pos{p2}")
    print(f"    Distance: {dist} hops → {'ROUTABLE ✓' if adjacent else 'BLOCKED ✗'}")

# Build flat vector state
print(f"\n  --- Flat Vector State (Approach A) ---")
N = 5
mapping_norm = [m / N for m in mapping]
print(f"  Mapping (normalized): {[f'{v:.2f}' for v in mapping_norm]}")

urgency = [0.0] * N
for gi in front_layer:
    q1, q2 = example_gates[gi]
    p1, p2 = mapping[q1], mapping[q2]
    d = dist_5q[p1][p2]
    urgency[q1] = max(urgency[q1], 1.0 / (d + 1))
    urgency[q2] = max(urgency[q2], 1.0 / (d + 1))

print(f"  Urgency:             {[f'{v:.2f}' for v in urgency]}")
for q in range(N):
    if urgency[q] > 0:
        print(f"    q{q}: urgency={urgency[q]:.2f} — needs to interact, partner is {int(1/urgency[q] - 1)} hops away")
    else:
        print(f"    q{q}: urgency=0.00 — not in front layer, no immediate need")

progress = sum(executed) / len(example_gates)
print(f"  Progress:            [{progress:.2f}] ({sum(executed)}/{len(example_gates)} gates done)")

state_vector = mapping_norm + urgency + [progress]
print(f"\n  COMPLETE STATE VECTOR (what the neural network receives):")
print(f"  {[f'{v:.2f}' for v in state_vector]}")
print(f"  Length: {len(state_vector)} numbers")

# Build matrix state
print(f"\n  --- Matrix State (Approach B) ---")
print(f"\n  Channel 0: Adjacency (the hardware topology)")
adj_matrix = np.zeros((N, N), dtype=int)
for i, j in edges_5q:
    adj_matrix[i][j] = 1
    adj_matrix[j][i] = 1
print(f"       pos0 pos1 pos2 pos3 pos4")
for i in range(N):
    print(f"  pos{i}: {list(adj_matrix[i])}")

print(f"\n  Channel 1: Qubit Assignment (current mapping)")
assign_matrix = np.zeros((N, N), dtype=int)
for q, p in enumerate(mapping):
    assign_matrix[q][p] = 1
print(f"       pos0 pos1 pos2 pos3 pos4")
for q in range(N):
    print(f"  q{q}:   {list(assign_matrix[q])}")

print(f"\n  Channel 2: Gate Demand (front layer needs)")
demand_matrix = np.zeros((N, N), dtype=int)
for gi in front_layer:
    q1, q2 = example_gates[gi]
    p1, p2 = mapping[q1], mapping[q2]
    demand_matrix[p1][p2] = 1
    demand_matrix[p2][p1] = 1
print(f"       pos0 pos1 pos2 pos3 pos4")
for i in range(N):
    print(f"  pos{i}: {list(demand_matrix[i])}")
print(f"\n  Reading channel 2: pos0↔pos3 has demand (q1 at 0 needs q3 at 3)")
print(f"                     pos1↔pos3 has demand (q0 at 1 needs q3 at 3)")

print(f"\n  COMPLETE MATRIX STATE: shape (3, {N}, {N}) = {3*N*N} numbers")
print(f"  This is what the CNN receives — like a tiny 3-channel {N}×{N} image.")

# GNN state
print(f"\n  --- Graph State (Approach C: GNN) ---")
print(f"\n  Node features (one row per physical position):")
print(f"  {'Pos':<5} {'Qubit Here':<15} {'Front Layer?':<14} {'Future Gates':<14} {'Degree':<8}")
print(f"  {'-'*56}")

for pos in range(N):
    qubit_here = mapping.index(pos)
    in_front = any(
        mapping[example_gates[gi][0]] == pos or mapping[example_gates[gi][1]] == pos
        for gi in front_layer
    )
    future = sum(1 for i, (q1, q2) in enumerate(example_gates) 
                 if not executed[i] and (q1 == qubit_here or q2 == qubit_here))
    degree = G_5q.degree(pos)
    print(f"  pos{pos:<2} q{qubit_here} (one-hot)   {'Yes' if in_front else 'No':<14} {future:<14} {degree:<8}")

print(f"\n  Edge list (connections between nodes):")
print(f"  {edges_5q}")
print(f"\n  The GNN takes: {N} nodes × (5 one-hot + 3 features) = {N*8} numbers")
print(f"  Plus the edge list telling it which nodes are connected.")
print(f"  Same GNN works for ANY topology — just change the edge list.")


print("\n" + "=" * 70)
print("PART 7: THE ACTION AND REWARD")
print("=" * 70)

print(f"\n  Available actions right now:")
for i, (p1, p2) in enumerate(edges_5q):
    lq1 = mapping.index(p1)
    lq2 = mapping.index(p2)
    print(f"    Action {i}: SWAP(pos{p1}, pos{p2}) — swaps q{lq1} and q{lq2}")
print(f"    Action {len(edges_5q)}: EXECUTE all routable gates")

print(f"\n  Let's say the agent picks Action 2: SWAP(pos2, pos3)")
new_mapping = mapping.copy()
p1, p2 = 2, 3
lq1 = new_mapping.index(p1)
lq2 = new_mapping.index(p2)
new_mapping[lq1], new_mapping[lq2] = new_mapping[lq2], new_mapping[lq1]

print(f"    Before: q{lq1}→pos{p1}, q{lq2}→pos{p2}")
print(f"    After:  q{lq1}→pos{p2}, q{lq2}→pos{p1}")
print(f"    Full mapping: {['q'+str(q)+'→pos'+str(p) for q, p in enumerate(new_mapping)]}")
print(f"\n    Reward: -1 (one SWAP inserted = 3 extra noisy CNOTs on real hardware)")

# Check if anything became routable
print(f"\n  After this SWAP, check front layer again:")
for gi in front_layer:
    q1, q2 = example_gates[gi]
    p1, p2 = new_mapping[q1], new_mapping[q2]
    dist = dist_5q[p1][p2]
    print(f"    Gate {gi}: q{q1}(pos{p1}) ↔ q{q2}(pos{p2}) → distance={dist} {'→ ROUTABLE!' if dist==1 else '→ still blocked'}")


print("\n" + "=" * 70)
print("PART 8: FULL EPISODE SUMMARY")
print("=" * 70)
print(f"""
  One complete RL episode:
  
  1. LOAD circuit (from QASMBench or random generation)
     → Gives us the list of gates and their dependencies
     
  2. SET topology (e.g., 27-qubit Heavy-Hex)
     → Gives us the hardware graph and distance matrix
     
  3. SET initial mapping (via SABRE placement or trivial)
     → Tells us where each logical qubit starts on the chip
     
  4. REPEAT until all gates executed:
     a. Compute front layer (which gates are ready)
     b. Build state (flat vector / matrix / graph)
     c. Agent picks action (SWAP or EXECUTE)
     d. Update mapping if SWAP, execute gates if EXECUTE
     e. Compute reward (-1 per SWAP, +0.1 per gate, +5 on completion)
     
  5. RECORD total SWAPs → this is what we minimize
  
  Training: thousands of episodes with random circuits
  Testing:  QASMBench circuits, compare SWAP count vs SABRE
""")

# Save a visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: 5-qubit topology
ax = axes[0]
pos_5q = {0: (0, 0), 1: (1, 0), 2: (2, 0), 3: (3, 0), 4: (4, 0)}
nx.draw(G_5q, pos_5q, ax=ax, with_labels=True, node_color='#6c63ff',
        node_size=600, font_color='white', font_weight='bold',
        edge_color='#475569', width=2)
ax.set_title("5-Qubit Linear Chain\n(4 edges = 4 possible SWAPs)", fontsize=12)

# Plot 2: 27-qubit topology
ax = axes[1]
pos_27q = nx.kamada_kawai_layout(G_27q)
nx.draw(G_27q, pos_27q, ax=ax, with_labels=True, node_color='#6c63ff',
        node_size=200, font_size=7, font_color='white', font_weight='bold',
        edge_color='#475569', width=1)
ax.set_title(f"27-Qubit IBM Heavy-Hex\n({len(edges_27q)} edges = {len(edges_27q)} possible SWAPs)", fontsize=12)

# Plot 3: Distance heatmap for 27q
ax = axes[2]
num_nodes_27 = len(G_27q.nodes())
sorted_nodes = sorted(G_27q.nodes())
dist_matrix_27 = np.zeros((num_nodes_27, num_nodes_27))
for i_idx, i in enumerate(sorted_nodes):
    for j_idx, j in enumerate(sorted_nodes):
        dist_matrix_27[i_idx][j_idx] = dist_27q[i][j]
im = ax.imshow(dist_matrix_27, cmap='viridis')
ax.set_title(f"{num_nodes_27}-Qubit IBM Heavy-Hex\nMax distance = {int(dist_matrix_27.max())} hops", fontsize=12)
ax.set_xlabel("Physical position")
ax.set_ylabel("Physical position")
plt.colorbar(im, ax=ax, label="Hops")

plt.tight_layout()
plt.savefig("/home/claude/data_exploration.png", dpi=150, bbox_inches='tight')
print("  Visualization saved to data_exploration.png")

print("\n" + "=" * 70)
print("DONE. You now understand the complete data pipeline.")
print("=" * 70)