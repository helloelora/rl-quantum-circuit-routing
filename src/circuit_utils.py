"""
Circuit utilities for quantum circuit routing.

Handles loading circuits, extracting two-qubit gates, building the dependency
DAG, computing front layers, and computing DAG depths.
"""

import numpy as np
import networkx as nx
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.circuit.random import random_circuit
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


def load_circuit(path):
    """Load a quantum circuit from a QASM file."""
    return QuantumCircuit.from_qasm_file(path)


def generate_random_circuit(num_qubits, depth, seed=None):
    """Generate a random quantum circuit for training."""
    return random_circuit(num_qubits, depth, max_operands=2, seed=seed)


def extract_two_qubit_gates(circuit):
    """
    Extract two-qubit gates from a circuit in dependency order.

    Returns:
        list of tuples: [(q_a, q_b), ...] where q_a, q_b are logical qubit indices.
        The list is in topological (dependency) order.
    """
    dag = circuit_to_dag(circuit)
    gates = []
    for node in dag.topological_op_nodes():
        if len(node.qargs) == 2:
            q_a = circuit.qubits.index(node.qargs[0])
            q_b = circuit.qubits.index(node.qargs[1])
            gates.append((q_a, q_b))
    return gates


def build_dependency_graph(gates):
    """
    Build a dependency graph for a list of two-qubit gates.

    A gate g_b depends on gate g_a if:
    - g_a appears earlier in the gate list than g_b
    - They share at least one qubit
    - g_a is the most recent earlier gate on that shared qubit

    Returns:
        predecessors: dict mapping gate_index -> set of direct predecessor gate indices
        successors: dict mapping gate_index -> set of direct successor gate indices
    """
    n_gates = len(gates)
    predecessors = {i: set() for i in range(n_gates)}
    successors = {i: set() for i in range(n_gates)}

    # For each qubit, track the most recent gate that used it
    last_gate_on_qubit = {}

    for i, (q_a, q_b) in enumerate(gates):
        for q in (q_a, q_b):
            if q in last_gate_on_qubit:
                pred = last_gate_on_qubit[q]
                predecessors[i].add(pred)
                successors[pred].add(i)
            last_gate_on_qubit[q] = i

    return predecessors, successors


def compute_front_layer(gates, executed, predecessors):
    """
    Compute the front layer: gates whose dependencies are all satisfied.

    A gate is in the front layer if:
    - It has not been executed
    - All its predecessors have been executed

    Args:
        gates: list of (q_a, q_b) tuples
        executed: set of gate indices that have been executed
        predecessors: dict mapping gate_index -> set of predecessor indices

    Returns:
        list of gate indices in the front layer
    """
    front = []
    for i in range(len(gates)):
        if i in executed:
            continue
        if all(pred in executed for pred in predecessors[i]):
            front.append(i)
    return front


def compute_dag_depths(gates, executed, predecessors, successors):
    """
    Compute the depth of each remaining gate in the remaining DAG.

    Depth = topological level. Gates in the front layer have depth 0.
    For any other gate: depth = max(depth of all remaining predecessors) + 1.

    Uses BFS level-by-level from the front layer.

    Args:
        gates: list of (q_a, q_b) tuples
        executed: set of executed gate indices
        predecessors: dict mapping gate_index -> set of predecessor indices
        successors: dict mapping gate_index -> set of successor indices

    Returns:
        dict mapping gate_index -> depth (only for remaining gates)
    """
    remaining = set(range(len(gates))) - executed
    depths = {}

    # Front layer = remaining gates whose predecessors are all executed
    current_level = []
    for i in remaining:
        remaining_preds = predecessors[i] - executed
        if not remaining_preds:
            current_level.append(i)

    d = 0
    while current_level:
        for gate_idx in current_level:
            depths[gate_idx] = d

        next_level_set = set()
        for gate_idx in current_level:
            for succ in successors[gate_idx]:
                if succ in executed or succ in depths or succ in next_level_set:
                    continue
                # Check if all remaining predecessors of succ have depths assigned
                remaining_preds = predecessors[succ] - executed
                if all(p in depths for p in remaining_preds):
                    next_level_set.add(succ)

        current_level = list(next_level_set)
        d += 1

    return depths


def build_coupling_graph(coupling_map):
    """
    Build a NetworkX graph and compute all-pairs shortest path distances.

    Args:
        coupling_map: Qiskit CouplingMap

    Returns:
        edges: sorted list of undirected edges as (i, j) tuples
        distance_matrix: 2D numpy array where [i][j] = shortest path length
        graph: NetworkX Graph
    """
    raw_edges = coupling_map.get_edges()
    edges = sorted(set(tuple(sorted(e)) for e in raw_edges))

    n = coupling_map.size()
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(edges)

    dist_dict = dict(nx.all_pairs_shortest_path_length(graph))
    distance_matrix = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(n):
            distance_matrix[i][j] = dist_dict[i][j]

    return edges, distance_matrix, graph


def get_coupling_map(topology_name):
    """
    Get a CouplingMap for a named topology.

    Supported formats:
        Named:      'heavy_hex_19', 'heavy_hex_27'
        Linear:     'linear_N'      (e.g. 'linear_5', 'linear_10')
        Ring:       'ring_N'        (e.g. 'ring_5', 'ring_10')
        Grid:       'grid_RxC'      (e.g. 'grid_3x3', 'grid_4x4')

    Returns:
        CouplingMap
    """
    if topology_name == 'heavy_hex_19':
        return CouplingMap.from_heavy_hex(3)
    elif topology_name == 'heavy_hex_27':
        return CouplingMap.from_heavy_hex(5)
    elif topology_name.startswith('linear_'):
        n = int(topology_name.split('_')[1])
        return CouplingMap.from_line(n)
    elif topology_name.startswith('ring_'):
        n = int(topology_name.split('_')[1])
        return CouplingMap.from_ring(n)
    elif topology_name.startswith('grid_'):
        dims = topology_name.split('_')[1]
        r, c = dims.split('x')
        return CouplingMap.from_grid(int(r), int(c))
    else:
        raise ValueError(f"Unknown topology: {topology_name}")


def get_sabre_initial_mapping(circuit, coupling_map):
    """
    Run SABRE layout to get an initial qubit-to-position mapping.

    Args:
        circuit: Qiskit QuantumCircuit
        coupling_map: Qiskit CouplingMap

    Returns:
        list: mapping where mapping[logical_qubit] = physical_position
    """
    pm = generate_preset_pass_manager(
        optimization_level=1,
        coupling_map=coupling_map,
    )
    transpiled = pm.run(circuit)
    init_layout = transpiled.layout.initial_layout
    mapping = [init_layout[circuit.qubits[i]] for i in range(circuit.num_qubits)]
    return mapping


def get_sabre_swap_count(circuit, coupling_map):
    """
    Run SABRE on a circuit and return the number of SWAPs inserted.

    Args:
        circuit: Qiskit QuantumCircuit
        coupling_map: Qiskit CouplingMap

    Returns:
        int: number of SWAPs inserted by SABRE
    """
    pm = generate_preset_pass_manager(
        optimization_level=1,
        coupling_map=coupling_map,
    )
    transpiled = pm.run(circuit)
    ops = transpiled.count_ops()
    return ops.get('swap', 0)
