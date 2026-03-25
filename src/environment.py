"""
Gymnasium environment for RL-based quantum circuit routing.

Implements the QubitRoutingEnv with:
- 3-channel N×N state (adjacency, mapping, depth-decayed gate demand)
- SWAP-only actions with automatic gate execution
- AlphaRouter-style reward with Pozzi distance shaping
- Multi-topology support: can switch topology each episode
- Configurable matrix size N (padded) for fixed CNN input shape
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

try:
    from .circuit_utils import (
        extract_two_qubit_gates,
        build_dependency_graph,
        compute_front_layer,
        compute_dag_depths,
        build_coupling_graph,
        generate_random_circuit,
        get_coupling_map,
        get_sabre_initial_mapping,
    )
except ImportError:
    from circuit_utils import (
        extract_two_qubit_gates,
        build_dependency_graph,
        compute_front_layer,
        compute_dag_depths,
        build_coupling_graph,
        generate_random_circuit,
        get_coupling_map,
        get_sabre_initial_mapping,
    )


class QubitRoutingEnv(gym.Env):
    """
    Quantum circuit routing environment.

    The agent inserts SWAP gates on hardware edges to make all two-qubit
    gates in a quantum circuit executable. After each SWAP, routable
    front-layer gates execute automatically.

    Supports two modes:
    - Single-topology: pass one topology_name or coupling_map.
    - Multi-topology:  pass a list of topology names via `topologies`.
      Each reset() randomly picks one topology from the list.

    In both cases the observation shape is always (3, N, N) where N =
    matrix_size, and the action space is Discrete(max_edges) where
    max_edges is the largest edge count across all topologies.

    State: (3, N, N) float32 array
        Channel 0: Hardware adjacency matrix (binary, constant per episode)
        Channel 1: Qubit assignment / mapping permutation matrix (binary)
        Channel 2: Depth-decayed gate demand (continuous [0, 1])

    Actions: Discrete(max_edges) — one action per hardware edge (SWAP).
             Actions beyond the current topology's edge count are invalid.

    Reward per step:
        r_t = gate_reward_coeff * gates_auto_executed
              + distance_reward_coeff * delta_distance
              + step_penalty
              + reverse_swap_penalty (if immediate backtracking)
              + capped(repeat_swap_penalty_coeff * (same_edge_streak - 1))
              + capped(no_progress_penalty_coeff * no_progress_streak)
              [+ completion_bonus if done]
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        coupling_map=None,
        topology_name="heavy_hex_19",
        topologies=None,
        topology_sampling_weights=None,
        circuit=None,
        num_qubits=None,
        circuit_depth=20,
        max_steps=500,
        gamma_decay=0.5,
        distance_reward_coeff=0.01,
        completion_bonus=8.0,
        timeout_penalty=-25.0,
        gate_reward_coeff=1.0,
        step_penalty=-0.05,
        reverse_swap_penalty=-0.2,
        repeat_swap_penalty_coeff=-0.1,
        repeat_swap_penalty_cap=-2.0,
        no_progress_penalty_coeff=-0.03,
        no_progress_penalty_cap=-1.5,
        no_progress_terminate_streak=0,
        max_steps_per_two_qubit_gate=0.0,
        max_steps_min=0,
        max_steps_max=0,
        min_two_qubit_gates=0,
        circuit_generation_attempts=16,
        matrix_size=27,
        initial_mapping_strategy="mixed",
        seed=None,
    ):
        """
        Args:
            coupling_map: Qiskit CouplingMap. If None, built from topology_name.
                          Ignored if topologies is provided.
            topology_name: Name of topology (used if coupling_map and
                           topologies are both None).
            topologies: List of topology names for multi-topology training.
                        e.g. ["heavy_hex_19", "grid_3x3", "linear_5"].
                        Each reset() randomly picks one from this list.
            topology_sampling_weights: Optional list of non-negative weights
                         (same length as `topologies`) used for weighted
                         topology sampling at reset(). If None, uniform sampling
                         is used.
            circuit: Qiskit QuantumCircuit to route. If None, a random one is
                     generated each reset().
            num_qubits: Number of qubits for random circuits. Defaults to
                        the current episode's topology qubit count.
            circuit_depth: Depth of random circuits (ignored if circuit given).
            max_steps: Maximum SWAP steps per episode before timeout.
            gamma_decay: Decay factor for depth-weighted gate demand channel.
            distance_reward_coeff: Coefficient for distance reduction shaping.
            completion_bonus: Reward bonus when all gates are routed.
            timeout_penalty: Penalty when max_steps is reached.
            gate_reward_coeff: Multiplier for executed front-layer gates.
            step_penalty: Constant per-step penalty to discourage long routes.
            reverse_swap_penalty: Extra penalty if the same SWAP edge is
                         selected in consecutive steps.
            repeat_swap_penalty_coeff: Progressive penalty coefficient for
                         consecutive reuse of the same physical SWAP edge.
                         Applied as coeff * (same_edge_streak - 1).
            repeat_swap_penalty_cap: Lower bound (negative cap) for the
                         progressive repeated-edge penalty.
            no_progress_penalty_coeff: Progressive penalty coefficient applied
                         when no new gate is executed at a step.
            no_progress_penalty_cap: Lower bound (negative cap) for the
                         progressive no-progress penalty.
            no_progress_terminate_streak: If > 0, truncate episode early when
                         no gate has been executed for this many consecutive
                         steps. Timeout penalty is applied.
            max_steps_per_two_qubit_gate: Optional dynamic episode cap.
                         If > 0, per-episode max steps are computed as:
                         ceil(num_2q_gates * this_factor), then clamped with
                         max_steps_min / max_steps_max (when > 0).
            max_steps_min: Optional lower clamp for dynamic max steps.
            max_steps_max: Optional upper clamp for dynamic max steps.
            min_two_qubit_gates: Minimum number of 2-qubit gates required
                         for randomly generated circuits.
            circuit_generation_attempts: Number of random samples to try to
                         satisfy `min_two_qubit_gates`.
            matrix_size: Side length N of the padded state matrices.
                         Must be >= largest topology's qubit count. Default 27.
            initial_mapping_strategy: How to set the initial qubit-to-position
                         mapping each episode. One of:
                         - "random": random permutation
                         - "identity": logical qubit q → physical position q
                         - "sabre": use SABRE's layout pass (realistic, slower)
                         - "mixed": 80% random, 20% SABRE (default)
            seed: Random seed for reproducibility.
        """
        super().__init__()

        self.N = matrix_size
        self.initial_mapping_strategy = initial_mapping_strategy

        # --- Build topology data for all topologies ---
        # Each topology is stored as a dict with: coupling_map, n_physical,
        # edges, distance_matrix, graph, adjacency_channel, num_edges
        self._topologies = []

        if topologies is not None:
            # Multi-topology mode
            topology_names = topologies
        elif coupling_map is not None:
            # Single topology from explicit coupling_map
            topology_names = None
        else:
            # Single topology from name
            topology_names = [topology_name]

        if topology_names is not None:
            for topo_name in topology_names:
                cmap = get_coupling_map(topo_name)
                self._topologies.append(self._build_topology_data(cmap, topo_name))
        else:
            self._topologies.append(
                self._build_topology_data(coupling_map, "custom")
            )
        self._topology_sampling_probs = self._build_topology_sampling_probs(
            topology_sampling_weights
        )

        # Validate that all topologies fit in the matrix
        max_physical = max(t["n_physical"] for t in self._topologies)
        assert self.N >= max_physical, (
            f"matrix_size ({self.N}) must be >= largest topology's qubit "
            f"count ({max_physical})"
        )

        # Action space = max edges across all topologies
        # (actions beyond current topology's edge count are masked)
        self.max_edges = max(t["num_edges"] for t in self._topologies)

        # --- Spaces ---
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(3, self.N, self.N), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.max_edges)

        # --- Episode parameters ---
        self.fixed_circuit = circuit
        self._num_qubits_override = num_qubits
        self.circuit_depth = circuit_depth
        self.max_steps = max_steps
        self.gamma_decay = gamma_decay
        self.distance_reward_coeff = distance_reward_coeff
        self.completion_bonus = completion_bonus
        self.timeout_penalty = timeout_penalty
        self.gate_reward_coeff = gate_reward_coeff
        self.step_penalty = step_penalty
        self.reverse_swap_penalty = reverse_swap_penalty
        self.repeat_swap_penalty_coeff = repeat_swap_penalty_coeff
        self.repeat_swap_penalty_cap = repeat_swap_penalty_cap
        self.no_progress_penalty_coeff = no_progress_penalty_coeff
        self.no_progress_penalty_cap = no_progress_penalty_cap
        self.no_progress_terminate_streak = max(0, int(no_progress_terminate_streak))
        self.max_steps_per_two_qubit_gate = max_steps_per_two_qubit_gate
        self.max_steps_min = max_steps_min
        self.max_steps_max = max_steps_max
        self.min_two_qubit_gates = max(0, int(min_two_qubit_gates))
        self.circuit_generation_attempts = max(1, int(circuit_generation_attempts))
        self.norm_factor = 1.0 / (1.0 - gamma_decay)  # e.g., 2.0 for γ=0.5

        # --- RNG ---
        self._rng = np.random.default_rng(seed)

        # --- Current episode topology (set in reset) ---
        self._current_topo = None
        self.gates = None
        self.predecessors = None
        self.successors = None
        self.mapping = None          # mapping[logical_qubit] = physical_position
        self.reverse_mapping = None  # reverse_mapping[physical_pos] = logical_qubit
        self.executed = None
        self.step_count = 0
        self.total_swaps = 0
        self.total_gates_executed = 0
        self.n_gates = 0
        self._last_action = None
        self._last_edge = None
        self._same_edge_streak = 0
        self._no_progress_streak = 0
        self._episode_max_steps = int(max_steps)

    def _build_topology_data(self, coupling_map, name):
        """Pre-compute all static data for a topology."""
        n_physical = coupling_map.size()
        edges, distance_matrix, graph = build_coupling_graph(coupling_map)
        num_edges = len(edges)

        adjacency_channel = np.zeros((self.N, self.N), dtype=np.float32)
        for i, j in edges:
            adjacency_channel[i, j] = 1.0
            adjacency_channel[j, i] = 1.0

        return {
            "name": name,
            "coupling_map": coupling_map,
            "n_physical": n_physical,
            "edges": edges,
            "distance_matrix": distance_matrix,
            "graph": graph,
            "adjacency_channel": adjacency_channel,
            "num_edges": num_edges,
        }

    def _build_topology_sampling_probs(self, topology_sampling_weights):
        """Build normalized probabilities for topology sampling at reset()."""
        n_topos = len(self._topologies)
        if n_topos <= 0:
            raise ValueError("At least one topology must be provided.")

        if topology_sampling_weights is None:
            return np.full(n_topos, 1.0 / n_topos, dtype=np.float64)

        weights = np.asarray(topology_sampling_weights, dtype=np.float64).reshape(-1)
        if weights.size != n_topos:
            raise ValueError(
                "topology_sampling_weights must match number of topologies: "
                f"got {weights.size}, expected {n_topos}."
            )
        if np.any(weights < 0):
            raise ValueError("topology_sampling_weights must be non-negative.")

        total = float(np.sum(weights))
        if total <= 0:
            raise ValueError("topology_sampling_weights must sum to a positive value.")
        return weights / total

    def _resolve_episode_max_steps(self):
        """Compute per-episode max steps from fixed/dynamic settings."""
        if self.max_steps_per_two_qubit_gate <= 0:
            return int(self.max_steps)

        computed = int(np.ceil(self.n_gates * self.max_steps_per_two_qubit_gate))
        lower = int(self.max_steps_min) if self.max_steps_min > 0 else 1
        upper = int(self.max_steps_max) if self.max_steps_max > 0 else int(self.max_steps)
        if upper < lower:
            upper = lower
        return int(min(max(computed, lower), upper))

    def reset(self, seed=None, options=None):
        """
        Reset the environment for a new episode.

        Picks a topology (randomly if multi-topology), generates a new
        circuit, sets the initial mapping, and auto-executes any
        initially routable gates.

        Args:
            seed: Optional seed override.
            options: Optional dict. Can contain:
                - "circuit": a specific QuantumCircuit to route
                - "initial_mapping": list where mapping[q] = p
                - "topology_index": int, force a specific topology

        Returns:
            observation: (3, N, N) state array
            info: dict with episode metadata
        """
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # --- Pick topology for this episode ---
        if options and "topology_index" in options:
            topo_idx = int(options["topology_index"])
        else:
            topo_idx = int(
                self._rng.choice(
                    len(self._topologies),
                    p=self._topology_sampling_probs,
                )
            )
        if not (0 <= topo_idx < len(self._topologies)):
            raise ValueError(
                f"Invalid topology_index={topo_idx}; expected in [0, {len(self._topologies)-1}]."
            )
        self._current_topo = self._topologies[topo_idx]

        n_physical = self._current_topo["n_physical"]

        # --- Get circuit ---
        if options and "circuit" in options:
            circuit = options["circuit"]
        elif self.fixed_circuit is not None:
            circuit = self.fixed_circuit
        else:
            num_qubits = (
                self._num_qubits_override
                if self._num_qubits_override
                else n_physical
            )
            circuit = generate_random_circuit(
                num_qubits,
                self.circuit_depth,
                seed=int(self._rng.integers(0, 2**31)),
                min_two_qubit_gates=self.min_two_qubit_gates,
                max_attempts=self.circuit_generation_attempts,
            )

        # --- Extract gates and build DAG ---
        self.gates = extract_two_qubit_gates(circuit)
        self.n_gates = len(self.gates)

        if self.n_gates == 0:
            self.predecessors = {}
            self.successors = {}
            self.executed = set()
            self.mapping = list(range(n_physical))
            self.reverse_mapping = list(range(n_physical))
            self.step_count = 0
            self.total_swaps = 0
            self.total_gates_executed = 0
            self._last_action = None
            self._last_edge = None
            self._same_edge_streak = 0
            self._no_progress_streak = 0
            self._episode_max_steps = 0
            obs = self._compute_state()
            return obs, self._get_info(done=True)

        self.predecessors, self.successors = build_dependency_graph(self.gates)
        self.executed = set()

        # --- Initial mapping ---
        if options and "initial_mapping" in options:
            self.mapping = list(options["initial_mapping"])
        else:
            strategy = self.initial_mapping_strategy
            if strategy == "mixed":
                # 80% random, 20% SABRE
                strategy = (
                    "sabre" if self._rng.random() < 0.2 else "random"
                )

            if strategy == "sabre":
                sabre_mapping = get_sabre_initial_mapping(
                    circuit, self._current_topo["coupling_map"]
                )
                self.mapping = [int(x) for x in sabre_mapping]
            elif strategy == "random":
                self.mapping = [int(x) for x in self._rng.permutation(n_physical)]
            else:
                # identity
                self.mapping = list(range(n_physical))

        self.reverse_mapping = [0] * n_physical
        for q, p in enumerate(self.mapping):
            self.reverse_mapping[p] = q

        # --- Auto-execute any initially routable gates ---
        self.step_count = 0
        self.total_swaps = 0
        self.total_gates_executed = 0
        self._last_action = None
        self._last_edge = None
        self._same_edge_streak = 0
        self._no_progress_streak = 0
        self._episode_max_steps = self._resolve_episode_max_steps()
        self._auto_execute_gates()

        obs = self._compute_state()
        info = self._get_info(done=self._is_done())
        return obs, info

    def step(self, action):
        """
        Execute one SWAP action.

        Args:
            action: int, index into the current topology's edge list.

        Returns:
            observation, reward, terminated, truncated, info
        """
        topo = self._current_topo
        assert 0 <= action < topo["num_edges"], (
            f"Invalid action {action}: current topology '{topo['name']}' "
            f"has {topo['num_edges']} edges (indices 0-{topo['num_edges']-1})"
        )

        # --- Compute distance sum BEFORE the SWAP (for shaping reward) ---
        dist_before = self._compute_front_layer_distance()
        was_immediate_backtrack = (
            self._last_action is not None and int(action) == int(self._last_action)
        )

        # --- Perform the SWAP ---
        p1, p2 = topo["edges"][action]
        edge_key = (min(int(p1), int(p2)), max(int(p1), int(p2)))
        if self._last_edge is not None and edge_key == self._last_edge:
            self._same_edge_streak += 1
        else:
            self._same_edge_streak = 1
        self._last_edge = edge_key

        q1 = self.reverse_mapping[p1]
        q2 = self.reverse_mapping[p2]

        self.mapping[q1] = p2
        self.mapping[q2] = p1
        self.reverse_mapping[p1] = q2
        self.reverse_mapping[p2] = q1

        self.step_count += 1
        self.total_swaps += 1
        self._last_action = int(action)

        # --- Auto-execute routable gates ---
        gates_executed = self._auto_execute_gates()

        # --- Compute distance sum AFTER (for shaping reward) ---
        dist_after = self._compute_front_layer_distance()
        delta_dist = dist_before - dist_after  # positive if qubits moved closer

        # --- Reward ---
        done = self._is_done()
        if gates_executed > 0 or done:
            self._no_progress_streak = 0
        else:
            self._no_progress_streak += 1

        reward = (
            self.gate_reward_coeff * gates_executed
            + self.distance_reward_coeff * delta_dist
            + self.step_penalty
        )
        if was_immediate_backtrack:
            reward += self.reverse_swap_penalty
        if self._same_edge_streak > 1:
            repeat_penalty = (
                self.repeat_swap_penalty_coeff * (self._same_edge_streak - 1)
            )
            reward += max(self.repeat_swap_penalty_cap, repeat_penalty)
        if self._no_progress_streak > 0:
            no_progress_penalty = (
                self.no_progress_penalty_coeff * self._no_progress_streak
            )
            reward += max(self.no_progress_penalty_cap, no_progress_penalty)
        if done:
            reward += self.completion_bonus

        # --- Truncation check ---
        truncated = False
        truncated_reason = ""
        if (
            not done
            and self.no_progress_terminate_streak > 0
            and self._no_progress_streak >= self.no_progress_terminate_streak
        ):
            truncated = True
            truncated_reason = "no_progress_streak"
            reward += self.timeout_penalty
        if not done and (not truncated) and self.step_count >= self._episode_max_steps:
            truncated = True
            truncated_reason = "max_steps"
            reward += self.timeout_penalty

        obs = self._compute_state()
        info = self._get_info(done=done)
        if truncated:
            info["truncated_reason"] = truncated_reason

        return obs, reward, done, truncated, info

    def _auto_execute_gates(self):
        """
        Execute all routable front-layer gates. Repeat until no more can execute.

        Returns:
            int: number of gates executed in this round.
        """
        adj = self._current_topo["adjacency_channel"]
        total_executed = 0
        while True:
            front = compute_front_layer(
                self.gates, self.executed, self.predecessors
            )
            executed_this_round = []
            for gate_idx in front:
                q_a, q_b = self.gates[gate_idx]
                p_a = self.mapping[q_a]
                p_b = self.mapping[q_b]
                if adj[p_a, p_b] == 1.0:
                    executed_this_round.append(gate_idx)

            if not executed_this_round:
                break

            for gate_idx in executed_this_round:
                self.executed.add(gate_idx)
            total_executed += len(executed_this_round)

        self.total_gates_executed += total_executed
        return total_executed

    def _compute_state(self):
        """
        Build the 3-channel N×N state observation.

        Channel 0: Adjacency (from current topology, padded)
        Channel 1: Mapping permutation matrix
        Channel 2: Depth-decayed gate demand
        """
        topo = self._current_topo
        state = np.zeros((3, self.N, self.N), dtype=np.float32)

        # Channel 0: Adjacency
        state[0] = topo["adjacency_channel"]

        # Channel 1: Mapping permutation matrix
        for q in range(topo["n_physical"]):
            p = self.mapping[q]
            state[1, q, p] = 1.0

        # Channel 2: Depth-decayed gate demand
        if self.gates and not self._is_done():
            depths = compute_dag_depths(
                self.gates, self.executed, self.predecessors, self.successors
            )
            for gate_idx, depth in depths.items():
                q_a, q_b = self.gates[gate_idx]
                p_a = self.mapping[q_a]
                p_b = self.mapping[q_b]
                value = self.gamma_decay ** depth
                state[2, p_a, p_b] += value
                state[2, p_b, p_a] += value

            # Normalize to [0, 1]
            state[2] /= self.norm_factor

        return state

    def _compute_front_layer_distance(self):
        """
        Sum of shortest-path distances for all front-layer gate qubit pairs.
        Used for the distance reduction shaping reward.
        """
        if self._is_done():
            return 0.0

        dist_matrix = self._current_topo["distance_matrix"]
        front = compute_front_layer(
            self.gates, self.executed, self.predecessors
        )
        total_dist = 0.0
        for gate_idx in front:
            q_a, q_b = self.gates[gate_idx]
            p_a = self.mapping[q_a]
            p_b = self.mapping[q_b]
            total_dist += dist_matrix[p_a, p_b]
        return total_dist

    def _is_done(self):
        """Check if all gates have been executed."""
        return len(self.executed) == self.n_gates

    def _get_info(self, done=False):
        """Build info dict for monitoring."""
        return {
            "step_count": self.step_count,
            "total_swaps": self.total_swaps,
            "total_gates_executed": self.total_gates_executed,
            "n_gates": self.n_gates,
            "remaining_gates": self.n_gates - len(self.executed),
            "done": done,
            "topology": self._current_topo["name"],
            "n_physical": self._current_topo["n_physical"],
            "num_edges": self._current_topo["num_edges"],
            "episode_max_steps": self._episode_max_steps,
            "same_edge_streak": self._same_edge_streak,
            "no_progress_streak": self._no_progress_streak,
            "no_progress_terminate_streak": self.no_progress_terminate_streak,
        }

    def get_action_mask(self):
        """
        Return a boolean mask over the action space.

        Actions for edges that exist in the current topology are True.
        Actions beyond the current topology's edge count are False
        (these correspond to edges that only exist in larger topologies).

        Returns:
            np.ndarray of shape (max_edges,), dtype bool
        """
        mask = np.zeros(self.max_edges, dtype=bool)
        num_edges = self._current_topo["num_edges"]
        mask[:num_edges] = True
        return mask

    def render(self):
        """Print a text summary of the current state."""
        topo = self._current_topo
        front = compute_front_layer(
            self.gates, self.executed, self.predecessors
        )
        print(f"Topology: {topo['name']} ({topo['n_physical']}q, "
              f"{topo['num_edges']} edges)")
        print(f"Step {self.step_count} | "
              f"SWAPs: {self.total_swaps} | "
              f"Executed: {len(self.executed)}/{self.n_gates} | "
              f"Front layer size: {len(front)}")
        print(f"Mapping: {self.mapping[:topo['n_physical']]}")
        if front:
            front_gates = [self.gates[i] for i in front]
            print(f"Front layer gates (logical): {front_gates}")
            front_physical = [
                (self.mapping[q_a], self.mapping[q_b])
                for q_a, q_b in front_gates
            ]
            print(f"Front layer gates (physical): {front_physical}")
