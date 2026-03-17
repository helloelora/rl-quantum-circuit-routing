from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from qiskit import QuantumCircuit

from topology import build_falcon_27_graph, compute_distance_matrix


class QuantumRoutingEnv(gym.Env):
    """
    Minimal Gymnasium environment for RL-based quantum circuit routing.

    Action space:
    - 0..E-1: apply a SWAP on hardware edge i
    - E: execute all currently executable front-layer gates
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        qasm_files: list[str] | None = None,
        lookahead_k: int = 4,
        max_steps: int = 2048,
    ) -> None:
        super().__init__()
        self.graph = build_falcon_27_graph()
        self.distance_matrix = compute_distance_matrix(self.graph)
        self.edges = list(self.graph.edges())
        self.num_qubits = self.graph.number_of_nodes()

        self.lookahead_k = lookahead_k
        self.max_steps = max_steps
        self.qasm_files = qasm_files or []
        self._episode_idx = 0

        # Action E means "Execute".
        self.execute_action = len(self.edges)
        self.action_space = spaces.Discrete(len(self.edges) + 1)

        # Observation = mapping + lookahead gate ids + flattened distances + progress.
        obs_size = self.num_qubits + (2 * self.lookahead_k) + self.num_qubits * self.num_qubits + 1
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_size,),
            dtype=np.float32,
        )

        self.mapping: np.ndarray = np.arange(self.num_qubits, dtype=np.int32)
        self.inverse_mapping: np.ndarray = np.arange(self.num_qubits, dtype=np.int32)
        self.pending_twoq_gates: list[tuple[int, int]] = []
        self.step_count = 0
        self.executed_gates = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.mapping = np.arange(self.num_qubits, dtype=np.int32)
        self.inverse_mapping = np.arange(self.num_qubits, dtype=np.int32)
        self.step_count = 0
        self.executed_gates = 0
        self.pending_twoq_gates = self._load_episode_gates()
        return self._get_obs(), self._get_info()

    def step(self, action: int):
        reward = -0.01  # time penalty
        self.step_count += 1

        if action == self.execute_action:
            executed_now = self._execute_ready_gates()
            reward += 0.1 * executed_now
            if executed_now == 0:
                reward -= 0.2
        else:
            u, v = self.edges[action]
            self._swap_physical_qubits(u, v)
            reward -= 1.0

        terminated = len(self.pending_twoq_gates) == 0
        truncated = self.step_count >= self.max_steps

        if terminated:
            reward += 5.0

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        return None

    def valid_actions(self) -> np.ndarray:
        """Action mask helper: SWAPs are always valid; execute valid if at least one gate is executable."""
        mask = np.ones(self.action_space.n, dtype=bool)
        mask[self.execute_action] = self._has_executable_gate()
        return mask

    def _load_episode_gates(self) -> list[tuple[int, int]]:
        if not self.qasm_files:
            # Tiny fallback workload when benchmarks folder is empty.
            return [(0, 5), (1, 4), (2, 3), (6, 9)]

        qasm_path = Path(self.qasm_files[self._episode_idx % len(self.qasm_files)])
        self._episode_idx += 1

        circuit = QuantumCircuit.from_qasm_file(str(qasm_path))
        twoq_gates: list[tuple[int, int]] = []
        for instruction in circuit.data:
            qargs = instruction.qubits
            if len(qargs) != 2:
                continue
            q0 = circuit.find_bit(qargs[0]).index
            q1 = circuit.find_bit(qargs[1]).index
            twoq_gates.append((q0, q1))
        return twoq_gates

    def _has_executable_gate(self) -> bool:
        if not self.pending_twoq_gates:
            return False
        lq0, lq1 = self.pending_twoq_gates[0]
        pq0 = int(self.mapping[lq0])
        pq1 = int(self.mapping[lq1])
        return self.graph.has_edge(pq0, pq1)

    def _execute_ready_gates(self) -> int:
        executed = 0
        while self.pending_twoq_gates:
            lq0, lq1 = self.pending_twoq_gates[0]
            pq0 = int(self.mapping[lq0])
            pq1 = int(self.mapping[lq1])
            if not self.graph.has_edge(pq0, pq1):
                break
            self.pending_twoq_gates.pop(0)
            executed += 1
            self.executed_gates += 1
        return executed

    def _swap_physical_qubits(self, physical_u: int, physical_v: int) -> None:
        logical_u = int(self.inverse_mapping[physical_u])
        logical_v = int(self.inverse_mapping[physical_v])

        self.mapping[logical_u], self.mapping[logical_v] = (
            self.mapping[logical_v],
            self.mapping[logical_u],
        )
        self.inverse_mapping[physical_u], self.inverse_mapping[physical_v] = (
            self.inverse_mapping[physical_v],
            self.inverse_mapping[physical_u],
        )

    def _get_obs(self) -> np.ndarray:
        mapping_part = self.mapping.astype(np.float32) / max(self.num_qubits - 1, 1)

        lookahead = np.full(2 * self.lookahead_k, -1.0, dtype=np.float32)
        for i, (q0, q1) in enumerate(self.pending_twoq_gates[: self.lookahead_k]):
            lookahead[2 * i] = q0 / max(self.num_qubits - 1, 1)
            lookahead[2 * i + 1] = q1 / max(self.num_qubits - 1, 1)

        max_distance = np.max(self.distance_matrix)
        distance_part = self.distance_matrix.flatten().astype(np.float32) / max(max_distance, 1.0)

        progress = np.array([self.step_count / max(self.max_steps, 1)], dtype=np.float32)

        return np.concatenate([mapping_part, lookahead, distance_part, progress], dtype=np.float32)

    def _get_info(self) -> dict[str, Any]:
        return {
            "remaining_twoq_gates": len(self.pending_twoq_gates),
            "executed_twoq_gates": self.executed_gates,
            "valid_actions": self.valid_actions(),
        }
