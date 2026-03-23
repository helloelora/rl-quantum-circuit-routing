"""
DQN agent for quantum circuit routing.

Implements a symmetry-aware Q-network with:
- Input: (3, N, N) state tensor
- Output: edge-level Q-values gathered from a symmetric N x N score map
- Actions: SWAP edges only, with topology-validity masking

The training/evaluation/tracing interface mirrors PPOAgent so existing
run logging and analysis files remain compatible.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .circuit_utils import (
        count_two_qubit_gates,
        generate_random_circuit,
        get_sabre_initial_mapping,
        get_sabre_swap_count,
    )
except ImportError:
    from circuit_utils import (
        count_two_qubit_gates,
        generate_random_circuit,
        get_sabre_initial_mapping,
        get_sabre_swap_count,
    )


@dataclass
class DQNConfig:
    total_timesteps: int = 200_000
    # Number of env steps grouped into one logging/update block.
    update_steps: int = 4096
    learning_rate: float = 1e-4
    gamma: float = 0.99
    replay_capacity: int = 100_000
    min_replay_size: int = 5_000
    batch_size: int = 128
    train_frequency_steps: int = 1
    gradient_steps: int = 1
    target_update_interval_steps: int = 2_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 200_000
    use_double_dqn: bool = True
    huber_delta: float = 1.0
    max_grad_norm: float = 0.5
    log_interval_updates: int = 5
    checkpoint_interval_updates: int = 25
    eval_interval_updates: int = 10
    eval_circuits_per_topology: int = 4
    eval_circuit_depth: int = 20
    eval_min_two_qubit_gates: int = 0
    eval_circuit_generation_attempts: int = 16
    # Chosen above training seed range [0, 2**31) to avoid overlap by design.
    eval_seed_base: int = 3_000_000_000
    # Periodic episode traces for behavior diagnostics during training.
    trace_interval_updates: int = 20
    trace_cases_per_topology: int = 1
    trace_max_steps: int = 500
    trace_dir_name: str = "traces"
    trace_alert_dom_threshold: float = 0.60
    trace_alert_backtrack_threshold: float = 0.50
    trace_alert_patience: int = 2
    # Reward shaping annealing across training.
    distance_reward_coeff_start: float = 0.03
    distance_reward_coeff_end: float = 0.015
    run_dir: str = ""
    seed: int = 42
    device: str = "cpu"


class SymmetricCNNQNetwork(nn.Module):
    """CNN Q-network with symmetry-aware edge scoring."""

    def __init__(self, matrix_size: int):
        super().__init__()
        self.matrix_size = matrix_size
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.q_map_head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (B, 3, N, N)
        Returns:
            q_map_sym: (B, N, N)
        """
        x = self.backbone(obs)
        q_map = self.q_map_head(x).squeeze(1)
        return 0.5 * (q_map + q_map.transpose(-1, -2))

    @staticmethod
    def gather_edge_q_values(
        q_map_sym: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gather edge-level Q-values from symmetric map.

        Args:
            q_map_sym: (B, N, N)
            edge_index: (B, A, 2)
        Returns:
            q_values: (B, A)
        """
        batch_size, num_actions, _ = edge_index.shape
        b_idx = torch.arange(batch_size, device=q_map_sym.device).unsqueeze(1)
        i_idx = edge_index[..., 0]
        j_idx = edge_index[..., 1]
        q_values = q_map_sym[b_idx, i_idx, j_idx]
        assert q_values.shape == (batch_size, num_actions)
        return q_values

    def get_q_values(
        self,
        obs: torch.Tensor,
        edge_index: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        q_map_sym = self.forward(obs)
        q_values = self.gather_edge_q_values(q_map_sym, edge_index)
        return q_values.masked_fill(~action_mask, -1e9)


class ReplayBuffer:
    """Memory-efficient ring buffer for DQN transitions."""

    def __init__(
        self,
        capacity: int,
        obs_shape: Tuple[int, ...],
        max_actions: int,
        seed: int,
    ) -> None:
        self.capacity = int(capacity)
        self.obs_shape = tuple(int(x) for x in obs_shape)
        self.max_actions = int(max_actions)
        self.rng = np.random.default_rng(seed)

        self.obs = np.empty((self.capacity, *self.obs_shape), dtype=np.float16)
        self.edge_index = np.empty((self.capacity, self.max_actions, 2), dtype=np.int16)
        self.action_mask = np.empty((self.capacity, self.max_actions), dtype=bool)
        self.actions = np.empty((self.capacity,), dtype=np.int64)
        self.rewards = np.empty((self.capacity,), dtype=np.float32)
        self.dones = np.empty((self.capacity,), dtype=np.float32)
        self.next_obs = np.empty((self.capacity, *self.obs_shape), dtype=np.float16)
        self.next_edge_index = np.empty((self.capacity, self.max_actions, 2), dtype=np.int16)
        self.next_action_mask = np.empty((self.capacity, self.max_actions), dtype=bool)

        self.size = 0
        self.pos = 0

    def add(
        self,
        obs: np.ndarray,
        edge_index: np.ndarray,
        action_mask: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        next_obs: np.ndarray,
        next_edge_index: np.ndarray,
        next_action_mask: np.ndarray,
    ) -> None:
        i = self.pos
        self.obs[i] = obs.astype(np.float16, copy=False)
        self.edge_index[i] = edge_index.astype(np.int16, copy=False)
        self.action_mask[i] = action_mask.astype(bool, copy=False)
        self.actions[i] = int(action)
        self.rewards[i] = float(reward)
        self.dones[i] = float(done)
        self.next_obs[i] = next_obs.astype(np.float16, copy=False)
        self.next_edge_index[i] = next_edge_index.astype(np.int16, copy=False)
        self.next_action_mask[i] = next_action_mask.astype(bool, copy=False)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        idx = self.rng.integers(0, self.size, size=int(batch_size))
        return {
            "obs": self.obs[idx],
            "edge_index": self.edge_index[idx],
            "action_mask": self.action_mask[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "dones": self.dones[idx],
            "next_obs": self.next_obs[idx],
            "next_edge_index": self.next_edge_index[idx],
            "next_action_mask": self.next_action_mask[idx],
        }


class DQNAgent:
    """Off-policy DQN trainer for QubitRoutingEnv."""

    def __init__(self, env, config: DQNConfig, model_state_dict=None):
        self.env = env
        self.cfg = config
        self.device = torch.device(config.device)
        self._set_seed(config.seed)

        matrix_size = int(self._unwrap_env().N)
        self.max_actions = int(self.env.action_space.n)
        self.q_net = SymmetricCNNQNetwork(matrix_size).to(self.device)
        self.target_net = SymmetricCNNQNetwork(matrix_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=config.learning_rate)

        if model_state_dict is not None:
            self.q_net.load_state_dict(model_state_dict)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.replay = ReplayBuffer(
            capacity=config.replay_capacity,
            obs_shape=(3, matrix_size, matrix_size),
            max_actions=self.max_actions,
            seed=config.seed,
        )

        self.run_dir = Path(config.run_dir) if config.run_dir else None
        self.metrics_path = None
        self.best_train_mean_episode_return = -np.inf
        self.best_eval_key = (-np.inf, -np.inf, -np.inf)
        self.trace_alert_streak = 0
        self._running_ep_return = 0.0
        self._running_ep_len = 0
        self.eval_cases: List[Dict] = []
        self.trace_cases: List[Dict] = []
        self.trace_dir = None
        self._init_logging()
        self.eval_cases = self._build_eval_cases()
        self.trace_cases = self._build_trace_cases()

    def _unwrap_env(self):
        env = self.env
        while hasattr(env, "env"):
            env = env.env
        return env

    def _set_seed(self, seed: int) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _distance_reward_coeff(self, global_step: int) -> float:
        frac = min(max(global_step / max(1, self.cfg.total_timesteps), 0.0), 1.0)
        start = self.cfg.distance_reward_coeff_start
        end = self.cfg.distance_reward_coeff_end
        return start + frac * (end - start)

    def _epsilon(self, global_step: int) -> float:
        frac = min(max(global_step / max(1, self.cfg.epsilon_decay_steps), 0.0), 1.0)
        return self.cfg.epsilon_start + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    def _init_logging(self) -> None:
        if self.run_dir is None:
            return

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.csv"
        self.trace_dir = self.run_dir / self.cfg.trace_dir_name
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        with self.metrics_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "update",
                    "global_step",
                    "mean_step_reward",
                    "done_rate",
                    "mean_episode_return",
                    "episodes_completed",
                    "policy_loss",
                    "value_loss",
                    "entropy",
                    "approx_kl",
                    "entropy_coef",
                    "distance_reward_coeff",
                    "explained_var",
                    "eval_cases",
                    "eval_mean_ppo_swaps",
                    "eval_mean_sabre_swaps",
                    "eval_improvement_pct",
                    "eval_win_rate",
                    "eval_timeout_rate",
                    "eval_mean_two_qubit_gates",
                    "trace_cases",
                    "trace_done_rate",
                    "trace_timeout_rate",
                    "trace_backtrack_rate",
                    "trace_action_dom_ratio",
                    "trace_mean_remaining_gates_end",
                    "trace_mean_ppo_swaps",
                    "trace_mean_sabre_swaps",
                    "trace_improvement_pct",
                    "trace_alert_flag",
                    "trace_alert_streak",
                ]
            )

    def _current_edges_and_mask(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build edge index tensor and validity mask for current topology.
        Returns:
            edge_index: (A, 2), int64
            action_mask: (A,), bool
        """
        unwrapped = self._unwrap_env()
        topo = unwrapped._current_topo
        edge_index = np.zeros((self.max_actions, 2), dtype=np.int64)
        num_edges = topo["num_edges"]
        if num_edges > 0:
            edge_index[:num_edges] = np.asarray(topo["edges"], dtype=np.int64)
        action_mask = unwrapped.get_action_mask().astype(bool)
        return edge_index, action_mask

    def _select_action_epsilon_greedy(
        self,
        obs: np.ndarray,
        epsilon: float,
    ) -> Tuple[int, np.ndarray, np.ndarray]:
        edge_index, action_mask = self._current_edges_and_mask()
        valid_actions = np.flatnonzero(action_mask)
        if valid_actions.size <= 0:
            raise RuntimeError("No valid action available for current topology.")

        if self.replay.rng.random() < epsilon:
            action = int(self.replay.rng.choice(valid_actions))
            return action, edge_index, action_mask

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        edge_t = torch.as_tensor(edge_index, dtype=torch.long, device=self.device).unsqueeze(0)
        mask_t = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net.get_q_values(obs_t, edge_t, mask_t)
            action = int(torch.argmax(q_values, dim=-1).item())
        return action, edge_index, action_mask

    def _greedy_action(self, obs: np.ndarray) -> int:
        edge_index, action_mask = self._current_edges_and_mask()
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        edge_t = torch.as_tensor(edge_index, dtype=torch.long, device=self.device).unsqueeze(0)
        mask_t = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net.get_q_values(obs_t, edge_t, mask_t)
            action = int(torch.argmax(q_values, dim=-1).item())
        return action

    def _greedy_action_with_prob(self, obs: np.ndarray) -> Tuple[int, float]:
        edge_index, action_mask = self._current_edges_and_mask()
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        edge_t = torch.as_tensor(edge_index, dtype=torch.long, device=self.device).unsqueeze(0)
        mask_t = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net.get_q_values(obs_t, edge_t, mask_t)
            logits = q_values.squeeze(0)
            probs = torch.softmax(logits, dim=-1)
            action = int(torch.argmax(logits, dim=-1).item())
            action_prob = float(probs[action].item())
        return action, action_prob

    def _train_step(self) -> Tuple[float, float, float]:
        batch = self.replay.sample(self.cfg.batch_size)

        obs_t = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        edge_t = torch.as_tensor(batch["edge_index"], dtype=torch.long, device=self.device)
        mask_t = torch.as_tensor(batch["action_mask"], dtype=torch.bool, device=self.device)
        actions_t = torch.as_tensor(batch["actions"], dtype=torch.long, device=self.device)
        rewards_t = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)
        next_obs_t = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        next_edge_t = torch.as_tensor(batch["next_edge_index"], dtype=torch.long, device=self.device)
        next_mask_t = torch.as_tensor(batch["next_action_mask"], dtype=torch.bool, device=self.device)

        q_values = self.q_net.get_q_values(obs_t, edge_t, mask_t)
        q_pred = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.cfg.use_double_dqn:
                next_online_q = self.q_net.get_q_values(next_obs_t, next_edge_t, next_mask_t)
                next_actions = torch.argmax(next_online_q, dim=1)
                next_target_q = self.target_net.get_q_values(next_obs_t, next_edge_t, next_mask_t)
                next_q = next_target_q.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                next_target_q = self.target_net.get_q_values(next_obs_t, next_edge_t, next_mask_t)
                next_q = torch.max(next_target_q, dim=1).values

            targets = rewards_t + self.cfg.gamma * (1.0 - dones_t) * next_q

        loss = F.smooth_l1_loss(q_pred, targets, beta=self.cfg.huber_delta)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()

        with torch.no_grad():
            td_abs = torch.mean(torch.abs(targets - q_pred)).item()
            probs = torch.softmax(q_values, dim=-1)
            entropy = torch.mean(-torch.sum(probs * torch.log(probs + 1e-8), dim=-1)).item()

        return float(loss.item()), float(td_abs), float(entropy)

    def _save_checkpoint(self, name: str) -> None:
        if self.run_dir is None:
            return
        checkpoint_path = self.run_dir / name
        torch.save(self.q_net.state_dict(), checkpoint_path)

    def _append_metrics_row(
        self,
        update_idx: int,
        global_step: int,
        mean_step_reward: float,
        done_rate: float,
        mean_episode_return: float,
        episodes_completed: int,
        metrics: Dict[str, float],
        eval_metrics: Dict[str, float] | None,
        trace_metrics: Dict[str, float] | None,
        trace_alert_flag: bool,
        trace_alert_streak: int,
        distance_reward_coeff: float,
    ) -> None:
        if self.metrics_path is None:
            return

        eval_cases = float("nan")
        eval_mean_ppo_swaps = float("nan")
        eval_mean_sabre_swaps = float("nan")
        eval_improvement_pct = float("nan")
        eval_win_rate = float("nan")
        eval_timeout_rate = float("nan")
        eval_mean_two_qubit_gates = float("nan")
        if eval_metrics is not None:
            eval_cases = eval_metrics["eval_cases"]
            eval_mean_ppo_swaps = eval_metrics["eval_mean_ppo_swaps"]
            eval_mean_sabre_swaps = eval_metrics["eval_mean_sabre_swaps"]
            eval_improvement_pct = eval_metrics["eval_improvement_pct"]
            eval_win_rate = eval_metrics["eval_win_rate"]
            eval_timeout_rate = eval_metrics["eval_timeout_rate"]
            eval_mean_two_qubit_gates = eval_metrics["eval_mean_two_qubit_gates"]

        trace_cases = float("nan")
        trace_done_rate = float("nan")
        trace_timeout_rate = float("nan")
        trace_backtrack_rate = float("nan")
        trace_action_dom_ratio = float("nan")
        trace_mean_remaining_gates_end = float("nan")
        trace_mean_ppo_swaps = float("nan")
        trace_mean_sabre_swaps = float("nan")
        trace_improvement_pct = float("nan")
        if trace_metrics is not None:
            trace_cases = trace_metrics["trace_cases"]
            trace_done_rate = trace_metrics["trace_done_rate"]
            trace_timeout_rate = trace_metrics["trace_timeout_rate"]
            trace_backtrack_rate = trace_metrics["trace_backtrack_rate"]
            trace_action_dom_ratio = trace_metrics["trace_action_dom_ratio"]
            trace_mean_remaining_gates_end = trace_metrics["trace_mean_remaining_gates_end"]
            trace_mean_ppo_swaps = trace_metrics["trace_mean_ppo_swaps"]
            trace_mean_sabre_swaps = trace_metrics["trace_mean_sabre_swaps"]
            trace_improvement_pct = trace_metrics["trace_improvement_pct"]
        trace_alert_flag_i = int(bool(trace_alert_flag))
        trace_alert_streak_i = int(trace_alert_streak)

        with self.metrics_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    update_idx,
                    global_step,
                    mean_step_reward,
                    done_rate,
                    mean_episode_return,
                    episodes_completed,
                    metrics["policy_loss"],
                    metrics["value_loss"],
                    metrics["entropy"],
                    metrics["approx_kl"],
                    metrics["entropy_coef"],
                    distance_reward_coeff,
                    metrics["explained_var"],
                    eval_cases,
                    eval_mean_ppo_swaps,
                    eval_mean_sabre_swaps,
                    eval_improvement_pct,
                    eval_win_rate,
                    eval_timeout_rate,
                    eval_mean_two_qubit_gates,
                    trace_cases,
                    trace_done_rate,
                    trace_timeout_rate,
                    trace_backtrack_rate,
                    trace_action_dom_ratio,
                    trace_mean_remaining_gates_end,
                    trace_mean_ppo_swaps,
                    trace_mean_sabre_swaps,
                    trace_improvement_pct,
                    trace_alert_flag_i,
                    trace_alert_streak_i,
                ]
            )

    def _build_eval_cases(self) -> List[Dict]:
        """
        Build a fixed random holdout set for periodic evaluation.
        This set uses a disjoint seed range to avoid overlap with training data.
        """
        if self.cfg.eval_circuits_per_topology <= 0:
            return []

        unwrapped = self._unwrap_env()
        cases: List[Dict] = []
        seed = int(self.cfg.eval_seed_base)

        for topo_idx, topo in enumerate(unwrapped._topologies):
            n_q = int(topo["n_physical"])
            cmap = topo["coupling_map"]
            for _ in range(self.cfg.eval_circuits_per_topology):
                circuit = generate_random_circuit(
                    num_qubits=n_q,
                    depth=self.cfg.eval_circuit_depth,
                    seed=seed,
                    min_two_qubit_gates=self.cfg.eval_min_two_qubit_gates,
                    max_attempts=self.cfg.eval_circuit_generation_attempts,
                )
                seed += 1

                sabre_init = get_sabre_initial_mapping(circuit, cmap)
                sabre_swaps = int(get_sabre_swap_count(circuit, cmap))
                twoq_count = int(count_two_qubit_gates(circuit))

                cases.append(
                    {
                        "topology_index": topo_idx,
                        "circuit": circuit,
                        "initial_mapping": [int(x) for x in sabre_init],
                        "sabre_swaps": sabre_swaps,
                        "two_qubit_gates": twoq_count,
                    }
                )
        return cases

    def _build_trace_cases(self) -> List[Dict]:
        """
        Build a fixed small subset of eval cases for periodic step-by-step tracing.
        """
        if self.cfg.trace_cases_per_topology <= 0:
            return []
        if not self.eval_cases:
            return []

        grouped: Dict[int, List[Dict]] = {}
        for case in self.eval_cases:
            topo_idx = int(case["topology_index"])
            grouped.setdefault(topo_idx, []).append(case)

        trace_cases: List[Dict] = []
        per_topo = int(self.cfg.trace_cases_per_topology)
        for topo_idx in sorted(grouped.keys()):
            topo_cases = grouped[topo_idx][:per_topo]
            for case_idx, case in enumerate(topo_cases):
                trace_cases.append(
                    {
                        "topology_index": topo_idx,
                        "case_index": int(case_idx),
                        "circuit": case["circuit"],
                        "initial_mapping": [int(x) for x in case["initial_mapping"]],
                        "sabre_swaps": int(case["sabre_swaps"]),
                        "two_qubit_gates": int(case["two_qubit_gates"]),
                    }
                )
        return trace_cases

    def _trace_case(self, case: Dict, update_idx: int) -> Dict[str, float]:
        unwrapped = self._unwrap_env()
        topo_idx = int(case["topology_index"])
        topo_name = str(unwrapped._topologies[topo_idx]["name"])
        case_idx = int(case["case_index"])

        obs, info = self.env.reset(
            options={
                "topology_index": topo_idx,
                "circuit": case["circuit"],
                "initial_mapping": case["initial_mapping"],
            }
        )

        rows: List[Dict] = []
        done = False
        truncated = False
        step_reward_sum = 0.0
        prev_total_exec = int(info.get("total_gates_executed", 0))
        last_action = None
        action_counts: Dict[int, int] = {}
        step_count_cap = max(1, int(self.cfg.trace_max_steps))

        while not done and not truncated:
            action, action_prob = self._greedy_action_with_prob(obs)
            current_topo = unwrapped._current_topo
            edge_i, edge_j = current_topo["edges"][action]
            front_before = float(unwrapped._compute_front_layer_distance())

            next_obs, reward, done, truncated, info = self.env.step(action)
            front_after = float(unwrapped._compute_front_layer_distance())
            delta_dist = front_before - front_after
            step_exec = int(info["total_gates_executed"]) - prev_total_exec
            prev_total_exec = int(info["total_gates_executed"])
            was_immediate_backtrack = int(last_action is not None and action == last_action)
            last_action = int(action)
            step_reward_sum += float(reward)
            action_counts[action] = action_counts.get(action, 0) + 1

            rows.append(
                {
                    "update": int(update_idx),
                    "topology": topo_name,
                    "trace_case_index": case_idx,
                    "step": int(info["step_count"]),
                    "action_index": int(action),
                    "edge_i": int(edge_i),
                    "edge_j": int(edge_j),
                    "action_prob": float(action_prob),
                    "reward": float(reward),
                    "step_gates_executed": int(step_exec),
                    "total_gates_executed": int(info["total_gates_executed"]),
                    "remaining_gates": int(info["remaining_gates"]),
                    "front_dist_before": float(front_before),
                    "front_dist_after": float(front_after),
                    "delta_dist": float(delta_dist),
                    "was_immediate_backtrack": int(was_immediate_backtrack),
                    "done": int(done),
                    "truncated": int(truncated),
                }
            )
            obs = next_obs

            if int(info.get("step_count", 0)) >= step_count_cap and not done:
                truncated = True

        ppo_swaps = int(info.get("total_swaps", 0))
        sabre_swaps = int(case["sabre_swaps"])
        improvement_pct = float(100.0 * (sabre_swaps - ppo_swaps) / max(1, sabre_swaps))
        backtrack_rate = float(np.mean([r["was_immediate_backtrack"] for r in rows])) if rows else 0.0
        dominant_action_ratio = (
            float(max(action_counts.values()) / max(1, len(rows)))
            if action_counts
            else 0.0
        )

        summary = {
            "update": int(update_idx),
            "topology": topo_name,
            "trace_case_index": int(case_idx),
            "steps": int(info.get("step_count", 0)),
            "episode_return": float(step_reward_sum),
            "done": bool(done),
            "truncated": bool(truncated),
            "ppo_swaps": int(ppo_swaps),
            "sabre_swaps": int(sabre_swaps),
            "improvement_pct_vs_sabre": float(improvement_pct),
            "remaining_gates_end": int(info.get("remaining_gates", 0)),
            "total_gates_executed_end": int(info.get("total_gates_executed", 0)),
            "backtrack_rate": float(backtrack_rate),
            "dominant_action_ratio": float(dominant_action_ratio),
            "two_qubit_gates": int(case["two_qubit_gates"]),
        }

        if self.trace_dir is not None:
            update_dir = self.trace_dir / f"update_{update_idx:05d}"
            update_dir.mkdir(parents=True, exist_ok=True)
            base = f"{topo_name}_case{case_idx:02d}"
            trace_csv = update_dir / f"{base}_trace.csv"
            summary_json = update_dir / f"{base}_summary.json"

            with trace_csv.open("w", newline="", encoding="utf-8") as f:
                if rows:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)

            with summary_json.open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

        return summary

    def _run_periodic_traces(self, update_idx: int) -> Dict[str, float] | None:
        if not self.trace_cases:
            return None
        if self.cfg.trace_interval_updates <= 0:
            return None
        if update_idx % self.cfg.trace_interval_updates != 0:
            return None

        summaries = [self._trace_case(case, update_idx) for case in self.trace_cases]
        if not summaries:
            return None

        ppo_arr = np.asarray([s["ppo_swaps"] for s in summaries], dtype=np.float32)
        sabre_arr = np.asarray([s["sabre_swaps"] for s in summaries], dtype=np.float32)
        done_arr = np.asarray([1.0 if s["done"] else 0.0 for s in summaries], dtype=np.float32)
        trunc_arr = np.asarray([1.0 if s["truncated"] else 0.0 for s in summaries], dtype=np.float32)
        backtrack_arr = np.asarray([s["backtrack_rate"] for s in summaries], dtype=np.float32)
        dom_ratio_arr = np.asarray([s["dominant_action_ratio"] for s in summaries], dtype=np.float32)
        remaining_arr = np.asarray([s["remaining_gates_end"] for s in summaries], dtype=np.float32)
        improve_arr = np.asarray([s["improvement_pct_vs_sabre"] for s in summaries], dtype=np.float32)

        return {
            "trace_cases": float(len(summaries)),
            "trace_done_rate": float(np.mean(done_arr)),
            "trace_timeout_rate": float(np.mean(trunc_arr)),
            "trace_backtrack_rate": float(np.mean(backtrack_arr)),
            "trace_action_dom_ratio": float(np.mean(dom_ratio_arr)),
            "trace_mean_remaining_gates_end": float(np.mean(remaining_arr)),
            "trace_mean_ppo_swaps": float(np.mean(ppo_arr)),
            "trace_mean_sabre_swaps": float(np.mean(sabre_arr)),
            "trace_improvement_pct": float(np.mean(improve_arr)),
        }

    def _evaluate_against_sabre(self) -> Dict[str, float] | None:
        if not self.eval_cases:
            return None

        ppo_swaps: List[int] = []
        sabre_swaps: List[int] = []
        twoq_counts: List[int] = []
        timeout_count = 0

        for case in self.eval_cases:
            obs, _ = self.env.reset(
                options={
                    "topology_index": case["topology_index"],
                    "circuit": case["circuit"],
                    "initial_mapping": case["initial_mapping"],
                }
            )
            done = False
            truncated = False
            info = {}
            while not done and not truncated:
                action = self._greedy_action(obs)
                obs, _, done, truncated, info = self.env.step(action)
            if truncated:
                timeout_count += 1

            ppo_swaps.append(int(info.get("total_swaps", 0)))
            sabre_swaps.append(int(case["sabre_swaps"]))
            twoq_counts.append(int(case["two_qubit_gates"]))

        ppo_arr = np.asarray(ppo_swaps, dtype=np.float32)
        sabre_arr = np.asarray(sabre_swaps, dtype=np.float32)
        safe_sabre = np.where(sabre_arr <= 0, 1.0, sabre_arr)
        improvement_pct = ((sabre_arr - ppo_arr) / safe_sabre) * 100.0
        win_rate = float(np.mean(ppo_arr <= sabre_arr))

        return {
            "eval_cases": float(len(self.eval_cases)),
            "eval_mean_ppo_swaps": float(np.mean(ppo_arr)),
            "eval_mean_sabre_swaps": float(np.mean(sabre_arr)),
            "eval_improvement_pct": float(np.mean(improvement_pct)),
            "eval_win_rate": win_rate,
            "eval_timeout_rate": float(timeout_count / len(self.eval_cases)),
            "eval_mean_two_qubit_gates": float(np.mean(np.asarray(twoq_counts, dtype=np.float32))),
        }

    @staticmethod
    def _eval_selection_key(eval_metrics: Dict[str, float]) -> Tuple[float, float, float]:
        """
        Lexicographic key for best-model selection aligned with project goal.
        Priority:
        1) higher improvement vs SABRE
        2) higher win rate vs SABRE
        3) lower timeout rate
        """
        return (
            float(eval_metrics["eval_improvement_pct"]),
            float(eval_metrics["eval_win_rate"]),
            -float(eval_metrics["eval_timeout_rate"]),
        )

    def _maybe_save_best_eval_model(
        self,
        eval_metrics: Dict[str, float],
        update_idx: int,
        global_step: int,
    ) -> bool:
        key = self._eval_selection_key(eval_metrics)
        if key <= self.best_eval_key:
            return False

        self.best_eval_key = key
        self._save_checkpoint("best_model.pt")
        if self.run_dir is not None:
            payload = {
                "update": int(update_idx),
                "global_step": int(global_step),
                "selection_key": [float(v) for v in key],
                "metrics": {
                    "eval_improvement_pct": float(eval_metrics["eval_improvement_pct"]),
                    "eval_win_rate": float(eval_metrics["eval_win_rate"]),
                    "eval_timeout_rate": float(eval_metrics["eval_timeout_rate"]),
                    "eval_mean_ppo_swaps": float(eval_metrics["eval_mean_ppo_swaps"]),
                    "eval_mean_sabre_swaps": float(eval_metrics["eval_mean_sabre_swaps"]),
                    "eval_cases": float(eval_metrics["eval_cases"]),
                },
            }
            with (self.run_dir / "best_eval_metrics.json").open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        return True

    def train(self) -> None:
        obs, _ = self.env.reset(seed=self.cfg.seed)
        global_step = 0
        update_idx = 0
        start_time = time.time()

        while global_step < self.cfg.total_timesteps:
            current_dist_coeff = self._distance_reward_coeff(global_step)
            self._unwrap_env().distance_reward_coeff = current_dist_coeff

            rewards_window: List[float] = []
            dones_window: List[float] = []
            ep_returns: List[float] = []
            q_losses: List[float] = []
            td_errors: List[float] = []
            entropies: List[float] = []

            for _ in range(self.cfg.update_steps):
                epsilon = self._epsilon(global_step)
                action, edge_index, action_mask = self._select_action_epsilon_greedy(obs, epsilon)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = bool(terminated or truncated)

                if done:
                    next_edge_index = np.zeros_like(edge_index)
                    next_action_mask = np.zeros_like(action_mask, dtype=bool)
                else:
                    next_edge_index, next_action_mask = self._current_edges_and_mask()

                self.replay.add(
                    obs=obs,
                    edge_index=edge_index,
                    action_mask=action_mask,
                    action=action,
                    reward=float(reward),
                    done=done,
                    next_obs=next_obs,
                    next_edge_index=next_edge_index,
                    next_action_mask=next_action_mask,
                )

                self._running_ep_return += float(reward)
                self._running_ep_len += 1
                rewards_window.append(float(reward))
                dones_window.append(float(done))

                obs = next_obs
                if done:
                    ep_returns.append(self._running_ep_return)
                    self._running_ep_return = 0.0
                    self._running_ep_len = 0
                    obs, _ = self.env.reset()

                if (
                    self.replay.size >= self.cfg.min_replay_size
                    and global_step % self.cfg.train_frequency_steps == 0
                ):
                    for _ in range(self.cfg.gradient_steps):
                        q_loss, td_abs, entropy = self._train_step()
                        q_losses.append(q_loss)
                        td_errors.append(td_abs)
                        entropies.append(entropy)

                if global_step % self.cfg.target_update_interval_steps == 0:
                    self.target_net.load_state_dict(self.q_net.state_dict())

                global_step += 1
                if global_step >= self.cfg.total_timesteps:
                    break

            update_idx += 1
            mean_step_reward = float(np.mean(rewards_window)) if rewards_window else float("nan")
            done_rate = float(np.mean(dones_window)) if dones_window else float("nan")
            mean_episode_return = float(np.mean(ep_returns)) if ep_returns else float("nan")
            epsilon_now = self._epsilon(global_step)

            metrics = {
                # Keep column compatibility with PPO logs.
                "policy_loss": float(np.mean(q_losses)) if q_losses else float("nan"),
                "value_loss": float(np.mean(td_errors)) if td_errors else float("nan"),
                "entropy": float(np.mean(entropies)) if entropies else float("nan"),
                "approx_kl": 0.0,
                "entropy_coef": float(epsilon_now),
                "explained_var": float("nan"),
            }

            eval_metrics = None
            best_eval_updated = False
            if (
                self.cfg.eval_interval_updates > 0
                and self.eval_cases
                and update_idx % self.cfg.eval_interval_updates == 0
            ):
                eval_metrics = self._evaluate_against_sabre()
                if eval_metrics is not None:
                    best_eval_updated = self._maybe_save_best_eval_model(
                        eval_metrics=eval_metrics,
                        update_idx=update_idx,
                        global_step=global_step,
                    )

            trace_metrics = self._run_periodic_traces(update_idx)
            trace_alert_flag = self.trace_alert_streak >= self.cfg.trace_alert_patience
            if trace_metrics is not None:
                dom = float(trace_metrics["trace_action_dom_ratio"])
                backtrack = float(trace_metrics["trace_backtrack_rate"])
                if (
                    dom >= self.cfg.trace_alert_dom_threshold
                    and backtrack >= self.cfg.trace_alert_backtrack_threshold
                ):
                    self.trace_alert_streak += 1
                else:
                    self.trace_alert_streak = 0
                trace_alert_flag = self.trace_alert_streak >= self.cfg.trace_alert_patience

            self._append_metrics_row(
                update_idx=update_idx,
                global_step=global_step,
                mean_step_reward=mean_step_reward,
                done_rate=done_rate,
                mean_episode_return=mean_episode_return,
                episodes_completed=len(ep_returns),
                metrics=metrics,
                eval_metrics=eval_metrics,
                trace_metrics=trace_metrics,
                trace_alert_flag=trace_alert_flag,
                trace_alert_streak=self.trace_alert_streak,
                distance_reward_coeff=current_dist_coeff,
            )

            if ep_returns and mean_episode_return > self.best_train_mean_episode_return:
                self.best_train_mean_episode_return = mean_episode_return
                self._save_checkpoint("best_train_model.pt")
                if not self.eval_cases:
                    # Fallback for setups with no periodic eval configured.
                    self._save_checkpoint("best_model.pt")

            if (
                self.cfg.checkpoint_interval_updates > 0
                and update_idx % self.cfg.checkpoint_interval_updates == 0
            ):
                self._save_checkpoint(f"checkpoint_update_{update_idx:05d}.pt")

            if update_idx % self.cfg.log_interval_updates == 0:
                elapsed_s = max(1e-6, time.time() - start_time)
                progress = min(1.0, global_step / max(1, self.cfg.total_timesteps))
                eta_s = (elapsed_s / progress - elapsed_s) if progress > 0 else float("nan")
                eval_msg = ""
                if eval_metrics is not None:
                    eval_msg = (
                        f" eval_improve={eval_metrics['eval_improvement_pct']:+.2f}%"
                        f" eval_win_rate={eval_metrics['eval_win_rate']:.2f}"
                        f" eval_timeout={eval_metrics['eval_timeout_rate']:.2f}"
                        f" eval_2q={eval_metrics['eval_mean_two_qubit_gates']:.1f}"
                    )
                    if best_eval_updated:
                        eval_msg += " best_eval=1"
                trace_msg = ""
                if trace_metrics is not None:
                    trace_msg = (
                        f" trace_timeout={trace_metrics['trace_timeout_rate']:.2f}"
                        f" trace_backtrack={trace_metrics['trace_backtrack_rate']:.2f}"
                        f" trace_dom={trace_metrics['trace_action_dom_ratio']:.2f}"
                        f" trace_improve={trace_metrics['trace_improvement_pct']:+.2f}%"
                    )
                    if trace_alert_flag:
                        trace_msg += f" trace_alert=1(streak={self.trace_alert_streak})"

                print(
                    f"[Update {update_idx:04d}] "
                    f"steps={global_step:>7d} "
                    f"mean_step_reward={mean_step_reward:+.4f} "
                    f"done_rate={done_rate:.3f} "
                    f"mean_ep_return={mean_episode_return:+.4f} "
                    f"q_loss={metrics['policy_loss']:.4f} "
                    f"td_abs={metrics['value_loss']:.4f} "
                    f"entropy={metrics['entropy']:.4f} "
                    f"epsilon={metrics['entropy_coef']:.4f}"
                    f" dist_coeff={current_dist_coeff:.4f}"
                    f" replay={self.replay.size}"
                    f" elapsed_min={elapsed_s/60.0:.1f}"
                    f" eta_min={eta_s/60.0:.1f}"
                    f"{eval_msg}"
                    f"{trace_msg}"
                )
                if trace_alert_flag:
                    if trace_metrics is not None:
                        print(
                            "[Alert] Loop-pattern risk detected from traces: "
                            f"dom={trace_metrics['trace_action_dom_ratio']:.3f}, "
                            f"backtrack={trace_metrics['trace_backtrack_rate']:.3f}, "
                            f"streak={self.trace_alert_streak}."
                        )
                    else:
                        print(
                            "[Alert] Loop-pattern risk detected from traces: "
                            f"streak={self.trace_alert_streak}."
                        )

        self._save_checkpoint("last_model.pt")
