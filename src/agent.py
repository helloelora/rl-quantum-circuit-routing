"""
PPO agent for quantum circuit routing.

Implements a symmetry-aware actor-critic network:
- Input: (3, N, N) state tensor
- Policy: predicts an N x N SWAP score map, then enforces undirected symmetry
          with S = 0.5 * (S + S^T)
- Actions: edge scores are gathered from the score map using the current
           topology edge list; non-existing edges are masked out.
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
from torch.distributions import Categorical

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
class PPOConfig:
    total_timesteps: int = 200_000
    rollout_steps: int = 2048
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    update_epochs: int = 4
    minibatch_size: int = 256
    entropy_coef_start: float = 0.01
    entropy_coef_end: float = 0.001
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.015
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


class SymmetricCNNActorCritic(nn.Module):
    """CNN actor-critic with symmetry-aware action scoring."""

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

        # Produces one score per (physical_i, physical_j) pair.
        self.policy_map_head = nn.Conv2d(32, 1, kernel_size=1)

        hidden_dim = 256
        self.value_head = nn.Sequential(
            nn.Linear(32 * matrix_size * matrix_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: (B, 3, N, N)
        Returns:
            score_map_sym: (B, N, N)
            values: (B,)
        """
        x = self.backbone(obs)
        score_map = self.policy_map_head(x).squeeze(1)  # (B, N, N)
        score_map_sym = 0.5 * (score_map + score_map.transpose(-1, -2))

        values = self.value_head(torch.flatten(x, start_dim=1)).squeeze(-1)
        return score_map_sym, values

    @staticmethod
    def gather_edge_logits(
        score_map_sym: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Gather action logits from symmetric score map using edge list.

        Args:
            score_map_sym: (B, N, N)
            edge_index: (B, A, 2), with endpoints (i, j) per action index
        Returns:
            edge_logits: (B, A)
        """
        batch_size, num_actions, _ = edge_index.shape
        b_idx = torch.arange(batch_size, device=score_map_sym.device).unsqueeze(1)
        i_idx = edge_index[..., 0]
        j_idx = edge_index[..., 1]
        edge_logits = score_map_sym[b_idx, i_idx, j_idx]
        assert edge_logits.shape == (batch_size, num_actions)
        return edge_logits

    def get_action_distribution(
        self,
        obs: torch.Tensor,
        edge_index: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> Tuple[Categorical, torch.Tensor]:
        score_map_sym, values = self.forward(obs)
        logits = self.gather_edge_logits(score_map_sym, edge_index)

        # Topology-validity masking only (no strategy mask by design).
        masked_logits = logits.masked_fill(~action_mask, -1e9)
        dist = Categorical(logits=masked_logits)
        return dist, values


class PPOAgent:
    """On-policy PPO trainer for QubitRoutingEnv."""

    def __init__(self, env, config: PPOConfig, model_state_dict=None):
        self.env = env
        self.cfg = config

        self.device = torch.device(config.device)
        self._set_seed(config.seed)

        matrix_size = int(self._unwrap_env().N)
        self.max_actions = int(self.env.action_space.n)

        self.model = SymmetricCNNActorCritic(matrix_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        if model_state_dict is not None:
            self.model.load_state_dict(model_state_dict)

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

    def _entropy_coef(self, global_step: int) -> float:
        frac = min(max(global_step / max(1, self.cfg.total_timesteps), 0.0), 1.0)
        start = self.cfg.entropy_coef_start
        end = self.cfg.entropy_coef_end
        return start + frac * (end - start)

    def _distance_reward_coeff(self, global_step: int) -> float:
        frac = min(max(global_step / max(1, self.cfg.total_timesteps), 0.0), 1.0)
        start = self.cfg.distance_reward_coeff_start
        end = self.cfg.distance_reward_coeff_end
        return start + frac * (end - start)

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

    def _collect_rollout(
        self, obs: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, List[float], List[int]]:
        data: Dict[str, List] = {
            "obs": [],
            "edge_index": [],
            "action_mask": [],
            "actions": [],
            "logprobs": [],
            "rewards": [],
            "dones": [],
            "values": [],
        }
        completed_episode_returns: List[float] = []
        completed_episode_lengths: List[int] = []

        for _ in range(self.cfg.rollout_steps):
            edge_index, action_mask = self._current_edges_and_mask()

            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            edge_t = torch.as_tensor(edge_index, dtype=torch.long, device=self.device).unsqueeze(0)
            mask_t = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device).unsqueeze(0)

            with torch.no_grad():
                dist, value = self.model.get_action_distribution(obs_t, edge_t, mask_t)
                action = dist.sample()
                logprob = dist.log_prob(action)

            next_obs, reward, terminated, truncated, _ = self.env.step(int(action.item()))
            done = bool(terminated or truncated)

            self._running_ep_return += float(reward)
            self._running_ep_len += 1

            data["obs"].append(obs.copy())
            data["edge_index"].append(edge_index.copy())
            data["action_mask"].append(action_mask.copy())
            data["actions"].append(int(action.item()))
            data["logprobs"].append(float(logprob.item()))
            data["rewards"].append(float(reward))
            data["dones"].append(float(done))
            data["values"].append(float(value.item()))

            obs = next_obs
            if done:
                completed_episode_returns.append(self._running_ep_return)
                completed_episode_lengths.append(self._running_ep_len)
                self._running_ep_return = 0.0
                self._running_ep_len = 0
                obs, _ = self.env.reset()

        rollout = {k: np.asarray(v) for k, v in data.items()}
        return rollout, obs, completed_episode_returns, completed_episode_lengths

    def _compute_gae(
        self,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
        last_value: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_nonterminal = 1.0 - dones[t]
                next_value = last_value
            else:
                next_nonterminal = 1.0 - dones[t]
                next_value = values[t + 1]

            delta = rewards[t] + self.cfg.gamma * next_value * next_nonterminal - values[t]
            last_gae = delta + self.cfg.gamma * self.cfg.gae_lambda * next_nonterminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def _update(self, rollout: Dict[str, np.ndarray], global_step: int) -> Dict[str, float]:
        obs = torch.as_tensor(rollout["obs"], dtype=torch.float32, device=self.device)
        edge_index = torch.as_tensor(rollout["edge_index"], dtype=torch.long, device=self.device)
        action_mask = torch.as_tensor(rollout["action_mask"], dtype=torch.bool, device=self.device)
        actions = torch.as_tensor(rollout["actions"], dtype=torch.long, device=self.device)
        old_logprobs = torch.as_tensor(rollout["logprobs"], dtype=torch.float32, device=self.device)
        old_values = torch.as_tensor(rollout["values"], dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(rollout["advantages"], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(rollout["returns"], dtype=torch.float32, device=self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        batch_size = obs.shape[0]
        indices = np.arange(batch_size)

        entropy_coef = self._entropy_coef(global_step)
        approx_kl = 0.0
        pg_loss_val = 0.0
        vf_loss_val = 0.0
        entropy_val = 0.0

        for _ in range(self.cfg.update_epochs):
            np.random.shuffle(indices)

            for start in range(0, batch_size, self.cfg.minibatch_size):
                mb_idx = indices[start:start + self.cfg.minibatch_size]
                mb_idx_t = torch.as_tensor(mb_idx, dtype=torch.long, device=self.device)

                dist, values = self.model.get_action_distribution(
                    obs[mb_idx_t],
                    edge_index[mb_idx_t],
                    action_mask[mb_idx_t],
                )
                new_logprobs = dist.log_prob(actions[mb_idx_t])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logprobs - old_logprobs[mb_idx_t])
                adv = advantages[mb_idx_t]
                pg_loss_1 = ratio * adv
                pg_loss_2 = torch.clamp(
                    ratio,
                    1.0 - self.cfg.clip_range,
                    1.0 + self.cfg.clip_range,
                ) * adv
                pg_loss = -torch.min(pg_loss_1, pg_loss_2).mean()

                value_loss = F.mse_loss(values, returns[mb_idx_t])

                loss = pg_loss + self.cfg.value_coef * value_loss - entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    this_kl = (old_logprobs[mb_idx_t] - new_logprobs).mean().item()
                approx_kl = this_kl
                pg_loss_val = pg_loss.item()
                vf_loss_val = value_loss.item()
                entropy_val = entropy.item()

            if self.cfg.target_kl > 0 and approx_kl > self.cfg.target_kl:
                break

        explained_var = 1.0 - torch.var(returns - old_values) / (torch.var(returns) + 1e-8)
        explained_var_val = float(explained_var.item())

        return {
            "policy_loss": pg_loss_val,
            "value_loss": vf_loss_val,
            "entropy": entropy_val,
            "approx_kl": approx_kl,
            "entropy_coef": entropy_coef,
            "explained_var": explained_var_val,
        }

    def _save_checkpoint(self, name: str) -> None:
        if self.run_dir is None:
            return
        checkpoint_path = self.run_dir / name
        torch.save(self.model.state_dict(), checkpoint_path)

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

    def _greedy_action(self, obs: np.ndarray) -> int:
        edge_index, action_mask = self._current_edges_and_mask()
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        edge_t = torch.as_tensor(edge_index, dtype=torch.long, device=self.device).unsqueeze(0)
        mask_t = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device).unsqueeze(0)

        with torch.no_grad():
            dist, _ = self.model.get_action_distribution(obs_t, edge_t, mask_t)
            action = int(torch.argmax(dist.logits, dim=-1).item())
        return action

    def _greedy_action_with_prob(self, obs: np.ndarray) -> Tuple[int, float]:
        edge_index, action_mask = self._current_edges_and_mask()
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        edge_t = torch.as_tensor(edge_index, dtype=torch.long, device=self.device).unsqueeze(0)
        mask_t = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device).unsqueeze(0)

        with torch.no_grad():
            dist, _ = self.model.get_action_distribution(obs_t, edge_t, mask_t)
            logits = dist.logits.squeeze(0)
            probs = torch.softmax(logits, dim=-1)
            action = int(torch.argmax(logits, dim=-1).item())
            action_prob = float(probs[action].item())
        return action, action_prob

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

            rollout, obs, ep_returns, _ = self._collect_rollout(obs)

            # Bootstrap from the final state of the rollout.
            edge_index, action_mask = self._current_edges_and_mask()
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            edge_t = torch.as_tensor(edge_index, dtype=torch.long, device=self.device).unsqueeze(0)
            mask_t = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
            with torch.no_grad():
                _, last_value_t = self.model.get_action_distribution(obs_t, edge_t, mask_t)
            last_value = float(last_value_t.item())

            advantages, returns = self._compute_gae(
                rewards=rollout["rewards"],
                dones=rollout["dones"],
                values=rollout["values"],
                last_value=last_value,
            )
            rollout["advantages"] = advantages
            rollout["returns"] = returns

            metrics = self._update(rollout, global_step)

            global_step += self.cfg.rollout_steps
            update_idx += 1

            mean_step_reward = float(np.mean(rollout["rewards"]))
            done_rate = float(np.mean(rollout["dones"]))
            if ep_returns:
                mean_episode_return = float(np.mean(ep_returns))
            else:
                mean_episode_return = float("nan")

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
            trace_alert_flag = (
                self.trace_alert_streak >= self.cfg.trace_alert_patience
            )
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
                    f"pi_loss={metrics['policy_loss']:+.4f} "
                    f"v_loss={metrics['value_loss']:.4f} "
                    f"entropy={metrics['entropy']:.4f} "
                    f"kl={metrics['approx_kl']:.5f}"
                    f" dist_coeff={current_dist_coeff:.4f}"
                    f" elapsed_min={elapsed_s/60.0:.1f}"
                    f" eta_min={eta_s/60.0:.1f}"
                    f"{eval_msg}"
                    f"{trace_msg}"
                )
                if trace_alert_flag:
                    print(
                        "[Alert] Loop-pattern risk detected from traces: "
                        f"dom={trace_metrics['trace_action_dom_ratio']:.3f}, "
                        f"backtrack={trace_metrics['trace_backtrack_rate']:.3f}, "
                        f"streak={self.trace_alert_streak}."
                    )

        self._save_checkpoint("last_model.pt")
