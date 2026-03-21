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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


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
    entropy_coef_start: float = 0.02
    entropy_coef_end: float = 0.005
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.015
    log_interval_updates: int = 5
    checkpoint_interval_updates: int = 25
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

    def __init__(self, env, config: PPOConfig):
        self.env = env
        self.cfg = config

        self.device = torch.device(config.device)
        self._set_seed(config.seed)

        matrix_size = int(self._unwrap_env().N)
        self.max_actions = int(self.env.action_space.n)

        self.model = SymmetricCNNActorCritic(matrix_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

        self.run_dir = Path(config.run_dir) if config.run_dir else None
        self.metrics_path = None
        self.best_mean_episode_return = -np.inf
        self._running_ep_return = 0.0
        self._running_ep_len = 0
        self._init_logging()

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

    def _init_logging(self) -> None:
        if self.run_dir is None:
            return

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.csv"
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
                    "explained_var",
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
    ) -> None:
        if self.metrics_path is None:
            return
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
                    metrics["explained_var"],
                ]
            )

    def train(self) -> None:
        obs, _ = self.env.reset(seed=self.cfg.seed)
        global_step = 0
        update_idx = 0

        while global_step < self.cfg.total_timesteps:
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

            self._append_metrics_row(
                update_idx=update_idx,
                global_step=global_step,
                mean_step_reward=mean_step_reward,
                done_rate=done_rate,
                mean_episode_return=mean_episode_return,
                episodes_completed=len(ep_returns),
                metrics=metrics,
            )

            if ep_returns and mean_episode_return > self.best_mean_episode_return:
                self.best_mean_episode_return = mean_episode_return
                self._save_checkpoint("best_model.pt")

            if (
                self.cfg.checkpoint_interval_updates > 0
                and update_idx % self.cfg.checkpoint_interval_updates == 0
            ):
                self._save_checkpoint(f"checkpoint_update_{update_idx:05d}.pt")

            if update_idx % self.cfg.log_interval_updates == 0:
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
                )

        self._save_checkpoint("last_model.pt")
