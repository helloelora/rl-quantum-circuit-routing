"""D3QN Agent: Double Dueling DQN with Prioritized Experience Replay."""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from networks import DuelingCNN
from replay_buffer import PrioritizedReplayBuffer


class D3QNAgent:
    """Double Dueling DQN agent with PER for quantum circuit routing."""

    def __init__(self, config, num_actions: int):
        self.config = config
        self.num_actions = num_actions

        # Device
        if config.device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(config.device)

        # Networks
        self.online_net = DuelingCNN(
            matrix_size=config.matrix_size,
            num_actions=num_actions,
            conv_channels=config.conv_channels,
            dueling_hidden=config.dueling_hidden,
        ).to(self.device)

        self.target_net = DuelingCNN(
            matrix_size=config.matrix_size,
            num_actions=num_actions,
            conv_channels=config.conv_channels,
            dueling_hidden=config.dueling_hidden,
        ).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.online_net.parameters(), lr=config.lr
        )

        # LR scheduler
        self.lr_schedule = getattr(config, "lr_schedule", "constant")
        self.scheduler = None
        if self.lr_schedule == "cosine":
            lr_min = getattr(config, "lr_min", 1e-5)
            # T_max = estimated total gradient steps
            est_steps = config.total_episodes * 50  # rough estimate
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=est_steps, eta_min=lr_min
            )

        # Replay buffer
        from networks import NUM_STATE_CHANNELS
        state_shape = (NUM_STATE_CHANNELS, config.matrix_size, config.matrix_size)
        self.buffer = PrioritizedReplayBuffer(
            capacity=config.buffer_capacity,
            state_shape=state_shape,
            num_actions=num_actions,
            alpha=config.per_alpha,
            beta_start=config.per_beta_start,
            beta_end=config.per_beta_end,
            beta_anneal_steps=config.per_beta_anneal_steps,
            epsilon=config.per_epsilon,
        )

        # N-step returns
        self.n_step = getattr(config, "n_step", 1)
        self._nstep_buf = []

        # Epsilon
        self.epsilon = config.epsilon_start
        self._epsilon_step = 0

        # Step counter
        self._train_steps = 0

    def select_action(self, state: np.ndarray, action_mask: np.ndarray,
                      deterministic: bool = False) -> int:
        """Epsilon-greedy action selection with masking."""
        if not deterministic and np.random.random() < self.epsilon:
            valid = np.where(action_mask)[0]
            return int(np.random.choice(valid))

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.online_net(state_t).cpu().numpy()[0]

        q[~action_mask] = -np.inf
        return int(np.argmax(q))

    def store_transition(self, state, action, reward, next_state, done,
                         next_action_mask):
        """Store transition in replay buffer (with n-step accumulation)."""
        if self.n_step <= 1:
            self.buffer.add(state, action, reward, next_state, done,
                            next_action_mask)
            return

        self._nstep_buf.append(
            (state, action, reward, next_state, done, next_action_mask)
        )

        if done:
            # Episode ended — flush all accumulated transitions
            while self._nstep_buf:
                self._pop_nstep()
            return

        if len(self._nstep_buf) >= self.n_step:
            self._pop_nstep()

    def _pop_nstep(self):
        """Pop the oldest transition, compute its n-step return, store in PER."""
        n = min(len(self._nstep_buf), self.n_step)
        s0, a0 = self._nstep_buf[0][0], self._nstep_buf[0][1]
        R = 0.0
        final_ns, final_done, final_nm = None, False, None
        for i in range(n):
            _, _, r, ns, d, nm = self._nstep_buf[i]
            R += (self.config.gamma ** i) * r
            final_ns, final_done, final_nm = ns, d, nm
            if d:
                break
        self.buffer.add(s0, a0, R, final_ns, final_done, final_nm)
        self._nstep_buf.pop(0)

    def end_episode(self):
        """Flush n-step buffer at episode end (needed for truncated episodes)."""
        if self.n_step > 1:
            while self._nstep_buf:
                self._pop_nstep()

    def train_step(self) -> dict:
        """
        One training step: sample from PER, compute Double DQN targets,
        Huber loss weighted by IS weights, update priorities.

        Returns dict with metrics, or empty dict if not enough samples.
        """
        if len(self.buffer) < self.config.train_start:
            return {}

        self._train_steps += 1

        batch, is_weights, tree_indices = self.buffer.sample(
            self.config.batch_size, self._train_steps
        )

        states = torch.FloatTensor(batch["states"]).to(self.device)
        actions = torch.LongTensor(batch["actions"]).to(self.device)
        rewards = torch.FloatTensor(batch["rewards"]).to(self.device)
        next_states = torch.FloatTensor(batch["next_states"]).to(self.device)
        dones = torch.FloatTensor(
            batch["dones"].astype(np.float32)
        ).to(self.device)
        next_masks = torch.BoolTensor(
            batch["next_action_masks"]
        ).to(self.device)
        is_weights_t = torch.FloatTensor(is_weights).to(self.device)

        # Current Q(s, a)
        q_values = self.online_net(states)
        q_current = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target
        with torch.no_grad():
            # Online net selects best action (masked)
            q_online_next = self.online_net(next_states)
            q_online_next[~next_masks] = -float("inf")
            best_actions = q_online_next.argmax(dim=1)

            # Target net evaluates
            q_target_next = self.target_net(next_states)
            q_next = q_target_next.gather(
                1, best_actions.unsqueeze(1)
            ).squeeze(1)

            gamma_n = self.config.gamma ** self.n_step
            targets = rewards + gamma_n * q_next * (1 - dones)

        # TD errors for priority update
        td_errors = (q_current - targets).detach().cpu().numpy()

        # Huber loss weighted by IS weights
        element_loss = F.smooth_l1_loss(q_current, targets, reduction="none")
        loss = (is_weights_t * element_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.online_net.parameters(), self.config.grad_clip_norm
        )
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        self.buffer.update_priorities(tree_indices, td_errors)

        current_lr = self.optimizer.param_groups[0]["lr"]
        return {
            "loss": loss.item(),
            "mean_q": q_current.mean().item(),
            "mean_td_error": float(np.abs(td_errors).mean()),
            "epsilon": self.epsilon,
            "lr": current_lr,
            "beta": min(
                self.config.per_beta_end,
                self.config.per_beta_start
                + (self.config.per_beta_end - self.config.per_beta_start)
                * self._train_steps / self.config.per_beta_anneal_steps,
            ),
        }

    def update_target_network(self):
        """Hard copy or soft Polyak update."""
        if self.config.tau >= 1.0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        else:
            for tp, op in zip(
                self.target_net.parameters(), self.online_net.parameters()
            ):
                tp.data.copy_(
                    self.config.tau * op.data
                    + (1 - self.config.tau) * tp.data
                )

    def update_epsilon(self):
        """Linear epsilon decay."""
        self._epsilon_step += 1
        self.epsilon = max(
            self.config.epsilon_end,
            self.config.epsilon_start
            - (self.config.epsilon_start - self.config.epsilon_end)
            * self._epsilon_step / self.config.epsilon_decay_steps,
        )

    def save_checkpoint(self, path: str, episode: int, extra: dict = None):
        """Save model checkpoint (.pt) and optionally buffer (.buf.npz)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "episode": episode,
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "epsilon_step": self._epsilon_step,
            "train_steps": self._train_steps,
            "num_actions": self.num_actions,
        }
        if extra:
            checkpoint.update(extra)

        torch.save(checkpoint, str(path))

        if self.config.save_buffer:
            n = len(self.buffer)
            buf_path = path.with_suffix(".buf.npz")
            np.savez_compressed(
                str(buf_path),
                states=self.buffer.states[:n],
                actions=self.buffer.actions[:n],
                rewards=self.buffer.rewards[:n],
                next_states=self.buffer.next_states[:n],
                dones=self.buffer.dones[:n],
                next_action_masks=self.buffer.next_action_masks[:n],
                tree_data=self.buffer.tree.tree,
                ptr=self.buffer._ptr,
                size=self.buffer._size,
                max_priority=self.buffer.tree.max_priority,
            )

        return str(path)

    def load_checkpoint(self, path: str) -> int:
        """Load checkpoint. Returns the episode number."""
        checkpoint = torch.load(
            str(path), map_location=self.device, weights_only=False
        )

        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self._epsilon_step = checkpoint["epsilon_step"]
        self._train_steps = checkpoint["train_steps"]

        buf_path = Path(path).with_suffix(".buf.npz")
        if buf_path.exists():
            data = np.load(str(buf_path))
            n = int(data["size"])
            self.buffer.states[:n] = data["states"]
            self.buffer.actions[:n] = data["actions"]
            self.buffer.rewards[:n] = data["rewards"]
            self.buffer.next_states[:n] = data["next_states"]
            self.buffer.dones[:n] = data["dones"]
            self.buffer.next_action_masks[:n] = data["next_action_masks"]
            self.buffer.tree.tree[:] = data["tree_data"]
            self.buffer._ptr = int(data["ptr"])
            self.buffer._size = n
            self.buffer.tree.size = n
            self.buffer.tree.data_pointer = int(data["ptr"])
            self.buffer.tree.max_priority = float(data["max_priority"])

        return checkpoint.get("episode", 0)
