from __future__ import annotations

from typing import Any

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm


def build_agent(env, algorithm: str = "dqn", **kwargs: Any) -> BaseAlgorithm:
    """Build a Deep RL model (DQN or PPO) for the routing environment."""
    algo = algorithm.lower()

    if algo == "dqn":
        default_kwargs: dict[str, Any] = {
            "learning_rate": 1e-3,
            "buffer_size": 100_000,
            "batch_size": 64,
            "learning_starts": 1_000,
            "gamma": 0.99,
            "target_update_interval": 500,
            "train_freq": 4,
            "verbose": 1,
        }
        default_kwargs.update(kwargs)
        return DQN("MlpPolicy", env, **default_kwargs)

    if algo == "ppo":
        default_kwargs = {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "verbose": 1,
        }
        default_kwargs.update(kwargs)
        return PPO("MlpPolicy", env, **default_kwargs)

    raise ValueError(f"Unsupported algorithm '{algorithm}'. Use 'dqn' or 'ppo'.")


def train_agent(model: BaseAlgorithm, total_timesteps: int = 100_000) -> BaseAlgorithm:
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    return model
