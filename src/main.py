"""Training entrypoint for PPO-based quantum circuit routing."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

try:
    import torchmetrics  # noqa: F401

    TORCHMETRICS_OK = True
    TORCHMETRICS_ERR = ""
except Exception as exc:  # pragma: no cover - optional dependency
    TORCHMETRICS_OK = False
    TORCHMETRICS_ERR = str(exc)

try:
    from .agent import PPOAgent, PPOConfig
    from .environment import QubitRoutingEnv
except ImportError:
    from agent import PPOAgent, PPOConfig
    from environment import QubitRoutingEnv


def setup_project_paths(in_colab: bool, project_root_arg: str):
    if in_colab:
        try:
            from google.colab import drive  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "--in-colab was passed but google.colab is not available."
            ) from exc
        drive.mount("/content/drive")

    if project_root_arg:
        project_root = Path(project_root_arg)
    elif in_colab:
        project_root = Path("/content/drive/MyDrive/rl_quantum_circuit_routing")
    else:
        project_root = Path.cwd()

    data_root = project_root / "data"
    run_root = project_root / "runs"
    data_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    return project_root, data_root, run_root


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO for qubit routing")

    parser.add_argument("--in-colab", action="store_true")
    parser.add_argument(
        "--project-root",
        type=str,
        default="",
        help="Optional project root. If omitted: cwd (local) or default Drive path (Colab).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Run folder name under runs/. If empty, timestamp is used.",
    )

    parser.add_argument(
        "--topologies",
        type=str,
        default="heavy_hex_19,grid_3x3,linear_5",
        help="Comma-separated topology names for multi-topology training",
    )
    parser.add_argument("--matrix-size", type=int, default=27)
    parser.add_argument("--circuit-depth", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--distance-reward-coeff", type=float, default=0.01)
    parser.add_argument("--completion-bonus", type=float, default=5.0)
    parser.add_argument("--timeout-penalty", type=float, default=-10.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--total-timesteps", type=int, default=200000)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--entropy-coef-start", type=float, default=0.02)
    parser.add_argument("--entropy-coef-end", type=float, default=0.005)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.015)
    parser.add_argument("--log-interval-updates", type=int, default=5)
    parser.add_argument("--checkpoint-interval-updates", type=int, default=25)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-path", type=str, default="")

    return parser.parse_args()


def main():
    args = parse_args()
    project_root, data_root, run_root = setup_project_paths(
        in_colab=args.in_colab,
        project_root_arg=args.project_root,
    )

    run_name = (
        args.run_name
        if args.run_name
        else datetime.now().strftime("ppo_%Y%m%d_%H%M%S")
    )
    run_dir = run_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    topologies = [t.strip() for t in args.topologies.split(",") if t.strip()]
    if not topologies:
        raise ValueError("No valid topology names were provided.")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    env = QubitRoutingEnv(
        topologies=topologies,
        matrix_size=args.matrix_size,
        circuit_depth=args.circuit_depth,
        max_steps=args.max_steps,
        gamma_decay=0.5,
        distance_reward_coeff=args.distance_reward_coeff,
        completion_bonus=args.completion_bonus,
        timeout_penalty=args.timeout_penalty,
        initial_mapping_strategy="mixed",
        seed=args.seed,
    )

    cfg = PPOConfig(
        total_timesteps=args.total_timesteps,
        rollout_steps=args.rollout_steps,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        entropy_coef_start=args.entropy_coef_start,
        entropy_coef_end=args.entropy_coef_end,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        log_interval_updates=args.log_interval_updates,
        checkpoint_interval_updates=args.checkpoint_interval_updates,
        run_dir=str(run_dir),
        seed=args.seed,
        device=device,
    )

    config_dump = vars(args).copy()
    config_dump["resolved_device"] = device
    config_dump["project_root"] = str(project_root)
    config_dump["data_root"] = str(data_root)
    config_dump["run_root"] = str(run_root)
    config_dump["run_dir"] = str(run_dir)
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config_dump, f, indent=2)

    print("Starting PPO training with settings:")
    print(f"  PROJECT_ROOT={project_root}")
    print(f"  DATA_ROOT={data_root}")
    print(f"  RUN_ROOT={run_root}")
    print(f"  RUN_DIR={run_dir}")
    print(f"  topologies={topologies}")
    print("  initial_mapping_strategy=mixed (80% random, 20% SABRE)")
    print("  strategy_masking=off (topology validity mask only)")
    print("  gamma_decay=0.5")
    print(f"  device={cfg.device}")
    print(f"  torch_version={torch.__version__}")
    print(f"  cuda_available={torch.cuda.is_available()}")
    print(
        "  torchmetrics="
        + ("OK" if TORCHMETRICS_OK else f"MISSING ({TORCHMETRICS_ERR})")
    )

    agent = PPOAgent(env, cfg)
    agent.train()

    if args.save_path:
        torch.save(agent.model.state_dict(), args.save_path)
        print(f"Saved model weights to: {args.save_path}")


if __name__ == "__main__":
    main()
