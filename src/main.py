from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3.common.monitor import Monitor

from agent import build_agent, train_agent
from environment import QuantumRoutingEnv


def _collect_qasm_files(benchmarks_dir: Path) -> list[str]:
    if not benchmarks_dir.exists():
        return []
    return sorted(str(path) for path in benchmarks_dir.rglob("*.qasm"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RL agents for quantum circuit routing.")
    parser.add_argument("--train", action="store_true", help="Run model training.")
    parser.add_argument("--algo", choices=["dqn", "ppo"], default="dqn", help="RL algorithm.")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Training timesteps.")
    parser.add_argument(
        "--benchmarks-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "benchmarks",
        help="Directory containing QASMBench .qasm files.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models" / "routing_agent",
        help="Model save/load path without extension.",
    )
    args = parser.parse_args()

    qasm_files = _collect_qasm_files(args.benchmarks_dir)
    env = Monitor(QuantumRoutingEnv(qasm_files=qasm_files))

    args.model_path.parent.mkdir(parents=True, exist_ok=True)

    if args.train:
        model = build_agent(env, algorithm=args.algo)
        train_agent(model, total_timesteps=args.timesteps)
        model.save(str(args.model_path))
        print(f"Model saved to: {args.model_path}")
        return

    print("No action requested. Use --train to start training.")


if __name__ == "__main__":
    main()
