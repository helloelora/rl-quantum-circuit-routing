"""Training entrypoint for PPO-based quantum circuit routing."""

from __future__ import annotations

import argparse
import copy
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


def parse_topologies(topologies_csv: str):
    return [t.strip() for t in topologies_csv.split(",") if t.strip()]


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
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--stage1-topologies", type=str, default="linear_5")
    parser.add_argument("--stage2-topologies", type=str, default="linear_5,grid_3x3")
    parser.add_argument("--stage1-steps", type=int, default=200000)
    parser.add_argument("--stage2-steps", type=int, default=200000)
    parser.add_argument("--stage3-steps", type=int, default=600000)
    parser.add_argument("--stage1-depth", type=int, default=10)
    parser.add_argument("--stage2-depth", type=int, default=14)
    parser.add_argument("--stage3-depth", type=int, default=20)
    parser.add_argument("--matrix-size", type=int, default=27)
    parser.add_argument("--circuit-depth", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--distance-reward-coeff", type=float, default=0.01)
    parser.add_argument("--distance-reward-coeff-start", type=float, default=0.03)
    parser.add_argument("--distance-reward-coeff-end", type=float, default=0.015)
    parser.add_argument("--completion-bonus", type=float, default=8.0)
    parser.add_argument("--timeout-penalty", type=float, default=-25.0)
    parser.add_argument("--min-two-qubit-gates", type=int, default=6)
    parser.add_argument("--circuit-generation-attempts", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--total-timesteps", type=int, default=200000)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--entropy-coef-start", type=float, default=0.01)
    parser.add_argument("--entropy-coef-end", type=float, default=0.001)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.015)
    parser.add_argument("--log-interval-updates", type=int, default=5)
    parser.add_argument("--checkpoint-interval-updates", type=int, default=25)
    parser.add_argument("--eval-interval-updates", type=int, default=10)
    parser.add_argument("--eval-circuits-per-topology", type=int, default=12)
    parser.add_argument("--eval-circuit-depth", type=int, default=20)
    parser.add_argument(
        "--eval-min-two-qubit-gates",
        type=int,
        default=-1,
        help="Min 2-qubit gates for eval holdout circuits; -1 means use --min-two-qubit-gates.",
    )
    parser.add_argument("--eval-circuit-generation-attempts", type=int, default=16)
    parser.add_argument("--eval-seed-base", type=int, default=3000000000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-path", type=str, default="")

    return parser.parse_args()


def train_phase(
    *,
    phase_name: str,
    phase_run_dir: Path,
    topologies: list[str],
    circuit_depth: int,
    total_timesteps: int,
    args,
    device: str,
    model_state_dict,
    eval_seed_offset: int,
    eval_circuit_depth: int,
    min_two_qubit_gates: int,
    eval_min_two_qubit_gates: int,
):
    env = QubitRoutingEnv(
        topologies=topologies,
        matrix_size=args.matrix_size,
        circuit_depth=circuit_depth,
        max_steps=args.max_steps,
        gamma_decay=0.5,
        # Start value; PPO agent will anneal it.
        distance_reward_coeff=args.distance_reward_coeff_start,
        completion_bonus=args.completion_bonus,
        timeout_penalty=args.timeout_penalty,
        min_two_qubit_gates=min_two_qubit_gates,
        circuit_generation_attempts=args.circuit_generation_attempts,
        initial_mapping_strategy="mixed",
        seed=args.seed,
    )

    cfg = PPOConfig(
        total_timesteps=total_timesteps,
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
        eval_interval_updates=args.eval_interval_updates,
        eval_circuits_per_topology=args.eval_circuits_per_topology,
        eval_circuit_depth=eval_circuit_depth,
        eval_min_two_qubit_gates=eval_min_two_qubit_gates,
        eval_circuit_generation_attempts=args.eval_circuit_generation_attempts,
        eval_seed_base=args.eval_seed_base + eval_seed_offset,
        distance_reward_coeff_start=args.distance_reward_coeff_start,
        distance_reward_coeff_end=args.distance_reward_coeff_end,
        run_dir=str(phase_run_dir),
        seed=args.seed,
        device=device,
    )

    print(f"\n=== {phase_name} ===")
    print(f"  topologies={topologies}")
    print(f"  circuit_depth={circuit_depth}")
    print(f"  min_two_qubit_gates={min_two_qubit_gates}")
    print(f"  eval_circuit_depth={eval_circuit_depth}")
    print(f"  eval_min_two_qubit_gates={eval_min_two_qubit_gates}")
    print(f"  total_timesteps={total_timesteps}")
    print(f"  run_dir={phase_run_dir}")

    agent = PPOAgent(env, cfg, model_state_dict=model_state_dict)
    agent.train()
    return {k: v.detach().cpu() for k, v in agent.model.state_dict().items()}


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

    topologies = parse_topologies(args.topologies)
    if not topologies:
        raise ValueError("No valid topology names were provided.")
    if args.min_two_qubit_gates < 0:
        raise ValueError("--min-two-qubit-gates must be >= 0.")
    if args.eval_min_two_qubit_gates < -1:
        raise ValueError("--eval-min-two-qubit-gates must be >= -1.")
    if args.circuit_generation_attempts < 1:
        raise ValueError("--circuit-generation-attempts must be >= 1.")
    if args.eval_circuit_generation_attempts < 1:
        raise ValueError("--eval-circuit-generation-attempts must be >= 1.")

    resolved_eval_min_twoq = (
        args.min_two_qubit_gates
        if args.eval_min_two_qubit_gates < 0
        else args.eval_min_two_qubit_gates
    )

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    config_dump = vars(args).copy()
    config_dump["resolved_device"] = device
    config_dump["project_root"] = str(project_root)
    config_dump["data_root"] = str(data_root)
    config_dump["run_root"] = str(run_root)
    config_dump["run_dir"] = str(run_dir)
    config_dump["resolved_eval_min_two_qubit_gates"] = int(resolved_eval_min_twoq)
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config_dump, f, indent=2)

    print("Starting PPO training with settings:")
    print(f"  PROJECT_ROOT={project_root}")
    print(f"  DATA_ROOT={data_root}")
    print(f"  RUN_ROOT={run_root}")
    print(f"  RUN_DIR={run_dir}")
    print(f"  topologies={topologies} (stage3 if curriculum)")
    print("  initial_mapping_strategy=mixed (80% random, 20% SABRE)")
    print("  strategy_masking=off (topology validity mask only)")
    print("  gamma_decay=0.5")
    print(f"  completion_bonus={args.completion_bonus}")
    print(f"  timeout_penalty={args.timeout_penalty}")
    print(f"  distance_reward_coeff schedule={args.distance_reward_coeff_start} -> {args.distance_reward_coeff_end}")
    print(f"  min_two_qubit_gates(train)={args.min_two_qubit_gates}")
    print(f"  min_two_qubit_gates(eval)={resolved_eval_min_twoq}")
    print(f"  entropy_coef schedule={args.entropy_coef_start} -> {args.entropy_coef_end}")
    print(f"  device={device}")
    print(f"  eval_interval_updates={args.eval_interval_updates}")
    print(f"  eval_circuits_per_topology={args.eval_circuits_per_topology}")
    print(f"  torch_version={torch.__version__}")
    print(f"  cuda_available={torch.cuda.is_available()}")
    print(
        "  torchmetrics="
        + ("OK" if TORCHMETRICS_OK else f"MISSING ({TORCHMETRICS_ERR})")
    )

    final_state = None
    if args.curriculum:
        stage1_topos = parse_topologies(args.stage1_topologies)
        stage2_topos = parse_topologies(args.stage2_topologies)
        if not stage1_topos or not stage2_topos:
            raise ValueError("Curriculum topologies cannot be empty.")

        phases = [
            ("stage1_easy", stage1_topos, args.stage1_depth, args.stage1_steps),
            ("stage2_mid", stage2_topos, args.stage2_depth, args.stage2_steps),
            ("stage3_full", topologies, args.stage3_depth, args.stage3_steps),
        ]

        model_state = None
        eval_seed_offset = 0
        for phase_name, phase_topos, phase_depth, phase_steps in phases:
            if phase_steps <= 0:
                continue
            phase_run_dir = run_dir / phase_name
            phase_run_dir.mkdir(parents=True, exist_ok=True)
            phase_eval_depth = max(1, min(args.eval_circuit_depth, phase_depth))
            phase_train_min_twoq = max(0, min(args.min_two_qubit_gates, phase_depth))
            phase_eval_min_twoq = max(0, min(resolved_eval_min_twoq, phase_eval_depth))
            model_state = train_phase(
                phase_name=phase_name,
                phase_run_dir=phase_run_dir,
                topologies=phase_topos,
                circuit_depth=phase_depth,
                total_timesteps=phase_steps,
                args=args,
                device=device,
                model_state_dict=model_state,
                eval_seed_offset=eval_seed_offset,
                eval_circuit_depth=phase_eval_depth,
                min_two_qubit_gates=phase_train_min_twoq,
                eval_min_two_qubit_gates=phase_eval_min_twoq,
            )
            eval_seed_offset += 1_000_000
        final_state = copy.deepcopy(model_state)
    else:
        phase_run_dir = run_dir / "single_stage"
        phase_run_dir.mkdir(parents=True, exist_ok=True)
        phase_eval_depth = max(1, args.eval_circuit_depth)
        phase_train_min_twoq = max(0, min(args.min_two_qubit_gates, args.circuit_depth))
        phase_eval_min_twoq = max(0, min(resolved_eval_min_twoq, phase_eval_depth))
        final_state = train_phase(
            phase_name="single_stage",
            phase_run_dir=phase_run_dir,
            topologies=topologies,
            circuit_depth=args.circuit_depth,
            total_timesteps=args.total_timesteps,
            args=args,
            device=device,
            model_state_dict=None,
            eval_seed_offset=0,
            eval_circuit_depth=phase_eval_depth,
            min_two_qubit_gates=phase_train_min_twoq,
            eval_min_two_qubit_gates=phase_eval_min_twoq,
        )

    final_model_path = run_dir / "final_model.pt"
    if final_state is not None:
        torch.save(final_state, final_model_path)
        print(f"Saved final model weights to: {final_model_path}")

    if args.save_path and final_state is not None:
        torch.save(final_state, args.save_path)
        print(f"Saved model weights to: {args.save_path}")


if __name__ == "__main__":
    main()
