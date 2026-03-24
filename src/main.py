"""Training entrypoint for PPO-based quantum circuit routing."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
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


def build_topology_sampling_weights(topologies: list[str], args) -> list[float]:
    """Build normalized per-topology sampling weights for env.reset()."""
    raw_weights: list[float] = []
    for topo in topologies:
        topo_l = topo.lower()
        if "linear" in topo_l:
            w = args.linear_topology_weight
        elif "grid" in topo_l:
            w = args.grid_topology_weight
        elif "heavy_hex" in topo_l or "heavy-hex" in topo_l:
            w = args.heavy_hex_topology_weight
        else:
            w = args.other_topology_weight
        raw_weights.append(max(0.0, float(w)))

    total = float(sum(raw_weights))
    if total <= 0:
        raise ValueError(
            "Topology sampling weights sum to zero. Increase at least one of "
            "--linear-topology-weight / --grid-topology-weight / "
            "--heavy-hex-topology-weight / --other-topology-weight."
        )
    return [w / total for w in raw_weights]


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
    parser.add_argument(
        "--stage2-topologies",
        type=str,
        default="linear_5,grid_3x3",
    )
    parser.add_argument("--stage1-steps", type=int, default=150000)
    parser.add_argument("--stage2-steps", type=int, default=250000)
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
    parser.add_argument("--completion-bonus", type=float, default=15.0)
    parser.add_argument("--timeout-penalty", type=float, default=-8.0)
    parser.add_argument("--gate-reward-coeff", type=float, default=1.0)
    parser.add_argument("--step-penalty", type=float, default=-0.05)
    parser.add_argument("--reverse-swap-penalty", type=float, default=-0.2)
    parser.add_argument(
        "--repeat-swap-penalty-coeff",
        type=float,
        default=-0.15,
        help=(
            "Progressive penalty coefficient for consecutive reuse of the same "
            "physical SWAP edge."
        ),
    )
    parser.add_argument(
        "--repeat-swap-penalty-cap",
        type=float,
        default=-2.0,
        help="Lower bound (negative cap) for repeated-edge penalty.",
    )
    parser.add_argument(
        "--no-progress-penalty-coeff",
        type=float,
        default=-0.03,
        help=(
            "Progressive penalty coefficient when no new gate is executed "
            "at a step."
        ),
    )
    parser.add_argument(
        "--no-progress-penalty-cap",
        type=float,
        default=-1.5,
        help="Lower bound (negative cap) for no-progress penalty.",
    )
    parser.add_argument(
        "--max-steps-per-two-qubit-gate",
        type=float,
        default=0.0,
        help=(
            "If > 0, uses dynamic per-episode max steps: "
            "ceil(num_2q_gates * factor)."
        ),
    )
    parser.add_argument(
        "--max-steps-min",
        type=int,
        default=0,
        help="Optional lower bound for dynamic max steps (0 disables).",
    )
    parser.add_argument(
        "--max-steps-max",
        type=int,
        default=0,
        help="Optional upper bound for dynamic max steps (0 uses --max-steps).",
    )
    parser.add_argument("--linear-topology-weight", type=float, default=0.5)
    parser.add_argument("--grid-topology-weight", type=float, default=1.5)
    parser.add_argument("--heavy-hex-topology-weight", type=float, default=1.5)
    parser.add_argument("--other-topology-weight", type=float, default=1.0)
    parser.add_argument("--min-two-qubit-gates", type=int, default=6)
    parser.add_argument("--circuit-generation-attempts", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--total-timesteps", type=int, default=200000)
    parser.add_argument("--rollout-steps", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.97)
    parser.add_argument("--clip-range", type=float, default=0.15)
    parser.add_argument("--update-epochs", type=int, default=8)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--entropy-coef-start", type=float, default=0.003)
    parser.add_argument("--entropy-coef-end", type=float, default=0.0001)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.015)
    parser.add_argument("--log-interval-updates", type=int, default=5)
    parser.add_argument("--checkpoint-interval-updates", type=int, default=25)
    parser.add_argument("--eval-interval-updates", type=int, default=20)
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
    parser.add_argument("--trace-interval-updates", type=int, default=20)
    parser.add_argument("--trace-cases-per-topology", type=int, default=1)
    parser.add_argument(
        "--trace-max-steps",
        type=int,
        default=500,
        help="Per-trace episode cap for periodic diagnostic traces.",
    )
    parser.add_argument("--trace-alert-dom-threshold", type=float, default=0.60)
    parser.add_argument("--trace-alert-backtrack-threshold", type=float, default=0.50)
    parser.add_argument("--trace-alert-patience", type=int, default=2)
    parser.add_argument("--best-model-max-eval-timeout-rate", type=float, default=0.95)
    parser.add_argument("--best-model-max-trace-timeout-rate", type=float, default=0.95)
    parser.add_argument("--best-model-max-trace-dom-ratio", type=float, default=0.90)
    parser.add_argument("--best-model-max-trace-backtrack-rate", type=float, default=0.80)
    parser.add_argument(
        "--best-model-require-trace",
        action="store_true",
        help="Require trace metrics to consider checkpoint exploitable for best-model selection.",
    )
    parser.add_argument(
        "--best-model-reject-trace-alert",
        dest="best_model_reject_trace_alert",
        action="store_true",
        help="Reject checkpoints flagged by trace loop-risk alert in best-model selection.",
    )
    parser.add_argument(
        "--no-best-model-reject-trace-alert",
        dest="best_model_reject_trace_alert",
        action="store_false",
    )
    parser.add_argument(
        "--best-model-use-latest-trace",
        dest="best_model_use_latest_trace",
        action="store_true",
        help="Use last available trace metrics when eval and trace intervals are not aligned.",
    )
    parser.add_argument(
        "--no-best-model-use-latest-trace",
        dest="best_model_use_latest_trace",
        action="store_false",
    )
    parser.set_defaults(
        best_model_reject_trace_alert=True,
        best_model_use_latest_trace=True,
    )
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
    topology_sampling_weights = build_topology_sampling_weights(topologies, args)
    env = QubitRoutingEnv(
        topologies=topologies,
        topology_sampling_weights=topology_sampling_weights,
        matrix_size=args.matrix_size,
        circuit_depth=circuit_depth,
        max_steps=args.max_steps,
        gamma_decay=0.5,
        # Start value; PPO agent will anneal it.
        distance_reward_coeff=args.distance_reward_coeff_start,
        completion_bonus=args.completion_bonus,
        timeout_penalty=args.timeout_penalty,
        gate_reward_coeff=args.gate_reward_coeff,
        step_penalty=args.step_penalty,
        reverse_swap_penalty=args.reverse_swap_penalty,
        repeat_swap_penalty_coeff=args.repeat_swap_penalty_coeff,
        repeat_swap_penalty_cap=args.repeat_swap_penalty_cap,
        no_progress_penalty_coeff=args.no_progress_penalty_coeff,
        no_progress_penalty_cap=args.no_progress_penalty_cap,
        max_steps_per_two_qubit_gate=args.max_steps_per_two_qubit_gate,
        max_steps_min=args.max_steps_min,
        max_steps_max=args.max_steps_max,
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
        trace_interval_updates=args.trace_interval_updates,
        trace_cases_per_topology=args.trace_cases_per_topology,
        trace_max_steps=args.trace_max_steps,
        trace_alert_dom_threshold=args.trace_alert_dom_threshold,
        trace_alert_backtrack_threshold=args.trace_alert_backtrack_threshold,
        trace_alert_patience=args.trace_alert_patience,
        best_model_max_eval_timeout_rate=args.best_model_max_eval_timeout_rate,
        best_model_max_trace_timeout_rate=args.best_model_max_trace_timeout_rate,
        best_model_max_trace_dom_ratio=args.best_model_max_trace_dom_ratio,
        best_model_max_trace_backtrack_rate=args.best_model_max_trace_backtrack_rate,
        best_model_reject_trace_alert=args.best_model_reject_trace_alert,
        best_model_require_trace=args.best_model_require_trace,
        best_model_use_latest_trace=args.best_model_use_latest_trace,
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
    topo_weight_view = {
        topo: round(float(w), 3)
        for topo, w in zip(topologies, topology_sampling_weights)
    }
    print(f"  topology_sampling_weights={topo_weight_view}")
    print(f"  trace_interval_updates={args.trace_interval_updates}")
    print(f"  trace_cases_per_topology={args.trace_cases_per_topology}")
    print(f"  trace_alert_thresholds=(dom>={args.trace_alert_dom_threshold}, backtrack>={args.trace_alert_backtrack_threshold}, patience={args.trace_alert_patience})")
    print(
        "  best_model_exploitability="
        f"(eval_timeout<={args.best_model_max_eval_timeout_rate}, "
        f"trace_timeout<={args.best_model_max_trace_timeout_rate}, "
        f"trace_dom<={args.best_model_max_trace_dom_ratio}, "
        f"trace_backtrack<={args.best_model_max_trace_backtrack_rate}, "
        f"require_trace={args.best_model_require_trace}, "
        f"reject_trace_alert={args.best_model_reject_trace_alert}, "
        f"use_latest_trace={args.best_model_use_latest_trace})"
    )
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
    if args.repeat_swap_penalty_coeff > 0:
        raise ValueError("--repeat-swap-penalty-coeff must be <= 0 (penalty).")
    if args.repeat_swap_penalty_cap > 0:
        raise ValueError("--repeat-swap-penalty-cap must be <= 0.")
    if args.no_progress_penalty_coeff > 0:
        raise ValueError("--no-progress-penalty-coeff must be <= 0 (penalty).")
    if args.no_progress_penalty_cap > 0:
        raise ValueError("--no-progress-penalty-cap must be <= 0.")
    if args.max_steps_per_two_qubit_gate < 0:
        raise ValueError("--max-steps-per-two-qubit-gate must be >= 0.")
    if args.max_steps_min < 0:
        raise ValueError("--max-steps-min must be >= 0.")
    if args.max_steps_max < 0:
        raise ValueError("--max-steps-max must be >= 0.")
    if args.max_steps_min > 0 and args.max_steps_max > 0 and args.max_steps_max < args.max_steps_min:
        raise ValueError("--max-steps-max must be >= --max-steps-min.")
    if args.linear_topology_weight < 0:
        raise ValueError("--linear-topology-weight must be >= 0.")
    if args.grid_topology_weight < 0:
        raise ValueError("--grid-topology-weight must be >= 0.")
    if args.heavy_hex_topology_weight < 0:
        raise ValueError("--heavy-hex-topology-weight must be >= 0.")
    if args.other_topology_weight < 0:
        raise ValueError("--other-topology-weight must be >= 0.")
    if args.trace_interval_updates < 0:
        raise ValueError("--trace-interval-updates must be >= 0.")
    if args.trace_cases_per_topology < 0:
        raise ValueError("--trace-cases-per-topology must be >= 0.")
    if args.trace_max_steps < 1:
        raise ValueError("--trace-max-steps must be >= 1.")
    if not (0.0 <= args.trace_alert_dom_threshold <= 1.0):
        raise ValueError("--trace-alert-dom-threshold must be in [0, 1].")
    if not (0.0 <= args.trace_alert_backtrack_threshold <= 1.0):
        raise ValueError("--trace-alert-backtrack-threshold must be in [0, 1].")
    if args.trace_alert_patience < 1:
        raise ValueError("--trace-alert-patience must be >= 1.")
    if not (0.0 <= args.best_model_max_eval_timeout_rate <= 1.0):
        raise ValueError("--best-model-max-eval-timeout-rate must be in [0, 1].")
    if not (0.0 <= args.best_model_max_trace_timeout_rate <= 1.0):
        raise ValueError("--best-model-max-trace-timeout-rate must be in [0, 1].")
    if not (0.0 <= args.best_model_max_trace_dom_ratio <= 1.0):
        raise ValueError("--best-model-max-trace-dom-ratio must be in [0, 1].")
    if not (0.0 <= args.best_model_max_trace_backtrack_rate <= 1.0):
        raise ValueError("--best-model-max-trace-backtrack-rate must be in [0, 1].")

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
    print(f"  gate_reward_coeff={args.gate_reward_coeff}")
    print(f"  step_penalty={args.step_penalty}")
    print(f"  reverse_swap_penalty={args.reverse_swap_penalty}")
    print(f"  repeat_swap_penalty_coeff={args.repeat_swap_penalty_coeff}")
    print(f"  repeat_swap_penalty_cap={args.repeat_swap_penalty_cap}")
    print(f"  no_progress_penalty_coeff={args.no_progress_penalty_coeff}")
    print(f"  no_progress_penalty_cap={args.no_progress_penalty_cap}")
    print(
        "  topology_weights="
        f"(linear={args.linear_topology_weight}, "
        f"grid={args.grid_topology_weight}, "
        f"heavy_hex={args.heavy_hex_topology_weight}, "
        f"other={args.other_topology_weight})"
    )
    print(f"  distance_reward_coeff schedule={args.distance_reward_coeff_start} -> {args.distance_reward_coeff_end}")
    print(f"  min_two_qubit_gates(train)={args.min_two_qubit_gates}")
    print(f"  min_two_qubit_gates(eval)={resolved_eval_min_twoq}")
    if args.max_steps_per_two_qubit_gate > 0:
        dynamic_max_cfg = (
            f"{args.max_steps_per_two_qubit_gate}x2q "
            f"(min={args.max_steps_min if args.max_steps_min > 0 else 1}, "
            f"max={args.max_steps_max if args.max_steps_max > 0 else args.max_steps})"
        )
    else:
        dynamic_max_cfg = "off"
    print(f"  dynamic_max_steps={dynamic_max_cfg}")
    print(f"  entropy_coef schedule={args.entropy_coef_start} -> {args.entropy_coef_end}")
    print(f"  device={device}")
    print(f"  eval_interval_updates={args.eval_interval_updates}")
    print(f"  eval_circuits_per_topology={args.eval_circuits_per_topology}")
    print(f"  trace_interval_updates={args.trace_interval_updates}")
    print(f"  trace_cases_per_topology={args.trace_cases_per_topology}")
    print(f"  trace_max_steps={args.trace_max_steps}")
    print(
        "  trace_alert_thresholds="
        f"(dom>={args.trace_alert_dom_threshold}, "
        f"backtrack>={args.trace_alert_backtrack_threshold}, "
        f"patience={args.trace_alert_patience})"
    )
    print(
        "  best_model_exploitability="
        f"(eval_timeout<={args.best_model_max_eval_timeout_rate}, "
        f"trace_timeout<={args.best_model_max_trace_timeout_rate}, "
        f"trace_dom<={args.best_model_max_trace_dom_ratio}, "
        f"trace_backtrack<={args.best_model_max_trace_backtrack_rate}, "
        f"require_trace={args.best_model_require_trace}, "
        f"reject_trace_alert={args.best_model_reject_trace_alert}, "
        f"use_latest_trace={args.best_model_use_latest_trace})"
    )
    print(f"  torch_version={torch.__version__}")
    print(f"  cuda_available={torch.cuda.is_available()}")
    print(
        "  torchmetrics="
        + ("OK" if TORCHMETRICS_OK else f"MISSING ({TORCHMETRICS_ERR})")
    )

    final_state = None
    last_phase_run_dir: Path | None = None
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
            last_phase_run_dir = phase_run_dir
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
        last_phase_run_dir = phase_run_dir
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

    if last_phase_run_dir is not None:
        best_eval_model = last_phase_run_dir / "best_model.pt"
        if best_eval_model.exists():
            shutil.copy2(best_eval_model, run_dir / "best_model.pt")
            print(f"Copied best eval model to: {run_dir / 'best_model.pt'}")
        best_eval_meta = last_phase_run_dir / "best_eval_metrics.json"
        if best_eval_meta.exists():
            shutil.copy2(best_eval_meta, run_dir / "best_eval_metrics.json")
        best_train_model = last_phase_run_dir / "best_train_model.pt"
        if best_train_model.exists():
            shutil.copy2(best_train_model, run_dir / "best_train_model.pt")

    final_model_path = run_dir / "final_model.pt"
    if final_state is not None:
        torch.save(final_state, final_model_path)
        print(f"Saved final model weights to: {final_model_path}")

    if args.save_path and final_state is not None:
        torch.save(final_state, args.save_path)
        print(f"Saved model weights to: {args.save_path}")


if __name__ == "__main__":
    main()
