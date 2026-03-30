"""CLI entry point for D3QN+PER quantum circuit routing."""

import argparse
import sys
from pathlib import Path

# Add src/ to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def cmd_train(args):
    from config import (
        TrainConfig, linear5_sanity_config,
        heavy_hex_config, multi_topology_config,
    )
    from train import train

    presets = {
        "linear5": linear5_sanity_config,
        "heavy_hex": heavy_hex_config,
        "multi": multi_topology_config,
    }

    if args.config:
        config = TrainConfig.load(args.config)
    else:
        config = presets[args.preset]()

    # CLI overrides
    if args.output_dir:
        config.output_base = args.output_dir
    if args.episodes is not None:
        config.total_episodes = args.episodes
    if args.device:
        config.device = args.device
    if args.seed is not None:
        config.seed = args.seed
    if args.lr is not None:
        config.lr = args.lr
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.save_buffer:
        config.save_buffer = True

    train(config, resume_from=args.resume, finetune_from=args.finetune)


def cmd_evaluate(args):
    from config import TrainConfig
    from dqn_agent import D3QNAgent
    from environment import QubitRoutingEnv
    from evaluate import (
        run_evaluation, run_qasmbench_evaluation, save_eval_results,
    )

    # Load config: try run dir first, then checkpoint dir
    if args.config:
        config = TrainConfig.load(args.config)
    else:
        ckpt_path = Path(args.checkpoint)
        # Check: outputs/run_NNN/checkpoints/file.pt -> outputs/run_NNN/config.json
        run_dir = ckpt_path.parent.parent
        config_path = run_dir / "config.json"
        if not config_path.exists():
            # Fallback: config next to checkpoint
            config_path = ckpt_path.parent / "config.json"
        if config_path.exists():
            config = TrainConfig.load(str(config_path))
        else:
            print("No config.json found. Use --config.")
            sys.exit(1)

    config.device = args.device or config.device

    env = QubitRoutingEnv(
        topologies=config.topologies,
        circuit_depth=config.circuit_depth,
        max_steps=config.max_steps,
        gamma_decay=config.gamma_decay,
        distance_reward_coeff=config.distance_reward_coeff,
        completion_bonus=config.completion_bonus,
        timeout_penalty=config.timeout_penalty,
        repetition_penalty=config.repetition_penalty,
        gate_execution_reward=getattr(config, 'gate_execution_reward', 1.0),
        matrix_size=config.matrix_size,
        initial_mapping_strategy=config.initial_mapping_strategy,
        topology_weights=config.topology_weights or None,
    )

    agent = D3QNAgent(config, env.max_edges)
    agent.load_checkpoint(args.checkpoint)
    print(f"Loaded checkpoint: {args.checkpoint}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nEvaluating on {args.episodes} random circuits per topology...")
    eval_out = run_evaluation(
        agent, env, config, args.episodes,
        log_trajectories=args.save_trajectories,
    )
    summary = eval_out["summary"]
    print(f"  Completion: {summary['completion_rate']:.0%}")
    print(f"  Agent SWAPs: {summary['mean_agent_swaps']:.1f}")
    print(f"  SABRE SWAPs: {summary['mean_sabre_swaps']:.1f}")
    print(f"  Ratio: {summary['mean_swap_ratio']:.3f}")

    save_eval_results(eval_out, output_dir / "random_eval.json")

    if args.qasmbench:
        print(f"\nEvaluating on QASMBench circuits from {args.qasmbench}...")
        qasm_out = run_qasmbench_evaluation(
            agent, env, config, args.qasmbench,
        )
        qsummary = qasm_out["summary"]
        print(f"  Circuits tested: {qsummary['total_circuits']}")
        print(f"  Completed: {qsummary['completed']}")
        if qsummary.get("mean_swap_ratio"):
            print(f"  Ratio: {qsummary['mean_swap_ratio']:.3f}")
        save_eval_results(qasm_out, output_dir / "qasmbench_eval.json")


def cmd_visualize(args):
    from visualize import (
        plot_training_curves, plot_eval_comparison,
        plot_swap_ratio_distribution, create_routing_gif,
        create_side_by_side_gif, create_routing_summary_table,
    )

    # If given a run dir, auto-detect log/eval/figures paths
    if args.run_dir:
        run = Path(args.run_dir)
        args.log_dir = args.log_dir or str(run / "logs")
        output_dir = run / "figures"
        # Find latest eval
        eval_dir = run / "eval"
        if eval_dir.exists():
            evals = sorted(eval_dir.glob("eval_ep*.json"))
            if evals and not args.eval_results:
                args.eval_results = str(evals[-1])
    else:
        output_dir = Path(args.output_dir)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.log_dir:
        print("Plotting training curves...")
        plot_training_curves(args.log_dir, output_dir)

    if args.eval_results:
        import json
        with open(args.eval_results) as f:
            data = json.load(f)

        results = data.get("results", [])
        if results:
            print("Plotting eval comparison...")
            plot_eval_comparison(results, output_dir / "eval_comparison.png")
            plot_swap_ratio_distribution(
                results, output_dir / "swap_ratio_dist.png"
            )
            table = create_routing_summary_table(results)
            (output_dir / "eval_summary.md").write_text(table)
            print(f"Summary table: {output_dir / 'eval_summary.md'}")

        if args.gif and data.get("trajectories"):
            for i, traj in enumerate(data["trajectories"][:5]):
                create_routing_gif(
                    traj, output_dir / f"routing_{i}.gif", fps=args.fps
                )
                create_side_by_side_gif(
                    traj, output_dir / f"routing_vs_sabre_{i}.gif",
                    fps=args.fps,
                )

    if args.trajectory:
        import json
        with open(args.trajectory) as f:
            traj = json.load(f)
        create_routing_gif(traj, output_dir / "routing.gif", fps=args.fps)


def main():
    parser = argparse.ArgumentParser(
        description="D3QN+PER Quantum Circuit Routing"
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- Train ---
    tp = subparsers.add_parser("train", help="Train the agent")
    tp.add_argument(
        "--preset", choices=["linear5", "heavy_hex", "multi"],
        default="heavy_hex",
    )
    tp.add_argument("--config", type=str, help="Path to config.json")
    tp.add_argument("--resume", type=str, help="Checkpoint to resume from")
    tp.add_argument("--finetune", type=str,
                    help="Checkpoint to fine-tune from (loads weights only, "
                         "fresh optimizer and epsilon, new run dir)")
    tp.add_argument("--output-dir", type=str, help="Base output dir (default: outputs)")
    tp.add_argument("--episodes", type=int)
    tp.add_argument("--device", type=str)
    tp.add_argument("--seed", type=int)
    tp.add_argument("--lr", type=float)
    tp.add_argument("--batch-size", type=int)
    tp.add_argument("--save-buffer", action="store_true")

    # --- Evaluate ---
    ep = subparsers.add_parser("evaluate", help="Evaluate a checkpoint")
    ep.add_argument("--checkpoint", type=str, required=True)
    ep.add_argument("--config", type=str)
    ep.add_argument("--episodes", type=int, default=50)
    ep.add_argument("--qasmbench", type=str, help="Path to QASMBench dir")
    ep.add_argument("--save-trajectories", action="store_true")
    ep.add_argument("--output-dir", type=str, default="eval_results")
    ep.add_argument("--device", type=str)

    # --- Visualize ---
    vp = subparsers.add_parser("visualize", help="Generate visualizations")
    vp.add_argument("--run-dir", type=str, help="Path to outputs/run_NNN")
    vp.add_argument("--log-dir", type=str)
    vp.add_argument("--eval-results", type=str, help="Path to eval JSON")
    vp.add_argument("--trajectory", type=str, help="Path to trajectory JSON")
    vp.add_argument("--output-dir", type=str, default="figures")
    vp.add_argument("--gif", action="store_true")
    vp.add_argument("--fps", type=int, default=1)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "visualize":
        cmd_visualize(args)


if __name__ == "__main__":
    main()
