"""Training loop for D3QN+PER quantum circuit routing agent."""

import json
import signal
import sys
import time
import numpy as np
import torch
from pathlib import Path

from environment import QubitRoutingEnv
from dqn_agent import D3QNAgent
from config import setup_run_dir


def _print(msg):
    """Print with immediate flush (needed for Colab/notebook output)."""
    print(msg, flush=True)


class TrainingLogger:
    """Writes JSONL logs for episodes, train steps, and evaluations."""

    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._ep_file = open(self.log_dir / "episodes.jsonl", "a")
        self._eval_file = open(self.log_dir / "evaluations.jsonl", "a")
        self._step_file = open(self.log_dir / "train_steps.jsonl", "a")

    def log_episode(self, episode, data):
        data["episode"] = episode
        self._ep_file.write(json.dumps(data) + "\n")
        self._ep_file.flush()

    def log_train_step(self, step, data):
        data["step"] = step
        self._step_file.write(json.dumps(data) + "\n")
        self._step_file.flush()

    def log_evaluation(self, episode, data):
        data["episode"] = episode
        self._eval_file.write(json.dumps(data) + "\n")
        self._eval_file.flush()

    def close(self):
        self._ep_file.close()
        self._eval_file.close()
        self._step_file.close()


def train(config, resume_from=None, finetune_from=None):
    """Main training loop."""

    # If resuming, reuse the existing run directory; otherwise create new
    if resume_from:
        ckpt_path = Path(resume_from)
        # checkpoints/file.pt -> run_NNN/
        run_dir = ckpt_path.parent.parent
        config_path = run_dir / "config.json"
        if config_path.exists() and run_dir.name.startswith("run_"):
            # Reuse existing run dir and its subdirectories
            config.run_dir = str(run_dir)
            config.log_dir = str(run_dir / "logs")
            config.checkpoint_dir = str(run_dir / "checkpoints")
            config.figures_dir = str(run_dir / "figures")
            config.eval_dir = str(run_dir / "eval")
            for d in [config.log_dir, config.checkpoint_dir,
                      config.figures_dir, config.eval_dir]:
                Path(d).mkdir(parents=True, exist_ok=True)
        else:
            config = setup_run_dir(config)
    else:
        config = setup_run_dir(config)
    _print(f"Run directory: {config.run_dir}")

    # Seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    # Create environment
    _print("Creating environment...")
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
        seed=config.seed,
    )

    # Curriculum setup
    curriculum_depths = getattr(config, 'curriculum_depths', [])
    curriculum_milestones = getattr(config, 'curriculum_milestones', [])
    use_curriculum = bool(curriculum_depths) and bool(curriculum_milestones)
    if use_curriculum:
        _print(f"Curriculum: depths={curriculum_depths}, "
               f"milestones={curriculum_milestones}")
        env.circuit_depth = curriculum_depths[0]
    num_actions = env.max_edges
    _print("Environment OK.")

    # Create agent
    _print("Creating agent...")
    agent = D3QNAgent(config, num_actions)
    param_count = sum(p.numel() for p in agent.online_net.parameters())
    _print(f"Device: {agent.device}")
    _print(f"Parameters: {param_count:,}")
    _print(f"Actions: {num_actions} | Topologies: {config.topologies}")

    # Resume or fine-tune
    start_episode = 0
    resume_global_step = 0
    resume_elapsed = 0.0
    if resume_from:
        start_episode = agent.load_checkpoint(resume_from)
        # Load extra training state if saved
        ckpt = torch.load(str(resume_from), map_location="cpu", weights_only=False)
        resume_global_step = ckpt.get("global_step", 0)
        resume_elapsed = ckpt.get("elapsed_time", 0.0)
        _print(f"Resumed from episode {start_episode} "
               f"(env steps: {resume_global_step}, elapsed: {resume_elapsed:.0f}s)")
    elif finetune_from:
        # Load network weights only — fresh optimizer, fresh epsilon, new run dir
        ckpt = torch.load(str(finetune_from), map_location=agent.device,
                          weights_only=False)
        agent.online_net.load_state_dict(ckpt["online_net"])
        agent.target_net.load_state_dict(ckpt["target_net"])
        _print(f"Fine-tuning from {finetune_from} "
               f"(weights only, fresh optimizer @ LR={config.lr})")

    # Logger
    logger = TrainingLogger(config.log_dir)

    # Ctrl+C handler
    interrupted = [False]
    current_episode = [start_episode]

    def signal_handler(sig, frame):
        if interrupted[0]:
            raise SystemExit(1)
        interrupted[0] = True
        _print("\nInterrupted — saving emergency checkpoint...")

    old_handler = signal.signal(signal.SIGINT, signal_handler)

    # Rolling stats
    recent_rewards = []
    recent_swaps = []
    recent_completions = []
    recent_gates_pct = []  # % of gates routed per episode
    recent_loss = []
    recent_q = []

    global_step = resume_global_step if resume_global_step else 0
    t_start = time.time() - resume_elapsed
    total_eps = config.total_episodes - start_episode

    # Try to use tqdm for progress bar
    try:
        from tqdm.auto import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    _print(f"\nStarting training: episodes {start_episode} → {config.total_episodes}")
    _print(f"{'='*60}")

    if has_tqdm:
        pbar = tqdm(
            range(start_episode, config.total_episodes),
            desc="Training",
            unit="ep",
            initial=0,
            total=total_eps,
            dynamic_ncols=True,
        )
    else:
        pbar = range(start_episode, config.total_episodes)

    try:
        for episode in pbar:
            if interrupted[0]:
                break

            current_episode[0] = episode
            ep_t0 = time.time()
            obs, info = env.reset()

            # Skip episodes with no two-qubit gates
            if info["done"]:
                logger.log_episode(episode, {
                    "reward": 0.0, "steps": 0, "swaps": 0,
                    "gates": info["n_gates"], "completed": True,
                    "topology": info["topology"],
                    "epsilon": round(agent.epsilon, 4),
                })
                continue

            action_mask = env.get_action_mask()
            episode_reward = 0.0
            ep_steps = 0

            while True:
                action = agent.select_action(obs, action_mask)
                next_obs, reward, terminated, truncated, info = env.step(action)
                next_mask = env.get_action_mask()

                # Store with terminated (not truncated) as done flag
                # so truncated episodes still bootstrap Q(s')
                agent.store_transition(
                    obs, action, reward, next_obs, terminated, next_mask
                )

                global_step += 1
                ep_steps += 1

                # Train step
                if global_step % config.train_freq == 0:
                    metrics = agent.train_step()
                    if metrics:
                        recent_loss.append(metrics["loss"])
                        recent_q.append(metrics["mean_q"])
                        if agent._train_steps % 100 == 0:
                            logger.log_train_step(agent._train_steps, metrics)
                        if agent._train_steps % config.target_update_freq == 0:
                            agent.update_target_network()

                agent.update_epsilon()
                episode_reward += reward
                obs = next_obs
                action_mask = next_mask

                if terminated or truncated:
                    agent.end_episode()
                    break

            # Curriculum: update circuit depth based on progress
            if use_curriculum:
                progress = (episode + 1) / config.total_episodes
                depth_idx = 0
                for mi, ms in enumerate(curriculum_milestones):
                    if progress >= ms:
                        depth_idx = mi + 1
                depth_idx = min(depth_idx, len(curriculum_depths) - 1)
                new_depth = curriculum_depths[depth_idx]
                if new_depth != env.circuit_depth:
                    env.circuit_depth = new_depth
                    if has_tqdm:
                        from tqdm.auto import tqdm as _tqdm
                        _tqdm.write(f"  Curriculum: depth → {new_depth} at ep{episode+1}")
                    else:
                        _print(f"  Curriculum: depth → {new_depth} at ep{episode+1}")

            # Log episode
            gates_pct = (info["total_gates_executed"] / info["n_gates"]
                         if info["n_gates"] > 0 else 1.0)
            ep_data = {
                "reward": round(episode_reward, 3),
                "steps": info["step_count"],
                "swaps": info["total_swaps"],
                "gates": info["n_gates"],
                "gates_routed": info["total_gates_executed"],
                "gates_pct": round(gates_pct, 3),
                "completed": info["done"],
                "topology": info["topology"],
                "epsilon": round(agent.epsilon, 4),
            }
            logger.log_episode(episode, ep_data)

            recent_rewards.append(episode_reward)
            recent_swaps.append(info["total_swaps"])
            recent_completions.append(int(info["done"]))
            if info["n_gates"] > 0:
                recent_gates_pct.append(
                    info["total_gates_executed"] / info["n_gates"]
                )
            else:
                recent_gates_pct.append(1.0)

            ep_time = time.time() - ep_t0

            # Update tqdm progress bar
            if has_tqdm:
                n = min(len(recent_rewards), config.log_every)
                postfix = {
                    "R": f"{np.mean(recent_rewards[-n:]):.1f}",
                    "SWAPs": f"{np.mean(recent_swaps[-n:]):.0f}",
                    "Gates": f"{np.mean(recent_gates_pct[-n:]):.0%}",
                    "Done": f"{np.mean(recent_completions[-n:]):.0%}",
                    "\u03b5": f"{agent.epsilon:.3f}",
                }
                if recent_loss:
                    postfix["Loss"] = f"{np.mean(recent_loss[-n:]):.3f}"
                    postfix["Q"] = f"{np.mean(recent_q[-n:]):.1f}"
                pbar.set_postfix(postfix, refresh=True)

            # Detailed console output every log_every episodes
            if (episode + 1) % config.log_every == 0:
                n = min(len(recent_rewards), config.log_every)
                elapsed = time.time() - t_start
                loss_str = f"{np.mean(recent_loss[-n:]):.3f}" if recent_loss else "n/a"
                q_str = f"{np.mean(recent_q[-n:]):.1f}" if recent_q else "n/a"
                msg = (
                    f"Ep {episode+1}/{config.total_episodes} "
                    f"({elapsed:.0f}s) | "
                    f"R: {np.mean(recent_rewards[-n:]):.1f} | "
                    f"SWAPs: {np.mean(recent_swaps[-n:]):.0f} | "
                    f"Gates: {np.mean(recent_gates_pct[-n:]):.0%} | "
                    f"Done: {np.mean(recent_completions[-n:]):.0%} | "
                    f"Loss: {loss_str} | Q: {q_str} | "
                    f"\u03b5: {agent.epsilon:.3f} | "
                    f"Steps: {global_step} | "
                    f"ep_time: {ep_time:.1f}s"
                )
                if has_tqdm:
                    tqdm.write(msg)
                else:
                    _print(msg)

            # Periodic evaluation + figures + save eval results
            if (episode + 1) % config.eval_every == 0:
                if has_tqdm:
                    tqdm.write(f"\n--- Evaluation at episode {episode+1} ---")
                else:
                    _print(f"\n--- Evaluation at episode {episode+1} ---")
                _run_periodic_eval(
                    agent, env, config, episode, logger, has_tqdm
                )

            # Checkpoint
            if (episode + 1) % config.checkpoint_every == 0:
                ckpt_path = (
                    Path(config.checkpoint_dir)
                    / f"checkpoint_ep{episode+1}.pt"
                )
                agent.save_checkpoint(str(ckpt_path), episode + 1, extra={
                    "global_step": global_step,
                    "elapsed_time": time.time() - t_start,
                })
                msg = f"  Saved checkpoint: {ckpt_path}"
                if has_tqdm:
                    tqdm.write(msg)
                else:
                    _print(msg)

    finally:
        if has_tqdm and hasattr(pbar, 'close'):
            pbar.close()

        ep = current_episode[0]
        extra = {
            "global_step": global_step,
            "elapsed_time": time.time() - t_start,
        }
        if interrupted[0]:
            ckpt_path = (
                Path(config.checkpoint_dir) / "checkpoint_emergency.pt"
            )
            agent.save_checkpoint(str(ckpt_path), ep, extra=extra)
            _print(f"Emergency checkpoint: {ckpt_path}")
        else:
            ckpt_path = (
                Path(config.checkpoint_dir) / "checkpoint_final.pt"
            )
            agent.save_checkpoint(str(ckpt_path), config.total_episodes, extra=extra)
            _print(f"Final checkpoint: {ckpt_path}")

        # Generate final figures
        try:
            from visualize import plot_training_curves
            plot_training_curves(config.log_dir, config.figures_dir)
        except Exception as e:
            _print(f"Warning: could not generate final figures: {e}")

        logger.close()
        signal.signal(signal.SIGINT, old_handler)

    elapsed = time.time() - t_start
    _print(f"Training complete. {elapsed:.0f}s total.")
    _print(f"All outputs in: {config.run_dir}")

    # Save results summary
    try:
        _save_results_summary(config, elapsed, global_step)
    except Exception as e:
        _print(f"Warning: could not save results summary: {e}")


def _save_results_summary(config, elapsed_seconds, global_step):
    """Write results_summary.json with all key metrics from the run."""
    run_dir = Path(config.run_dir)
    log_dir = Path(config.log_dir)

    summary = {
        "run_dir": str(run_dir),
        "topologies": config.topologies,
        "total_episodes": config.total_episodes,
        "elapsed_seconds": round(elapsed_seconds, 1),
        "elapsed_human": f"{elapsed_seconds/3600:.1f}h",
        "global_steps": global_step,
    }

    # --- Training metrics from episodes.jsonl (last 500 episodes) ---
    ep_file = log_dir / "episodes.jsonl"
    if ep_file.exists():
        episodes = []
        with open(ep_file) as f:
            for line in f:
                try:
                    episodes.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if episodes:
            last_n = episodes[-500:]
            summary["training"] = {
                "total_logged_episodes": len(episodes),
                "final_500ep": {
                    "mean_reward": round(np.mean([e["reward"] for e in last_n]), 2),
                    "mean_swaps": round(np.mean([e["swaps"] for e in last_n]), 1),
                    "mean_gates_pct": round(np.mean([e.get("gates_pct", 0) for e in last_n]), 3),
                    "completion_rate": round(np.mean([int(e["completed"]) for e in last_n]), 3),
                    "mean_steps": round(np.mean([e["steps"] for e in last_n]), 1),
                },
                "final_epsilon": episodes[-1].get("epsilon", None),
            }

    # --- All evaluation checkpoints from evaluations.jsonl ---
    eval_file = log_dir / "evaluations.jsonl"
    if eval_file.exists():
        evals = []
        with open(eval_file) as f:
            for line in f:
                try:
                    evals.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if evals:
            # Find best eval by swap ratio
            best = min(evals, key=lambda e: e.get("mean_swap_ratio", 999))
            latest = evals[-1]

            summary["evaluations"] = {
                "total_evals": len(evals),
                "best": {
                    "episode": best.get("episode"),
                    "completion_rate": best.get("completion_rate"),
                    "mean_agent_swaps": best.get("mean_agent_swaps"),
                    "mean_sabre_swaps": best.get("mean_sabre_swaps"),
                    "mean_swap_ratio": best.get("mean_swap_ratio"),
                },
                "final": {
                    "episode": latest.get("episode"),
                    "completion_rate": latest.get("completion_rate"),
                    "mean_agent_swaps": latest.get("mean_agent_swaps"),
                    "mean_sabre_swaps": latest.get("mean_sabre_swaps"),
                    "mean_swap_ratio": latest.get("mean_swap_ratio"),
                },
                "progression": [
                    {
                        "episode": e.get("episode"),
                        "completion_rate": e.get("completion_rate"),
                        "mean_swap_ratio": e.get("mean_swap_ratio"),
                    }
                    for e in evals
                ],
            }

    # --- Config snapshot (key hyperparameters only) ---
    summary["config"] = {
        "lr": config.lr,
        "epsilon_end": config.epsilon_end,
        "epsilon_decay_steps": config.epsilon_decay_steps,
        "conv_channels": config.conv_channels,
        "dueling_hidden": config.dueling_hidden,
        "batch_size": config.batch_size,
        "buffer_capacity": config.buffer_capacity,
        "initial_mapping_strategy": config.initial_mapping_strategy,
        "distance_reward_coeff": config.distance_reward_coeff,
        "repetition_penalty": config.repetition_penalty,
        "tau": config.tau,
        "target_update_freq": config.target_update_freq,
    }

    out_path = run_dir / "results_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    _print(f"Results summary: {out_path}")


def _run_periodic_eval(agent, env, config, episode, logger, has_tqdm=False):
    """Run eval, log summary, save results + update figures."""
    from evaluate import run_evaluation, save_eval_results
    from visualize import plot_training_curves, plot_eval_comparison

    eval_results = run_evaluation(
        agent, env, config, config.eval_episodes,
        log_trajectories=True,
    )
    summary = eval_results["summary"]
    logger.log_evaluation(episode, summary)

    msg = (
        f"  EVAL | Agent: {summary['mean_agent_swaps']:.1f} "
        f"SABRE: {summary['mean_sabre_swaps']:.1f} | "
        f"Ratio: {summary['mean_swap_ratio']:.2f} | "
        f"Done: {summary['completion_rate']:.0%}"
    )
    if has_tqdm:
        from tqdm.auto import tqdm
        tqdm.write(msg)
    else:
        _print(msg)

    # Save eval results with episode number
    eval_path = Path(config.eval_dir) / f"eval_ep{episode+1}.json"
    save_eval_results(eval_results, eval_path)

    # Update training curves figure (overwrites previous)
    try:
        plot_training_curves(config.log_dir, config.figures_dir)
    except Exception:
        pass

    # Save eval comparison chart for this checkpoint
    try:
        results = eval_results.get("results", [])
        if results:
            fig_path = (
                Path(config.figures_dir)
                / f"eval_comparison_ep{episode+1}.png"
            )
            plot_eval_comparison(results, str(fig_path))
    except Exception:
        pass
