"""Training loop for D3QN+PER quantum circuit routing agent."""

import json
import signal
import time
import numpy as np
import torch
from pathlib import Path

from environment import QubitRoutingEnv
from dqn_agent import D3QNAgent
from config import setup_run_dir


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


def train(config, resume_from=None):
    """Main training loop."""

    # Set up unified run directory: outputs/run_NNN/
    config = setup_run_dir(config)
    print(f"Run directory: {config.run_dir}")

    # Seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    # Create environment
    env = QubitRoutingEnv(
        topologies=config.topologies,
        circuit_depth=config.circuit_depth,
        max_steps=config.max_steps,
        gamma_decay=config.gamma_decay,
        distance_reward_coeff=config.distance_reward_coeff,
        completion_bonus=config.completion_bonus,
        timeout_penalty=config.timeout_penalty,
        matrix_size=config.matrix_size,
        initial_mapping_strategy=config.initial_mapping_strategy,
        seed=config.seed,
    )
    num_actions = env.max_edges

    # Create agent
    agent = D3QNAgent(config, num_actions)
    param_count = sum(p.numel() for p in agent.online_net.parameters())
    print(f"Device: {agent.device}")
    print(f"Parameters: {param_count:,}")
    print(f"Actions: {num_actions} | Topologies: {config.topologies}")

    # Resume
    start_episode = 0
    if resume_from:
        start_episode = agent.load_checkpoint(resume_from)
        print(f"Resumed from episode {start_episode}")

    # Logger
    logger = TrainingLogger(config.log_dir)

    # Ctrl+C handler
    interrupted = [False]
    current_episode = [start_episode]

    def signal_handler(sig, frame):
        if interrupted[0]:
            raise SystemExit(1)
        interrupted[0] = True
        print("\nInterrupted — saving emergency checkpoint...")

    old_handler = signal.signal(signal.SIGINT, signal_handler)

    # Rolling stats
    recent_rewards = []
    recent_swaps = []
    recent_completions = []

    global_step = agent._train_steps
    t_start = time.time()

    try:
        for episode in range(start_episode, config.total_episodes):
            if interrupted[0]:
                break

            current_episode[0] = episode
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

                # Train step
                if global_step % config.train_freq == 0:
                    metrics = agent.train_step()
                    if metrics:
                        if agent._train_steps % 100 == 0:
                            logger.log_train_step(agent._train_steps, metrics)
                        if agent._train_steps % config.target_update_freq == 0:
                            agent.update_target_network()

                agent.update_epsilon()
                episode_reward += reward
                obs = next_obs
                action_mask = next_mask

                if terminated or truncated:
                    break

            # Log episode
            ep_data = {
                "reward": round(episode_reward, 3),
                "steps": info["step_count"],
                "swaps": info["total_swaps"],
                "gates": info["n_gates"],
                "completed": info["done"],
                "topology": info["topology"],
                "epsilon": round(agent.epsilon, 4),
            }
            logger.log_episode(episode, ep_data)

            recent_rewards.append(episode_reward)
            recent_swaps.append(info["total_swaps"])
            recent_completions.append(int(info["done"]))

            # Console output
            if (episode + 1) % config.log_every == 0:
                n = min(len(recent_rewards), config.log_every)
                elapsed = time.time() - t_start
                print(
                    f"Ep {episode+1}/{config.total_episodes} "
                    f"({elapsed:.0f}s) | "
                    f"R: {np.mean(recent_rewards[-n:]):.1f} | "
                    f"SWAPs: {np.mean(recent_swaps[-n:]):.1f} | "
                    f"Done: {np.mean(recent_completions[-n:]):.0%} | "
                    f"\u03b5: {agent.epsilon:.3f} | "
                    f"Buf: {len(agent.buffer)}"
                )

            # Periodic evaluation + figures + save eval results
            if (episode + 1) % config.eval_every == 0:
                _run_periodic_eval(
                    agent, env, config, episode, logger
                )

            # Checkpoint
            if (episode + 1) % config.checkpoint_every == 0:
                ckpt_path = (
                    Path(config.checkpoint_dir)
                    / f"checkpoint_ep{episode+1}.pt"
                )
                agent.save_checkpoint(str(ckpt_path), episode + 1)
                print(f"  Saved: {ckpt_path}")

    finally:
        ep = current_episode[0]
        if interrupted[0]:
            ckpt_path = (
                Path(config.checkpoint_dir) / "checkpoint_emergency.pt"
            )
            agent.save_checkpoint(str(ckpt_path), ep)
            print(f"Emergency checkpoint: {ckpt_path}")
        else:
            ckpt_path = (
                Path(config.checkpoint_dir) / "checkpoint_final.pt"
            )
            agent.save_checkpoint(str(ckpt_path), config.total_episodes)
            print(f"Final checkpoint: {ckpt_path}")

        # Generate final figures
        try:
            from visualize import plot_training_curves
            plot_training_curves(config.log_dir, config.figures_dir)
        except Exception as e:
            print(f"Warning: could not generate final figures: {e}")

        logger.close()
        signal.signal(signal.SIGINT, old_handler)

    elapsed = time.time() - t_start
    print(f"Training complete. {elapsed:.0f}s total.")
    print(f"All outputs in: {config.run_dir}")


def _run_periodic_eval(agent, env, config, episode, logger):
    """Run eval, log summary, save results + update figures."""
    from evaluate import run_evaluation, save_eval_results
    from visualize import plot_training_curves, plot_eval_comparison

    eval_results = run_evaluation(
        agent, env, config, config.eval_episodes,
        log_trajectories=True,
    )
    summary = eval_results["summary"]
    logger.log_evaluation(episode, summary)

    print(
        f"  EVAL | Agent: {summary['mean_agent_swaps']:.1f} "
        f"SABRE: {summary['mean_sabre_swaps']:.1f} | "
        f"Ratio: {summary['mean_swap_ratio']:.2f} | "
        f"Done: {summary['completion_rate']:.0%}"
    )

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
