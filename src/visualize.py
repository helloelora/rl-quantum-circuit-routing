"""
Minimal visualization for PPO training runs on La Ruche.
Reads metrics.csv + eval/*.json and produces two PNGs.

Usage:
    python -m src.visualize <run_dir>
    python -m src.visualize $WORKDIR/rl_qrouting_runs/runs/<run_name>/<phase>
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on HPC
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def smooth(values, weight=0.9):
    """Exponential moving average."""
    s = []
    last = values.iloc[0] if len(values) > 0 else 0
    for v in values:
        if np.isnan(v):
            s.append(np.nan)
        else:
            last = weight * last + (1 - weight) * v
            s.append(last)
    return s


def plot_training_curves(df, out_path):
    """2x3 grid: return, PPO vs SABRE swaps, win rate, policy loss, value loss, entropy."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("PPO Training Curves", fontsize=14)

    # --- Row 0, Col 0: Mean episode return ---
    ax = axes[0, 0]
    ret = df["mean_episode_return"].dropna()
    if len(ret) > 0:
        ax.plot(ret.index, ret.values, alpha=0.3, color="steelblue")
        ax.plot(ret.index, smooth(ret), color="steelblue", linewidth=1.5)
    ax.set_xlabel("Update")
    ax.set_ylabel("Return")
    ax.set_title("Mean Episode Return")

    # --- Row 0, Col 1: PPO vs SABRE swaps ---
    ax = axes[0, 1]
    evals = df.dropna(subset=["eval_mean_ppo_swaps"])
    if len(evals) > 0:
        ax.plot(evals["update"], evals["eval_mean_ppo_swaps"], "o-",
                label="PPO", color="steelblue", markersize=3)
        ax.plot(evals["update"], evals["eval_mean_sabre_swaps"], "o-",
                label="SABRE", color="coral", markersize=3)
        ax.legend(fontsize=8)
    ax.set_xlabel("Update")
    ax.set_ylabel("Mean SWAPs")
    ax.set_title("Eval: PPO vs SABRE")

    # --- Row 0, Col 2: Win rate + improvement ---
    ax = axes[0, 2]
    if len(evals) > 0:
        ax.plot(evals["update"], evals["eval_win_rate"] * 100, "o-",
                label="Win rate %", color="green", markersize=3)
        ax.plot(evals["update"], evals["eval_improvement_pct"], "s-",
                label="Improvement %", color="purple", markersize=3)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        ax.legend(fontsize=8)
    ax.set_xlabel("Update")
    ax.set_ylabel("%")
    ax.set_title("Eval: Win Rate & Improvement")

    # --- Row 1, Col 0: Policy loss ---
    ax = axes[1, 0]
    pl = df["policy_loss"].dropna()
    if len(pl) > 0:
        ax.plot(pl.index, pl.values, alpha=0.3, color="steelblue")
        ax.plot(pl.index, smooth(pl), color="steelblue", linewidth=1.5)
    ax.set_xlabel("Update")
    ax.set_ylabel("Loss")
    ax.set_title("Policy Loss")

    # --- Row 1, Col 1: Value loss ---
    ax = axes[1, 1]
    vl = df["value_loss"].dropna()
    if len(vl) > 0:
        ax.plot(vl.index, vl.values, alpha=0.3, color="coral")
        ax.plot(vl.index, smooth(vl), color="coral", linewidth=1.5)
    ax.set_xlabel("Update")
    ax.set_ylabel("Loss")
    ax.set_title("Value Loss")

    # --- Row 1, Col 2: Entropy ---
    ax = axes[1, 2]
    ent = df["entropy"].dropna()
    if len(ent) > 0:
        ax.plot(ent.index, ent.values, alpha=0.3, color="green")
        ax.plot(ent.index, smooth(ent), color="green", linewidth=1.5)
    ax.set_xlabel("Update")
    ax.set_ylabel("Entropy")
    ax.set_title("Policy Entropy")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_eval_comparison(eval_dir, out_path):
    """Bar chart: PPO vs SABRE per topology from the latest eval JSON."""
    eval_files = sorted(eval_dir.glob("update_*.json"))
    if not eval_files:
        print("No eval snapshots found, skipping eval_comparison.png")
        return

    # Use the latest eval snapshot
    with eval_files[-1].open() as f:
        data = json.load(f)

    per_topo = data.get("per_topology", {})
    if not per_topo:
        print("No per-topology data in eval snapshot, skipping eval_comparison.png")
        return

    topos = sorted(per_topo.keys())
    ppo_swaps = [per_topo[t]["mean_ppo_swaps"] for t in topos]
    sabre_swaps = [per_topo[t]["mean_sabre_swaps"] for t in topos]

    x = np.arange(len(topos))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, ppo_swaps, width, label="PPO", color="steelblue")
    bars2 = ax.bar(x + width / 2, sabre_swaps, width, label="SABRE", color="coral")

    # Add ratio labels on PPO bars
    for i, (p, s) in enumerate(zip(ppo_swaps, sabre_swaps)):
        if s > 0:
            ratio = p / s
            ax.text(x[i] - width / 2, p + 0.5, f"{ratio:.2f}x",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Mean SWAPs")
    ax.set_title(f"PPO vs SABRE (update {data.get('update', '?')})")
    ax.set_xticks(x)
    ax.set_xticklabels(topos)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot PPO training curves")
    parser.add_argument("run_dir", type=str, help="Path to the phase run directory")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    metrics_path = run_dir / "metrics.csv"
    eval_dir = run_dir / "eval"

    if not metrics_path.exists():
        print(f"Error: {metrics_path} not found")
        sys.exit(1)

    df = pd.read_csv(metrics_path)
    plot_training_curves(df, run_dir / "training_curves.png")

    if eval_dir.exists():
        plot_eval_comparison(eval_dir, run_dir / "eval_comparison.png")


if __name__ == "__main__":
    main()
