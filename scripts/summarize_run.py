"""
Quick summary of a PPO run. Run on La Ruche:
    apptainer exec --bind $WORKDIR:$WORKDIR --pwd $PWD $WORKDIR/rl_qrouting.sif \
        python scripts/summarize_run.py $WORKDIR/rl_qrouting_runs/runs/ppo_linear5_*/single_stage

Or for all runs:
    for d in $WORKDIR/rl_qrouting_runs/runs/*/single_stage; do
        apptainer exec --bind $WORKDIR:$WORKDIR --pwd $PWD $WORKDIR/rl_qrouting.sif \
            python scripts/summarize_run.py "$d"
    done
"""
import json
import sys
from pathlib import Path

import pandas as pd


def summarize(run_dir):
    run_dir = Path(run_dir)
    name = run_dir.parent.name

    metrics = run_dir / "metrics.csv"
    if not metrics.exists():
        print(f"[{name}] No metrics.csv found")
        return

    df = pd.read_csv(metrics)
    total_steps = int(df["global_step"].max())
    total_updates = int(df["update"].max())

    # Training stats (last 20 updates)
    tail = df.tail(20)
    mean_return = tail["mean_episode_return"].mean()
    mean_done_rate = tail["done_rate"].mean()
    mean_entropy = tail["entropy"].mean()
    mean_pi_loss = tail["policy_loss"].mean()
    mean_v_loss = tail["value_loss"].mean()

    # Eval progression (non-NaN rows)
    evals = df.dropna(subset=["eval_mean_ppo_swaps"])
    if len(evals) == 0:
        print(f"[{name}] No eval data")
        return

    first_eval = evals.iloc[0]
    best_idx = evals["eval_improvement_pct"].idxmax()
    best_eval = evals.loc[best_idx]
    last_eval = evals.iloc[-1]

    # Best eval JSON for per-topology breakdown
    eval_dir = run_dir / "eval"
    per_topo_summary = ""
    if eval_dir.exists():
        eval_files = sorted(eval_dir.glob("update_*.json"))
        if eval_files:
            with eval_files[-1].open() as f:
                data = json.load(f)
            pt = data.get("per_topology", {})
            for t in sorted(pt):
                s = pt[t]
                per_topo_summary += (
                    f"    {t}: ppo={s['eval_mean_ppo_swaps']:.1f} "
                    f"sabre={s['eval_mean_sabre_swaps']:.1f} "
                    f"imp={s['eval_improvement_pct']:+.1f}% "
                    f"win={s['eval_win_rate']:.2f} "
                    f"to={s['eval_timeout_rate']:.2f}\n"
                )

    print(f"{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Steps: {total_steps:,} | Updates: {total_updates}")
    print()
    print(f"  --- Training (last 20 updates) ---")
    print(f"  mean_return={mean_return:.1f}  done_rate={mean_done_rate:.3f}  entropy={mean_entropy:.3f}")
    print(f"  pi_loss={mean_pi_loss:.4f}  v_loss={mean_v_loss:.4f}")
    print()
    print(f"  --- Eval progression ---")
    print(f"  First:  step={int(first_eval['global_step']):>8,}  imp={first_eval['eval_improvement_pct']:+.1f}%  win={first_eval['eval_win_rate']:.2f}  to={first_eval['eval_timeout_rate']:.2f}")
    print(f"  Best:   step={int(best_eval['global_step']):>8,}  imp={best_eval['eval_improvement_pct']:+.1f}%  win={best_eval['eval_win_rate']:.2f}  to={best_eval['eval_timeout_rate']:.2f}")
    print(f"  Last:   step={int(last_eval['global_step']):>8,}  imp={last_eval['eval_improvement_pct']:+.1f}%  win={last_eval['eval_win_rate']:.2f}  to={last_eval['eval_timeout_rate']:.2f}")
    print()
    if per_topo_summary:
        print(f"  --- Per topology (last eval) ---")
        print(per_topo_summary, end="")
    print()


if __name__ == "__main__":
    for path in sys.argv[1:]:
        summarize(path)
