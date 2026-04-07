"""Generate training diagnostics figures from PPO/DQN run artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _safe_float(value: str) -> float:
    if value is None:
        return float("nan")
    v = str(value).strip()
    if v == "" or v.lower() in {"nan", "none"}:
        return float("nan")
    try:
        return float(v)
    except ValueError:
        return float("nan")


def _load_metrics_csv(path: Path) -> Dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}

    cols = rows[0].keys()
    out: Dict[str, List[float]] = {c: [] for c in cols}
    for row in rows:
        for c in cols:
            out[c].append(_safe_float(row.get(c)))
    return {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}


def _valid_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _smooth(y: np.ndarray, window: int) -> np.ndarray:
    if y.size <= 1 or window <= 1:
        return y
    w = min(window, int(y.size))
    if w <= 1:
        return y
    kernel = np.ones(w, dtype=np.float32) / float(w)
    return np.convolve(y, kernel, mode="valid")


def _plot_stage_dashboard(
    stage_name: str,
    metrics: Dict[str, np.ndarray],
    output_path: Path,
    smooth_window: int,
) -> Dict[str, float]:
    if not metrics or "update" not in metrics:
        return {}

    x = metrics["update"]
    gs = lambda key: metrics.get(key, np.full_like(x, np.nan, dtype=np.float32))

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle(f"Run Diagnostics - {stage_name}", fontsize=15, fontweight="bold")

    # 1) Reward + done rate
    ax = axes[0, 0]
    y_reward = gs("mean_step_reward")
    xv, yv = _valid_xy(x, y_reward)
    if yv.size:
        ax.plot(xv, yv, color="tab:blue", alpha=0.25, label="mean_step_reward (raw)")
        sm = _smooth(yv, smooth_window)
        ax.plot(xv[len(xv) - len(sm):], sm, color="tab:blue", linewidth=2, label="smoothed")
    y_done = gs("done_rate")
    xv2, yv2 = _valid_xy(x, y_done)
    if yv2.size:
        ax2 = ax.twinx()
        ax2.plot(xv2, yv2, color="tab:green", linewidth=1.5, label="done_rate")
        ax2.set_ylabel("done_rate", color="tab:green")
        ax2.tick_params(axis="y", labelcolor="tab:green")
    ax.set_title("Reward and Completion")
    ax.set_xlabel("Update")
    ax.set_ylabel("mean_step_reward")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)

    # 2) Episode return
    ax = axes[0, 1]
    y_ep = gs("mean_episode_return")
    xv, yv = _valid_xy(x, y_ep)
    if yv.size:
        ax.plot(xv, yv, color="tab:purple", alpha=0.25, label="mean_episode_return (raw)")
        sm = _smooth(yv, smooth_window)
        ax.plot(xv[len(xv) - len(sm):], sm, color="tab:purple", linewidth=2, label="smoothed")
    ax.set_title("Episode Return")
    ax.set_xlabel("Update")
    ax.set_ylabel("mean_episode_return")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    # 3) Eval metrics
    ax = axes[0, 2]
    y_eval_imp = gs("eval_improvement_pct")
    xv, yv = _valid_xy(x, y_eval_imp)
    if yv.size:
        ax.plot(xv, yv, marker="o", color="tab:red", linewidth=2, label="eval_improvement_pct")
    y_eval_win = gs("eval_win_rate")
    xv2, yv2 = _valid_xy(x, y_eval_win)
    if yv2.size:
        ax.plot(xv2, yv2 * 100.0, marker="s", color="tab:green", linewidth=2, label="eval_win_rate (%)")
    y_eval_to = gs("eval_timeout_rate")
    xv3, yv3 = _valid_xy(x, y_eval_to)
    if yv3.size:
        ax.plot(xv3, yv3 * 100.0, marker="^", color="tab:orange", linewidth=2, label="eval_timeout_rate (%)")
    ax.set_title("Eval vs SABRE")
    ax.set_xlabel("Update")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    # 4) Losses
    ax = axes[1, 0]
    y_pi = gs("policy_loss")
    xv, yv = _valid_xy(x, y_pi)
    if yv.size:
        ax.plot(xv, yv, color="tab:blue", label="policy_loss")
    y_v = gs("value_loss")
    xv2, yv2 = _valid_xy(x, y_v)
    if yv2.size:
        ax.plot(xv2, yv2, color="tab:red", label="value_loss")
    ax.set_title("Losses")
    ax.set_xlabel("Update")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    # 5) Entropy / KL / distance coeff
    ax = axes[1, 1]
    y_ent = gs("entropy")
    xv, yv = _valid_xy(x, y_ent)
    if yv.size:
        ax.plot(xv, yv, color="tab:purple", label="entropy")
    y_kl = gs("approx_kl")
    xv2, yv2 = _valid_xy(x, y_kl)
    if yv2.size:
        ax.plot(xv2, yv2, color="tab:brown", label="approx_kl")
    y_dist = gs("distance_reward_coeff")
    xv3, yv3 = _valid_xy(x, y_dist)
    if yv3.size:
        ax2 = ax.twinx()
        ax2.plot(xv3, yv3, color="tab:cyan", linestyle="--", label="distance_reward_coeff")
        ax2.set_ylabel("distance_reward_coeff", color="tab:cyan")
        ax2.tick_params(axis="y", labelcolor="tab:cyan")
    ax.set_title("Exploration and Coefficients")
    ax.set_xlabel("Update")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)

    # 6) Trace collapse diagnostics
    ax = axes[1, 2]
    y_dom = gs("trace_action_dom_ratio")
    xv, yv = _valid_xy(x, y_dom)
    if yv.size:
        ax.plot(xv, yv, marker="o", color="tab:red", label="trace_dom_ratio")
    y_back = gs("trace_backtrack_rate")
    xv2, yv2 = _valid_xy(x, y_back)
    if yv2.size:
        ax.plot(xv2, yv2, marker="s", color="tab:orange", label="trace_backtrack")
    y_tout = gs("trace_timeout_rate")
    xv3, yv3 = _valid_xy(x, y_tout)
    if yv3.size:
        ax.plot(xv3, yv3, marker="^", color="tab:gray", label="trace_timeout")
    y_streak = gs("trace_alert_streak")
    xv4, yv4 = _valid_xy(x, y_streak)
    if yv4.size:
        ax2 = ax.twinx()
        ax2.plot(xv4, yv4, color="tab:blue", linestyle="--", label="trace_alert_streak")
        ax2.set_ylabel("alert_streak", color="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax.axhline(0.60, color="tab:red", linestyle=":", linewidth=1)
    ax.axhline(0.50, color="tab:orange", linestyle=":", linewidth=1)
    ax.set_title("Trace Collapse Signals")
    ax.set_xlabel("Update")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    best_eval = float(np.nanmax(y_eval_imp)) if np.isfinite(y_eval_imp).any() else float("nan")
    last_eval_idx = np.where(np.isfinite(y_eval_imp))[0]
    last_eval = float(y_eval_imp[last_eval_idx[-1]]) if last_eval_idx.size else float("nan")
    return {
        "best_eval_improvement_pct": best_eval,
        "last_eval_improvement_pct": last_eval,
    }


def _read_trace_summaries_for_latest_update(stage_dir: Path) -> List[Dict]:
    trace_dir = stage_dir / "traces"
    if not trace_dir.exists():
        return []
    update_dirs = sorted([d for d in trace_dir.glob("update_*") if d.is_dir()])
    if not update_dirs:
        return []
    latest = update_dirs[-1]
    summaries: List[Dict] = []
    for p in sorted(latest.glob("*_summary.json")):
        try:
            summaries.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return summaries


def _plot_latest_trace_topology(stage_name: str, stage_dir: Path, output_path: Path) -> bool:
    rows = _read_trace_summaries_for_latest_update(stage_dir)
    if not rows:
        return False

    grouped: Dict[str, List[Dict]] = {}
    for r in rows:
        topo = str(r.get("topology", "unknown"))
        grouped.setdefault(topo, []).append(r)

    tops = sorted(grouped.keys())
    ppo = [float(np.mean([x.get("ppo_swaps", np.nan) for x in grouped[t]])) for t in tops]
    sabre = [float(np.mean([x.get("sabre_swaps", np.nan) for x in grouped[t]])) for t in tops]
    done = [float(np.mean([1.0 if x.get("done") else 0.0 for x in grouped[t]])) for t in tops]
    back = [float(np.mean([x.get("backtrack_rate", np.nan) for x in grouped[t]])) for t in tops]
    dom = [float(np.mean([x.get("dominant_action_ratio", np.nan) for x in grouped[t]])) for t in tops]

    x = np.arange(len(tops))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, 2 + 1.8 * len(tops)), 6))
    ax.bar(x - width / 2, ppo, width, label="Agent SWAPs", color="tab:blue")
    ax.bar(x + width / 2, sabre, width, label="SABRE SWAPs", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels(tops, rotation=20, ha="right")
    ax.set_ylabel("SWAP count")
    ax.set_title(f"Latest Trace Cases by Topology - {stage_name}")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left")

    for i, topo in enumerate(tops):
        txt = f"done={done[i]:.2f}\ndom={dom[i]:.2f}\nback={back[i]:.2f}"
        ax.text(i, max(ppo[i], sabre[i]) + 2.0, txt, ha="center", va="bottom", fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def _discover_stages(run_dir: Path) -> List[Tuple[str, Path, Path]]:
    out: List[Tuple[str, Path, Path]] = []
    root_metrics = run_dir / "metrics.csv"
    if root_metrics.exists():
        out.append(("single_stage", run_dir, root_metrics))
    for d in sorted(run_dir.iterdir()):
        if d.is_dir():
            m = d / "metrics.csv"
            if m.exists():
                out.append((d.name, d, m))
    # deduplicate if both root and subdirs
    seen = set()
    uniq: List[Tuple[str, Path, Path]] = []
    for x in out:
        key = str(x[2].resolve())
        if key not in seen:
            uniq.append(x)
            seen.add(key)
    return uniq


def _plot_run_overview(stage_stats: Dict[str, Dict[str, float]], output_path: Path):
    if not stage_stats:
        return
    names = list(stage_stats.keys())
    best_vals = [stage_stats[n].get("best_eval_improvement_pct", np.nan) for n in names]
    last_vals = [stage_stats[n].get("last_eval_improvement_pct", np.nan) for n in names]
    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, 2 + 1.8 * len(names)), 5))
    ax.bar(x - width / 2, best_vals, width, label="Best eval improvement %", color="tab:green")
    ax.bar(x + width / 2, last_vals, width, label="Last eval improvement %", color="tab:red")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Improvement vs SABRE (%)")
    ax.set_title("Run Overview Across Stages")
    ax.axhline(0.0, color="black", linestyle=":", linewidth=1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize PPO/DQN run diagnostics.")
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Run directory (e.g., runs/ppo_run9_... or runs/ppo_curriculum_...)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Optional output directory. Default: <run-dir>/figures_diag",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=7,
        help="Smoothing window for noisy training series.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    out_dir = Path(args.output_dir) if args.output_dir else (run_dir / "figures_diag")
    out_dir.mkdir(parents=True, exist_ok=True)

    stages = _discover_stages(run_dir)
    if not stages:
        raise FileNotFoundError(
            f"No metrics.csv found in run directory or first-level subdirs: {run_dir}"
        )

    stage_stats: Dict[str, Dict[str, float]] = {}
    for stage_name, stage_dir, metrics_path in stages:
        metrics = _load_metrics_csv(metrics_path)
        dashboard_path = out_dir / f"{stage_name}_dashboard.png"
        stats = _plot_stage_dashboard(
            stage_name=stage_name,
            metrics=metrics,
            output_path=dashboard_path,
            smooth_window=max(1, int(args.smooth_window)),
        )
        stage_stats[stage_name] = stats
        print(f"Saved: {dashboard_path}")

        trace_plot = out_dir / f"{stage_name}_latest_trace_topology.png"
        if _plot_latest_trace_topology(stage_name, stage_dir, trace_plot):
            print(f"Saved: {trace_plot}")

    overview_path = out_dir / "run_overview_eval.png"
    _plot_run_overview(stage_stats, overview_path)
    print(f"Saved: {overview_path}")
    print(f"Done. Figures in: {out_dir}")


if __name__ == "__main__":
    main()
