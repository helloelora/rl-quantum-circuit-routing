"""Visualization tools: training curves, agent vs SABRE charts, routing GIFs."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
import networkx as nx
from pathlib import Path

from collections import defaultdict

from circuit_utils import compute_front_layer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_per_topology_eval_data(log_dir):
    """
    Load per-topology metrics from eval JSON files.

    Looks for eval_ep*.json in the eval/ directory (sibling of log_dir).
    Each file contains a "results" list with per-circuit entries that include
    "topology", and a "summary" dict with an "episode" field.

    Returns:
        dict mapping topology name -> list of dicts sorted by episode, each
        with keys: episode, swap_ratio, completion_rate.
        Returns empty dict if no multi-topology data found.
    """
    eval_dir = Path(log_dir).parent / "eval"
    if not eval_dir.exists():
        return {}

    eval_files = sorted(eval_dir.glob("eval_ep*.json"))
    if not eval_files:
        return {}

    # Collect per-topology metrics across all eval checkpoints
    topo_data = defaultdict(list)  # topology -> list of (episode, ratio, comp)

    for fpath in eval_files:
        try:
            with open(fpath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        results = data.get("results", [])
        summary = data.get("summary", {})
        episode = summary.get("episode")
        if episode is None or not results:
            continue

        # Group results by topology
        by_topo = defaultdict(list)
        for r in results:
            topo = r.get("topology")
            if topo:
                by_topo[topo].append(r)

        # Only useful if there are multiple topologies
        if len(by_topo) < 2:
            continue

        for topo, entries in by_topo.items():
            completed = [e for e in entries if e.get("completed")]
            valid = [
                e for e in completed
                if e.get("sabre_swaps", 0) > 0
            ]
            ratios = [
                e["agent_swaps"] / e["sabre_swaps"] for e in valid
            ]
            comp_rate = len(completed) / len(entries) if entries else 0.0
            mean_ratio = float(np.mean(ratios)) if ratios else float("nan")

            topo_data[topo].append({
                "episode": episode,
                "swap_ratio": mean_ratio,
                "completion_rate": comp_rate,
            })

    # Sort each topology's data by episode
    for topo in topo_data:
        topo_data[topo].sort(key=lambda d: d["episode"])

    return dict(topo_data)


def _smooth(values, window=50):
    if len(values) < window:
        window = max(1, len(values))
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def _fig_to_array(fig):
    """Render a matplotlib figure to a numpy RGB array."""
    fig.canvas.draw()
    return np.array(fig.canvas.buffer_rgba())[:, :, :3].copy()


# ---------------------------------------------------------------------------
# 1. Training curves  (2x3, or 2x4 when multi-topology eval data exists)
# ---------------------------------------------------------------------------

def plot_training_curves(log_dir, output_dir=None, window=50):
    """
    Plot training dashboard.

    Default 2x3 layout:
        Row 1: Reward | SWAP count + SABRE baseline | Completion rate
        Row 2: Training loss | Epsilon (exploration) | Mean Q-value

    When multi-topology eval data is available, expands to 2x4:
        Row 1: Reward | SWAP count | Completion | Per-Topology Swap Ratio
        Row 2: Loss   | Epsilon    | Q-value    | Per-Topology Completion

    Every line is labelled in a legend so you know what you are looking at.
    """
    log_dir = Path(log_dir)
    if output_dir is None:
        output_dir = log_dir / "figures"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = _load_jsonl(log_dir / "episodes.jsonl")
    if not episodes:
        print("No episode data found.")
        return

    eps = [e["episode"] for e in episodes]
    rewards = [e["reward"] for e in episodes]
    swaps = [e["swaps"] for e in episodes]
    epsilons = [e.get("epsilon", 0) for e in episodes]
    completions = [1 if e.get("completed") else 0 for e in episodes]

    # Load optional log files
    step_path = log_dir / "train_steps.jsonl"
    steps, s_idx = [], []
    if step_path.exists():
        steps = _load_jsonl(step_path)
        if steps:
            s_idx = [s["step"] for s in steps]

    eval_path = log_dir / "evaluations.jsonl"
    evals = []
    if eval_path.exists():
        evals = _load_jsonl(eval_path)

    # Check for per-topology eval data (multi-topology runs)
    topo_data = _load_per_topology_eval_data(log_dir)
    has_topo = len(topo_data) >= 2

    if has_topo:
        n_cols = 4
        fig, axes = plt.subplots(2, 4, figsize=(26, 10))
    else:
        n_cols = 3
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle("Training Dashboard", fontsize=16, fontweight="bold")

    # ---- (0,0) Reward ----
    ax = axes[0, 0]
    ax.plot(eps, rewards, alpha=0.15, color="royalblue", label="Per-episode (raw)")
    if len(rewards) >= window:
        sm = _smooth(rewards, window)
        ax.plot(eps[window - 1:], sm, color="royalblue", linewidth=2,
                label=f"Smoothed (window={window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Episode Reward")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- (0,1) SWAP count + SABRE ----
    ax = axes[0, 1]
    ax.plot(eps, swaps, alpha=0.15, color="darkorange",
            label="Agent SWAPs (raw)")
    if len(swaps) >= window:
        sm = _smooth(swaps, window)
        ax.plot(eps[window - 1:], sm, color="darkorange", linewidth=2,
                label=f"Agent SWAPs (smoothed)")
    if evals:
        eval_eps = [e["episode"] for e in evals]
        sabre_m = [e.get("mean_sabre_swaps", 0) for e in evals]
        agent_m = [e.get("mean_agent_swaps", 0) for e in evals]
        ax.plot(eval_eps, sabre_m, "r--", linewidth=2, marker="o",
                markersize=4, label="SABRE baseline (eval avg)")
        ax.plot(eval_eps, agent_m, "b-", linewidth=2, marker="s",
                markersize=4, label="Agent (eval avg)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("SWAP Count")
    ax.set_title("SWAP Count per Episode")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- (0,2) Completion rate ----
    ax = axes[0, 2]
    if len(completions) >= window:
        sm = _smooth(completions, window)
        ax.plot(eps[window - 1:], sm * 100, color="seagreen", linewidth=2,
                label=f"Completion % (smoothed)")
    ax.plot(eps, [c * 100 for c in completions], alpha=0.1, color="seagreen",
            label="Per-episode")
    if evals:
        eval_comp = [e.get("completion_rate", 0) * 100 for e in evals]
        ax.plot(eval_eps, eval_comp, "ko-", markersize=5, linewidth=1.5,
                label="Eval completion %")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Completion %")
    ax.set_title("Episode Completion Rate")
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- (1,0) Loss ----
    ax = axes[1, 0]
    if steps:
        losses = [s["loss"] for s in steps]
        ax.plot(s_idx, losses, alpha=0.2, color="crimson", label="Loss (raw)")
        if len(losses) >= 20:
            sm = _smooth(losses, 20)
            ax.plot(s_idx[19:], sm, color="crimson", linewidth=2,
                    label="Loss (smoothed)")
    ax.set_xlabel("Gradient Step")
    ax.set_ylabel("Huber Loss")
    ax.set_title("Training Loss (Huber)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- (1,1) Epsilon ----
    ax = axes[1, 1]
    ax.plot(eps, epsilons, color="mediumorchid", linewidth=2,
            label="Epsilon (exploration rate)")
    ax.axhline(y=epsilons[-1] if epsilons else 0.05, color="gray",
               linestyle=":", alpha=0.5, label=f"Current: {epsilons[-1]:.3f}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.set_title("Exploration Rate (Epsilon)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- (1,2) Mean Q-value ----
    ax = axes[1, 2]
    if steps:
        mean_qs = [s.get("mean_q", 0) for s in steps]
        ax.plot(s_idx, mean_qs, alpha=0.3, color="teal", label="Mean Q (raw)")
        if len(mean_qs) >= 20:
            sm = _smooth(mean_qs, 20)
            ax.plot(s_idx[19:], sm, color="teal", linewidth=2,
                    label="Mean Q (smoothed)")
        # Also show mean TD error if available
        td_errors = [s.get("mean_td_error", 0) for s in steps]
        if any(td_errors):
            ax2 = ax.twinx()
            ax2.plot(s_idx, td_errors, alpha=0.3, color="salmon",
                     label="Mean |TD error| (raw)")
            if len(td_errors) >= 20:
                sm = _smooth(td_errors, 20)
                ax2.plot(s_idx[19:], sm, color="salmon", linewidth=1.5,
                         label="Mean |TD error| (smoothed)")
            ax2.set_ylabel("Mean |TD Error|", color="salmon", fontsize=9)
            ax2.tick_params(axis="y", labelcolor="salmon")
            ax2.legend(fontsize=7, loc="upper right")
    ax.set_xlabel("Gradient Step")
    ax.set_ylabel("Mean Q-value", color="teal")
    ax.tick_params(axis="y", labelcolor="teal")
    ax.set_title("Q-value & TD Error")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    # ---- Per-topology panels (only when multi-topology data exists) ----
    if has_topo:
        # Consistent color per topology across both panels
        topo_names = sorted(topo_data.keys())
        topo_cmap = plt.cm.get_cmap("tab10", max(len(topo_names), 1))
        topo_colors = {
            name: topo_cmap(i) for i, name in enumerate(topo_names)
        }

        # ---- (0,3) Per-Topology Swap Ratio ----
        ax = axes[0, 3]
        for topo in topo_names:
            entries = topo_data[topo]
            t_eps = [d["episode"] for d in entries]
            t_ratios = [d["swap_ratio"] for d in entries]
            ax.plot(t_eps, t_ratios, marker="o", markersize=3,
                    linewidth=1.5, color=topo_colors[topo], label=topo)
        ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1.5,
                   alpha=0.7, label="SABRE baseline (1.0)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Swap Ratio (agent / SABRE)")
        ax.set_title("Per-Topology Swap Ratio")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

        # ---- (1,3) Per-Topology Completion Rate ----
        ax = axes[1, 3]
        for topo in topo_names:
            entries = topo_data[topo]
            t_eps = [d["episode"] for d in entries]
            t_comp = [d["completion_rate"] * 100 for d in entries]
            ax.plot(t_eps, t_comp, marker="s", markersize=3,
                    linewidth=1.5, color=topo_colors[topo], label=topo)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Completion %")
        ax.set_title("Per-Topology Completion Rate")
        ax.set_ylim(-5, 105)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = output_dir / "training_curves.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# 2 & 3. Bar chart / histogram  (unchanged but kept here)
# ---------------------------------------------------------------------------

def plot_eval_comparison(eval_results, output_path=None):
    """Grouped bar chart: agent vs SABRE swap counts per eval episode."""
    if not eval_results:
        return

    n = len(eval_results)
    agent_swaps = [r["agent_swaps"] for r in eval_results]
    sabre_swaps = [r["sabre_swaps"] for r in eval_results]
    labels = [r.get("circuit", f"Ep {i}") for i, r in enumerate(eval_results)]

    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, n * 0.5), 6))
    ax.bar(x - width / 2, agent_swaps, width, label="Agent (D3QN)",
           color="steelblue")
    ax.bar(x + width / 2, sabre_swaps, width, label="SABRE (Qiskit)",
           color="coral")
    ax.set_xlabel("Circuit")
    ax.set_ylabel("SWAP Count")
    ax.set_title("Agent vs SABRE: SWAP Counts per Circuit")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")
    plt.close(fig)


def plot_swap_ratio_distribution(eval_results, output_path=None):
    """Histogram of agent_swaps / sabre_swaps ratios."""
    ratios = []
    for r in eval_results:
        if r.get("completed") and r.get("sabre_swaps", 0) > 0:
            ratios.append(r["agent_swaps"] / r["sabre_swaps"])
    if not ratios:
        print("No valid ratios to plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ratios, bins=20, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(1.0, color="red", linestyle="--", linewidth=2,
               label="SABRE baseline (ratio = 1.0)")
    ax.axvline(np.mean(ratios), color="green", linestyle="-", linewidth=2,
               label=f"Agent mean ({np.mean(ratios):.2f})")
    ax.set_xlabel("Agent / SABRE Swap Ratio  (<1 = agent wins)")
    ax.set_ylabel("Count")
    ax.set_title("Swap Ratio Distribution across Eval Circuits")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")
    plt.close(fig)


def create_routing_summary_table(eval_results):
    """Return a markdown table summarizing eval results."""
    lines = [
        "| Circuit | Topology | Gates | Agent SWAPs | SABRE SWAPs | Ratio | Done |",
        "|---------|----------|-------|-------------|-------------|-------|------|",
    ]
    for r in eval_results:
        name = r.get("circuit", "-")
        topo = r.get("topology", "-")
        gates = r.get("n_gates", "-")
        a_sw = r["agent_swaps"]
        s_sw = r["sabre_swaps"]
        ratio = f"{a_sw / s_sw:.2f}" if s_sw > 0 else "N/A"
        done = "Yes" if r.get("completed") else "No"
        lines.append(
            f"| {name} | {topo} | {gates} | {a_sw} | {s_sw} | {ratio} | {done} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Shared GIF helpers
# ---------------------------------------------------------------------------

def _build_graph_and_layout(n_phys, edges):
    G = nx.Graph()
    G.add_nodes_from(range(n_phys))
    G.add_edges_from(edges)
    pos = nx.kamada_kawai_layout(G)
    return G, pos


def _qubit_palette(n):
    cmap = plt.cm.get_cmap("hsv", n + 1)
    return [mcolors.to_hex(cmap(i)) for i in range(n)]


def _reverse_mapping(mapping, n_phys):
    rev = [0] * n_phys
    for q, p in enumerate(mapping):
        if q < n_phys and p < n_phys:
            rev[p] = q
    return rev


def _reconstruct_predecessors(preds_raw):
    predecessors = {}
    for k, v in preds_raw.items():
        predecessors[int(k)] = set(v)
    return predecessors


def _gate_status_text(gates, executed_set, predecessors, n_phys, mapping):
    """Build a multi-line gate checklist string."""
    front = set(compute_front_layer(gates, executed_set, predecessors))
    lines = []
    for i, (qa, qb) in enumerate(gates):
        if i in executed_set:
            mark = "\u2713"    # checkmark
            status = "done"
        elif i in front:
            mark = "\u25b8"    # right-pointing triangle
            status = "READY"
        else:
            mark = " "
            status = "waiting"

        # Show logical qubits and their current physical positions
        pa = mapping[qa] if qa < len(mapping) else "?"
        pb = mapping[qb] if qb < len(mapping) else "?"
        line = f" {mark}  g{i}: q{qa}\u2013q{qb}  (pos {pa}\u2013{pb})  [{status}]"
        lines.append(line)
    return "\n".join(lines)


def _draw_legend(ax):
    """Add a colour/line legend to the graph axes."""
    legend_items = [
        mlines.Line2D([], [], color="red", linewidth=4, alpha=0.9,
                       label="SWAP action"),
        mlines.Line2D([], [], color="limegreen", linewidth=3, alpha=0.8,
                       label="Gate executed (CNOT)"),
        mlines.Line2D([], [], color="dodgerblue", linewidth=1.5,
                       linestyle="--", alpha=0.6,
                       label="Gate demand (front layer)"),
        mlines.Line2D([], [], color="lightgray", linewidth=1, alpha=0.6,
                       label="Hardware edge"),
    ]
    ax.legend(handles=legend_items, loc="lower left", fontsize=8,
              framealpha=0.85, edgecolor="gray")


# ---------------------------------------------------------------------------
# 5. Step-by-step routing GIF
# ---------------------------------------------------------------------------

def create_routing_gif(trajectory, output_path, fps=1):
    """
    Animated GIF of the agent routing a circuit step-by-step.

    Each frame shows:
      LEFT  – hardware graph  (nodes = physical qubits, label = logical qubit)
              Red edge   = SWAP being performed
              Green edge = CNOT gate(s) that just executed
              Blue dashed = front-layer gate demands (need to become adjacent)
      RIGHT – gate checklist   (\u2713 done, \u25b8 ready, ' ' waiting)
              + action description  + cumulative reward
    """
    try:
        import imageio.v2 as imageio
    except ImportError:
        try:
            import imageio
        except ImportError:
            print("imageio not installed.  pip install imageio")
            return

    n_phys = trajectory["n_physical"]
    edges = [tuple(e) for e in trajectory["edges"]]
    gates = [tuple(g) for g in trajectory["gates"]]
    steps = trajectory["steps"]
    n_gates = trajectory.get("n_gates", len(gates))
    sabre_swaps = trajectory.get("sabre_swaps", 0)
    predecessors = _reconstruct_predecessors(
        trajectory.get("predecessors", {})
    )

    G, pos = _build_graph_and_layout(n_phys, edges)
    qcolors = _qubit_palette(n_phys)

    # ------------------------------------------------------------------
    def render_frame(mapping, executed_set, swap_edge, just_executed,
                     step_num, reward, cum_reward, total_swaps,
                     action_text):

        fig = plt.figure(figsize=(15, 8))
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 2], figure=fig)
        ax_graph = fig.add_subplot(gs[0, 0])
        ax_info  = fig.add_subplot(gs[0, 1])

        rev = _reverse_mapping(mapping, n_phys)

        # --- Graph panel ---
        # hardware edges (gray)
        nx.draw_networkx_edges(G, pos, ax=ax_graph, edge_color="lightgray",
                               width=1, alpha=0.5)

        # front-layer demands (blue dashed)
        front = compute_front_layer(gates, executed_set, predecessors)
        for gi in front:
            qa, qb = gates[gi]
            if qa < len(mapping) and qb < len(mapping):
                pa, pb = mapping[qa], mapping[qb]
                if pa in pos and pb in pos:
                    ax_graph.plot(
                        [pos[pa][0], pos[pb][0]],
                        [pos[pa][1], pos[pb][1]],
                        color="dodgerblue", linestyle="--",
                        alpha=0.6, linewidth=1.5, zorder=2,
                    )

        # just-executed gates (green)
        for gi in just_executed:
            qa, qb = gates[gi]
            if qa < len(mapping) and qb < len(mapping):
                pa, pb = mapping[qa], mapping[qb]
                if pa in pos and pb in pos:
                    ax_graph.plot(
                        [pos[pa][0], pos[pb][0]],
                        [pos[pa][1], pos[pb][1]],
                        color="limegreen", linewidth=4, alpha=0.85,
                        zorder=3,
                    )

        # SWAP edge (red)
        if swap_edge is not None:
            nx.draw_networkx_edges(
                G, pos, edgelist=[swap_edge], ax=ax_graph,
                edge_color="red", width=5, alpha=0.9, style="solid",
            )

        # nodes coloured by logical qubit
        nc = [qcolors[rev[p] % n_phys] for p in range(n_phys)]
        nx.draw_networkx_nodes(G, pos, ax=ax_graph, node_color=nc,
                               node_size=500, edgecolors="black",
                               linewidths=1.5)
        labels = {p: f"q{rev[p]}" for p in range(n_phys)}
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax_graph,
                                font_size=9, font_weight="bold")

        _draw_legend(ax_graph)

        ax_graph.set_title(
            f"Step {step_num}  |  SWAPs: {total_swaps}  |  "
            f"Gates: {len(executed_set)}/{n_gates}",
            fontsize=13, fontweight="bold",
        )
        ax_graph.axis("off")

        # --- Info panel (text) ---
        ax_info.axis("off")

        info_lines = []
        info_lines.append(f"ACTION:  {action_text}")
        info_lines.append(f"Reward this step: {reward:+.2f}")
        info_lines.append(f"Cumulative reward: {cum_reward:+.2f}")
        info_lines.append(f"SABRE total SWAPs: {sabre_swaps}")
        info_lines.append("")
        info_lines.append(
            f"GATE PROGRESS  ({len(executed_set)}/{n_gates}):"
        )
        info_lines.append("-" * 38)

        # Gate checklist
        gate_text = _gate_status_text(
            gates, executed_set, predecessors, n_phys, mapping
        )
        info_lines.append(gate_text)

        full_text = "\n".join(info_lines)
        ax_info.text(
            0.02, 0.98, full_text, transform=ax_info.transAxes,
            fontsize=9, verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="gray", alpha=0.9),
        )

        plt.tight_layout()
        buf = _fig_to_array(fig)
        plt.close(fig)
        return buf
    # ------------------------------------------------------------------

    frames = []
    init_mapping = trajectory["initial_mapping"]
    init_executed = set(trajectory.get("initial_executed", []))
    cum_reward = 0.0

    # Initial frame
    frames.append(render_frame(
        init_mapping, init_executed, None, list(init_executed),
        0, 0.0, 0.0, 0,
        "Initial state (no action yet)",
    ))

    prev_executed = set(init_executed)
    total_swaps = 0

    for i, step in enumerate(steps):
        action = step["action"]
        reward = step["reward"]
        mapping = step["mapping"]
        executed = set(step["executed"])
        total_swaps += 1
        cum_reward += reward

        swap_edge = tuple(edges[action]) if action < len(edges) else None
        just_exec = sorted(executed - prev_executed)

        # Build action description
        if swap_edge is not None:
            p1, p2 = swap_edge
            rev_before = _reverse_mapping(
                steps[i - 1]["mapping"] if i > 0 else init_mapping, n_phys
            )
            lq1, lq2 = rev_before[p1], rev_before[p2]
            action_text = (
                f"SWAP on edge ({p1},{p2})  —  "
                f"logical q{lq1} \u2194 q{lq2}"
            )
        else:
            action_text = f"Action {action}"

        if just_exec:
            gate_strs = [
                f"q{gates[gi][0]}\u2013q{gates[gi][1]}" for gi in just_exec
            ]
            action_text += f"\n  \u2192 Executed: {', '.join(gate_strs)}"

        frames.append(render_frame(
            mapping, executed, swap_edge, just_exec,
            i + 1, reward, cum_reward, total_swaps, action_text,
        ))
        prev_executed = executed

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(output_path), frames, fps=fps, loop=0)
    print(f"Saved GIF: {output_path} ({len(frames)} frames, {fps} fps)")


# ---------------------------------------------------------------------------
# 6. Side-by-side GIF  (agent graph + SABRE comparison + gate list)
# ---------------------------------------------------------------------------

def create_side_by_side_gif(trajectory, output_path, fps=1):
    """
    Agent routing on left, SABRE bar comparison centre, gate checklist right.
    """
    try:
        import imageio.v2 as imageio
    except ImportError:
        try:
            import imageio
        except ImportError:
            print("imageio not installed.")
            return

    n_phys = trajectory["n_physical"]
    edges = [tuple(e) for e in trajectory["edges"]]
    gates = [tuple(g) for g in trajectory["gates"]]
    steps = trajectory["steps"]
    n_gates = trajectory.get("n_gates", len(gates))
    sabre_swaps = trajectory.get("sabre_swaps", 0)
    predecessors = _reconstruct_predecessors(
        trajectory.get("predecessors", {})
    )

    G, pos = _build_graph_and_layout(n_phys, edges)
    qcolors = _qubit_palette(n_phys)

    frames = []
    init_mapping = trajectory["initial_mapping"]
    prev_executed = set(trajectory.get("initial_executed", []))
    cum_reward = 0.0

    for i, step in enumerate(steps):
        mapping = step["mapping"]
        executed = set(step["executed"])
        action = step["action"]
        reward = step["reward"]
        cum_reward += reward

        fig = plt.figure(figsize=(18, 8))
        gs = gridspec.GridSpec(1, 3, width_ratios=[3, 1.2, 2], figure=fig)
        ax_graph = fig.add_subplot(gs[0, 0])
        ax_bar   = fig.add_subplot(gs[0, 1])
        ax_info  = fig.add_subplot(gs[0, 2])

        rev = _reverse_mapping(mapping, n_phys)

        # --- LEFT: graph ---
        nx.draw_networkx_edges(G, pos, ax=ax_graph, edge_color="lightgray",
                               width=1, alpha=0.5)

        front = compute_front_layer(gates, executed, predecessors)
        for gi in front:
            qa, qb = gates[gi]
            if qa < len(mapping) and qb < len(mapping):
                pa, pb = mapping[qa], mapping[qb]
                if pa in pos and pb in pos:
                    ax_graph.plot(
                        [pos[pa][0], pos[pb][0]],
                        [pos[pa][1], pos[pb][1]],
                        color="dodgerblue", linestyle="--",
                        alpha=0.6, linewidth=1.5, zorder=2,
                    )

        just_exec = sorted(executed - prev_executed)
        for gi in just_exec:
            qa, qb = gates[gi]
            if qa < len(mapping) and qb < len(mapping):
                pa, pb = mapping[qa], mapping[qb]
                if pa in pos and pb in pos:
                    ax_graph.plot(
                        [pos[pa][0], pos[pb][0]],
                        [pos[pa][1], pos[pb][1]],
                        color="limegreen", linewidth=4, alpha=0.85,
                        zorder=3,
                    )

        swap_edge = tuple(edges[action]) if action < len(edges) else None
        if swap_edge:
            nx.draw_networkx_edges(
                G, pos, edgelist=[swap_edge], ax=ax_graph,
                edge_color="red", width=5, alpha=0.9,
            )

        nc = [qcolors[rev[p] % n_phys] for p in range(n_phys)]
        nx.draw_networkx_nodes(G, pos, ax=ax_graph, node_color=nc,
                               node_size=500, edgecolors="black",
                               linewidths=1.5)
        lbl = {p: f"q{rev[p]}" for p in range(n_phys)}
        nx.draw_networkx_labels(G, pos, labels=lbl, ax=ax_graph,
                                font_size=9, font_weight="bold")

        _draw_legend(ax_graph)

        # Action description
        if swap_edge:
            p1, p2 = swap_edge
            rev_b = _reverse_mapping(
                steps[i - 1]["mapping"] if i > 0 else init_mapping, n_phys
            )
            action_str = (
                f"SWAP ({p1},{p2}): q{rev_b[p1]} \u2194 q{rev_b[p2]}"
            )
        else:
            action_str = f"Action {action}"

        ax_graph.set_title(
            f"Step {i+1}  |  {action_str}  |  "
            f"Gates: {len(executed)}/{n_gates}",
            fontsize=12, fontweight="bold",
        )
        ax_graph.axis("off")

        # --- CENTRE: SWAP comparison bar ---
        agent_sw = i + 1
        bars = ax_bar.barh(
            ["Agent\n(so far)", "SABRE\n(total)"],
            [agent_sw, sabre_swaps],
            color=["steelblue", "coral"], edgecolor="black", linewidth=0.5,
        )
        # Add count labels on bars
        for bar, val in zip(bars, [agent_sw, sabre_swaps]):
            ax_bar.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                        str(val), va="center", fontweight="bold", fontsize=11)
        ax_bar.set_xlabel("SWAP Count", fontsize=10)
        ax_bar.set_title("Agent vs SABRE", fontsize=11, fontweight="bold")
        max_x = max(sabre_swaps, agent_sw, 3) * 1.4
        ax_bar.set_xlim(0, max_x)
        ax_bar.grid(True, alpha=0.3, axis="x")

        # --- RIGHT: gate checklist ---
        ax_info.axis("off")
        info_lines = [
            f"Reward: {reward:+.2f}  (cum: {cum_reward:+.2f})",
            "",
            f"GATE PROGRESS  ({len(executed)}/{n_gates}):",
            "-" * 36,
            _gate_status_text(gates, executed, predecessors, n_phys, mapping),
        ]
        ax_info.text(
            0.02, 0.98, "\n".join(info_lines),
            transform=ax_info.transAxes, fontsize=9,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="gray", alpha=0.9),
        )

        plt.tight_layout()
        frames.append(_fig_to_array(fig))
        plt.close(fig)
        prev_executed = executed

    if frames:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(output_path), frames, fps=fps, loop=0)
        print(f"Saved side-by-side GIF: {output_path} "
              f"({len(frames)} frames, {fps} fps)")
