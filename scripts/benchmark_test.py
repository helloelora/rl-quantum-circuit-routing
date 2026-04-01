"""
Full benchmark: evaluate agent on generated algorithm circuits + QASMBench.

Two test suites run independently:
  Suite A — Generated circuits (110): QFT, VQE, GHZ, BV, QV, Structured, Random
  Suite B — QASMBench circuits (~40 usable): community-written real algorithms

Produces per-suite figures, combined figures, and detailed markdown/JSON logs.

Usage:
    python3 scripts/benchmark_test.py \
        --checkpoint outputs/run_030/checkpoints/checkpoint_best.pt \
        --qasmbench QASMBench \
        --output-dir outputs/tests/benchmark_002

    # Generated suite only (no QASMBench):
    python3 scripts/benchmark_test.py \
        --checkpoint outputs/run_030/checkpoints/checkpoint_best.pt
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qiskit import QuantumCircuit, qasm2
from qiskit.circuit.library import EfficientSU2, QFT, QuantumVolume
from qiskit.transpiler import CouplingMap
import torch

from config import TrainConfig
from circuit_utils import extract_two_qubit_gates, get_coupling_map, generate_random_circuit
from environment import QubitRoutingEnv
from dqn_agent import D3QNAgent
from evaluate import get_sabre_results, extend_mapping


# ── Color palette ───────────────────────────────────────────────────

SUITE_COLORS = {"Generated": "#2196F3", "QASMBench": "#9C27B0"}
AGENT_COLOR = "#2196F3"
SABRE_COLOR = "#FF9800"
WIN_COLOR = "#4CAF50"
LOSE_COLOR = "#F44336"
TIE_COLOR = "#9E9E9E"


def get_category_colors(categories):
    cmap = plt.cm.tab20
    return {cat: cmap(i / max(len(categories) - 1, 1))
            for i, cat in enumerate(categories)}


# ── Circuit generators ──────────────────────────────────────────────

def generate_benchmark_circuits(max_qubits):
    """Suite A: programmatically generated algorithm circuits."""
    circuits = []

    # QFT — all-to-all interaction
    for n in [4, 8, 12, 16, max_qubits]:
        if n > max_qubits:
            continue
        qft = QFT(n).decompose()
        circuits.append({
            "name": f"qft_{n}", "category": "QFT", "circuit": qft,
            "description": f"QFT on {n} qubits",
        })

    # VQE linear
    for n in [4, 8, 12, 16, max_qubits]:
        if n > max_qubits:
            continue
        for reps in [1, 3, 5]:
            c = EfficientSU2(n, reps=reps, entanglement="linear").decompose()
            circuits.append({
                "name": f"vqe_linear_{n}q_{reps}r", "category": "VQE",
                "circuit": c,
                "description": f"EfficientSU2 {n}q, {reps} reps, linear",
            })

    # VQE circular
    for n in [4, 8, 12, 16, max_qubits]:
        if n > max_qubits:
            continue
        for reps in [1, 3, 5]:
            c = EfficientSU2(n, reps=reps, entanglement="circular").decompose()
            circuits.append({
                "name": f"vqe_circular_{n}q_{reps}r", "category": "VQE",
                "circuit": c,
                "description": f"EfficientSU2 {n}q, {reps} reps, circular",
            })

    # VQE full
    for n in [4, 8, 12, max_qubits]:
        if n > max_qubits:
            continue
        for reps in [1, 2, 3]:
            c = EfficientSU2(n, reps=reps, entanglement="full").decompose()
            circuits.append({
                "name": f"vqe_full_{n}q_{reps}r", "category": "VQE",
                "circuit": c,
                "description": f"EfficientSU2 {n}q, {reps} reps, full",
            })

    # GHZ
    for n in [4, 8, 12, 16, max_qubits]:
        if n > max_qubits:
            continue
        ghz = QuantumCircuit(n)
        ghz.h(0)
        for i in range(n - 1):
            ghz.cx(i, i + 1)
        circuits.append({
            "name": f"ghz_{n}", "category": "GHZ", "circuit": ghz,
            "description": f"GHZ state on {n} qubits",
        })

    # Bernstein-Vazirani
    for n in [4, 8, 12, 16, max_qubits]:
        if n > max_qubits:
            continue
        bv = QuantumCircuit(n)
        bv.h(range(n - 1))
        bv.x(n - 1)
        bv.h(n - 1)
        for i in range(n - 1):
            bv.cx(i, n - 1)
        bv.h(range(n - 1))
        circuits.append({
            "name": f"bernstein_vazirani_{n}", "category": "Bernstein-Vazirani",
            "circuit": bv,
            "description": f"Bernstein-Vazirani {n}q — star CX pattern",
        })

    # Quantum Volume
    for n in [4, 8, 12, max_qubits]:
        if n > max_qubits:
            continue
        for seed in range(5):
            try:
                qv = QuantumVolume(n, depth=3, seed=seed).decompose().decompose()
                circuits.append({
                    "name": f"quantum_volume_{n}_s{seed}",
                    "category": "Quantum Volume", "circuit": qv,
                    "description": f"Quantum Volume {n}q, depth 3, seed {seed}",
                })
            except Exception:
                pass

    # Structured ring
    for reps in [3, 5, 10]:
        stair = QuantumCircuit(max_qubits)
        for _ in range(reps):
            for i in range(max_qubits):
                stair.cx(i, (i + 1) % max_qubits)
        circuits.append({
            "name": f"cnot_ring_{max_qubits}_{reps}rep", "category": "Structured",
            "circuit": stair,
            "description": f"Ring CNOT staircase {max_qubits}q, {reps} reps",
        })

    # Random baseline (30 circuits)
    for seed in range(30):
        circ = generate_random_circuit(max_qubits, 20, seed=seed)
        circuits.append({
            "name": f"random_d20_s{seed}", "category": "Random",
            "circuit": circ,
            "description": f"Random depth-20, {max_qubits}q (seed {seed})",
        })

    return circuits


def load_qasmbench_circuits(qasm_dir, max_qubits):
    """Suite B: load usable circuits from QASMBench repository.

    Uses subprocess isolation because some .qasm files crash the Qiskit
    Rust parser with an unrecoverable panic.
    """
    qasm_dir = Path(qasm_dir)
    circuits = []
    skipped = []

    # Collect all non-transpiled .qasm files
    qasm_files = []
    for size in ["small", "medium", "large"]:
        size_dir = qasm_dir / size
        if not size_dir.exists():
            continue
        for d in sorted(size_dir.iterdir()):
            if not d.is_dir():
                continue
            candidates = [f for f in d.glob("*.qasm") if "transpiled" not in f.name]
            if candidates:
                qasm_files.append(candidates[0])

    for qf in qasm_files:
        # Probe in subprocess to survive Rust panics
        try:
            proc = subprocess.run(
                [sys.executable, "-c",
                 f"from qiskit import qasm2; import json; "
                 f"c = qasm2.load('{qf}', custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS); "
                 f"n2q = sum(1 for i in c.data if len(i.qubits)==2); "
                 f"print(json.dumps({{'q': c.num_qubits, 'g': n2q}}))"],
                capture_output=True, text=True, timeout=30,
            )
            if proc.returncode != 0:
                skipped.append((qf.stem, "parse_crash"))
                continue
            info = json.loads(proc.stdout.strip())
        except (subprocess.TimeoutExpired, Exception) as e:
            skipped.append((qf.stem, str(e)[:40]))
            continue

        if info["q"] > max_qubits:
            skipped.append((qf.stem, f"too_large ({info['q']}q)"))
            continue
        if info["g"] == 0:
            skipped.append((qf.stem, "no_2q_gates"))
            continue

        # Now load for real (we know it works from the probe)
        try:
            circ = qasm2.load(str(qf), custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
            circuits.append({
                "name": qf.stem,
                "category": f"QB:{_classify_qasmbench(qf.stem)}",
                "circuit": circ,
                "description": f"QASMBench: {qf.name}",
            })
        except Exception:
            skipped.append((qf.stem, "load_fail"))

    return circuits, skipped


def _classify_qasmbench(name):
    """Group QASMBench circuit by algorithm family."""
    families = {
        "qft": "QFT/QPE", "qpe": "QFT/QPE", "inverseqft": "QFT/QPE",
        "pea": "QFT/QPE", "ipea": "QFT/QPE",
        "vqe": "VQE/Variational", "variational": "VQE/Variational",
        "qaoa": "VQE/Variational",
        "grover": "Search", "simon": "Search", "deutsch": "Search",
        "shor": "Factoring/Arithmetic", "qf21": "Factoring/Arithmetic",
        "adder": "Factoring/Arithmetic", "multiplier": "Factoring/Arithmetic",
        "multiply": "Factoring/Arithmetic", "square_root": "Factoring/Arithmetic",
        "hhl": "Linear Algebra", "linearsolver": "Linear Algebra",
        "ising": "Simulation", "basis_trotter": "Simulation",
        "quantumwalks": "Simulation",
        "error_correctiond3": "Error Correction", "qec_en": "Error Correction",
        "qec_sm": "Error Correction", "qec9xz": "Error Correction",
        "dnn": "ML/Classification", "knn": "ML/Classification",
        "cat_state": "State Prep", "bell": "State Prep", "ghz": "State Prep",
        "wstate": "State Prep", "teleportation": "State Prep",
    }
    base = name.rsplit("_n", 1)[0] if "_n" in name else name
    return families.get(base, "Other")


# ── Evaluation ───────────────────────────────────────────────────

def evaluate_circuit(agent, env, circuit, coupling_map, n_physical, topo_idx):
    """Route a single circuit with agent and SABRE. Returns result dict."""
    gates = extract_two_qubit_gates(circuit)
    n_2q_gates = len(gates)

    if n_2q_gates == 0:
        return {
            "n_2q_gates": 0, "agent_swaps": 0, "sabre_swaps": 0,
            "completed": True, "ratio": 1.0, "agent_steps": 0,
        }

    # SABRE baseline
    try:
        sabre_mapping, sabre_swaps = get_sabre_results(circuit, coupling_map)
        full_mapping = extend_mapping(sabre_mapping, n_physical)
    except Exception:
        full_mapping = list(range(n_physical))
        sabre_swaps = -1

    # Agent
    obs, info = env.reset(options={
        "circuit": circuit,
        "initial_mapping": full_mapping,
        "topology_index": topo_idx,
    })

    if info["done"]:
        return {
            "n_2q_gates": n_2q_gates, "agent_swaps": 0,
            "sabre_swaps": sabre_swaps, "completed": True,
            "ratio": 0.0 if sabre_swaps > 0 else 1.0, "agent_steps": 0,
        }

    mask = env.get_action_mask()
    steps = 0
    while True:
        action = agent.select_action(obs, mask, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        mask = env.get_action_mask()
        steps += 1
        if terminated or truncated:
            break

    agent_swaps = info["total_swaps"]
    completed = info["done"]
    ratio = agent_swaps / sabre_swaps if sabre_swaps > 0 and completed else float("nan")

    return {
        "n_2q_gates": n_2q_gates, "agent_swaps": agent_swaps,
        "sabre_swaps": sabre_swaps, "completed": completed,
        "ratio": ratio, "agent_steps": steps,
    }


def run_suite(agent, env, benchmarks, coupling_map, n_physical, suite_name):
    """Run evaluation on a list of benchmark circuits."""
    results = []
    t_start = time.time()

    for i, bench in enumerate(benchmarks):
        circuit = bench["circuit"]
        n_2q = len(extract_two_qubit_gates(circuit))
        print(f"  [{i+1:3d}/{len(benchmarks)}] {bench['name']:40s} "
              f"({circuit.num_qubits}q, {n_2q:>3d} 2q-gates) ... ",
              end="", flush=True)

        t0 = time.time()
        result = evaluate_circuit(agent, env, circuit, coupling_map, n_physical, 0)
        dt = time.time() - t0

        result.update({
            "name": bench["name"],
            "category": bench["category"],
            "n_qubits": circuit.num_qubits,
            "description": bench["description"],
            "suite": suite_name,
            "time_s": round(dt, 2),
        })
        results.append(result)

        status = "OK" if result["completed"] else "TIMEOUT"
        ratio_str = f"{result['ratio']:.3f}" if not np.isnan(result["ratio"]) else "---"
        print(f"agent={result['agent_swaps']:4d}  sabre={result['sabre_swaps']:4d}  "
              f"ratio={ratio_str:>7s}  [{status}]  ({dt:.1f}s)")

    elapsed = time.time() - t_start
    print(f"  Suite '{suite_name}' done in {elapsed:.1f}s "
          f"({len(results)} circuits)\n")
    return results


# ── Figures ────────────────────────────────────────────────────────

def _valid(results):
    """Filter to completed results with valid ratios."""
    return [r for r in results
            if r["completed"] and r["sabre_swaps"] > 0
            and not np.isnan(r["ratio"])]


def _stats(ratios):
    if not ratios:
        return {}
    return {
        "mean": float(np.mean(ratios)),
        "median": float(np.median(ratios)),
        "std": float(np.std(ratios)),
        "min": float(np.min(ratios)),
        "max": float(np.max(ratios)),
        "wins": sum(1 for r in ratios if r < 1.0),
        "ties": sum(1 for r in ratios if r == 1.0),
        "losses": sum(1 for r in ratios if r > 1.0),
        "n": len(ratios),
    }


def plot_suite(results, output_dir, suite_name, prefix):
    """Generate all figures for one suite."""
    output_dir = Path(output_dir)
    valid = _valid(results)
    if not valid:
        return

    categories = sorted(set(r["category"] for r in results))
    cat_colors = get_category_colors(categories)

    # ── Fig 1: Per-circuit bar chart ──
    fig, ax = plt.subplots(figsize=(max(16, len(valid) * 0.45), 7))
    names = [r["name"] for r in valid]
    a_swaps = [r["agent_swaps"] for r in valid]
    s_swaps = [r["sabre_swaps"] for r in valid]
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w/2, a_swaps, w, label="Agent (D3QN)", color=AGENT_COLOR, alpha=0.85)
    ax.bar(x + w/2, s_swaps, w, label="SABRE", color=SABRE_COLOR, alpha=0.85)
    ax.set_ylabel("SWAP Count", fontsize=12)
    ax.set_title(f"{suite_name} — Agent vs SABRE Per Circuit", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=70, ha="right", fontsize=6)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_per_circuit.png", dpi=150)
    plt.close(fig)

    # ── Fig 2: Ratio box plot by category ──
    cat_ratios = {}
    for cat in categories:
        ratios = [r["ratio"] for r in valid if r["category"] == cat]
        if ratios:
            cat_ratios[cat] = ratios

    if cat_ratios:
        fig, ax = plt.subplots(figsize=(max(10, len(cat_ratios) * 1.2), 6))
        positions = list(range(len(cat_ratios)))
        bp = ax.boxplot(cat_ratios.values(), positions=positions,
                        patch_artist=True, widths=0.5)
        for i, (cat, patch) in enumerate(zip(cat_ratios.keys(), bp["boxes"])):
            patch.set_facecolor(cat_colors.get(cat, "#aaa"))
            patch.set_alpha(0.7)
        # Overlay individual points
        for i, (cat, ratios) in enumerate(cat_ratios.items()):
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(ratios))
            ax.scatter(i + jitter, ratios, c=[cat_colors.get(cat, "#aaa")],
                       s=25, alpha=0.6, edgecolors="black", linewidths=0.3, zorder=3)
        ax.set_xticks(positions)
        ax.set_xticklabels(cat_ratios.keys(), fontsize=9, rotation=30, ha="right")
        ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.7, label="SABRE = 1.0")
        ax.set_ylabel("Agent/SABRE Ratio", fontsize=12)
        ax.set_title(f"{suite_name} — Ratio Distribution by Category", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"{prefix}_ratio_by_category.png", dpi=150)
        plt.close(fig)

    # ── Fig 3: Ratio vs 2Q gates (scatter) ──
    fig, ax = plt.subplots(figsize=(10, 6))
    for cat in categories:
        cr = [r for r in valid if r["category"] == cat]
        if cr:
            ax.scatter([r["n_2q_gates"] for r in cr], [r["ratio"] for r in cr],
                       c=[cat_colors[cat]], label=cat, s=60, alpha=0.8,
                       edgecolors="black", linewidths=0.5)
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.7)
    ax.set_xlabel("Number of 2-Qubit Gates", fontsize=12)
    ax.set_ylabel("Agent/SABRE Ratio", fontsize=12)
    ax.set_title(f"{suite_name} — Ratio vs Circuit Complexity", fontsize=14)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_ratio_vs_gates.png", dpi=150)
    plt.close(fig)

    # ── Fig 4: Ratio vs qubit count ──
    fig, ax = plt.subplots(figsize=(10, 6))
    for cat in categories:
        cr = [r for r in valid if r["category"] == cat]
        if cr:
            ax.scatter([r["n_qubits"] for r in cr], [r["ratio"] for r in cr],
                       c=[cat_colors[cat]], label=cat, s=60, alpha=0.8,
                       edgecolors="black", linewidths=0.5)
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.7)
    ax.set_xlabel("Number of Qubits", fontsize=12)
    ax.set_ylabel("Agent/SABRE Ratio", fontsize=12)
    ax.set_title(f"{suite_name} — Ratio vs Qubit Count", fontsize=14)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_ratio_vs_qubits.png", dpi=150)
    plt.close(fig)

    # ── Fig 5: Agent vs SABRE scatter (y=x line) ──
    fig, ax = plt.subplots(figsize=(8, 8))
    for cat in categories:
        cr = [r for r in valid if r["category"] == cat]
        if cr:
            ax.scatter([r["sabre_swaps"] for r in cr],
                       [r["agent_swaps"] for r in cr],
                       c=[cat_colors[cat]], label=cat, s=50, alpha=0.7,
                       edgecolors="black", linewidths=0.5)
    all_swaps = [r["agent_swaps"] for r in valid] + [r["sabre_swaps"] for r in valid]
    lim = max(all_swaps) * 1.1 if all_swaps else 100
    ax.plot([0, lim], [0, lim], "r--", alpha=0.5, label="y=x (tied)")
    ax.set_xlabel("SABRE SWAPs", fontsize=12)
    ax.set_ylabel("Agent SWAPs", fontsize=12)
    ax.set_title(f"{suite_name} — Agent vs SABRE SWAP Count", fontsize=14)
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_agent_vs_sabre.png", dpi=150)
    plt.close(fig)

    # ── Fig 6: Agent steps histogram ──
    steps = [r["agent_steps"] for r in valid if r["agent_steps"] > 0]
    if steps:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(steps, bins=30, color=AGENT_COLOR, alpha=0.7, edgecolor="black")
        ax.axvline(np.mean(steps), color="red", linestyle="--",
                    label=f"Mean: {np.mean(steps):.0f}")
        ax.set_xlabel("Agent Steps to Complete", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(f"{suite_name} — Agent Steps Distribution", fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"{prefix}_steps_hist.png", dpi=150)
        plt.close(fig)


def plot_combined(all_results, output_dir):
    """Generate combined figures across both suites."""
    output_dir = Path(output_dir)
    valid = _valid(all_results)
    if not valid:
        return

    suites = sorted(set(r["suite"] for r in valid))
    categories = sorted(set(r["category"] for r in valid))
    cat_colors = get_category_colors(categories)

    # ── Combined Fig 1: Grand summary dashboard (2x3) ──
    fig = plt.figure(figsize=(20, 13))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # 1a: Mean ratio by category (all categories from both suites)
    ax = fig.add_subplot(gs[0, 0])
    cat_means = {}
    for cat in categories:
        ratios = [r["ratio"] for r in valid if r["category"] == cat]
        if ratios:
            cat_means[cat] = np.mean(ratios)
    if cat_means:
        sorted_cats = sorted(cat_means.keys(), key=lambda c: cat_means[c])
        vals = [cat_means[c] for c in sorted_cats]
        colors = [WIN_COLOR if v < 1.0 else LOSE_COLOR for v in vals]
        bars = ax.barh(range(len(sorted_cats)), vals, color=colors, alpha=0.8)
        ax.set_yticks(range(len(sorted_cats)))
        ax.set_yticklabels(sorted_cats, fontsize=8)
        ax.axvline(x=1.0, color="red", linestyle="--", alpha=0.7)
        ax.set_xlabel("Mean SWAP Ratio")
        ax.set_title("Mean Ratio by Category (all suites)")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=8)

    # 1b: Win/Tie/Loss pie chart
    ax = fig.add_subplot(gs[0, 1])
    ratios = [r["ratio"] for r in valid]
    wins = sum(1 for r in ratios if r < 1.0)
    ties = sum(1 for r in ratios if r == 1.0)
    losses = sum(1 for r in ratios if r > 1.0)
    if wins + ties + losses > 0:
        sizes = [wins, ties, losses]
        labels = [f"Agent Wins\n({wins})", f"Ties\n({ties})", f"SABRE Wins\n({losses})"]
        colors_pie = [WIN_COLOR, TIE_COLOR, LOSE_COLOR]
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors_pie, autopct="%1.0f%%",
            startangle=90, textprops={"fontsize": 10})
        ax.set_title("Win/Tie/Loss Distribution")

    # 1c: Ratio distribution histogram (both suites overlaid)
    ax = fig.add_subplot(gs[0, 2])
    for suite in suites:
        sr = [r["ratio"] for r in valid if r["suite"] == suite]
        if sr:
            ax.hist(sr, bins=20, alpha=0.5, label=suite,
                    color=SUITE_COLORS.get(suite, "#999"), edgecolor="black")
    ax.axvline(x=1.0, color="red", linestyle="--", alpha=0.7, label="SABRE = 1.0")
    ax.set_xlabel("Agent/SABRE Ratio")
    ax.set_ylabel("Count")
    ax.set_title("Ratio Distribution by Suite")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 1d: Ratio vs 2Q gates (both suites, colored by suite)
    ax = fig.add_subplot(gs[1, 0])
    for suite in suites:
        sr = [r for r in valid if r["suite"] == suite]
        if sr:
            ax.scatter([r["n_2q_gates"] for r in sr], [r["ratio"] for r in sr],
                       c=SUITE_COLORS.get(suite, "#999"), label=suite,
                       s=40, alpha=0.7, edgecolors="black", linewidths=0.3)
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.7)
    ax.set_xlabel("2-Qubit Gates")
    ax.set_ylabel("Ratio")
    ax.set_title("Ratio vs Complexity (by suite)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 1e: Ratio vs qubits (both suites)
    ax = fig.add_subplot(gs[1, 1])
    for suite in suites:
        sr = [r for r in valid if r["suite"] == suite]
        if sr:
            ax.scatter([r["n_qubits"] for r in sr], [r["ratio"] for r in sr],
                       c=SUITE_COLORS.get(suite, "#999"), label=suite,
                       s=40, alpha=0.7, edgecolors="black", linewidths=0.3)
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.7)
    ax.set_xlabel("Qubits")
    ax.set_ylabel("Ratio")
    ax.set_title("Ratio vs Qubit Count (by suite)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 1f: Summary text
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    all_ratios = [r["ratio"] for r in valid]
    lines = []
    lines.append("COMBINED BENCHMARK SUMMARY")
    lines.append("=" * 44)
    lines.append(f"Total circuits:     {len(all_results)}")
    lines.append(f"Completed (>0 SWAPs): {len(valid)}")
    lines.append(f"Suites:             {', '.join(suites)}")
    lines.append("")
    lines.append(f"Overall mean ratio:  {np.mean(all_ratios):.4f}")
    lines.append(f"Overall median:      {np.median(all_ratios):.4f}")
    lines.append(f"Best ratio:          {np.min(all_ratios):.4f}")
    lines.append(f"Worst ratio:         {np.max(all_ratios):.4f}")
    lines.append(f"Std dev:             {np.std(all_ratios):.4f}")
    lines.append("")
    lines.append(f"Agent WINS:  {wins:3d} ({wins/len(all_ratios):.0%})")
    lines.append(f"TIES:        {ties:3d} ({ties/len(all_ratios):.0%})")
    lines.append(f"SABRE WINS:  {losses:3d} ({losses/len(all_ratios):.0%})")
    lines.append("")
    for suite in suites:
        sr = [r["ratio"] for r in valid if r["suite"] == suite]
        if sr:
            sw = sum(1 for r in sr if r < 1.0)
            lines.append(f"  {suite}: mean={np.mean(sr):.4f}  "
                         f"W/T/L={sw}/{sum(1 for r in sr if r==1.0)}/"
                         f"{sum(1 for r in sr if r>1.0)}")

    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle("Full Benchmark — D3QN Agent vs SABRE",
                 fontsize=16, fontweight="bold")
    fig.savefig(output_dir / "combined_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Combined Fig 2: Per-suite mean ratio comparison ──
    fig, ax = plt.subplots(figsize=(12, 6))
    # Group by category, show both suites side by side
    all_cats = sorted(set(r["category"] for r in valid))
    suite_list = sorted(suites)
    n_suites = len(suite_list)
    x = np.arange(len(all_cats))
    w = 0.35
    for si, suite in enumerate(suite_list):
        means = []
        for cat in all_cats:
            rs = [r["ratio"] for r in valid
                  if r["suite"] == suite and r["category"] == cat]
            means.append(np.mean(rs) if rs else 0)
        offset = (si - (n_suites - 1) / 2) * w
        bars = ax.bar(x + offset, means, w, label=suite,
                      color=SUITE_COLORS.get(suite, "#999"), alpha=0.8)
        for bar, val in zip(bars, means):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", fontsize=7, rotation=45)
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.7, label="SABRE = 1.0")
    ax.set_xticks(x)
    ax.set_xticklabels(all_cats, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean SWAP Ratio")
    ax.set_title("Mean Ratio by Category — Suite Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "combined_ratio_comparison.png", dpi=150)
    plt.close(fig)

    # ── Combined Fig 3: Agent vs SABRE scatter (colored by suite) ──
    fig, ax = plt.subplots(figsize=(8, 8))
    for suite in suites:
        sr = [r for r in valid if r["suite"] == suite]
        if sr:
            ax.scatter([r["sabre_swaps"] for r in sr],
                       [r["agent_swaps"] for r in sr],
                       c=SUITE_COLORS.get(suite, "#999"), label=suite,
                       s=50, alpha=0.7, edgecolors="black", linewidths=0.3)
    all_swaps = [r["agent_swaps"] for r in valid] + [r["sabre_swaps"] for r in valid]
    lim = max(all_swaps) * 1.1 if all_swaps else 100
    ax.plot([0, lim], [0, lim], "r--", alpha=0.5, label="y=x (tied)")
    ax.set_xlabel("SABRE SWAPs", fontsize=12)
    ax.set_ylabel("Agent SWAPs", fontsize=12)
    ax.set_title("Agent vs SABRE — Both Suites", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "combined_agent_vs_sabre.png", dpi=150)
    plt.close(fig)


# ── Logging ────────────────────────────────────────────────────────

def write_results(all_results, output_dir, qasmbench_skipped=None):
    """Write detailed JSON + markdown results."""
    output_dir = Path(output_dir)

    # ── JSON (full data) ──
    def _default(x):
        if isinstance(x, float) and np.isnan(x):
            return None
        if isinstance(x, np.floating):
            return float(x)
        if isinstance(x, np.integer):
            return int(x)
        return x

    with open(output_dir / "benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=_default)

    # ── Markdown ──
    suites = sorted(set(r["suite"] for r in all_results))
    lines = ["# Full Benchmark Results — D3QN Agent vs SABRE\n"]
    lines.append(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Grand summary
    valid = _valid(all_results)
    if valid:
        all_ratios = [r["ratio"] for r in valid]
        s = _stats(all_ratios)
        lines.append("## Overall Summary\n")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Total circuits | {len(all_results)} |")
        lines.append(f"| Completed (with SWAPs) | {len(valid)} |")
        lines.append(f"| Mean ratio | **{s['mean']:.4f}** |")
        lines.append(f"| Median ratio | {s['median']:.4f} |")
        lines.append(f"| Std dev | {s['std']:.4f} |")
        lines.append(f"| Best ratio | {s['min']:.4f} |")
        lines.append(f"| Worst ratio | {s['max']:.4f} |")
        lines.append(f"| Agent wins | {s['wins']} ({s['wins']/s['n']:.0%}) |")
        lines.append(f"| Ties | {s['ties']} ({s['ties']/s['n']:.0%}) |")
        lines.append(f"| SABRE wins | {s['losses']} ({s['losses']/s['n']:.0%}) |")
        lines.append("")

        # Per-suite summary table
        lines.append("### Per-Suite Summary\n")
        lines.append("| Suite | Circuits | Completed | Mean Ratio | Median | Wins | Ties | Losses |")
        lines.append("|-------|----------|-----------|------------|--------|------|------|--------|")
        for suite in suites:
            sr_all = [r for r in all_results if r["suite"] == suite]
            sr_valid = [r for r in valid if r["suite"] == suite]
            sr_ratios = [r["ratio"] for r in sr_valid]
            if sr_ratios:
                ss = _stats(sr_ratios)
                lines.append(f"| {suite} | {len(sr_all)} | {len(sr_valid)} | "
                             f"**{ss['mean']:.4f}** | {ss['median']:.4f} | "
                             f"{ss['wins']} | {ss['ties']} | {ss['losses']} |")
        lines.append("")

    # Per-suite detailed tables
    for suite in suites:
        suite_results = [r for r in all_results if r["suite"] == suite]
        suite_valid = _valid(suite_results)
        categories = sorted(set(r["category"] for r in suite_results))

        lines.append(f"---\n\n## Suite: {suite}\n")
        if suite_valid:
            sr = [r["ratio"] for r in suite_valid]
            lines.append(f"**{len(suite_results)} circuits**, "
                         f"{len(suite_valid)} completed with SWAPs, "
                         f"mean ratio {np.mean(sr):.4f}\n")

        # Category summary table
        lines.append("### Category Summary\n")
        lines.append("| Category | Circuits | Completed | Mean Ratio | Median | "
                     "Min | Max | Wins | Losses |")
        lines.append("|----------|----------|-----------|------------|--------|"
                     "-----|-----|------|--------|")
        for cat in categories:
            cr_all = [r for r in suite_results if r["category"] == cat]
            cr_valid = [r for r in suite_valid if r["category"] == cat]
            cr_ratios = [r["ratio"] for r in cr_valid]
            if cr_ratios:
                cs = _stats(cr_ratios)
                lines.append(f"| {cat} | {len(cr_all)} | {len(cr_valid)} | "
                             f"**{cs['mean']:.4f}** | {cs['median']:.4f} | "
                             f"{cs['min']:.3f} | {cs['max']:.3f} | "
                             f"{cs['wins']} | {cs['losses']} |")
            else:
                lines.append(f"| {cat} | {len(cr_all)} | 0 | — | — | — | — | 0 | 0 |")
        lines.append("")

        # Detailed per-circuit table
        for cat in categories:
            cr = [r for r in suite_results if r["category"] == cat]
            lines.append(f"### {cat}\n")
            lines.append("| Circuit | Qubits | 2Q Gates | Agent SWAPs | SABRE SWAPs | "
                         "Ratio | Steps | Time | Status |")
            lines.append("|---------|--------|----------|-------------|-------------|"
                         "-------|-------|------|--------|")
            for r in cr:
                ratio_str = f"{r['ratio']:.3f}" if not np.isnan(r["ratio"]) else "—"
                status = "OK" if r["completed"] else "TIMEOUT"
                # Highlight wins/losses
                if not np.isnan(r["ratio"]):
                    if r["ratio"] < 1.0:
                        ratio_str = f"**{ratio_str}**"
                lines.append(
                    f"| {r['name']} | {r['n_qubits']} | {r['n_2q_gates']} | "
                    f"{r['agent_swaps']} | {r['sabre_swaps']} | {ratio_str} | "
                    f"{r['agent_steps']} | {r['time_s']:.1f}s | {status} |")
            lines.append("")

    # QASMBench skipped circuits
    if qasmbench_skipped:
        lines.append("---\n\n## QASMBench — Skipped Circuits\n")
        lines.append(f"{len(qasmbench_skipped)} circuits could not be used:\n")
        lines.append("| Circuit | Reason |")
        lines.append("|---------|--------|")
        for name, reason in qasmbench_skipped:
            lines.append(f"| {name} | {reason} |")
        lines.append("")

    with open(output_dir / "benchmark_results.md", "w") as f:
        f.write("\n".join(lines))

    # ── Per-category stats JSON ──
    cat_stats = {}
    for r in valid:
        cat = r["category"]
        cat_stats.setdefault(cat, []).append(r["ratio"])
    summary = {cat: _stats(ratios) for cat, ratios in cat_stats.items()}
    with open(output_dir / "category_stats.json", "w") as f:
        json.dump(summary, f, indent=2, default=_default)


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Full benchmark: D3QN agent vs SABRE on generated + QASMBench circuits"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--topology", type=str, default=None)
    parser.add_argument("--qasmbench", type=str, default=None,
                        help="Path to QASMBench root directory")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="Max steps per circuit (default: 1000)")
    args = parser.parse_args()

    print("=" * 60)
    print("  FULL BENCHMARK — D3QN Agent vs SABRE")
    print("=" * 60)

    # ── Load config ──
    ckpt_path = Path(args.checkpoint)
    config_path = Path(args.config) if args.config else ckpt_path.parent.parent / "config.json"
    print(f"\nConfig:     {config_path}")
    config = TrainConfig.load(str(config_path))
    if args.device != "auto":
        config.device = args.device

    topology = args.topology or config.topologies[0]
    coupling_map = get_coupling_map(topology)
    n_physical = coupling_map.size()
    print(f"Topology:   {topology} ({n_physical} qubits, "
          f"{len(coupling_map.get_edges())//2} edges)")
    print(f"Checkpoint: {ckpt_path}")

    # ── Setup output dir ──
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        test_base = Path("outputs/tests")
        test_base.mkdir(parents=True, exist_ok=True)
        existing = sorted(test_base.glob("benchmark_*"))
        if existing:
            nums = [int(p.name.split("_")[1]) for p in existing
                    if p.name.split("_")[1].isdigit()]
            test_num = max(nums) + 1 if nums else 1
        else:
            test_num = 1
        output_dir = test_base / f"benchmark_{test_num:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "generated").mkdir(exist_ok=True)
    (output_dir / "qasmbench").mkdir(exist_ok=True)
    print(f"Output:     {output_dir}")

    # ── Create environment ──
    benchmark_max_steps = max(config.max_steps, args.max_steps)
    print(f"Max steps:  {benchmark_max_steps}")

    env = QubitRoutingEnv(
        topologies=[topology],
        circuit_depth=config.circuit_depth,
        max_steps=benchmark_max_steps,
        gamma_decay=config.gamma_decay,
        distance_reward_coeff=config.distance_reward_coeff,
        completion_bonus=config.completion_bonus,
        timeout_penalty=config.timeout_penalty,
        repetition_penalty=config.repetition_penalty,
        gate_execution_reward=config.gate_execution_reward,
        matrix_size=config.matrix_size,
        initial_mapping_strategy="random",
        seed=42,
    )

    # ── Load agent ──
    agent = D3QNAgent(config, env.max_edges)
    agent.load_checkpoint(str(ckpt_path))
    agent.online_net.eval()
    n_params = sum(p.numel() for p in agent.online_net.parameters())
    print(f"Agent:      {n_params:,} parameters\n")

    all_results = []

    # ── Suite A: Generated circuits ──
    print("=" * 60)
    print("  SUITE A: Generated Algorithm Circuits")
    print("=" * 60)
    gen_circuits = generate_benchmark_circuits(n_physical)
    print(f"  {len(gen_circuits)} circuits across "
          f"{len(set(c['category'] for c in gen_circuits))} categories\n")
    gen_results = run_suite(agent, env, gen_circuits, coupling_map,
                            n_physical, "Generated")
    all_results.extend(gen_results)

    print("  Generating Suite A figures...")
    plot_suite(gen_results, output_dir / "generated", "Suite A: Generated", "gen")

    # ── Suite B: QASMBench ──
    qb_skipped = []
    if args.qasmbench:
        qb_dir = Path(args.qasmbench)
        if qb_dir.exists():
            print("=" * 60)
            print("  SUITE B: QASMBench Circuits")
            print("=" * 60)
            qb_circuits, qb_skipped = load_qasmbench_circuits(qb_dir, n_physical)
            print(f"  {len(qb_circuits)} usable circuits, "
                  f"{len(qb_skipped)} skipped\n")
            if qb_circuits:
                qb_results = run_suite(agent, env, qb_circuits, coupling_map,
                                       n_physical, "QASMBench")
                all_results.extend(qb_results)
                print("  Generating Suite B figures...")
                plot_suite(qb_results, output_dir / "qasmbench",
                           "Suite B: QASMBench", "qb")
        else:
            print(f"\n  WARNING: QASMBench directory not found: {qb_dir}\n")

    # ── Combined figures ──
    print("Generating combined figures...")
    plot_combined(all_results, output_dir)

    # ── Write all results ──
    print("Writing results...")
    write_results(all_results, output_dir, qb_skipped)

    # Save test config
    test_meta = {
        "checkpoint": str(ckpt_path),
        "config": str(config_path),
        "topology": topology,
        "n_physical": n_physical,
        "max_steps": benchmark_max_steps,
        "n_circuits_total": len(all_results),
        "n_generated": sum(1 for r in all_results if r["suite"] == "Generated"),
        "n_qasmbench": sum(1 for r in all_results if r["suite"] == "QASMBench"),
        "n_qasmbench_skipped": len(qb_skipped),
        "total_time_s": sum(r["time_s"] for r in all_results),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(output_dir / "test_config.json", "w") as f:
        json.dump(test_meta, f, indent=2)

    # ── Final summary ──
    valid = _valid(all_results)
    if valid:
        all_ratios = [r["ratio"] for r in valid]
        wins = sum(1 for r in all_ratios if r < 1.0)
        ties = sum(1 for r in all_ratios if r == 1.0)
        losses = sum(1 for r in all_ratios if r > 1.0)
        total_time = sum(r["time_s"] for r in all_results)

        print(f"\n{'='*60}")
        print(f"  FULL BENCHMARK RESULTS")
        print(f"{'='*60}")
        print(f"  Total circuits:   {len(all_results)}")
        print(f"  Completed:        {len(valid)}")
        print(f"  Mean ratio:       {np.mean(all_ratios):.4f}")
        print(f"  Median ratio:     {np.median(all_ratios):.4f}")
        print(f"  Wins/Ties/Losses: {wins}/{ties}/{losses}")
        print(f"  Total time:       {total_time:.1f}s")
        print()
        for suite in sorted(set(r["suite"] for r in valid)):
            sr = [r["ratio"] for r in valid if r["suite"] == suite]
            sw = sum(1 for r in sr if r < 1.0)
            print(f"  {suite:12s}  mean={np.mean(sr):.4f}  median={np.median(sr):.4f}  "
                  f"W/T/L={sw}/{sum(1 for r in sr if r==1.0)}/{sum(1 for r in sr if r>1.0)}")
        print(f"\n  Output: {output_dir}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
