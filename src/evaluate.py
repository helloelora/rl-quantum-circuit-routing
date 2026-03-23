"""Evaluation: agent vs SABRE comparison + trajectory recording."""

import json
import numpy as np
from pathlib import Path

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from circuit_utils import (
    generate_random_circuit,
    get_coupling_map,
    load_circuit,
)
from environment import QubitRoutingEnv


def get_sabre_results(circuit, coupling_map):
    """Run SABRE once, return (initial_mapping, swap_count)."""
    pm = generate_preset_pass_manager(
        optimization_level=1, coupling_map=coupling_map
    )
    transpiled = pm.run(circuit)
    init_layout = transpiled.layout.initial_layout
    mapping = [
        init_layout[circuit.qubits[i]] for i in range(circuit.num_qubits)
    ]
    ops = transpiled.count_ops()
    swap_count = ops.get("swap", 0)
    return mapping, swap_count


def extend_mapping(partial_mapping, n_physical):
    """Extend partial mapping to a full permutation of n_physical positions."""
    used = set(partial_mapping)
    available = sorted(p for p in range(n_physical) if p not in used)
    return list(partial_mapping) + available


def run_evaluation(agent, env, config, eval_episodes=20,
                   log_trajectories=False):
    """
    Evaluate agent on random circuits and compare with SABRE.

    For each topology, generates eval_episodes random circuits. Both agent
    (greedy, SABRE initial mapping) and SABRE route the same circuit.

    Returns:
        dict with "results" (per-episode), "trajectories" (if requested),
        and "summary" (aggregated).
    """
    results = []
    trajectories = []

    for topo_idx, topo_name in enumerate(config.topologies):
        coupling_map = get_coupling_map(topo_name)
        n_physical = coupling_map.size()

        for ep in range(eval_episodes):
            circuit = generate_random_circuit(
                n_physical, config.circuit_depth,
                seed=np.random.randint(0, 2**31),
            )

            # SABRE baseline (same circuit)
            try:
                sabre_mapping, sabre_swaps = get_sabre_results(
                    circuit, coupling_map
                )
                full_mapping = extend_mapping(sabre_mapping, n_physical)
            except Exception:
                # Fallback if SABRE fails on a degenerate circuit
                full_mapping = list(range(n_physical))
                sabre_swaps = -1

            # Reset env with same circuit + SABRE initial mapping
            obs, info = env.reset(options={
                "circuit": circuit,
                "initial_mapping": full_mapping,
                "topology_index": topo_idx,
            })

            if info["done"]:
                results.append({
                    "topology": topo_name,
                    "n_gates": info["n_gates"],
                    "agent_swaps": 0,
                    "sabre_swaps": sabre_swaps,
                    "completed": True,
                    "total_reward": 0.0,
                })
                continue

            mask = env.get_action_mask()
            total_reward = 0.0
            traj_steps = []

            if log_trajectories:
                initial_mapping = list(env.mapping[:n_physical])
                initial_executed = sorted(env.executed)
                gates = [list(g) for g in env.gates]
                preds = {
                    str(k): sorted(v)
                    for k, v in env.predecessors.items()
                }

            while True:
                action = agent.select_action(obs, mask, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                mask = env.get_action_mask()
                total_reward += reward

                if log_trajectories:
                    traj_steps.append({
                        "action": int(action),
                        "reward": round(float(reward), 4),
                        "mapping": list(env.mapping[:n_physical]),
                        "executed": sorted(env.executed),
                    })

                if terminated or truncated:
                    break

            results.append({
                "topology": topo_name,
                "n_gates": info["n_gates"],
                "agent_swaps": info["total_swaps"],
                "sabre_swaps": sabre_swaps,
                "completed": info["done"],
                "total_reward": round(total_reward, 3),
            })

            if log_trajectories:
                trajectories.append({
                    "topology": topo_name,
                    "n_physical": n_physical,
                    "edges": [
                        list(e) for e in env._current_topo["edges"]
                    ],
                    "gates": gates,
                    "predecessors": preds,
                    "initial_mapping": initial_mapping,
                    "initial_executed": initial_executed,
                    "sabre_swaps": sabre_swaps,
                    "agent_swaps": info["total_swaps"],
                    "completed": info["done"],
                    "total_reward": round(total_reward, 3),
                    "n_gates": info["n_gates"],
                    "steps": traj_steps,
                })

    # Summary
    if results:
        completed = [r for r in results if r["completed"]]
        valid = [r for r in results if r["sabre_swaps"] > 0]
        ratios = [
            r["agent_swaps"] / r["sabre_swaps"]
            for r in valid if r["completed"]
        ]

        summary = {
            "total_episodes": len(results),
            "completion_rate": len(completed) / len(results),
            "mean_agent_swaps": np.mean(
                [r["agent_swaps"] for r in results]
            ),
            "mean_sabre_swaps": np.mean(
                [r["sabre_swaps"] for r in results if r["sabre_swaps"] >= 0]
            ),
            "mean_swap_ratio": np.mean(ratios) if ratios else float("nan"),
            "median_swap_ratio": (
                float(np.median(ratios)) if ratios else float("nan")
            ),
            "mean_reward": np.mean(
                [r["total_reward"] for r in results]
            ),
        }
    else:
        summary = {}

    return {
        "results": results,
        "trajectories": trajectories,
        "summary": summary,
    }


def run_qasmbench_evaluation(agent, env, config, qasm_dir, max_qubits=None):
    """
    Evaluate on QASMBench circuits.

    Loads all .qasm files from qasm_dir, filters by qubit count,
    routes each with agent and SABRE.
    """
    qasm_dir = Path(qasm_dir)
    if not qasm_dir.exists():
        print(f"QASMBench directory not found: {qasm_dir}")
        return {"results": [], "summary": {}}

    results = []

    for topo_idx, topo_name in enumerate(config.topologies):
        coupling_map = get_coupling_map(topo_name)
        n_physical = coupling_map.size()
        if max_qubits is None:
            max_q = n_physical
        else:
            max_q = min(max_qubits, n_physical)

        qasm_files = sorted(qasm_dir.glob("*.qasm"))

        for qasm_path in qasm_files:
            try:
                circuit = load_circuit(str(qasm_path))
            except Exception as e:
                print(f"  Skip {qasm_path.name}: {e}")
                continue

            if circuit.num_qubits > max_q:
                continue

            # SABRE
            try:
                sabre_mapping, sabre_swaps = get_sabre_results(
                    circuit, coupling_map
                )
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
                results.append({
                    "circuit": qasm_path.name,
                    "topology": topo_name,
                    "n_qubits": circuit.num_qubits,
                    "n_gates": info["n_gates"],
                    "agent_swaps": 0,
                    "sabre_swaps": sabre_swaps,
                    "completed": True,
                })
                continue

            mask = env.get_action_mask()

            while True:
                action = agent.select_action(obs, mask, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                mask = env.get_action_mask()
                if terminated or truncated:
                    break

            results.append({
                "circuit": qasm_path.name,
                "topology": topo_name,
                "n_qubits": circuit.num_qubits,
                "n_gates": info["n_gates"],
                "agent_swaps": info["total_swaps"],
                "sabre_swaps": sabre_swaps,
                "completed": info["done"],
            })

    # Summary
    completed = [r for r in results if r["completed"]]
    valid = [r for r in completed if r["sabre_swaps"] > 0]
    ratios = [r["agent_swaps"] / r["sabre_swaps"] for r in valid]

    summary = {
        "total_circuits": len(results),
        "completed": len(completed),
        "mean_agent_swaps": (
            np.mean([r["agent_swaps"] for r in results]) if results else 0
        ),
        "mean_sabre_swaps": (
            np.mean([r["sabre_swaps"] for r in valid]) if valid else 0
        ),
        "mean_swap_ratio": np.mean(ratios) if ratios else float("nan"),
    }

    return {"results": results, "summary": summary}


def save_eval_results(eval_output, path):
    """Save evaluation results to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    data = json.loads(json.dumps(eval_output, default=convert))
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
