"""Final evaluation script: PPO model vs SABRE on held-out QASM circuits."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

try:
    from .agent import SymmetricCNNActorCritic
    from .circuit_utils import (
        get_coupling_map,
        get_sabre_initial_mapping,
        get_sabre_swap_count,
        load_circuit,
    )
    from .environment import QubitRoutingEnv
except ImportError:
    from agent import SymmetricCNNActorCritic
    from circuit_utils import (
        get_coupling_map,
        get_sabre_initial_mapping,
        get_sabre_swap_count,
        load_circuit,
    )
    from environment import QubitRoutingEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PPO model against SABRE.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--qasmbench-root", type=str, required=True)
    parser.add_argument("--topologies", type=str, default="heavy_hex_19")
    parser.add_argument("--matrix-size", type=int, default=27)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-csv", type=str, default="")
    parser.add_argument("--summary-json", type=str, default="")
    return parser.parse_args()


def iter_qasm_files(root: Path) -> List[Path]:
    return sorted(root.rglob("*.qasm"))


def build_edge_index_and_mask(env, max_actions: int):
    topo = env._current_topo
    edge_index = np.zeros((max_actions, 2), dtype=np.int64)
    num_edges = topo["num_edges"]
    if num_edges > 0:
        edge_index[:num_edges] = np.asarray(topo["edges"], dtype=np.int64)
    action_mask = env.get_action_mask().astype(bool)
    return edge_index, action_mask


def greedy_action(model, obs, edge_index, action_mask, device):
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    edge_t = torch.as_tensor(edge_index, dtype=torch.long, device=device).unsqueeze(0)
    mask_t = torch.as_tensor(action_mask, dtype=torch.bool, device=device).unsqueeze(0)
    with torch.no_grad():
        dist, _ = model.get_action_distribution(obs_t, edge_t, mask_t)
        action = int(torch.argmax(dist.logits, dim=-1).item())
    return action


def evaluate_one_circuit(model, env, topo_idx: int, circuit, device) -> int:
    topo = env._topologies[topo_idx]
    sabre_init = get_sabre_initial_mapping(circuit, topo["coupling_map"])
    obs, _ = env.reset(
        options={
            "topology_index": topo_idx,
            "circuit": circuit,
            "initial_mapping": [int(x) for x in sabre_init],
        }
    )

    done = False
    truncated = False
    info: Dict = {}
    max_actions = int(env.action_space.n)
    while not done and not truncated:
        edge_index, action_mask = build_edge_index_and_mask(env, max_actions)
        action = greedy_action(model, obs, edge_index, action_mask, device)
        obs, _, done, truncated, info = env.step(action)

    return int(info.get("total_swaps", 0))


def main():
    args = parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    qasmbench_root = Path(args.qasmbench_root)
    topologies = [t.strip() for t in args.topologies.split(",") if t.strip()]

    if not qasmbench_root.exists():
        raise FileNotFoundError(f"QASM root not found: {qasmbench_root}")
    if not topologies:
        raise ValueError("No topology provided.")

    env = QubitRoutingEnv(
        topologies=topologies,
        matrix_size=args.matrix_size,
        max_steps=args.max_steps,
        gamma_decay=0.5,
        initial_mapping_strategy="sabre",
        seed=42,
    )

    model = SymmetricCNNActorCritic(matrix_size=env.N).to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()

    qasm_files = iter_qasm_files(qasmbench_root)
    if not qasm_files:
        raise ValueError(f"No .qasm files found in {qasmbench_root}")

    rows = []
    for topo_idx, topo_name in enumerate(topologies):
        cmap = get_coupling_map(topo_name)
        n_physical = cmap.size()
        for qasm_path in qasm_files:
            circuit = load_circuit(str(qasm_path))
            if circuit.num_qubits > n_physical:
                continue

            ppo_swaps = evaluate_one_circuit(model, env, topo_idx, circuit, device)
            sabre_swaps = int(get_sabre_swap_count(circuit, cmap))
            improvement_pct = (
                100.0 * (sabre_swaps - ppo_swaps) / max(1, sabre_swaps)
            )

            rows.append(
                {
                    "topology": topo_name,
                    "qasm_file": str(qasm_path),
                    "num_qubits": int(circuit.num_qubits),
                    "ppo_swaps": ppo_swaps,
                    "sabre_swaps": sabre_swaps,
                    "improvement_pct": improvement_pct,
                }
            )

    if not rows:
        raise ValueError("No compatible circuits found for selected topologies.")

    ppo_arr = np.asarray([r["ppo_swaps"] for r in rows], dtype=np.float32)
    sabre_arr = np.asarray([r["sabre_swaps"] for r in rows], dtype=np.float32)
    improvement_arr = np.asarray([r["improvement_pct"] for r in rows], dtype=np.float32)
    win_rate = float(np.mean(ppo_arr <= sabre_arr))

    summary = {
        "num_circuits_evaluated": int(len(rows)),
        "mean_ppo_swaps": float(np.mean(ppo_arr)),
        "mean_sabre_swaps": float(np.mean(sabre_arr)),
        "mean_improvement_pct": float(np.mean(improvement_arr)),
        "median_improvement_pct": float(np.median(improvement_arr)),
        "win_rate_vs_sabre": win_rate,
    }

    output_csv = Path(args.output_csv) if args.output_csv else Path("evaluation_vs_sabre.csv")
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "topology",
                "qasm_file",
                "num_qubits",
                "ppo_swaps",
                "sabre_swaps",
                "improvement_pct",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_json = Path(args.summary_json) if args.summary_json else Path("evaluation_summary.json")
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Evaluation finished.")
    print(f"CSV: {output_csv}")
    print(f"Summary JSON: {summary_json}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

