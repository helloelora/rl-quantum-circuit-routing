"""
Evaluate a supervised policy checkpoint against SABRE by rollout.

This script is designed for sanity-check V2 checkpoints and reports
routing-quality metrics (not just imitation accuracy):
  - mean model swaps
  - mean SABRE swaps
  - improvement % vs SABRE
  - win rate vs SABRE
  - timeout rate

It uses holdout random circuits with a dedicated eval seed base to avoid
overlap with training circuits.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from circuit_utils import (  # noqa: E402
    generate_random_circuit,
    get_coupling_map,
    get_sabre_initial_mapping,
    get_sabre_swap_count,
)
from dqn_agent import SymmetricCNNQNetwork  # noqa: E402
from environment import QubitRoutingEnv  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate supervised policy checkpoint by rollout vs SABRE."
    )
    parser.add_argument("--project-root", type=str, default=str(PROJECT_ROOT))
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--topologies", type=str, default="linear_5")
    parser.add_argument("--matrix-size", type=int, default=27)
    parser.add_argument("--circuits-per-topology", type=int, default=200)
    parser.add_argument("--circuit-depth", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--min-two-qubit-gates", type=int, default=8)
    parser.add_argument("--circuit-generation-attempts", type=int, default=16)
    parser.add_argument("--eval-seed-base", type=int, default=4_000_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def build_edge_index_and_mask(env, max_actions: int) -> Tuple[np.ndarray, np.ndarray]:
    topo = env._current_topo
    edge_index = np.zeros((max_actions, 2), dtype=np.int64)
    num_edges = int(topo["num_edges"])
    if num_edges > 0:
        edge_index[:num_edges] = np.asarray(topo["edges"], dtype=np.int64)
    action_mask = env.get_action_mask().astype(bool)
    return edge_index, action_mask


def strip_known_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())

    # From supervised_sanity_check(_v2): SymmetricCNNClassifier with attribute "net".
    if all(k.startswith("net.") for k in keys):
        return {k[len("net."):]: v for k, v in state_dict.items()}

    # Some checkpoints may nest inside q_net.
    if all(k.startswith("q_net.") for k in keys):
        return {k[len("q_net."):]: v for k, v in state_dict.items()}

    return state_dict


def load_q_network(model_path: Path, matrix_size: int, device: str) -> SymmetricCNNQNetwork:
    raw = torch.load(model_path, map_location=device)
    if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        raw = raw["state_dict"]
    if not isinstance(raw, dict):
        raise ValueError(f"Unsupported checkpoint format in {model_path}.")

    state_dict = strip_known_prefixes(raw)
    model = SymmetricCNNQNetwork(matrix_size=matrix_size).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def greedy_action(model, obs, edge_index, action_mask, device: str) -> int:
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    edge_t = torch.as_tensor(edge_index, dtype=torch.long, device=device).unsqueeze(0)
    mask_t = torch.as_tensor(action_mask, dtype=torch.bool, device=device).unsqueeze(0)
    with torch.no_grad():
        q_values = model.get_q_values(obs_t, edge_t, mask_t)
        action = int(torch.argmax(q_values, dim=-1).item())
    return action


def evaluate_case(
    env,
    topo_idx: int,
    circuit,
    sabre_init: List[int],
    model: SymmetricCNNQNetwork,
    device: str,
) -> Tuple[int, bool]:
    obs, _ = env.reset(
        options={
            "topology_index": topo_idx,
            "circuit": circuit,
            "initial_mapping": [int(x) for x in sabre_init],
        }
    )
    max_actions = int(env.action_space.n)

    done = False
    truncated = False
    info = {}
    while not done and not truncated:
        edge_index, action_mask = build_edge_index_and_mask(env, max_actions)
        action = greedy_action(model, obs, edge_index, action_mask, device)
        obs, _, done, truncated, info = env.step(action)

    swaps = int(info.get("total_swaps", 0))
    timeout = bool(truncated and not done)
    return swaps, timeout


def main():
    args = parse_args()
    device = resolve_device(args.device)
    topologies = [t.strip() for t in args.topologies.split(",") if t.strip()]
    if not topologies:
        raise ValueError("No topology provided.")

    project_root = Path(args.project_root)
    output_dir = Path(args.output_dir) if args.output_dir else (project_root / "results" / "supervised_rollout_eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    env = QubitRoutingEnv(
        topologies=topologies,
        matrix_size=args.matrix_size,
        circuit_depth=args.circuit_depth,
        max_steps=args.max_steps,
        min_two_qubit_gates=args.min_two_qubit_gates,
        circuit_generation_attempts=args.circuit_generation_attempts,
        initial_mapping_strategy="sabre",
        seed=args.seed,
    )
    model = load_q_network(model_path=model_path, matrix_size=env.N, device=device)

    rng = np.random.default_rng(args.seed)
    rows: List[Dict] = []

    for topo_idx, topo_name in enumerate(topologies):
        cmap = get_coupling_map(topo_name)
        n_qubits = int(cmap.size())
        for case_idx in range(int(args.circuits_per_topology)):
            cseed = int(args.eval_seed_base + topo_idx * 10_000 + case_idx)
            # Add controlled jitter per global seed to support repeated experiments.
            cseed = int((cseed + rng.integers(0, 2**20)) % (2**31 - 1))

            circuit = generate_random_circuit(
                num_qubits=n_qubits,
                depth=args.circuit_depth,
                seed=cseed,
                min_two_qubit_gates=args.min_two_qubit_gates,
                max_attempts=args.circuit_generation_attempts,
            )
            sabre_init = get_sabre_initial_mapping(circuit, cmap)
            sabre_swaps = int(get_sabre_swap_count(circuit, cmap))
            model_swaps, timeout = evaluate_case(
                env=env,
                topo_idx=topo_idx,
                circuit=circuit,
                sabre_init=sabre_init,
                model=model,
                device=device,
            )
            improve_pct = float(100.0 * (sabre_swaps - model_swaps) / max(1, sabre_swaps))
            win = int(model_swaps <= sabre_swaps)
            rows.append(
                {
                    "topology": topo_name,
                    "case_index": int(case_idx),
                    "seed": int(cseed),
                    "model_swaps": int(model_swaps),
                    "sabre_swaps": int(sabre_swaps),
                    "improvement_pct": improve_pct,
                    "win_vs_sabre": int(win),
                    "timeout": int(timeout),
                }
            )

    if not rows:
        raise RuntimeError("No evaluation rows were produced.")

    csv_path = output_dir / "supervised_rollout_eval.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "topology",
                "case_index",
                "seed",
                "model_swaps",
                "sabre_swaps",
                "improvement_pct",
                "win_vs_sabre",
                "timeout",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Global summary
    model_arr = np.asarray([r["model_swaps"] for r in rows], dtype=np.float32)
    sabre_arr = np.asarray([r["sabre_swaps"] for r in rows], dtype=np.float32)
    improve_arr = np.asarray([r["improvement_pct"] for r in rows], dtype=np.float32)
    win_arr = np.asarray([r["win_vs_sabre"] for r in rows], dtype=np.float32)
    timeout_arr = np.asarray([r["timeout"] for r in rows], dtype=np.float32)

    summary = {
        "model_path": str(model_path),
        "device": device,
        "topologies": topologies,
        "cases_total": int(len(rows)),
        "mean_model_swaps": float(np.mean(model_arr)),
        "mean_sabre_swaps": float(np.mean(sabre_arr)),
        "mean_improvement_pct": float(np.mean(improve_arr)),
        "median_improvement_pct": float(np.median(improve_arr)),
        "win_rate_vs_sabre": float(np.mean(win_arr)),
        "timeout_rate": float(np.mean(timeout_arr)),
    }

    # Per-topology summaries
    per_topology = {}
    for topo in topologies:
        sub = [r for r in rows if r["topology"] == topo]
        m = np.asarray([r["model_swaps"] for r in sub], dtype=np.float32)
        s = np.asarray([r["sabre_swaps"] for r in sub], dtype=np.float32)
        im = np.asarray([r["improvement_pct"] for r in sub], dtype=np.float32)
        w = np.asarray([r["win_vs_sabre"] for r in sub], dtype=np.float32)
        to = np.asarray([r["timeout"] for r in sub], dtype=np.float32)
        per_topology[topo] = {
            "cases": int(len(sub)),
            "mean_model_swaps": float(np.mean(m)),
            "mean_sabre_swaps": float(np.mean(s)),
            "mean_improvement_pct": float(np.mean(im)),
            "median_improvement_pct": float(np.median(im)),
            "win_rate_vs_sabre": float(np.mean(w)),
            "timeout_rate": float(np.mean(to)),
        }
    summary["per_topology"] = per_topology

    summary_path = output_dir / "supervised_rollout_eval_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Supervised rollout evaluation complete.")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

