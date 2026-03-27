"""Trace one PPO greedy episode step-by-step on a chosen topology/circuit."""

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
        generate_random_circuit,
        get_coupling_map,
        get_sabre_initial_mapping,
        get_sabre_swap_count,
        load_circuit,
    )
    from .environment import QubitRoutingEnv
except ImportError:
    from agent import SymmetricCNNActorCritic
    from circuit_utils import (
        generate_random_circuit,
        get_coupling_map,
        get_sabre_initial_mapping,
        get_sabre_swap_count,
        load_circuit,
    )
    from environment import QubitRoutingEnv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Trace one episode with a trained PPO policy."
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--topology", type=str, default="heavy_hex_19")
    parser.add_argument("--matrix-size", type=int, default=27)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--device", type=str, default="auto")

    # Circuit source
    parser.add_argument(
        "--qasm-path",
        type=str,
        default="",
        help="If provided, load this QASM. Otherwise generate a random circuit.",
    )
    parser.add_argument("--circuit-depth", type=int, default=20)
    parser.add_argument(
        "--num-qubits",
        type=int,
        default=0,
        help="For random circuit only; 0 means topology qubit count.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-two-qubit-gates", type=int, default=8)
    parser.add_argument("--circuit-generation-attempts", type=int, default=16)

    # Mapping setup
    parser.add_argument(
        "--initial-mapping",
        type=str,
        default="sabre",
        choices=["sabre", "random", "identity", "mixed"],
    )

    # Environment reward params (for logged rewards)
    parser.add_argument("--distance-reward-coeff", type=float, default=0.015)
    parser.add_argument("--completion-bonus", type=float, default=15.0)
    parser.add_argument("--timeout-penalty", type=float, default=-8.0)
    parser.add_argument("--gate-reward-coeff", type=float, default=1.0)
    parser.add_argument("--step-penalty", type=float, default=-0.05)
    parser.add_argument("--reverse-swap-penalty", type=float, default=-0.2)

    # Outputs
    parser.add_argument("--output-csv", type=str, default="trace_episode.csv")
    parser.add_argument(
        "--summary-json", type=str, default="trace_episode_summary.json"
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=1,
        help="Print one line every N steps. 0 disables per-step printing.",
    )
    return parser.parse_args()


def build_edge_index_and_mask(env, max_actions: int):
    topo = env._current_topo
    edge_index = np.zeros((max_actions, 2), dtype=np.int64)
    num_edges = topo["num_edges"]
    if num_edges > 0:
        edge_index[:num_edges] = np.asarray(topo["edges"], dtype=np.int64)
    action_mask = env.get_action_mask().astype(bool)
    return edge_index, action_mask


def greedy_action_with_stats(model, obs, edge_index, action_mask, device):
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    edge_t = torch.as_tensor(edge_index, dtype=torch.long, device=device).unsqueeze(0)
    mask_t = torch.as_tensor(action_mask, dtype=torch.bool, device=device).unsqueeze(0)
    with torch.no_grad():
        dist, _ = model.get_action_distribution(obs_t, edge_t, mask_t)
        logits = dist.logits.squeeze(0)
        probs = torch.softmax(logits, dim=-1)
        action = int(torch.argmax(logits).item())
        action_prob = float(probs[action].item())
    return action, action_prob


def make_circuit(args, topology_qubits: int):
    if args.qasm_path:
        circuit = load_circuit(args.qasm_path)
        return circuit, f"qasm:{args.qasm_path}"

    num_qubits = topology_qubits if args.num_qubits <= 0 else args.num_qubits
    circuit = generate_random_circuit(
        num_qubits=num_qubits,
        depth=args.circuit_depth,
        seed=args.seed,
        min_two_qubit_gates=args.min_two_qubit_gates,
        max_attempts=args.circuit_generation_attempts,
    )
    return circuit, "random"


def make_initial_mapping(args, circuit, coupling_map, n_physical):
    if args.initial_mapping == "sabre":
        return [int(x) for x in get_sabre_initial_mapping(circuit, coupling_map)]
    if args.initial_mapping == "identity":
        return list(range(n_physical))
    if args.initial_mapping == "random":
        rng = np.random.default_rng(args.seed)
        return [int(x) for x in rng.permutation(n_physical)]
    return None  # mixed: let env sample


def main():
    args = parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device

    cmap = get_coupling_map(args.topology)
    n_physical = int(cmap.size())

    circuit, circuit_source = make_circuit(args, topology_qubits=n_physical)
    if circuit.num_qubits > n_physical:
        raise ValueError(
            f"Circuit has {circuit.num_qubits} qubits but topology '{args.topology}' has only {n_physical}."
        )

    env = QubitRoutingEnv(
        topologies=[args.topology],
        matrix_size=args.matrix_size,
        max_steps=args.max_steps,
        gamma_decay=0.5,
        distance_reward_coeff=args.distance_reward_coeff,
        completion_bonus=args.completion_bonus,
        timeout_penalty=args.timeout_penalty,
        gate_reward_coeff=args.gate_reward_coeff,
        step_penalty=args.step_penalty,
        reverse_swap_penalty=args.reverse_swap_penalty,
        min_two_qubit_gates=args.min_two_qubit_gates,
        circuit_generation_attempts=args.circuit_generation_attempts,
        initial_mapping_strategy=args.initial_mapping,
        seed=args.seed,
    )

    model = SymmetricCNNActorCritic(matrix_size=env.N).to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    init_mapping = make_initial_mapping(
        args=args,
        circuit=circuit,
        coupling_map=cmap,
        n_physical=n_physical,
    )
    reset_options: Dict = {"topology_index": 0, "circuit": circuit}
    if init_mapping is not None:
        reset_options["initial_mapping"] = init_mapping

    obs, info = env.reset(seed=args.seed, options=reset_options)
    max_actions = int(env.action_space.n)

    rows: List[Dict] = []
    done = False
    truncated = False
    step_reward_sum = 0.0
    prev_total_exec = int(info.get("total_gates_executed", 0))
    last_action = None

    while not done and not truncated:
        edge_index, action_mask = build_edge_index_and_mask(env, max_actions)
        action, action_prob = greedy_action_with_stats(
            model=model,
            obs=obs,
            edge_index=edge_index,
            action_mask=action_mask,
            device=device,
        )
        topo = env._current_topo
        edge_i, edge_j = topo["edges"][action]
        front_dist_before = float(env._compute_front_layer_distance())

        next_obs, reward, done, truncated, info = env.step(action)
        front_dist_after = float(env._compute_front_layer_distance())
        delta_dist = front_dist_before - front_dist_after
        step_gates_executed = int(info["total_gates_executed"]) - prev_total_exec
        prev_total_exec = int(info["total_gates_executed"])
        was_immediate_backtrack = int(last_action is not None and action == last_action)
        last_action = action
        step_reward_sum += float(reward)

        row = {
            "step": int(info["step_count"]),
            "action_index": int(action),
            "edge_i": int(edge_i),
            "edge_j": int(edge_j),
            "action_prob": action_prob,
            "reward": float(reward),
            "step_gates_executed": step_gates_executed,
            "total_gates_executed": int(info["total_gates_executed"]),
            "remaining_gates": int(info["remaining_gates"]),
            "front_dist_before": front_dist_before,
            "front_dist_after": front_dist_after,
            "delta_dist": delta_dist,
            "was_immediate_backtrack": was_immediate_backtrack,
            "done": int(done),
            "truncated": int(truncated),
        }
        rows.append(row)

        if args.print_every > 0 and (row["step"] % args.print_every == 0):
            print(
                f"[Step {row['step']:03d}] action={row['action_index']:03d} "
                f"edge=({row['edge_i']},{row['edge_j']}) "
                f"p={row['action_prob']:.3f} reward={row['reward']:+.3f} "
                f"exec_step={row['step_gates_executed']} rem={row['remaining_gates']} "
                f"delta_dist={row['delta_dist']:+.2f} "
                f"backtrack={row['was_immediate_backtrack']} "
                f"done={row['done']} trunc={row['truncated']}"
            )

        obs = next_obs

    sabre_swaps = int(get_sabre_swap_count(circuit, cmap))
    ppo_swaps = int(info.get("total_swaps", 0))
    improvement_pct = 100.0 * (sabre_swaps - ppo_swaps) / max(1, sabre_swaps)
    backtrack_rate = (
        float(np.mean([r["was_immediate_backtrack"] for r in rows]))
        if rows
        else 0.0
    )

    summary = {
        "topology": args.topology,
        "circuit_source": circuit_source,
        "qasm_path": args.qasm_path,
        "circuit_qubits": int(circuit.num_qubits),
        "circuit_depth": int(circuit.depth()),
        "initial_mapping_mode": args.initial_mapping,
        "steps": int(info.get("step_count", 0)),
        "episode_return": float(step_reward_sum),
        "done": bool(done),
        "truncated": bool(truncated),
        "ppo_swaps": ppo_swaps,
        "sabre_swaps": sabre_swaps,
        "improvement_pct_vs_sabre": float(improvement_pct),
        "remaining_gates_end": int(info.get("remaining_gates", 0)),
        "total_gates_executed_end": int(info.get("total_gates_executed", 0)),
        "backtrack_rate": backtrack_rate,
    }

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nTrace finished.")
    print(f"Trace CSV: {output_csv}")
    print(f"Summary JSON: {summary_json}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

