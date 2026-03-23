"""
Supervised sanity check V2 (more rigorous):

- Teacher: SABRE step-by-step (replay SABRE SWAP sequence).
- Split: by circuits (train/val circuit IDs), not by shuffled states.
- Model: same symmetric CNN backbone as RL (edge-level logits).

This is a diagnostic to validate representation learnability, not a final
RL benchmark against SABRE.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from qiskit import transpile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from circuit_utils import generate_random_circuit, get_coupling_map  # noqa: E402
from dqn_agent import SymmetricCNNQNetwork  # noqa: E402
from environment import QubitRoutingEnv  # noqa: E402


@dataclass
class Sample:
    obs: np.ndarray
    edge_index: np.ndarray
    action_mask: np.ndarray
    label: int


@dataclass
class CircuitTrace:
    circuit_id: int
    seed: int
    sabre_swap_count: int
    collected_samples: int
    rollout_done: bool
    rollout_truncated: bool
    samples: List[Sample]


@dataclass
class SampleBatch:
    obs: np.ndarray
    edge_index: np.ndarray
    action_mask: np.ndarray
    labels: np.ndarray
    circuit_ids: np.ndarray


class SanityDataset(Dataset):
    def __init__(self, batch: SampleBatch):
        self.obs = torch.as_tensor(batch.obs, dtype=torch.float32)
        self.edge_index = torch.as_tensor(batch.edge_index, dtype=torch.long)
        self.action_mask = torch.as_tensor(batch.action_mask, dtype=torch.bool)
        self.labels = torch.as_tensor(batch.labels, dtype=torch.long)
        self.circuit_ids = torch.as_tensor(batch.circuit_ids, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):
        return (
            self.obs[idx],
            self.edge_index[idx],
            self.action_mask[idx],
            self.labels[idx],
            self.circuit_ids[idx],
        )


class SymmetricCNNClassifier(nn.Module):
    """Edge-action classifier using the same symmetry-aware CNN as DQN."""

    def __init__(self, matrix_size: int):
        super().__init__()
        self.net = SymmetricCNNQNetwork(matrix_size=matrix_size)

    def forward(
        self,
        obs: torch.Tensor,
        edge_index: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        q_map_sym = self.net.forward(obs)
        logits = SymmetricCNNQNetwork.gather_edge_q_values(q_map_sym, edge_index)
        return logits.masked_fill(~action_mask, -1e9)


def parse_args():
    parser = argparse.ArgumentParser(description="Supervised sanity check V2 (SABRE imitation).")
    parser.add_argument("--project-root", type=str, default=str(PROJECT_ROOT))
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--topology", type=str, default="linear_5")
    parser.add_argument("--matrix-size", type=int, default=27)
    parser.add_argument("--num-circuits", type=int, default=400)
    parser.add_argument("--target-samples", type=int, default=20000)
    parser.add_argument("--max-samples-per-circuit", type=int, default=400)
    parser.add_argument("--circuit-depth", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--min-two-qubit-gates", type=int, default=8)
    parser.add_argument("--circuit-generation-attempts", type=int, default=16)
    parser.add_argument("--initial-mapping-strategy", type=str, default="sabre")
    parser.add_argument("--train-split-circuits", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--sabre-optimization-level", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def get_sabre_mapping_and_swaps(
    circuit,
    coupling_map,
    optimization_level: int,
    seed: int,
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Run SABRE and extract:
    - initial logical->physical mapping
    - ordered list of SWAP physical edges
    """
    routed = transpile(
        circuit,
        coupling_map=coupling_map,
        routing_method="sabre",
        layout_method="sabre",
        optimization_level=int(optimization_level),
        seed_transpiler=int(seed),
    )

    layout_obj = getattr(routed, "layout", None)
    init_layout = getattr(layout_obj, "initial_layout", None)
    if init_layout is None:
        raise RuntimeError("SABRE initial layout not found on transpiled circuit.")

    mapping = [int(init_layout[circuit.qubits[i]]) for i in range(circuit.num_qubits)]

    swaps: List[Tuple[int, int]] = []
    for inst in routed.data:
        op = getattr(inst, "operation", None)
        qargs = getattr(inst, "qubits", ())
        if op is None:
            op = inst[0]
            qargs = inst[1]
        if getattr(op, "name", "") != "swap" or len(qargs) != 2:
            continue
        p1 = int(routed.find_bit(qargs[0]).index)
        p2 = int(routed.find_bit(qargs[1]).index)
        edge = (min(p1, p2), max(p1, p2))
        swaps.append(edge)

    return mapping, swaps


def collect_trace_for_circuit(
    env,
    circuit_id: int,
    circuit_seed: int,
    circuit,
    mapping: Sequence[int],
    sabre_swaps: Sequence[Tuple[int, int]],
    max_samples_per_circuit: int,
) -> CircuitTrace:
    max_actions = int(env.action_space.n)
    topo = env._topologies[0]
    edge_to_action = {tuple(edge): idx for idx, edge in enumerate(topo["edges"])}

    obs, _ = env.reset(
        options={
            "topology_index": 0,
            "circuit": circuit,
            "initial_mapping": [int(x) for x in mapping],
        }
    )

    samples: List[Sample] = []
    done = False
    truncated = False

    for edge in sabre_swaps:
        action = edge_to_action.get(tuple(edge), None)
        if action is None:
            # Should not happen on correct coupling map, but skip robustly.
            continue

        edge_index, action_mask = build_edge_index_and_mask(env, max_actions)
        if not action_mask[action]:
            # Skip invalid state/action pair (should be rare).
            continue

        samples.append(
            Sample(
                obs=obs.astype(np.float32, copy=False),
                edge_index=edge_index,
                action_mask=action_mask,
                label=int(action),
            )
        )
        if len(samples) >= max_samples_per_circuit:
            break

        obs, _, done, truncated, _ = env.step(int(action))
        if done or truncated:
            break

    return CircuitTrace(
        circuit_id=int(circuit_id),
        seed=int(circuit_seed),
        sabre_swap_count=int(len(sabre_swaps)),
        collected_samples=int(len(samples)),
        rollout_done=bool(done),
        rollout_truncated=bool(truncated),
        samples=samples,
    )


def flatten_circuit_traces(traces: Sequence[CircuitTrace]) -> SampleBatch:
    obs_list: List[np.ndarray] = []
    edge_list: List[np.ndarray] = []
    mask_list: List[np.ndarray] = []
    labels_list: List[int] = []
    circuit_ids: List[int] = []

    for tr in traces:
        for s in tr.samples:
            obs_list.append(s.obs)
            edge_list.append(s.edge_index)
            mask_list.append(s.action_mask)
            labels_list.append(int(s.label))
            circuit_ids.append(int(tr.circuit_id))

    if not labels_list:
        raise RuntimeError("No samples were collected from SABRE traces.")

    return SampleBatch(
        obs=np.asarray(obs_list, dtype=np.float32),
        edge_index=np.asarray(edge_list, dtype=np.int64),
        action_mask=np.asarray(mask_list, dtype=bool),
        labels=np.asarray(labels_list, dtype=np.int64),
        circuit_ids=np.asarray(circuit_ids, dtype=np.int64),
    )


def split_traces_by_circuit(
    traces: Sequence[CircuitTrace],
    train_split: float,
    seed: int,
) -> Tuple[List[CircuitTrace], List[CircuitTrace]]:
    if len(traces) < 2:
        raise RuntimeError("Need at least 2 usable circuits to split train/val by circuit.")
    idx = np.arange(len(traces))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_train = int(round(len(traces) * train_split))
    n_train = min(max(1, n_train), len(traces) - 1)
    train_idx = set(idx[:n_train].tolist())
    train = [traces[i] for i in range(len(traces)) if i in train_idx]
    val = [traces[i] for i in range(len(traces)) if i not in train_idx]
    return train, val


def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    kk = min(k, int(logits.shape[-1]))
    topk = torch.topk(logits, k=kk, dim=-1).indices
    ok = (topk == labels.unsqueeze(1)).any(dim=1)
    return float(ok.float().mean().item())


def evaluate_epoch(model, loader, device, criterion):
    model.eval()
    n = 0
    loss_sum = 0.0
    top1_sum = 0.0
    top3_sum = 0.0
    with torch.no_grad():
        for obs, edge_index, action_mask, labels, _cids in loader:
            obs = obs.to(device)
            edge_index = edge_index.to(device)
            action_mask = action_mask.to(device)
            labels = labels.to(device)
            logits = model(obs, edge_index, action_mask)
            loss = criterion(logits, labels)
            bs = labels.shape[0]
            n += bs
            loss_sum += float(loss.item()) * bs
            top1_sum += topk_accuracy(logits, labels, 1) * bs
            top3_sum += topk_accuracy(logits, labels, 3) * bs
    if n == 0:
        return {"loss": float("nan"), "top1": float("nan"), "top3": float("nan")}
    return {
        "loss": loss_sum / n,
        "top1": top1_sum / n,
        "top3": top3_sum / n,
    }


def train_model(
    model,
    train_loader,
    val_loader,
    device: str,
    epochs: int,
    lr: float,
    weight_decay: float,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: List[Dict[str, float]] = []
    best_state = None
    best_val_top1 = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_n = 0
        train_loss_sum = 0.0

        for obs, edge_index, action_mask, labels, _cids in train_loader:
            obs = obs.to(device)
            edge_index = edge_index.to(device)
            action_mask = action_mask.to(device)
            labels = labels.to(device)

            logits = model(obs, edge_index, action_mask)
            loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = labels.shape[0]
            train_n += bs
            train_loss_sum += float(loss.item()) * bs

        train_loss = train_loss_sum / max(1, train_n)
        val_metrics = evaluate_epoch(model, val_loader, device, criterion)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_top1": val_metrics["top1"],
            "val_top3": val_metrics["top3"],
        }
        history.append(row)
        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_top1={val_metrics['top1']:.4f} "
            f"val_top3={val_metrics['top3']:.4f}"
        )

        if val_metrics["top1"] > best_val_top1:
            best_val_top1 = float(val_metrics["top1"])
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    return history, best_state


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    rng = np.random.default_rng(args.seed)

    project_root = Path(args.project_root)
    run_name = args.run_name or f"sanity_v2_{int(time.time())}"
    run_dir = project_root / "results" / "sanity_checks_v2" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("Starting supervised sanity check V2:")
    print(f"  run_dir={run_dir}")
    print(f"  topology={args.topology}")
    print(f"  num_circuits={args.num_circuits}")
    print(f"  target_samples={args.target_samples}")
    print(f"  device={device}")

    cmap = get_coupling_map(args.topology)
    n_qubits = int(cmap.size())

    env = QubitRoutingEnv(
        topologies=[args.topology],
        matrix_size=args.matrix_size,
        circuit_depth=args.circuit_depth,
        max_steps=args.max_steps,
        min_two_qubit_gates=args.min_two_qubit_gates,
        circuit_generation_attempts=args.circuit_generation_attempts,
        initial_mapping_strategy=args.initial_mapping_strategy,
        seed=args.seed,
    )

    traces: List[CircuitTrace] = []
    skipped_no_swap = 0
    skipped_error = 0
    total_samples = 0

    t_collect_start = time.time()
    for circuit_id in range(int(args.num_circuits)):
        if total_samples >= int(args.target_samples):
            break
        cseed = int(rng.integers(0, 2**31))
        circuit = generate_random_circuit(
            num_qubits=n_qubits,
            depth=args.circuit_depth,
            seed=cseed,
            min_two_qubit_gates=args.min_two_qubit_gates,
            max_attempts=args.circuit_generation_attempts,
        )
        try:
            mapping, sabre_swaps = get_sabre_mapping_and_swaps(
                circuit=circuit,
                coupling_map=cmap,
                optimization_level=args.sabre_optimization_level,
                seed=cseed,
            )
        except Exception:
            skipped_error += 1
            continue

        if len(sabre_swaps) == 0:
            skipped_no_swap += 1
            continue

        trace = collect_trace_for_circuit(
            env=env,
            circuit_id=circuit_id,
            circuit_seed=cseed,
            circuit=circuit,
            mapping=mapping,
            sabre_swaps=sabre_swaps,
            max_samples_per_circuit=int(args.max_samples_per_circuit),
        )
        if trace.collected_samples <= 0:
            continue

        traces.append(trace)
        total_samples += trace.collected_samples

    collect_sec = time.time() - t_collect_start
    if len(traces) < 2:
        raise RuntimeError(
            f"Not enough usable circuits for split. usable={len(traces)}, "
            f"skipped_no_swap={skipped_no_swap}, skipped_error={skipped_error}"
        )

    train_traces, val_traces = split_traces_by_circuit(
        traces=traces,
        train_split=float(args.train_split_circuits),
        seed=int(args.seed),
    )

    train_batch = flatten_circuit_traces(train_traces)
    val_batch = flatten_circuit_traces(val_traces)

    train_ds = SanityDataset(train_batch)
    val_ds = SanityDataset(val_batch)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = SymmetricCNNClassifier(matrix_size=args.matrix_size).to(device)

    t_train_start = time.time()
    history, best_state = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=int(args.epochs),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    train_sec = time.time() - t_train_start

    if best_state is not None:
        model.load_state_dict(best_state)
    criterion = nn.CrossEntropyLoss()
    val_metrics = evaluate_epoch(model, val_loader, device, criterion)

    val_valid_counts = val_batch.action_mask.sum(axis=1).astype(np.float32)
    masked_random_top1 = float(np.mean(1.0 / np.maximum(1.0, val_valid_counts)))

    train_hist = np.bincount(train_batch.labels, minlength=train_batch.action_mask.shape[1])
    majority_label = int(np.argmax(train_hist))
    majority_top1 = float(np.mean(val_batch.labels == majority_label))

    best_model_path = run_dir / "best_model.pt"
    torch.save(model.state_dict(), best_model_path)

    history_path = run_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_top1", "val_top3"])
        writer.writeheader()
        writer.writerows(history)

    traces_meta_path = run_dir / "circuits_meta.csv"
    with traces_meta_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "circuit_id",
                "seed",
                "sabre_swap_count",
                "collected_samples",
                "rollout_done",
                "rollout_truncated",
                "split",
            ],
        )
        writer.writeheader()
        train_ids = {tr.circuit_id for tr in train_traces}
        for tr in traces:
            writer.writerow(
                {
                    "circuit_id": tr.circuit_id,
                    "seed": tr.seed,
                    "sabre_swap_count": tr.sabre_swap_count,
                    "collected_samples": tr.collected_samples,
                    "rollout_done": int(tr.rollout_done),
                    "rollout_truncated": int(tr.rollout_truncated),
                    "split": "train" if tr.circuit_id in train_ids else "val",
                }
            )

    summary = {
        "run_name": run_name,
        "device": device,
        "topology": args.topology,
        "matrix_size": int(args.matrix_size),
        "num_circuits_requested": int(args.num_circuits),
        "num_circuits_used": int(len(traces)),
        "num_circuits_train": int(len(train_traces)),
        "num_circuits_val": int(len(val_traces)),
        "samples_train": int(train_batch.labels.shape[0]),
        "samples_val": int(val_batch.labels.shape[0]),
        "collect_minutes": float(collect_sec / 60.0),
        "train_minutes": float(train_sec / 60.0),
        "val_top1": float(val_metrics["top1"]),
        "val_top3": float(val_metrics["top3"]),
        "val_loss": float(val_metrics["loss"]),
        "masked_random_baseline_top1": masked_random_top1,
        "majority_baseline_top1": majority_top1,
        "skipped_no_swap": int(skipped_no_swap),
        "skipped_error": int(skipped_error),
        "history_path": str(history_path),
        "best_model_path": str(best_model_path),
        "circuits_meta_path": str(traces_meta_path),
    }

    summary_path = run_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Sanity check V2 done.")
    print(f"  history: {history_path}")
    print(f"  summary: {summary_path}")
    print(f"  best_model: {best_model_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

