"""
Supervised sanity check for the routing state representation.

Goal:
- Build a dataset of (state, teacher_action) samples from the routing env.
- Train the same symmetric CNN backbone used by RL to predict teacher actions.
- Report top-1 / top-3 accuracy vs masked-random baseline.

This is a representation diagnostic:
if the model cannot learn this supervised proxy, RL is unlikely to work.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dqn_agent import SymmetricCNNQNetwork  # noqa: E402
from environment import QubitRoutingEnv  # noqa: E402


@dataclass
class SampleBatch:
    obs: np.ndarray
    edge_index: np.ndarray
    action_mask: np.ndarray
    labels: np.ndarray


class SanityDataset(Dataset):
    def __init__(self, batch: SampleBatch):
        self.obs = torch.as_tensor(batch.obs, dtype=torch.float32)
        self.edge_index = torch.as_tensor(batch.edge_index, dtype=torch.long)
        self.action_mask = torch.as_tensor(batch.action_mask, dtype=torch.bool)
        self.labels = torch.as_tensor(batch.labels, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):
        return (
            self.obs[idx],
            self.edge_index[idx],
            self.action_mask[idx],
            self.labels[idx],
        )


class SymmetricCNNClassifier(nn.Module):
    """
    Classification wrapper around the DQN symmetric score map.

    Produces edge-level logits (masked on invalid actions).
    """

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
    parser = argparse.ArgumentParser(description="Run supervised sanity check.")
    parser.add_argument("--project-root", type=str, default=str(PROJECT_ROOT))
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--topologies", type=str, default="linear_5")
    parser.add_argument("--matrix-size", type=int, default=27)
    parser.add_argument("--circuit-depth", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--min-two-qubit-gates", type=int, default=2)
    parser.add_argument("--circuit-generation-attempts", type=int, default=16)
    parser.add_argument("--initial-mapping-strategy", type=str, default="mixed")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument(
        "--teacher-rollout-prob",
        type=float,
        default=0.85,
        help="Probability of following teacher action while rolling out states.",
    )
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
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


def snapshot_env_state(env) -> Dict:
    return {
        "mapping": list(env.mapping),
        "reverse_mapping": list(env.reverse_mapping),
        "executed": set(env.executed),
        "step_count": int(env.step_count),
        "total_swaps": int(env.total_swaps),
        "total_gates_executed": int(env.total_gates_executed),
        "_last_action": None if env._last_action is None else int(env._last_action),
        "_last_edge": None if env._last_edge is None else tuple(env._last_edge),
        "_same_edge_streak": int(env._same_edge_streak),
        "_no_progress_streak": int(env._no_progress_streak),
    }


def restore_env_state(env, snap: Dict):
    env.mapping = list(snap["mapping"])
    env.reverse_mapping = list(snap["reverse_mapping"])
    env.executed = set(snap["executed"])
    env.step_count = int(snap["step_count"])
    env.total_swaps = int(snap["total_swaps"])
    env.total_gates_executed = int(snap["total_gates_executed"])
    env._last_action = snap["_last_action"]
    env._last_edge = snap["_last_edge"]
    env._same_edge_streak = int(snap["_same_edge_streak"])
    env._no_progress_streak = int(snap["_no_progress_streak"])


def pick_teacher_action(env, action_mask: np.ndarray) -> int:
    """
    One-step lookahead teacher:
    prioritize terminal/progressing actions, then lower front-layer distance.
    """
    valid_actions = np.flatnonzero(action_mask)
    if valid_actions.size == 0:
        return 0

    base = snapshot_env_state(env)
    dist_before = float(env._compute_front_layer_distance())
    exec_before = int(env.total_gates_executed)

    best_action = int(valid_actions[0])
    best_score = None

    for action in valid_actions:
        restore_env_state(env, base)
        _, reward, done, truncated, _ = env.step(int(action))
        exec_after = int(env.total_gates_executed)
        dist_after = float(env._compute_front_layer_distance())
        delta_dist = dist_before - dist_after
        gates_exec = exec_after - exec_before
        # Lexicographic score:
        # 1) solve episode, 2) execute gates now, 3) reduce front distance,
        # 4) avoid truncation, 5) immediate reward tie-break.
        score = (
            int(done),
            int(gates_exec),
            float(delta_dist),
            -int(truncated),
            float(reward),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_action = int(action)

    restore_env_state(env, base)
    return best_action


def collect_dataset(env, samples: int, teacher_rollout_prob: float, seed: int) -> SampleBatch:
    rng = np.random.default_rng(seed)
    max_actions = int(env.action_space.n)
    obs_list: List[np.ndarray] = []
    edge_list: List[np.ndarray] = []
    mask_list: List[np.ndarray] = []
    label_list: List[int] = []

    obs, _ = env.reset()
    while len(label_list) < samples:
        edge_index, action_mask = build_edge_index_and_mask(env, max_actions)
        teacher_action = pick_teacher_action(env, action_mask)

        obs_list.append(obs.astype(np.float32, copy=False))
        edge_list.append(edge_index)
        mask_list.append(action_mask)
        label_list.append(int(teacher_action))

        valid_actions = np.flatnonzero(action_mask)
        if valid_actions.size == 0:
            obs, _ = env.reset()
            continue

        if rng.random() < teacher_rollout_prob:
            rollout_action = int(teacher_action)
        else:
            rollout_action = int(rng.choice(valid_actions))

        obs, _, done, truncated, _ = env.step(rollout_action)
        if done or truncated:
            obs, _ = env.reset()

    return SampleBatch(
        obs=np.asarray(obs_list, dtype=np.float32),
        edge_index=np.asarray(edge_list, dtype=np.int64),
        action_mask=np.asarray(mask_list, dtype=bool),
        labels=np.asarray(label_list, dtype=np.int64),
    )


def split_indices(n: int, train_split: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(round(n * train_split))
    n_train = min(max(1, n_train), n - 1)
    return idx[:n_train], idx[n_train:]


def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    kk = min(k, int(logits.shape[-1]))
    topk = torch.topk(logits, k=kk, dim=-1).indices
    ok = (topk == labels.unsqueeze(1)).any(dim=1)
    return float(ok.float().mean().item())


def evaluate_epoch(model, loader, device, criterion):
    model.eval()
    loss_sum = 0.0
    n = 0
    top1_sum = 0.0
    top3_sum = 0.0
    with torch.no_grad():
        for obs, edge_index, action_mask, labels in loader:
            obs = obs.to(device)
            edge_index = edge_index.to(device)
            action_mask = action_mask.to(device)
            labels = labels.to(device)

            logits = model(obs, edge_index, action_mask)
            loss = criterion(logits, labels)
            bs = labels.shape[0]
            loss_sum += float(loss.item()) * bs
            top1_sum += topk_accuracy(logits, labels, k=1) * bs
            top3_sum += topk_accuracy(logits, labels, k=3) * bs
            n += bs

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
    device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    history: List[Dict[str, float]] = []
    best_state = None
    best_val_top1 = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        for obs, edge_index, action_mask, labels in train_loader:
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
            train_loss_sum += float(loss.item()) * bs
            train_n += bs

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
    topologies = [t.strip() for t in args.topologies.split(",") if t.strip()]
    if not topologies:
        raise ValueError("No topology provided.")

    project_root = Path(args.project_root)
    results_root = project_root / "results" / "sanity_checks"
    results_root.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or f"sanity_{int(time.time())}"
    run_dir = results_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("Starting supervised sanity check:")
    print(f"  run_dir={run_dir}")
    print(f"  device={device}")
    print(f"  topologies={topologies}")
    print(f"  samples={args.samples}")
    print(f"  epochs={args.epochs}")
    print(f"  teacher_rollout_prob={args.teacher_rollout_prob}")

    env = QubitRoutingEnv(
        topologies=topologies,
        matrix_size=args.matrix_size,
        circuit_depth=args.circuit_depth,
        max_steps=args.max_steps,
        min_two_qubit_gates=args.min_two_qubit_gates,
        circuit_generation_attempts=args.circuit_generation_attempts,
        initial_mapping_strategy=args.initial_mapping_strategy,
        seed=args.seed,
    )

    t0 = time.time()
    batch = collect_dataset(
        env=env,
        samples=int(args.samples),
        teacher_rollout_prob=float(args.teacher_rollout_prob),
        seed=int(args.seed),
    )
    collect_sec = time.time() - t0
    print(f"Dataset collected in {collect_sec/60.0:.2f} min.")

    n_total = int(batch.labels.shape[0])
    train_idx, val_idx = split_indices(n_total, args.train_split, args.seed)

    train_batch = SampleBatch(
        obs=batch.obs[train_idx],
        edge_index=batch.edge_index[train_idx],
        action_mask=batch.action_mask[train_idx],
        labels=batch.labels[train_idx],
    )
    val_batch = SampleBatch(
        obs=batch.obs[val_idx],
        edge_index=batch.edge_index[val_idx],
        action_mask=batch.action_mask[val_idx],
        labels=batch.labels[val_idx],
    )

    train_ds = SanityDataset(train_batch)
    val_ds = SanityDataset(val_batch)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = SymmetricCNNClassifier(matrix_size=args.matrix_size).to(device)
    t1 = time.time()
    history, best_state = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_sec = time.time() - t1

    if best_state is not None:
        model.load_state_dict(best_state)

    criterion = nn.CrossEntropyLoss()
    val_final = evaluate_epoch(model, val_loader, device, criterion)

    valid_counts = batch.action_mask.sum(axis=1).astype(np.float32)
    random_baseline = float(np.mean(1.0 / np.maximum(1.0, valid_counts)))
    labels = batch.labels
    label_hist = np.bincount(labels, minlength=batch.action_mask.shape[1]).astype(np.int64)
    dominant_ratio = float(label_hist.max() / max(1, n_total))

    best_model_path = run_dir / "best_model.pt"
    torch.save(model.state_dict(), best_model_path)

    history_path = run_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_top1", "val_top3"])
        writer.writeheader()
        writer.writerows(history)

    summary = {
        "run_name": run_name,
        "device": device,
        "topologies": topologies,
        "matrix_size": int(args.matrix_size),
        "samples_total": n_total,
        "samples_train": int(len(train_idx)),
        "samples_val": int(len(val_idx)),
        "collect_minutes": float(collect_sec / 60.0),
        "train_minutes": float(train_sec / 60.0),
        "val_top1": float(val_final["top1"]),
        "val_top3": float(val_final["top3"]),
        "val_loss": float(val_final["loss"]),
        "masked_random_baseline_top1": random_baseline,
        "label_dominant_ratio": dominant_ratio,
        "valid_actions_mean": float(np.mean(valid_counts)),
        "valid_actions_min": float(np.min(valid_counts)),
        "valid_actions_max": float(np.max(valid_counts)),
        "teacher_rollout_prob": float(args.teacher_rollout_prob),
        "initial_mapping_strategy": args.initial_mapping_strategy,
        "min_two_qubit_gates": int(args.min_two_qubit_gates),
        "circuit_depth": int(args.circuit_depth),
        "max_steps": int(args.max_steps),
        "seed": int(args.seed),
        "best_model_path": str(best_model_path),
        "history_path": str(history_path),
    }

    summary_path = run_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Supervised sanity check done.")
    print(f"  history: {history_path}")
    print(f"  summary: {summary_path}")
    print(f"  best_model: {best_model_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

