"""Training configuration for D3QN+PER quantum circuit routing."""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class TrainConfig:
    """All hyperparameters for D3QN+PER training."""

    # --- Environment ---
    topologies: list = field(default_factory=lambda: ["heavy_hex_19"])
    matrix_size: int = 27
    circuit_depth: int = 20
    max_steps: int = 500
    gamma_decay: float = 0.5
    distance_reward_coeff: float = 0.01
    completion_bonus: float = 5.0
    timeout_penalty: float = -10.0
    initial_mapping_strategy: str = "random"

    # --- Network ---
    conv_channels: list = field(default_factory=lambda: [32, 64, 32])
    dueling_hidden: int = 256

    # --- DQN ---
    gamma: float = 0.99
    lr: float = 1e-4
    batch_size: int = 64
    target_update_freq: int = 1000
    tau: float = 1.0  # 1.0 = hard copy, <1.0 = soft Polyak
    grad_clip_norm: float = 10.0

    # --- Epsilon ---
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50000

    # --- PER ---
    buffer_capacity: int = 100000
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_beta_anneal_steps: int = 100000
    per_epsilon: float = 1e-6

    # --- Training loop ---
    total_episodes: int = 20000
    train_start: int = 1000
    train_freq: int = 1
    log_every: int = 50
    eval_every: int = 500
    checkpoint_every: int = 1000

    # --- Eval ---
    eval_episodes: int = 20
    eval_deterministic: bool = True

    # --- Output (unified: logs + checkpoints + figures + evals) ---
    output_base: str = "outputs"
    save_buffer: bool = False

    # --- Device / Seed ---
    device: str = "auto"
    seed: int = 42

    # --- Derived (set by setup_run_dir, not serialized from user) ---
    # These are filled in automatically — don't set manually.
    checkpoint_dir: str = ""
    log_dir: str = ""
    figures_dir: str = ""
    eval_dir: str = ""
    run_dir: str = ""

    def save(self, path: str):
        """Save config as JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TrainConfig":
        """Load config from JSON."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


def setup_run_dir(config: TrainConfig) -> TrainConfig:
    """
    Create a numbered run directory under config.output_base.

    Structure:
        outputs/
        ├── run_001/
        │   ├── config.json
        │   ├── logs/          (episodes.jsonl, train_steps.jsonl, evaluations.jsonl)
        │   ├── checkpoints/   (.pt files)
        │   ├── figures/       (training_curves.png, etc.)
        │   └── eval/          (random_eval.json, trajectories, etc.)
        ├── run_002/
        ...

    Returns the config with all directory paths filled in.
    """
    base = Path(config.output_base)
    base.mkdir(parents=True, exist_ok=True)

    # Find next run number
    existing = sorted(base.glob("run_*"))
    if existing:
        last_num = max(
            int(p.name.split("_")[1])
            for p in existing
            if p.name.split("_")[1].isdigit()
        )
        run_num = last_num + 1
    else:
        run_num = 1

    run_dir = base / f"run_{run_num:03d}"
    run_dir.mkdir()

    config.run_dir = str(run_dir)
    config.log_dir = str(run_dir / "logs")
    config.checkpoint_dir = str(run_dir / "checkpoints")
    config.figures_dir = str(run_dir / "figures")
    config.eval_dir = str(run_dir / "eval")

    # Create subdirs
    for d in [config.log_dir, config.checkpoint_dir,
              config.figures_dir, config.eval_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # Save config into the run dir
    config.save(str(run_dir / "config.json"))

    return config


def linear5_sanity_config() -> TrainConfig:
    """Quick sanity check on linear_5 (4 edges, simple circuits)."""
    return TrainConfig(
        topologies=["linear_5"],
        circuit_depth=5,
        max_steps=100,
        total_episodes=2000,
        buffer_capacity=20000,
        epsilon_decay_steps=10000,
        per_beta_anneal_steps=20000,
        train_start=500,
        eval_every=200,
        checkpoint_every=500,
        log_every=20,
        eval_episodes=10,
    )


def heavy_hex_config() -> TrainConfig:
    """Primary training on heavy_hex_19."""
    return TrainConfig(
        topologies=["heavy_hex_19"],
        circuit_depth=20,
        max_steps=300,
        total_episodes=20000,
        buffer_capacity=100000,
        epsilon_decay_steps=1000000,
        per_beta_anneal_steps=200000,
        train_start=1000,
        train_freq=4,
        log_every=25,
        eval_every=500,
        eval_episodes=50,
        checkpoint_every=1000,
    )


def multi_topology_config() -> TrainConfig:
    """Multi-topology training (novel contribution)."""
    return TrainConfig(
        topologies=["linear_5", "grid_3x3", "heavy_hex_19"],
        circuit_depth=20,
        max_steps=300,
        total_episodes=30000,
        buffer_capacity=150000,
        epsilon_decay_steps=1500000,
        per_beta_anneal_steps=300000,
        train_start=2000,
        train_freq=4,
        log_every=25,
        eval_every=1000,
        eval_episodes=50,
        checkpoint_every=2000,
    )
