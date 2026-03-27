# Quantum Circuit Routing — Experiment Runs

## Overview

D3QN+PER agent trained to minimize SWAP gates on quantum hardware topologies.
Baseline: IBM SABRE compiler. Target: 15–25% fewer SWAPs.

---

## Run 1 — Heavy-Hex Baseline

| Field | Value |
|-------|-------|
| **Job ID** | 162244 |
| **Config** | `configs/run1_heavy_hex_fixed.json` |
| **Topology** | heavy_hex_19 (19 qubits, 20 edges) |
| **Episodes** | 30,000 |
| **LR** | 1e-4 |
| **Batch** | 128 |
| **Epsilon** | 1.0 → 0.10 over 3M steps |
| **Seed** | 42 |
| **Node** | sh00 |
| **Logs** | `outputs/slurm_162244.out` / `.err` |
| **Run dir** | `outputs/run_003/` |

**Purpose**: Baseline heavy_hex_19 training with default hyperparameters.

**Results**: _(pending)_

| Metric | Value |
|--------|-------|
| Final reward | |
| Completion rate | |
| Agent SWAPs (mean) | |
| SABRE SWAPs (mean) | |
| Swap ratio | |

---

## Run 2 — High Learning Rate

| Field | Value |
|-------|-------|
| **Job ID** | 162245 |
| **Config** | `configs/run2_heavy_hex_highLR.json` |
| **Topology** | heavy_hex_19 |
| **Episodes** | 30,000 |
| **LR** | **3e-4** (3× baseline) |
| **Batch** | 128 |
| **Epsilon** | 1.0 → 0.10 over 3M steps |
| **Seed** | 123 |
| **Node** | sh21 |
| **Logs** | `outputs/slurm_162245.out` / `.err` |
| **Run dir** | `outputs/run_004/` |

**Purpose**: Test whether higher LR speeds convergence or causes instability.

**What to watch**: Compare loss/Q-value curves vs Run 1. If loss spikes or Q diverges, LR is too high.

**Results**: _(pending)_

| Metric | Value |
|--------|-------|
| Final reward | |
| Completion rate | |
| Agent SWAPs (mean) | |
| SABRE SWAPs (mean) | |
| Swap ratio | |

---

## Run 3 — Long Training + Low Final Epsilon

| Field | Value |
|-------|-------|
| **Job ID** | 162246 |
| **Config** | `configs/run3_heavy_hex_long.json` |
| **Topology** | heavy_hex_19 |
| **Episodes** | **40,000** |
| **LR** | 1e-4 |
| **Batch** | 128 |
| **Epsilon** | 1.0 → **0.02** over 3M steps |
| **Seed** | 42 |
| **Node** | sh22 |
| **Logs** | `outputs/slurm_162246.out` / `.err` |
| **Run dir** | `outputs/run_005/` |

**Purpose**: More training + nearly greedy final policy (epsilon 0.02 vs 0.10). Tests whether Run 1 is undertrained.

**What to watch**: Does performance keep improving past 30k episodes? Does lower final epsilon help or hurt?

**Results**: _(pending)_

| Metric | Value |
|--------|-------|
| Final reward | |
| Completion rate | |
| Agent SWAPs (mean) | |
| SABRE SWAPs (mean) | |
| Swap ratio | |

---

## Run 4 — Multi-Topology Generalization

| Field | Value |
|-------|-------|
| **Job ID** | 162247 |
| **Config** | `configs/run4_multi_fixed.json` |
| **Topology** | **linear_5 + grid_3x3 + heavy_hex_19** |
| **Episodes** | **45,000** |
| **LR** | 1e-4 |
| **Batch** | 128 |
| **Epsilon** | 1.0 → 0.10 over 4.5M steps |
| **Buffer** | 300k (larger for multi-topology) |
| **Seed** | 42 |
| **Node** | _(pending resources)_ |
| **Logs** | `outputs/slurm_162247.out` / `.err` |
| **Run dir** | `outputs/run_006/` |

**Purpose**: Train a single agent on 3 topologies simultaneously. Tests generalization — can one model learn routing across different hardware graphs?

**What to watch**: Per-topology completion rates. Does training on easier topologies (linear_5) help or hurt heavy_hex performance?

**Results**: _(pending)_

| Metric | linear_5 | grid_3x3 | heavy_hex_19 |
|--------|----------|----------|--------------|
| Completion rate | | | |
| Agent SWAPs (mean) | | | |
| SABRE SWAPs (mean) | | | |
| Swap ratio | | | |

---

## Monitoring Commands

```bash
# Quick status for all runs
for f in outputs/slurm_16224{4,5,6,7}.out; do echo "=== $(basename $f) ===" && tail -n 1 "$f" 2>/dev/null || echo "not started"; done

# Job queue
squeue -u dor_ali | grep quantum

# Training curves (once run finishes)
python3 main.py visualize --run-dir outputs/run_NNN

# Full evaluation logs
cat outputs/run_NNN/logs/evaluations.jsonl | python3 -m json.tool
```

---

## Summary Table

| Run | Config | Topology | Episodes | LR | Key Change | Status | Swap Ratio |
|-----|--------|----------|----------|----|------------|--------|------------|
| 1 | run1_heavy_hex_fixed | heavy_hex_19 | 30k | 1e-4 | Baseline | Running | |
| 2 | run2_heavy_hex_highLR | heavy_hex_19 | 30k | 3e-4 | Higher LR | Running | |
| 3 | run3_heavy_hex_long | heavy_hex_19 | 40k | 1e-4 | Longer + low ε | Running | |
| 4 | run4_multi_fixed | 3 topologies | 45k | 1e-4 | Multi-topology | Pending | |
