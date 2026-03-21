# RL-Based Quantum Circuit Routing

## The goal
Quantum computers can only run two-qubit gates between adjacent physical qubits. We use Deep Reinforcement Learning to insert SWAPs efficiently, targeting a **15-25% SWAP reduction** against IBM SABRE.

## Hardware and benchmarks
- **Training setup:** multi-topology training with a shared model (Heavy-Hex, grid, linear), padded to a fixed `27 x 27` input.
- **Evaluation data:** QASMBench circuits (for generalization checks).

## Environment highlights
- **State:** 3-channel `N x N` tensor (topology adjacency, current mapping, depth-decayed demand).
- **Actions:** SWAP-only on hardware edges; routable gates auto-execute.
- **Mapping strategy (training):** mixed initialization (80% random, 20% SABRE).
- **Masking strategy:** topology-validity mask only (no strategy masking).
- **Gate-demand decay:** `gamma_decay = 0.5`.

## How to run PPO training
1. Install dependencies:
   `pip install -r requirements.txt`
2. Run default training:
   `python main.py`
3. Example custom run:
   `python main.py --topologies heavy_hex_19,grid_3x3,linear_5 --total-timesteps 300000`
4. Recommended curriculum run:
   `python main.py --curriculum --topologies heavy_hex_19,grid_3x3,linear_5 --stage1-topologies linear_5 --stage2-topologies linear_5,grid_3x3 --stage1-steps 120000 --stage2-steps 80000 --stage3-steps 120000`

## Tracking and artifacts
- Each run writes to `runs/<run_name>/`.
- Single-stage mode writes in `runs/<run_name>/single_stage/`.
- Curriculum mode writes in `runs/<run_name>/stage1_easy/`, `stage2_mid/`, `stage3_full/`.
- `metrics.csv` stores training metrics by update, including periodic holdout evaluation vs SABRE (`eval_improvement_pct`, `eval_win_rate`, `eval_timeout_rate`).
- `best_model.pt`, `last_model.pt`, and periodic checkpoints are saved in each stage folder.
- `final_model.pt` is saved at `runs/<run_name>/final_model.pt`.
- `config.json` stores the exact run configuration.

## Colab notebook
- Use `notebooks/train_ppo_colab.ipynb` for Drive mount, GPU training, and metric plots.

## Final evaluation vs SABRE
Use:
`python benchmark.py --model-path <path_to_final_model.pt> --qasmbench-root <path_to_qasm_files>`
