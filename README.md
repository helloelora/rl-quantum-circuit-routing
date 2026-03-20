# RL-Based quantum circuit routing

## The goal
Quantum computers can only run gates between adjacent qubits. We use Deep Reinforcement Learning to move qubits (via SWAPs) as efficiently as possible. Our target is a **15-25% reduction in SWAP gates** compared to IBM's SABRE compiler.

## Hardware & benchmarks
- **Target Topology:** 19-qubit Heavy-Hex (`CouplingMap.from_heavy_hex(3)`).
- **Data:** QASMBench (Grover, QFT, QAOA, VQE).

## The environment (MDP)
- **State:** 3-channel N×N matrix — adjacency (topology), qubit assignment (current mapping), gate demand (front layer interactions).
- **Actions:** Discrete SWAPs on hardware edges. Routable gates execute automatically after each SWAP.
- **Rewards:** -1 per SWAP, +1 per gate auto-executed, +0.01 × distance reduction, +5 on completion.

## How to run
1. Install dependencies: `pip install -r requirements.txt`
2. Train the agent: `python main.py --train`
3. Benchmark vs SABRE: `python benchmark.py`
