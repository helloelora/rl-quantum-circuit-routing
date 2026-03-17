# RL-Based quantum circuit routing

## The qoal
[cite_start]Quantum computers can only run gates between adjacent qubits[cite: 3]. We use Deep Reinforcement Learning to move qubits (via SWAPs) as efficiently as possible. [cite_start]Our target is a **15-25% reduction in SWAP gates** compared to IBM's SABRE compiler[cite: 8].

## Hardware & benchmarks
- [cite_start]**Target Topology:** IBM Falcon 27q (Heavy-Hex layout)[cite: 10, 11].
- [cite_start]**Data:** QASMBench (Grover, QFT, QAOA, VQE)[cite: 11].

## The environment (MDP)
- [cite_start]**State:** Current mapping ($\pi$), front layer of gates, lookahead window ($K$), and the hardware distance matrix[cite: 10].
- [cite_start]**Actions:** Discrete SWAPs on hardware edges or "Execute" ready gates[cite: 10].
- [cite_start]**Rewards:** -1 per SWAP, +0.1 per executed gate, +5 on completion, -0.01 per step[cite: 10].

## How to run
1. Install dependencies: `pip install -r requirements.txt`
2. Train the agent: `python main.py --train`
3. Benchmark vs SABRE: `python benchmark.py`
