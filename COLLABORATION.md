# Collaboration Guide: Who Does What?

To move fast, we are splitting the work into two tracks. We use the **Gymnasium** interface as our "handshake" to ensure our code connects perfectly.

### Track 1: Quantum environment (The "World")
- [cite_start]**Topology:** Build the IBM Falcon 27q graph and distance matrix[cite: 10].
- [cite_start]**Circuit Logic:** Use Qiskit to load QASMBench and find the "front layer" of gates[cite: 10].
- **Step Function:** Define how the mapping changes after a SWAP.
- [cite_start]**Done Criteria:** Signal when all gates are executed[cite: 10].

### Track 2: RL Agent (The "Brain")
- [cite_start]**The Model:** Set up the DQN or Maskable PPO architecture[cite: 12].
- [cite_start]**Action Masking:** Prevent the agent from choosing invalid SWAPs[cite: 10].
- [cite_start]**Reward Logic:** Implement the scoring system to drive learning[cite: 10].
- [cite_start]**Hyperparameters:** Tune the lookahead window ($K$) and learning rates[cite: 10].

### Sync Points
- We agree on the **State Shape** (how the mapping and gates are represented).
- We use **Stable Baselines3** to keep the training code standard.
