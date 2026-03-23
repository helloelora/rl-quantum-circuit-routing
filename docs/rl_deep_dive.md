# Reinforcement Learning Deep Dive: From Foundations to DQN Mastery

A ground-up guide for someone who knows ML but wants to truly understand RL — the math, the intuition, and the "why" behind every design choice.

---

## Part 1: The RL Problem — What Are We Actually Doing?

### 1.1 Supervised Learning vs RL

In supervised learning, you have:
- A dataset of (input, correct_answer) pairs
- You minimize a loss function comparing your prediction to the correct answer
- The "correct answer" is given to you

In RL, everything is different:
- There is **no dataset** — the agent generates its own data by interacting with an environment
- There is **no correct answer** — only a reward signal that tells you "that was good" or "that was bad"
- Actions have **consequences** — what you do now changes what you see next
- Rewards are **delayed** — the best action now might look bad immediately but pay off 50 steps later

This is fundamentally harder. You're learning while simultaneously exploring, and the data you learn from depends on your own decisions.

### 1.2 The MDP Framework

Every RL problem is formalized as a **Markov Decision Process (MDP)**:

- **S**: Set of states (everything the agent can observe)
- **A**: Set of actions (everything the agent can do)
- **P(s'|s, a)**: Transition function — probability of reaching state s' after taking action a in state s
- **R(s, a, s')**: Reward function — immediate reward for the transition
- **gamma (γ)**: Discount factor (0 < γ ≤ 1) — how much to value future rewards vs immediate rewards

The **Markov property** says: the future depends only on the current state, not on how you got there. This is crucial — it means state s contains all the information needed to make optimal decisions.

**In our quantum routing problem:**
- S = (3, 27, 27) matrix showing topology + current mapping + gate demands
- A = {0, 1, ..., 19} — which edge to SWAP
- P is deterministic — same state + same action = same next state
- R = gates_executed - 1 + distance_shaping + completion_bonus
- γ = 0.99

### 1.3 The Goal: Find the Optimal Policy

A **policy** π(a|s) tells the agent what to do in each state. It maps states to actions (or to probability distributions over actions).

The goal is to find the policy π* that maximizes the **expected cumulative discounted reward**:

```
G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + γ³·r_{t+3} + ...
    = Σ_{k=0}^{∞} γ^k · r_{t+k}
```

This is called the **return**. The discount factor γ controls the trade-off:
- γ = 0: Only care about immediate reward (greedy)
- γ = 1: Care equally about all future rewards (far-sighted)
- γ = 0.99: Care about future but gradually less — a reward 100 steps away is worth 0.99^100 ≈ 0.37 of an immediate reward

**Why discount?** Three reasons:
1. **Mathematical**: Ensures the sum converges (infinite series of bounded rewards)
2. **Practical**: Future is uncertain — a reward now is more reliable than a promised reward later
3. **Computational**: Without discounting, all policies that eventually finish look equally good. Discounting prefers policies that finish sooner.

---

## Part 2: Value Functions — The Core Idea

### 2.1 What Is a Value Function?

The central idea in RL: instead of directly learning what to do, learn **how good** each state (or state-action pair) is. Then just pick the action that leads to the best state.

**State value function V^π(s)**: "How much total reward can I expect from state s if I follow policy π?"

```
V^π(s) = E_π[G_t | s_t = s]
       = E_π[r_t + γ·r_{t+1} + γ²·r_{t+2} + ... | s_t = s]
```

**Action value function Q^π(s, a)**: "How much total reward can I expect if I'm in state s, take action a, then follow policy π?"

```
Q^π(s, a) = E_π[G_t | s_t = s, a_t = a]
           = E_π[r_t + γ·r_{t+1} + γ²·r_{t+2} + ... | s_t = s, a_t = a]
```

The relationship:
```
V^π(s) = Σ_a π(a|s) · Q^π(s, a)
```
(The value of a state = average over Q-values of all actions, weighted by how likely you are to take each action.)

### 2.2 The Bellman Equation — The Magic Recursion

Here's the key insight. The return G_t has a recursive structure:

```
G_t = r_t + γ · G_{t+1}
```

Total reward from now = immediate reward + discounted total reward from next step.

This gives us the **Bellman equation**:

```
Q^π(s, a) = E[r + γ · Q^π(s', π(s'))]
```

In words: the Q-value of taking action a in state s equals the immediate reward plus the discounted Q-value of the next state under the current policy.

**Why is this so powerful?** Because it turns a problem about infinite sums into a one-step relationship. Instead of estimating returns over entire episodes, we can learn Q-values by bootstrapping — using our current Q-estimates to improve themselves.

### 2.3 The Optimal Bellman Equation

The **optimal** Q-function Q* satisfies:

```
Q*(s, a) = E[r + γ · max_{a'} Q*(s', a')]
```

If we knew Q*, the optimal policy is trivial: in every state, just pick the action with the highest Q-value.

```
π*(s) = argmax_a Q*(s, a)
```

This is the entire foundation of DQN: approximate Q* with a neural network, then act greedily.

### 2.4 Why Q Over V?

You might wonder: why learn Q(s,a) instead of just V(s)?

With V(s), to pick the best action you need to:
1. For each possible action a, simulate what state s' you'd end up in
2. Look up V(s') for each
3. Pick the action leading to the highest V(s')

Step 1 requires knowing the transition function P(s'|s,a) — i.e., you need a model of the environment.

With Q(s,a), you just:
1. Compute Q(s,a) for each action
2. Pick argmax

No environment model needed. This is why Q-learning is called **model-free**.

---

## Part 3: Temporal Difference Learning — Learning Without Waiting

### 3.1 The Problem with Monte Carlo

The simplest way to learn V(s): play entire episodes, record the actual return G_t from each state visited, then average them.

```
V(s) ← V(s) + α · (G_t - V(s))
```

This is **Monte Carlo (MC)** learning. Problem: you have to wait until the episode ends to compute G_t. For our quantum routing problem, that could be 500 steps. And the variance is huge — one lucky episode doesn't mean the state is good.

### 3.2 TD(0) — The Breakthrough

**Temporal Difference (TD)** learning uses the Bellman equation to learn after every single step:

```
V(s) ← V(s) + α · (r + γ·V(s') - V(s))
                    ├───────────┤
                    │  TD target │
                    ├────────────────────┤
                    │       TD error      │
```

Instead of waiting for the full return G_t, we use `r + γ·V(s')` as an estimate. This is **bootstrapping** — using our current value estimate to update itself.

**The TD error** `δ = r + γ·V(s') - V(s)` is the surprise:
- δ > 0: "This was better than I expected" → increase V(s)
- δ < 0: "This was worse than I expected" → decrease V(s)
- δ = 0: "Exactly as expected" → no update

**Why TD is better than MC for our problem:**
- Updates after every step (don't wait for episode end)
- Lower variance (single step r + γ·V(s') vs full trajectory)
- Works for continuing tasks (no episode boundary needed)
- The cost: slight bias (V(s') might be wrong), but this bias shrinks as V improves

### 3.3 Q-Learning — TD for Q-Values (Off-Policy)

Apply TD to Q-values:

```
Q(s, a) ← Q(s, a) + α · (r + γ · max_{a'} Q(s', a') - Q(s, a))
```

This is **Q-learning** (Watkins, 1989). The critical insight: we use `max` over next actions, regardless of what action we actually took next. This makes Q-learning **off-policy** — it learns about the optimal policy while following a different (exploratory) policy.

This is huge: the agent can explore randomly but still learn what the best action is.

### 3.4 SARSA — The On-Policy Alternative

```
Q(s, a) ← Q(s, a) + α · (r + γ · Q(s', a') - Q(s, a))
```

SARSA uses the action a' that was actually taken (not the max). It learns the value of the policy it's actually following (on-policy). Name comes from (S, A, R, S', A') — the five things you need for one update.

**Q-learning vs SARSA:**
- Q-learning: learns Q* regardless of exploration policy. More aggressive.
- SARSA: learns Q^π for the current policy. More conservative, safer.
- For our problem: Q-learning is better — we want to find the optimal policy, not evaluate our exploratory policy.

---

## Part 4: Deep Q-Networks (DQN)

### 4.1 The Tabular Problem

Classic Q-learning stores Q(s,a) in a table — one entry per (state, action) pair. Our state space is a 3×27×27 continuous matrix. A table is impossible.

**Solution**: Approximate Q(s,a) with a neural network Q(s, a; θ) where θ are the network weights.

But this is where things get tricky — naively combining neural networks with Q-learning is unstable and often diverges.

### 4.2 Why Naive Deep Q-Learning Fails

Two fundamental problems:

**Problem 1: Correlated samples.** In supervised learning, we assume training samples are independent (i.i.d.). In RL, consecutive experiences (s_t, a_t, r_t, s_{t+1}) are highly correlated — state 5 looks a lot like state 6. Training a neural network on correlated sequential data causes it to overfit to recent experience and forget earlier lessons.

**Problem 2: Moving target.** The TD target `r + γ · max Q(s', a'; θ)` depends on the same network θ we're updating. Every time we improve θ, the target changes. It's like trying to hit a moving target — the network chases its own tail and oscillates or diverges.

### 4.3 DQN: Two Innovations That Fix Everything

**Innovation 1: Experience Replay Buffer**

Store every transition (s, a, r, s', done) in a large buffer (e.g., 100,000 transitions). When training, sample a random mini-batch from the buffer.

```python
# Collecting experience
buffer.store(state, action, reward, next_state, done)

# Training (sample random batch)
batch = buffer.sample(batch_size=64)
```

This breaks the correlation between consecutive samples. A batch might contain a transition from step 10, step 5000, and step 200 — they're independent. This is also incredibly **sample-efficient** — each experience can be replayed many times.

**Why "replay"?** Think of it like studying for an exam. Instead of only looking at today's lecture notes (sequential), you review random flashcards from the entire semester (replay). You learn patterns instead of memorizing sequences.

**Innovation 2: Target Network**

Keep two copies of the Q-network:
- **Online network** Q(s, a; θ) — updated every step
- **Target network** Q(s, a; θ⁻) — frozen copy, updated every C steps (e.g., C = 1000)

```python
# TD target uses the FROZEN target network
target = r + γ · max_{a'} Q(s', a'; θ⁻)    # θ⁻ is frozen

# Loss uses the ONLINE network
loss = (Q(s, a; θ) - target)²               # θ is updated

# Every C steps, sync
θ⁻ ← θ
```

The target is now stable for C steps. The network can converge toward a fixed target instead of chasing itself.

**Analogy**: Imagine calibrating an instrument. If the reference standard keeps changing while you calibrate, you'll never converge. You need to freeze the reference, calibrate, then update the reference. That's exactly what the target network does.

### 4.4 The Complete DQN Algorithm

```
Initialize replay buffer D (capacity N)
Initialize Q-network with random weights θ
Initialize target network with weights θ⁻ = θ

For each episode:
    Get initial state s from environment

    For each step:
        // ACTION SELECTION (epsilon-greedy)
        With probability ε: pick random action a
        Otherwise: a = argmax_a Q(s, a; θ)

        // ENVIRONMENT INTERACTION
        Execute a, observe reward r, next state s', done flag
        Store (s, a, r, s', done) in buffer D

        // LEARNING (if buffer has enough samples)
        Sample random mini-batch of (s_j, a_j, r_j, s'_j, done_j) from D

        For each sample j:
            If done_j:
                target_j = r_j
            Else:
                target_j = r_j + γ · max_{a'} Q(s'_j, a'; θ⁻)

        Loss = (1/batch) · Σ_j (Q(s_j, a_j; θ) - target_j)²
        Update θ by gradient descent on Loss

        // TARGET NETWORK UPDATE
        Every C steps: θ⁻ ← θ

        // EXPLORATION DECAY
        ε ← max(ε_min, ε - decay_rate)

        s ← s'
```

### 4.5 Epsilon-Greedy Exploration

The agent needs to **explore** (try random actions to discover what's good) and **exploit** (use what it's learned to maximize reward). Epsilon-greedy is the simplest strategy:

```
With probability ε: random action     (explore)
With probability 1-ε: argmax Q(s,a)   (exploit)
```

During training, ε decays from 1.0 (pure exploration) to 0.05 (mostly exploitation):
- Early training: ε ≈ 1.0, almost all random — the agent knows nothing, so it explores
- Late training: ε ≈ 0.05, mostly greedy — the agent knows a lot, but still occasionally tries new things

**Why not ε = 0 at the end?** Because the environment changes between episodes (different circuits). A small amount of exploration prevents the agent from getting stuck in suboptimal habits.

### 4.6 The Loss Function — What DQN Actually Optimizes

```
L(θ) = E[(target - Q(s, a; θ))²]

where target = r + γ · max_{a'} Q(s', a'; θ⁻)
```

This is just **mean squared error** between the predicted Q-value and the TD target. Standard gradient descent:

```
θ ← θ - α · ∇_θ L(θ)
  = θ - α · ∇_θ (target - Q(s, a; θ))²
  = θ + α · (target - Q(s, a; θ)) · ∇_θ Q(s, a; θ)
                 │                       │
                 TD error                gradient of network output
```

Note: we do NOT take the gradient through the target — the target is treated as a fixed number. This is why the target network is important — if we took gradients through it, we'd be optimizing a different objective.

---

## Part 5: Double DQN — Fixing the Overestimation Problem

### 5.1 The Problem: Overestimation Bias

Vanilla DQN uses `max_{a'} Q(s', a'; θ⁻)` as the target. Here's the problem:

The `max` operator is **positively biased** when applied to noisy estimates.

Imagine you have 20 actions and the true Q-value of each is 0. But your estimates have noise: Q-estimates are [-0.3, 0.5, -0.1, 0.4, ...]. The max of these is 0.5, which is above the true max of 0.

```
E[max(Q₁, Q₂, ..., Q_n)] ≥ max(E[Q₁], E[Q₂], ..., E[Q_n])
```

This is Jensen's inequality. The more actions you have and the noisier your estimates, the worse the overestimation.

**Why this is harmful**: The agent thinks some states are much better than they are. It develops overly optimistic Q-values that compound through bootstrapping. State A looks amazing because it leads to state B which has overestimated Q-values. Training becomes unstable.

### 5.2 The Fix: Decouple Selection from Evaluation

**Double DQN** (van Hasselt, 2015): Use the online network to SELECT the best action, but the target network to EVALUATE it.

```python
# Vanilla DQN target:
best_action = argmax_a Q(s', a; θ⁻)         # target network picks
target = r + γ · Q(s', best_action; θ⁻)      # target network evaluates
# Same network does both → overestimation

# Double DQN target:
best_action = argmax_a Q(s', a; θ)           # ONLINE network picks
target = r + γ · Q(s', best_action; θ⁻)      # TARGET network evaluates
# Different networks → decorrelated errors → less overestimation
```

**Intuition**: If the online network's noise causes it to overvalue action 7, the target network has different noise and will give action 7 a more realistic value. The two networks' errors are independent, so they don't compound.

**Implementation**: Literally one line changes in the target computation. The rest of DQN stays identical.

```python
# Vanilla DQN
target = reward + gamma * target_net(next_state).max(dim=1).values

# Double DQN
best_actions = online_net(next_state).argmax(dim=1)
target = reward + gamma * target_net(next_state).gather(1, best_actions.unsqueeze(1)).squeeze()
```

---

## Part 6: Dueling DQN — Separating "Where" from "What"

### 6.1 The Key Insight

In many states, the **action doesn't matter much**. Think about our routing problem: if all front-layer gate qubits are 5 hops from their targets, no single SWAP is going to execute anything. The state is just bad, regardless of which edge you swap.

In other states, the action matters a lot: a specific SWAP will execute 3 gates while all others execute 0.

Vanilla DQN learns Q(s, a) for every (state, action) pair independently. If a state is bad for all actions, DQN has to learn that separately for each action. That's wasteful.

### 6.2 The Dueling Architecture

**Dueling DQN** (Wang et al., 2016) decomposes Q(s, a) into two pieces:

```
Q(s, a) = V(s) + A(s, a)
```

- **V(s)** = State value: "How good is this state, regardless of what I do?"
- **A(s, a)** = Advantage: "How much better is action a compared to the average action in this state?"

The network splits into two heads after the shared CNN:

```
                Input: (3, 27, 27)
                       │
                  ┌────┴────┐
                  │   CNN    │  ← shared feature extraction
                  │  layers  │
                  └────┬────┘
                       │
              ┌────────┴────────┐
              │                 │
         ┌────┴────┐      ┌────┴────┐
         │ V head  │      │ A head  │
         │ Linear  │      │ Linear  │
         │ → 1     │      │ → 20    │
         └────┬────┘      └────┬────┘
              │                │
              └────────┬───────┘
                       │
              Q(s,a) = V(s) + A(s,a) - mean(A)
```

### 6.3 Why Subtract the Mean?

If we just used `Q = V + A`, the decomposition is not unique. You could shift any constant from V to A and get the same Q. For example:
- V=10, A=[2, -1, -1] → Q=[12, 9, 9]
- V=12, A=[0, -3, -3] → Q=[12, 9, 9]   (same Q, different V and A!)

To make V and A identifiable, we force A to have zero mean:

```
Q(s, a) = V(s) + (A(s, a) - mean_a[A(s, a)])
```

Now V(s) truly represents the average value of the state, and A(s,a) truly represents the advantage of each action relative to average.

### 6.4 Why This Helps Our Problem

In quantum circuit routing:
- Many states are "stuck" — all front-layer qubits are far from targets. V(s) learns "this is a bad state" once, not 20 times (once per action).
- When a state IS interesting (one SWAP could execute gates), the A(s,a) head focuses its capacity on distinguishing the good SWAP from the bad ones.
- The V head also helps the target computation: since V is shared, good value estimates propagate to all actions in that state simultaneously.

**Result**: Faster learning, especially in states where many actions are equivalent.

---

## Part 7: Prioritized Experience Replay — Learning from Surprises

### 7.1 The Problem with Uniform Replay

Vanilla DQN samples uniformly from the replay buffer. But not all experiences are equally useful:

- **Boring transition**: SWAP edge 5, nothing happens, reward = -1. The agent already predicted Q ≈ -1. TD error ≈ 0. Learning nothing.
- **Surprising transition**: SWAP edge 12, 3 gates execute, reward = +2. The agent predicted Q ≈ -1. TD error ≈ 3. Lots to learn!

With uniform sampling, the agent replays boring transitions 95% of the time. The rare, informative transitions get drowned out.

### 7.2 The Solution: Priority = |TD Error|

**Prioritized Experience Replay (PER)** (Schaul et al., 2015): Sample transitions with probability proportional to their TD error.

```
priority_i = |δ_i| + ε

where δ_i = r + γ · max Q(s', a') - Q(s, a)    (TD error)
      ε = small constant (e.g., 0.01) to ensure all transitions have nonzero priority
```

Sampling probability:

```
P(i) = priority_i^α / Σ_k priority_k^α

where α controls how much prioritization matters:
  α = 0: uniform sampling (no prioritization)
  α = 1: fully proportional to priority
  typical α = 0.6
```

### 7.3 The Importance Sampling Correction

Prioritized sampling introduces **bias**. We're no longer sampling uniformly, so the expected gradient is wrong — we're overweighting surprising transitions.

Fix: multiply each sample's loss by an **importance sampling weight**:

```
w_i = (1 / (N · P(i)))^β

where N = buffer size
      β = annealing parameter (starts at ~0.4, increases to 1.0 over training)
```

- β = 1: fully corrects the bias (unbiased gradient)
- β < 1: partial correction (allows some bias for faster learning early on)

We anneal β from 0.4 → 1.0 during training: early on, we accept bias for faster learning. Late in training, we want unbiased gradients for convergence.

```python
# The full PER loss computation
weights = (1.0 / (buffer_size * priorities)) ** beta
weights = weights / weights.max()  # normalize to [0, 1] for stability

loss = (weights * (target - Q(s, a)) ** 2).mean()
```

### 7.4 Implementation: Sum Tree

Efficiently sampling from a weighted distribution requires a **sum tree** — a binary tree where each leaf stores a priority and each internal node stores the sum of its children.

```
         42          ← root = sum of all priorities
        /  \
      29    13
     / \   / \
   12  17  6  7     ← leaf priorities (one per transition)
```

To sample proportionally:
1. Pick a random number u ∈ [0, 42)
2. Walk down the tree: if u < left child sum, go left; otherwise subtract left sum and go right
3. Reach a leaf = your sampled transition

This gives O(log N) sampling instead of O(N). For a buffer of 100,000, that's ~17 operations instead of 100,000.

### 7.5 Why This Matters for Quantum Routing

In early training:
- 95% of SWAPs do nothing (reward = -1, predicted ≈ -1, TD error ≈ 0)
- 5% of SWAPs execute gates (reward > 0, predicted ≈ -1, TD error large)

Without PER: the agent almost never replays the useful 5%.
With PER: the agent focuses on those critical transitions, learning much faster when SWAPs are actually helpful.

---

## Part 8: Putting It All Together — Double Dueling DQN with PER

### 8.1 The Complete Architecture

```
Input state: (3, 27, 27)
       │
       ▼
  ┌─────────────────────────────┐
  │  Conv2d(3→32, 3×3, pad=1)  │
  │  ReLU                       │
  │  Conv2d(32→64, 3×3, pad=1) │
  │  ReLU                       │
  │  Conv2d(64→32, 3×3, pad=1) │
  │  ReLU                       │
  └──────────┬──────────────────┘
             │
        Flatten (32 × 27 × 27 = 23,328)
             │
      ┌──────┴──────┐
      ▼              ▼
  ┌───────┐    ┌───────────┐
  │V head │    │  A head   │
  │Lin→256│    │ Lin→256   │
  │ ReLU  │    │  ReLU     │
  │Lin→1  │    │ Lin→20    │ (max_edges)
  └───┬───┘    └─────┬─────┘
      │              │
      └──────┬───────┘
             │
   Q(s,a) = V + (A - mean(A))
             │
   Apply action mask: Q[invalid] = -inf
             │
   Action = argmax Q (or random with prob ε)
```

### 8.2 The Complete Training Loop

```
Initialize:
    Online network Q(s, a; θ)
    Target network Q(s, a; θ⁻) = copy of θ
    Prioritized replay buffer D (capacity 100,000)
    ε = 1.0 (will decay to 0.05)
    β = 0.4 (will anneal to 1.0)

For each episode:
    state, info = env.reset()

    While not done:
        // 1. SELECT ACTION
        mask = env.get_action_mask()
        if random() < ε:
            action = random choice from valid actions (where mask=True)
        else:
            q_values = online_net(state)
            q_values[~mask] = -inf
            action = argmax(q_values)

        // 2. STEP ENVIRONMENT
        next_state, reward, done, truncated, info = env.step(action)

        // 3. STORE IN BUFFER (with max priority for new transitions)
        buffer.add(state, action, reward, next_state, done, priority=max_priority)

        // 4. LEARN (if buffer has enough samples)
        if len(buffer) > batch_size:
            // Sample prioritized batch
            indices, batch, weights = buffer.sample(batch_size, beta=β)

            states, actions, rewards, next_states, dones = batch

            // DOUBLE DQN target
            with no_grad():
                best_actions = online_net(next_states).argmax(dim=1)    // online selects
                next_q = target_net(next_states)
                next_q = next_q.gather(1, best_actions.unsqueeze(1)).squeeze()
                targets = rewards + γ * next_q * (1 - dones)

            // Current Q-values (DUELING architecture computes V + A - mean(A) internally)
            current_q = online_net(states).gather(1, actions.unsqueeze(1)).squeeze()

            // TD errors (for priority update)
            td_errors = (targets - current_q).abs().detach()
            buffer.update_priorities(indices, td_errors)

            // Weighted loss (IMPORTANCE SAMPLING correction)
            loss = (weights * (targets - current_q) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            // Gradient clipping (stabilizes training)
            clip_grad_norm_(online_net.parameters(), max_norm=10.0)
            optimizer.step()

        // 5. UPDATE TARGET NETWORK (every C steps)
        if step_count % C == 0:
            target_net.load_state_dict(online_net.state_dict())

        // 6. DECAY EXPLORATION
        ε = max(0.05, ε - decay_rate)
        β = min(1.0, β + beta_anneal_rate)

        state = next_state
```

### 8.3 How All the Pieces Connect

```
┌────────────────────────────────────────────────────────────┐
│                     TRAINING LOOP                          │
│                                                            │
│  ┌──────────┐  action   ┌──────────────┐  (s,a,r,s',d)   │
│  │  Agent   │ ────────→ │ Environment  │ ──────────────┐  │
│  │          │ ←──────── │              │               │  │
│  └──────────┘  state    └──────────────┘               │  │
│       │                                                │  │
│       │ Q-values                                       │  │
│       │                                                ▼  │
│  ┌────┴─────────────────┐              ┌──────────────────┐│
│  │   DUELING NETWORK    │              │  PRIORITIZED     ││
│  │                      │  sample      │  REPLAY BUFFER   ││
│  │  Online net (θ)      │◄─────────────│                  ││
│  │  Target net (θ⁻)     │              │  Sum tree for    ││
│  │                      │  update      │  fast sampling   ││
│  │  V(s) + A(s,a)       │  priorities  │                  ││
│  │  - mean(A)           │─────────────→│  IS weights for  ││
│  │                      │              │  bias correction  ││
│  └──────────────────────┘              └──────────────────┘│
│                                                            │
│  DOUBLE DQN: online picks action, target evaluates it      │
│  Target net syncs every C steps                            │
│  ε decays 1.0 → 0.05 (exploration → exploitation)         │
│  β anneals 0.4 → 1.0 (bias correction)                    │
└────────────────────────────────────────────────────────────┘
```

---

## Part 9: PPO — The Policy Gradient Alternative (For Reference)

Since Elora is implementing PPO, here's a brief overview so you understand how it differs.

### 9.1 Fundamental Difference: Value vs Policy Methods

**DQN** (value-based): Learn Q(s,a), then derive the policy as argmax Q.
**PPO** (policy-based): Directly learn the policy π(a|s) — a probability distribution over actions.

### 9.2 Why Policy Gradients Exist

Value methods have limitations:
- They output a deterministic policy (argmax). What if the optimal policy is stochastic?
- Action masking is a hack (set Q = -inf). Policy methods naturally assign zero probability.
- For continuous action spaces, you can't argmax over infinite actions (not our problem, but important generally).

Policy gradient methods directly optimize the policy by computing:

```
∇_θ J(θ) = E[∇_θ log π(a|s; θ) · A(s, a)]
```

Where A(s,a) is the **advantage** — how much better action a is compared to average. This is the **policy gradient theorem**: to improve the policy, increase the probability of actions with positive advantage, decrease for negative advantage.

### 9.3 PPO's Key Innovation: Clipped Surrogate Objective

Vanilla policy gradient is unstable — a large gradient step can destroy the policy. PPO prevents this with clipping:

```
L(θ) = min(ratio · A, clip(ratio, 1-ε, 1+ε) · A)

where ratio = π_new(a|s) / π_old(a|s)
      ε = 0.2 (clip range)
```

If the policy changes too much (ratio far from 1), the objective is clipped — the gradient is zeroed out. This prevents catastrophic updates.

### 9.4 DQN vs PPO Summary

| Aspect | DQN | PPO |
|--------|-----|-----|
| Learns | Q(s,a) values | π(a\|s) probabilities |
| Policy type | Deterministic (argmax) | Stochastic (sample) |
| Data usage | Off-policy (replay buffer) | On-policy (use once, discard) |
| Sample efficiency | Better (replay) | Worse (no replay) |
| Stability | Can be unstable (moving targets) | More stable (clipped updates) |
| Exploration | ε-greedy (crude) | Entropy bonus (smoother) |
| Action masking | Set Q = -inf | Set log prob = -inf |
| Credit assignment | Bootstrapping | GAE (Generalized Advantage Estimation) |

Both are valid for our problem. The comparison between your DQN and Elora's PPO is itself a valuable result for the project report.

---

## Part 10: Key Hyperparameters and What They Do

### 10.1 Hyperparameter Reference

| Parameter | Typical Value | What It Controls | Too Low | Too High |
|-----------|:------------:|------------------|---------|----------|
| **Learning rate (α)** | 1e-4 | Step size for gradient descent | Learns too slowly | Overshoots, unstable |
| **Discount (γ)** | 0.99 | How much to value future rewards | Myopic, greedy | Slow convergence, high variance |
| **Batch size** | 64 | Samples per gradient update | Noisy gradients | Slow updates, less exploration |
| **Buffer size** | 100,000 | Replay buffer capacity | Forgets old experience | Memory-heavy, stale data |
| **Target update (C)** | 1,000 | Steps between target network syncs | Moving target (unstable) | Stale targets (slow learning) |
| **ε start** | 1.0 | Initial exploration rate | Not enough exploration | — |
| **ε end** | 0.05 | Final exploration rate | — | Too much random behavior |
| **ε decay steps** | 50,000 | How fast exploration decays | Exploits too early (stuck in local optima) | Wastes time exploring |
| **PER α** | 0.6 | How much to prioritize surprising transitions | Near-uniform (wastes time on boring data) | Overfocuses on outliers |
| **PER β start** | 0.4 | Initial importance sampling correction | More bias (but faster early learning) | Slower early learning |
| **PER β end** | 1.0 | Final IS correction | Biased gradients (won't converge) | — |
| **Gradient clip** | 10.0 | Max gradient norm | — | Exploding gradients |

### 10.2 Debugging Your DQN — What to Watch

**Training metrics to log every episode:**
- Episode reward (should trend upward)
- Episode length / number of SWAPs (should trend downward)
- Gates executed (should reach n_gates consistently)
- Average Q-value (should increase, but watch for explosion)
- Average TD error (should decrease over time)
- ε value (should follow your decay schedule)

**Red flags:**
- Q-values exploding (> 100) → reduce learning rate, check reward scale
- Q-values collapsing to same value for all actions → network not learning, check gradients
- Episode reward stuck → ε too low (not exploring), or learning rate too high (overshooting)
- Loss not decreasing → learning rate too low, or target update too frequent (C too small)

### 10.3 Training Strategy for Our Problem

**Phase 1: Sanity check on linear_5**
- 5 qubits, 4 edges, simple circuits
- Should learn within ~5,000 episodes
- Verify: reward increases, SWAP count decreases, episodes complete

**Phase 2: Scale to heavy_hex_19**
- 19 qubits, 20 edges, deeper circuits
- Will need ~50,000-100,000 episodes
- May need to adjust learning rate and exploration schedule

**Phase 3: Multi-topology training**
- Mix of topologies
- Longer training, but tests generalization

**Phase 4: Compare against SABRE**
- Run trained agent on QASMBench circuits
- Compare SWAP counts

---

## Part 11: Mathematical Appendix

### 11.1 The Bellman Optimality Equation (Full Derivation)

Starting from the definition of Q*:

```
Q*(s, a) = E[G_t | s_t = s, a_t = a, following π*]
         = E[r_t + γ·G_{t+1} | s_t = s, a_t = a, following π*]
         = E[r_t + γ · max_{a'} Q*(s_{t+1}, a') | s_t = s, a_t = a]
         = Σ_{s'} P(s'|s,a) · [R(s,a,s') + γ · max_{a'} Q*(s', a')]
```

For our deterministic environment, P(s'|s,a) = 1 for exactly one s', so:

```
Q*(s, a) = R(s, a) + γ · max_{a'} Q*(s', a')
```

This is the equation DQN tries to satisfy.

### 11.2 Why the Max Operator Overestimates (Proof)

Let X₁, ..., X_n be random variables with E[X_i] = μ for all i.

```
E[max(X₁, ..., X_n)] ≥ max(E[X₁], ..., E[X_n]) = μ
```

Proof: max(X₁, ..., X_n) ≥ X_i for all i, so E[max] ≥ E[X_i] = μ for all i, therefore E[max] ≥ max(μ, ..., μ) = μ.

The bias grows with:
- n (more actions → more chances for noise to produce a high value)
- Var(X_i) (noisier estimates → more overestimation)

In DQN with 20 actions and noisy Q-estimates, this bias is substantial.

### 11.3 Importance Sampling Derivation

We want to compute E_p[f(x)] but sample from q(x):

```
E_p[f(x)] = Σ_x p(x) · f(x)
           = Σ_x q(x) · [p(x)/q(x)] · f(x)
           = E_q[w(x) · f(x)]

where w(x) = p(x) / q(x) are the importance sampling weights
```

In PER:
- p = uniform distribution (what we want)
- q = prioritized distribution (what we sample from)
- w_i = (1/N) / P(i) = 1/(N · P(i))

Raising to power β ∈ [0,1] partially corrects: w_i^β. At β=1, fully corrected. At β=0, no correction.

### 11.4 TD Error and Its Relationship to the Bellman Equation

If Q* satisfies the Bellman equation perfectly:

```
Q*(s, a) = r + γ · max Q*(s', a')
```

Then the TD error would be:

```
δ = r + γ · max Q*(s', a') - Q*(s, a) = 0
```

So TD error = 0 means the Bellman equation is satisfied. The TD error measures how far our Q-network is from satisfying the Bellman equation. Minimizing TD error ≈ finding Q*.

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────┐
│                    DQN FAMILY                       │
│                                                     │
│  Vanilla DQN                                        │
│    target = r + γ · max_a' Q_target(s', a')         │
│    ✗ Overestimates Q-values                         │
│                                                     │
│  + Double DQN                                       │
│    target = r + γ · Q_target(s', argmax Q_online)   │
│    ✓ Fixes overestimation                           │
│                                                     │
│  + Dueling DQN                                      │
│    Q(s,a) = V(s) + A(s,a) - mean(A)                │
│    ✓ Efficient when many actions are similar        │
│                                                     │
│  + Prioritized Replay                               │
│    P(i) ∝ |TD_error|^α                              │
│    ✓ Focuses on surprising transitions              │
│    ✓ Importance sampling weights correct bias       │
│                                                     │
│  = D3QN+PER (our implementation)                    │
│    All three improvements combined                  │
└─────────────────────────────────────────────────────┘
```
