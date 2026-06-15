# 🐦 Flappy Bird — Deep Q-Network (DQN)

A Deep Reinforcement Learning agent trained to play **Flappy Bird** using the **Deep Q-Network (DQN)** algorithm. Built with PyTorch and the `flappy-bird-gymnasium` environment.

---

## 📁 Project Structure

```
flappy_bird_DQN/
├── agent.py              # Core training & inference loop
├── dqn.py                # Neural network (policy & target)
├── experience_replay.py  # Replay memory buffer
├── parameters.yaml       # Hyperparameter configuration
└── runs/                 # Saved models & training logs
```

---

## 🧠 Theory & Algorithm

### 1. Reinforcement Learning Fundamentals

Reinforcement Learning (RL) frames the problem as an **agent** interacting with an **environment**:

- At each timestep $t$, the agent observes a **state** $s_t$
- It selects an **action** $a_t$ according to a **policy** $\pi$
- The environment transitions to $s_{t+1}$ and returns a **reward** $r_t$

The agent's goal is to maximise the expected **cumulative discounted return**:

$$G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$$

where $\gamma \in [0, 1)$ is the **discount factor** that controls how much future rewards are valued relative to immediate ones.

---

### 2. Q-Learning

**Q-Learning** is a model-free, off-policy RL algorithm. It learns the **action-value function** $Q(s, a)$ — the expected return from taking action $a$ in state $s$ and then following the optimal policy:

$$Q^*(s, a) = \mathbb{E}\left[r + \gamma \max_{a'} Q^*(s', a') \mid s, a\right]$$

This is the **Bellman Optimality Equation**. Q-Learning performs iterative updates toward this target:

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right]$$

where $\alpha$ is the **learning rate**.

---

### 3. Deep Q-Network (DQN)

Classic Q-Learning uses a table to store $Q(s, a)$ values, which becomes infeasible for large or continuous state spaces. **DQN** (Mnih et al., 2015) replaces the Q-table with a **neural network** $Q(s, a; \theta)$ parameterised by weights $\theta$.

The network takes the full state vector as input and outputs a Q-value for each possible action simultaneously.

#### Loss Function

The network is trained by minimising the **Mean Squared Error (MSE)** loss between the predicted Q-value and a **target Q-value**:

$$\mathcal{L}(\theta) = \mathbb{E}\left[\left(y_t - Q(s_t, a_t; \theta)\right)^2\right]$$

where the target $y_t$ is:

$$y_t = \begin{cases} r_t & \text{if } s_{t+1} \text{ is terminal} \\ r_t + \gamma \max_{a'} Q(s_{t+1}, a'; \theta^{-}) & \text{otherwise} \end{cases}$$

---

### 4. Key DQN Innovations

#### 4.1 Experience Replay

Naive online training (updating after every single step) causes two major problems:

1. **Correlated samples** — consecutive steps are highly similar, making gradient updates noisy and unstable.
2. **Catastrophic forgetting** — the network quickly forgets rare but important experiences.

**Experience Replay** solves this by storing all transitions $(s_t, a_t, s_{t+1}, r_t, \text{done})$ in a fixed-size circular **Replay Buffer** (FIFO queue). During training, a random **mini-batch** of transitions is sampled uniformly, breaking the temporal correlation between updates.

```
Replay Buffer  →  [random mini-batch sample]  →  gradient update
```

In this project:
- Buffer size: **100,000** transitions
- Mini-batch size: **32** transitions

#### 4.2 Target Network

A fundamental instability in DQN training comes from the fact that both the **predicted Q-values** and the **target Q-values** are computed by the same network $\theta$. As the network updates, the target values shift simultaneously, making optimisation chase a moving target — analogous to a dog chasing its own tail.

The **Target Network** ($\theta^-$) is a periodically frozen copy of the policy network $\theta$. It is used **only** to compute target Q-values, and is updated every $N$ steps by hard-copying the policy network weights:

$$\theta^{-} \leftarrow \theta \quad \text{every } N \text{ steps}$$

This decouples the target from the prediction, significantly stabilising training.

In this project:
- Sync rate: every **10** steps

---

### 5. Exploration vs. Exploitation: ε-Greedy Policy

The agent faces the **exploration-exploitation dilemma**:
- **Exploration**: Try random actions to discover better strategies.
- **Exploitation**: Use the current best-known policy to maximise reward.

The **ε-greedy** strategy balances this:

$$a_t = \begin{cases} \text{random action} & \text{with probability } \varepsilon \\ \arg\max_a Q(s_t, a; \theta) & \text{with probability } 1 - \varepsilon \end{cases}$$

Epsilon **decays** over training — starting high (full exploration) and gradually settling to a minimum (mostly exploitation):

$$\varepsilon_t = \max(\varepsilon_{\min},\; \varepsilon_{t-1} \times \varepsilon_{\text{decay}})$$

| Parameter | Value |
|---|---|
| `epsilon_init` | `1.0` (100% random) |
| `epsilon_min` | `0.05` (5% random) |
| `epsilon_decay_rate` | `0.9995` per episode |

---

### 6. The Neural Network Architecture

The policy network (`DQN` class in `dqn.py`) is a simple **Multi-Layer Perceptron (MLP)**:

```
Input Layer  →  [state_dim = 12 features]
Hidden Layer →  [256 neurons, ReLU activation]
Output Layer →  [action_dim = 2 Q-values: flap / no-flap]
```

The **ReLU** (Rectified Linear Unit) activation introduces non-linearity:

$$\text{ReLU}(x) = \max(0, x)$$

The output is a Q-value for each action. The action with the **highest Q-value** is selected greedily.

---

### 7. State Space (Observations)

The `FlappyBird-v0` environment from `flappy-bird-gymnasium` provides a **12-dimensional** observation vector including:

- Last pipe's horizontal position and top/bottom Y coordinates
- Next pipe's horizontal position and top/bottom Y coordinates
- Next-next pipe's horizontal position and top/bottom Y coordinates
- Player's vertical position
- Player's vertical velocity
- Player's rotation

---

### 8. Action Space

The agent has **2 discrete actions**:

| Action | Description |
|--------|-------------|
| `0` | Do nothing (fall due to gravity) |
| `1` | Flap wings (jump upward) |

---

### 9. Training Loop (Step-by-Step)

```
For each episode:
  1. Reset environment → get initial state s
  2. For each timestep:
       a. Choose action a via ε-greedy policy
       b. Execute a → observe (s', r, done)
       c. Store (s, a, s', r, done) in Replay Buffer
       d. If buffer has >= mini_batch_size transitions:
            → Sample random mini-batch
            → Compute target Q-values using Target Network
            → Compute predicted Q-values using Policy Network
            → Calculate MSE loss
            → Backpropagate & update Policy Network weights
       e. Every N steps: sync Target Network <- Policy Network
       f. Update s <- s'
       g. If done or reward > threshold: end episode
  3. Decay epsilon
  4. If episode reward > best_reward:
       → Save model checkpoint
       → Log best reward
```

---

### 10. Hyperparameters

| Parameter | Value | Role |
|---|---|---|
| `alpha` (learning rate) | `0.001` | Step size for Adam optimiser |
| `gamma` (discount) | `0.99` | Weight of future rewards |
| `epsilon_init` | `1.0` | Initial exploration rate |
| `epsilon_min` | `0.05` | Minimum exploration rate |
| `epsilon_decay_rate` | `0.9995` | Exploration decay per episode |
| `replay_memory_sizes` | `100,000` | Max transitions stored |
| `mini_batch_size` | `32` | Samples per training update |
| `network_sync_rate` | `10` | Steps between target network syncs |
| `reward_threshold` | `1000` | Max reward per episode cap |

---

## 🚀 How to Run

### Training
```bash
python agent.py flappybirdv0 --train
```

### Testing (with rendering)
```bash
python agent.py flappybirdv0
```

---

## 📦 Dependencies

- `torch` — Neural network & autograd
- `gymnasium` — RL environment interface
- `flappy-bird-gymnasium` — Flappy Bird environment
- `pyyaml` — Hyperparameter config loading

---
