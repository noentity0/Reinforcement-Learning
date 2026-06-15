# 🤖 Reinforcement Learning

A collection of Reinforcement Learning projects implemented from scratch in Python, progressing from classical tabular methods to deep learning approaches.

---

## 📂 Projects

### 1. 🗂️ Temporal Difference Learning — Cliff Walking

> **Location:** [`Temporal Difference Learning/`](./Temporal%20Difference%20Learning/)

Tabular RL on the `CliffWalking-v1` environment using a **4×12 grid** with a cliff, start, and goal. Both algorithms learn a Q-table of shape `(48 states × 4 actions)` and use an **ε-greedy** policy for exploration.

Two classic TD control algorithms are compared side by side:

| | [SARSA](./Temporal%20Difference%20Learning/SARSA_algo/) | [Q-Learning](./Temporal%20Difference%20Learning/Q-Learning_algo/) |
|---|---|---|
| **Type** | On-policy | Off-policy |
| **Update uses** | Actual next action $Q(s', a')$ | Best next action $\max Q(s', a')$ |
| **Path learned** | Safer (one row above cliff) | Riskier but optimal (cliff edge) |
| **Converges to** | Policy optimal for ε-greedy agent | True optimal policy |
| **Env** | `CliffWalking-v1` | `CliffWalking-v1` |
| **Implementation** | Jupyter Notebook | Jupyter Notebook |

**Core idea:** TD methods learn directly from experience, one step at a time, without waiting for an episode to finish — unlike Monte Carlo. They bootstrap: the update target is a mix of the immediate reward and the current estimate of the next state's value.

---

### 2. 🐦 Flappy Bird — Deep Q-Network (DQN)

> **Location:** [`flappy_bird_DQN/`](./flappy_bird_DQN/)

A DQN agent trained to play Flappy Bird on the `FlappyBird-v0` environment. This project scales up from tabular methods by replacing the Q-table with a **neural network**, making it possible to handle a continuous 12-dimensional state space.

**Key components:**

| Component | File | Role |
|---|---|---|
| Policy & Target Network | [`dqn.py`](./flappy_bird_DQN/dqn.py) | MLP: 12 → 256 → 2 Q-values |
| Training Agent | [`agent.py`](./flappy_bird_DQN/agent.py) | ε-greedy loop, optimisation, model saving |
| Replay Buffer | [`experience_replay.py`](./flappy_bird_DQN/experience_replay.py) | FIFO queue of 100k transitions |
| Hyperparameters | [`parameters.yaml`](./flappy_bird_DQN/parameters.yaml) | All tunable settings in one place |

**Why DQN over tabular Q-Learning?**  
Tabular Q-Learning stores one value per (state, action) pair — fine for 48 states, but impossible for high-dimensional or continuous spaces. DQN generalises across states by approximating the Q-function with a neural network. It also introduces two stabilising tricks — **Experience Replay** and a **Target Network** — that make neural network training tractable.

---

## 🗺️ Learning Progression

```
Tabular Methods                    Deep RL
──────────────────────────────────────────────────────────────►
   SARSA           Q-Learning              DQN
(on-policy TD)  (off-policy TD)   (Q-Learning + Neural Net
                                   + Experience Replay
                                   + Target Network)
```

Each project builds on the concepts of the previous one.

---

## 📦 Dependencies

```bash
pip install gymnasium numpy torch flappy-bird-gymnasium pyyaml
```
