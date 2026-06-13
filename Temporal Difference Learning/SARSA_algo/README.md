# SARSA — Cliff Walking 🧗

> **Algorithm:** SARSA (On-Policy Temporal Difference Control)  
> **Environment:** `CliffWalking-v1` (Gymnasium)

---

## 📌 What is SARSA?

SARSA stands for **S**tate–**A**ction–**R**eward–**S**tate–**A**ction. It is an **on-policy** TD control algorithm, meaning it learns the Q-value for the **same policy it is following** (ε-greedy).

The name comes from the 5 things used in each update step: `(s, a, r, s', a')`.

### Update Rule

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma Q(s', a') - Q(s, a) \right]$$

| Symbol | Meaning |
|--------|---------|
| `α` (alpha) | Learning rate — how fast we update Q-values |
| `γ` (gamma) | Discount factor — how much we value future rewards |
| `r` | Reward received after taking action `a` in state `s` |
| `s'` | Next state |
| `a'` | Next action chosen by **the same ε-greedy policy** |

The key insight: unlike Q-Learning, SARSA uses `Q(s', a')` — the value of the **actual next action** the agent will take — not the maximum. This makes it **on-policy** and more conservative.

---

## 🏔️ The Environment — CliffWalking-v1

The cliff walking grid is **4 rows × 12 columns = 48 states**.

```
o  o  o  o  o  o  o  o  o  o  o  o
o  o  o  o  o  o  o  o  o  o  o  o
o  o  o  o  o  o  o  o  o  o  o  o
S  C  C  C  C  C  C  C  C  C  C  G
```

- **S** = Start (state 36)
- **G** = Goal (state 47)
- **C** = Cliff (states 37–46) — stepping here gives reward **−100** and resets to Start
- Every non-cliff step gives reward **−1**
- **Actions:** `0=Up`, `1=Right`, `2=Down`, `3=Left`
- **Observation space:** 48 discrete states
- **Action space:** 4 discrete actions

---

## 🔧 Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `gamma` | `0.99` | Discount factor |
| `epsilon` | `0.1` | Exploration rate (ε-greedy) |
| `alpha` | `0.5` | Learning rate |
| `episodes` | `500` | Number of training episodes |

---

## 🗃️ Q-Table

```python
Q = np.zeros((48, 4))  # 48 states × 4 actions
```

Initialized to all zeros. Displayed at the start to confirm shape (48×4).

---

## 🎯 Epsilon-Greedy Policy

```python
def epsilon_greedy(state):
    if random.random() < epsilon:
        return env.action_space.sample()  # explore (random action)
    else:
        return np.argmax(Q[state])        # exploit (greedy best action)
```

With probability `ε = 0.1` → random action (explore)  
Otherwise → greedy best action (exploit)

---

## 🔄 Training Loop

```python
for episode in range(episodes):
    env = gym.make("CliffWalking-v1")
    state, _ = env.reset()
    action = epsilon_greedy(state)   # pick FIRST action before loop
    done = False

    while not done:
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_action = epsilon_greedy(next_state)   # pick NEXT action NOW

        # SARSA Update (on-policy — uses the actual next action)
        Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])

        state = next_state
        action = next_action   # carry forward the chosen next action
    env.close()
```

### What happens each step:
1. Take action `a` in state `s`
2. Observe `next_state (s')` and `reward (r)`
3. Choose `next_action (a')` using ε-greedy from `s'`
4. Update `Q[s, a]` using `Q[s', a']` — the **actual next action**
5. Move to `(s', a')`

---

## 🏁 Gymnasium Environment Lifecycle

As shown in the notebook:

| Step | Method |
|------|--------|
| 1. Create | `gym.make()` |
| 2. Reset | `env.reset()` |
| 3. Act | `env.step()` |
| 4. Render | `env.render()` |
| 5. Close | `env.close()` |

---

## 📊 Results

- SARSA converges stably around episode 65–100
- Settled reward per episode is typically around **−17 to −23**, slightly worse than the theoretical minimum of −13
- SARSA takes a **safer path** (one row above the cliff), because it accounts for the chance of accidentally stepping off due to ε-exploration

### Why SARSA ≠ −13 optimal?
Because SARSA is **on-policy**: it knows it sometimes takes random actions (ε=0.1), so it learns to stay **away from the cliff edge** as a safety buffer. This is the correct behavior for an ε-greedy agent.

---

## ⚡ Key Takeaway: SARSA vs Q-Learning

| Property | SARSA | Q-Learning |
|----------|-------|-----------|
| Policy type | On-policy | Off-policy |
| Next action in update | Actual (ε-greedy) | Best (greedy max) |
| Path found | **Safer** (away from cliff) | **Riskier** (cliff edge) |
| Reward during training | Higher (safer) | Lower (falls more) |
| Converges to | Policy for ε-greedy agent | Optimal policy |

---

## 📦 Dependencies

```bash
pip install gymnasium numpy
```

## ▶️ Run

Open `cliff-walking_SARSA.ipynb` in Jupyter and run all cells.
