# Q-Learning — Cliff Walking 🧗

> **Algorithm:** Q-Learning (Off-Policy Temporal Difference Control)  
> **Environment:** `CliffWalking-v1` (Gymnasium)

---

## 📌 What is Q-Learning?

Q-Learning is an **off-policy** TD control algorithm. It learns the **optimal action-value function** Q*(s, a) directly, regardless of what policy the agent is currently following.

### Update Rule

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

| Symbol | Meaning |
|--------|---------|
| `α` (alpha) | Learning rate — how fast we update Q-values |
| `γ` (gamma) | Discount factor — how much we value future rewards |
| `r` | Reward received after taking action `a` in state `s` |
| `s'` | Next state |
| `max Q(s', a')` | Best possible Q-value from next state (greedy) |

The key insight: the update uses `max Q(s', a')` — the **best** possible next action — not necessarily the one the agent will take. This makes it **off-policy**.

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
- **Optimal path length:** 13 steps → total reward = **−13**

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

Initialized to zero. Each cell `Q[state, action]` stores the expected cumulative discounted reward.

---

## 🎯 Epsilon-Greedy Policy

```python
def epsilon_greedy(state):
    if random.random() < epsilon:
        return env.action_space.sample()  # explore (random)
    else:
        return np.argmax(Q[state])        # exploit (greedy)
```

With probability `ε = 0.1` the agent picks a **random action** (exploration); otherwise it picks the **best known action** (exploitation).

---

## 🔄 Training Loop

```python
for episode in range(episodes):
    env = gym.make("CliffWalking-v1")
    state, _ = env.reset()
    done = False

    while not done:
        action = epsilon_greedy(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Q-Learning update (off-policy — uses max over next state)
        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])

        state = next_state
    env.close()
```

### What happens each step:
1. Choose action using ε-greedy policy
2. Take the action, get `next_state` and `reward`
3. Update `Q[state, action]` using the **maximum** Q-value of `next_state`
4. Move to `next_state`

---

## 📊 Results

- After ~25 episodes, the agent starts finding paths close to optimal
- By episode 500, the agent consistently achieves **total reward ≈ −13**, which is the **optimal** solution
- Q-Learning converges to the **optimal (shortest) path** along the cliff edge

### Learned Q-values (example)
```python
Q[36]  # Start state — strong preference for "Right" (action index 1 was NOT chosen; Up preferred away from cliff)
# array([ -12.25, -112.13, -13.13, -13.13])

Q[35]  # State just before goal
# array([-2.97, -1.99, -1.0, -2.97])  → prefers "Down" (action 2) to reach goal
```

---

## ⚡ Key Takeaway: Q-Learning vs SARSA

Q-Learning is **optimistic** — it always assumes the agent will take the **best possible** next action. This means it finds the **theoretically optimal path** (along the cliff edge), but during training it may fall off the cliff more often because ε-greedy exploration can push it over.

---

## 📦 Dependencies

```bash
pip install gymnasium numpy
```

## ▶️ Run

Open `Cliff-walking_Q_Learning.ipynb` in Jupyter and run all cells.
