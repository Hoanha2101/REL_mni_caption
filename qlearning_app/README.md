# ✂️ Solving the 2D Cutting Stock Problem using Q-Learning

This repository demonstrates how **Q-Learning**, a classic reinforcement learning algorithm, can be applied to the **2D Cutting Stock Problem (2D-CSP)** to optimize material usage, reduce trim loss, and minimize the number of stock sheets used.

---

## 📦 Problem Description

The **2D Cutting Stock Problem** involves cutting rectangular products from large rectangular stock sheets. The objectives are:

- Minimize **trim loss** (unused space).
- Fulfill all **product demands**.
- Reduce the **number of sheets used**.

Traditional optimization techniques struggle in large search spaces. We instead explore **Q-Learning**, allowing an agent to learn cutting strategies by interacting with a simulated environment.

---

## 🤖 Q-Learning Overview

Q-Learning is an off-policy reinforcement learning algorithm that learns an optimal action-value function (Q-table).

### 🔑 Key Concepts

- **Q-table**: A matrix \( Q(s, a) \) estimating the expected reward for taking action `a` in state `s`.
- **Epsilon-greedy**: Balances exploration (random actions) and exploitation (greedy actions).
- **Bellman Update**:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
$$

---

## 🧪 Cutting Stock Environment

The environment is built using the **Gymnasium** interface and includes:

### 🔹 Observation Space

- Each stock is a 2D grid:
  - `-2`: Out of bounds.
  - `-1`: Empty cell.
  - `>= 0`: Occupied by a product ID.
- Product list: sizes and quantities remaining.

### 🔹 Action Space

Each action is a dictionary:
```python
{
  "stock_idx": int,
  "size": (w, h),
  "position": (x, y)
}
```

## 🔹 Step Function & Termination

- Validates action → applies cut if valid → reduces product quantity.
- **Episode terminates** when all product quantities reach zero (fully cut).

---

## 🔹 Reward Function

Custom reward is calculated using:

- ✅ **Filled Ratio** – Encourages placing more products in used space.
- ✅ **Bonus for Unused Stocks** – Incentivizes using fewer sheets.
- ❌ **Trim Loss** – Penalizes leftover space in partially used stocks.

**Total Reward = FilledRatio + UnusedStockBonus − TrimLoss**

---

## 🧠 Q-Learning Agent

### 🧩 State Encoding

To reduce Q-table size, we extract simple features:

- `empty_space` = number of `-1` cells (free space).
- `remaining_products` = sum of product quantities left.

Then combine into a single state index:

```python
state = (empty_space * 1000 + remaining_products) % state_size
```

### 🎮 Action Mapping

Each action index maps to:

- A **product index** to cut.
- A **placement position (x, y)** on a selected stock sheet.

⚠️ **Invalid actions** (e.g., overlapping or out-of-bound cuts) are **skipped** or result in a **no-operation (no-op)**.

---

### 🔁 Learning Updates

The Q-learning agent follows the standard update rule based on the Bellman equation:

- **Learning rate (α)**: Controls how much newly acquired information overrides the old Q-value.
- **Discount factor (γ)**: Determines the importance of future rewards relative to immediate rewards.
- **Epsilon decay**: Gradually reduces the probability of random action selection (exploration) over time, favoring learned actions (exploitation).