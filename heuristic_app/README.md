# 2D Cutting Stock Problem - Heuristic Approaches

## 📌 Problem Description

The **2D Cutting Stock Problem (2CSP)** involves cutting rectangular products from large rectangular stock sheets, with the goal of:
- Maximizing stock material usage.
- Minimizing the number of stocks used.
- Reducing material waste.

This problem commonly appears in industries like:
- Wooden floor manufacturing.
- Glass cutting.
- Textile manufacturing.
- Metal fabrication.

---

## 🧠 Heuristic Algorithms Implemented

### 1. **First-Fit Algorithm**

#### 💡 Idea:
- Sort products by decreasing area.
- Sort available stocks by remaining usable area.
- For each product:
  - Iterate over existing stocks to find the **first available space**.
  - If no suitable stock is found, **open a new stock** and place the product in it.

#### ✅ Pros:
- Fast and simple.
- Easy to implement.

#### ❌ Cons:
- May lead to inefficient space usage.
- Can waste material due to premature placement decisions.

---

### 2. **Best-Fit Algorithm**

#### 💡 Idea:
- For each product:
  - Iterate over existing stocks to find the position that leaves the **least remaining area**.
  - If no suitable stock is found, **open a new one**.

#### ✅ Pros:
- Better space utilization than First-Fit.
- Reduces the number of stocks used.

#### ❌ Cons:
- Slower than First-Fit due to extra calculations.
- Still not guaranteed to find the global optimum.

---

### 3. **Combination Algorithm (FF + BF + Merging)**

#### 💡 Pipeline includes three phases:

1. **First-Fit Placement**  
   - Quickly place products in the first available stock that fits.
   - Speeds up the initial allocation process.

2. **Best-Fit Refinement**  
   - Refine the layout by finding placements that minimize `Sij` (used area enclosing the product).
   - Aims to reduce waste and improve packing.

3. **Stock Merging**  
   - If a partially used smaller stock can fit into a larger unused one, **move its content** to reduce the total number of stocks used.

#### ✅ Pros:
- Balances speed and optimization.
- Ideal for medium-sized real-world instances.

---

## 🧱 Function Structure

```python
first_fit_policy(observation, info)
best_fit_policy(observation, info)
combination_policy(observation, info)
