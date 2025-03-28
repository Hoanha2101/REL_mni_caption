# 🧠 Proximal Policy Optimization (PPO) for 2D Cutting Stock Problem

## 🔍 Overview of PPO

**Proximal Policy Optimization (PPO)** is a policy gradient method in reinforcement learning that aims to achieve **training stability and efficiency**. It belongs to the family of **actor-critic** algorithms, where:

- **Actor**: Learns a policy to select actions based on current environment states.
- **Critic**: Evaluates the value of states (V(s)) to guide and stabilize the actor's updates.

![image](https://github.com/user-attachments/assets/1d9b824b-3d1c-4939-8d1b-95024c74e190)

(Image's source: Proximal Policy Optimization (PPO) architecture) 

---

## 🎯 PPO Training Objective

PPO optimizes a surrogate loss function that constrains the magnitude of policy updates using a **clipping mechanism**. This prevents excessively large updates that could destabilize training.

The clipped objective is given by:

$$
\nabla_\theta L^{\mathrm{CLIP}}(\theta) 
= \nabla_\theta \min \Big( 
    r_t(\theta)\,A_t,\; 
    \mathrm{clip}\big(r_t(\theta), 1 - \varepsilon, 1 + \varepsilon\big)\,A_t 
\Big)
$$

Where:
- \( r_t(\theta) \): Ratio between new and old policy probabilities.
- \( A_t \): Advantage estimate at timestep \( t \).
- \( \varepsilon \): Clipping parameter (commonly 0.1 or 0.2).

---

## ✅ Key Advantages of PPO

- **Simple and effective**: Easy to implement with strong empirical performance.
- **Stable training**: The clipping mechanism avoids destructive updates.
- **Versatile**: Applied successfully in robotics, games (e.g. Atari), and complex optimization problems.

---

## ⚠️ Limitations

- **Sensitive to hyperparameters**: Learning rate, clipping epsilon, batch size, etc., need careful tuning.
- **Challenging in complex action spaces**: In high-dimensional or continuous spaces, network design and action distributions can be harder to model.

---

## 🔁 PPO Training Pipeline

1. Collect trajectories (experience rollouts) from interactions with the environment.
2. Estimate advantages using the difference between predicted value and actual return.
3. Update the policy using the clipped loss objective to ensure conservative improvement.
4. Repeat across multiple epochs and batches.

---

## 🪟 PPO for the Glass Cutting Environment

This repository applies PPO to solve a **2D Glass Cutting Problem**. The environment simulates the placement of rectangular glass pieces on stock sheets, where the goal is to **minimize waste and the number of sheets used**.

### 📦 Project Structure

| File | Description |
|------|-------------|
| `cutting_glass_env.py` | Custom OpenAI Gym environment defining state/action/reward logic for glass cutting. |
| `ppo_policy.py` | Defines the CNN/MLP-based Actor-Critic model used in PPO. |
| `ppo_agent.py` | Implements `PPOAgent`: memory buffer, action sampling, policy optimization. |
| `main.py` | Training script: collects data, trains the agent, and saves model checkpoints. |
| `visualize.py` | Evaluates the trained PPO model across 10 test datasets and visualizes results. |
| `test.py` | Visualizes the cutting result on a single dataset for manual inspection. |

---

### 🖼️ Model Architecture Highlights

- **Feature extraction**: Convolutional and/or MLP layers process the input grid of the stock.
- **Actor head**: Outputs probability distribution over valid cutting actions.
- **Critic head**: Outputs estimated state value \( V(s) \).

---

### 📊 Training Outputs

- Trained PPO models saved in the `Model/` directory.
- Loss and reward plots saved in `Loss_Plot/`, tracking agent performance over episodes.
- Visual cutting layouts generated by `test.py` or `visualize.py`.

![Cutting Results](https://github.com/user-attachments/assets/24b067f2-4720-4094-9b0c-75d4fcf72e9d)

---

## 📚 References & Resources

- [📝 PPO Original Paper (Schulman et al., 2017)](https://arxiv.org/pdf/1707.06347)
- [📖 Medium - A Comprehensive Guide to PPO](https://medium.com/%40oleglatypov/a-comprehensive-guide-to-proximal-policy-optimization-ppo-in-ai-82edab5db200)
- [🎥 PPO Explained (YouTube)](https://www.youtube.com/watch?v=hlv79rcHws0)

## Github branch

[PPO_using_Neuron_Network](https://github.com/dangchau2111/Reinforcement-Learning-Course/tree/main/Mini_Capstone_Cutting_Stock/PPO_using_Neuron_Network)
## 📄 License

This project is licensed under the MIT License.
