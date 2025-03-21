import os
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from env.CuttingStockEnvOptimized import CuttingStockEnvOptimized
from agents.q_learning_agent import QLearningAgent
from data.static_data import STATIC_DATA

def get_reward(observation, info):
    """
    Calculate the reward based on filled_ratio, trim_loss, and bonus for unused stocks.

    Args:
        observation (dict): Environment observation containing stocks.
        info (dict): Dictionary with keys such as "filled_ratio" and "trim_loss".

    Returns:
        float: Computed reward.
    """
    filled_ratio = info["filled_ratio"]
    trim_loss = info["trim_loss"]
    total_stocks = len(observation["stocks"])
    num_stocks_used = sum(1 for stock in observation["stocks"] if np.any(stock != -2))
    num_stocks_unused = total_stocks - num_stocks_used

    lambda_bonus = 0.2  # Bonus coefficient for unused stocks
    stock_bonus = lambda_bonus * (num_stocks_unused / total_stocks)

    return (filled_ratio - trim_loss) + stock_bonus

def compute_metrics(env, episode_steps, episode_reward):
    """
    Compute metrics from the final state of the environment.

    For each stock:
      - valid_area: number of valid cells (cell != -2)
      - free_area: number of free cells (cell == -1)
      - used_area: valid_area - free_area (if the stock has been used)
      - trim_loss: ratio of free_area to valid_area (if the stock has been used)

    Returns:
        dict: A dictionary containing:
            - steps (int): Number of steps in the episode.
            - total_trim_loss (float): Total trim loss of used stocks.
            - remaining_stock (int): Total free area across all stocks.
            - used_stocks (int): Number of stocks used (cut).
            - avg_used_stock_area (float): Average used area of the used stocks.
            - total_reward (float): Total reward of the episode.
    """
    used_stocks = int(np.sum(env.cutted_stocks))
    remaining_stock = 0
    used_areas = []
    total_trim_loss_val = 0
    for idx, stock in enumerate(env._stocks):
        valid_area = np.sum(stock != -2)
        free_area = np.sum(stock == -1)
        remaining_stock += free_area
        if env.cutted_stocks[idx] == 1:
            used_area = valid_area - free_area
            used_areas.append(used_area)
            tl = free_area / valid_area if valid_area > 0 else 0
            total_trim_loss_val += tl
    avg_used_stock_area = np.mean(used_areas) if used_areas else 0

    metrics = {
        "steps": episode_steps,
        "total_trim_loss": total_trim_loss_val,
        "remaining_stock": remaining_stock,
        "used_stocks": used_stocks,
        "avg_used_stock_area": avg_used_stock_area,
        "total_reward": episode_reward
    }
    return metrics

def train(num_episodes=10, state_size=100000, action_size=1000):
    """
    Train the Q-Learning agent for 10 batches using static_data.

    - In each batch, the environment is initialized with fixed max_w and max_h equal to 50.
    - Save the Q-table checkpoint of the episode with the highest reward.
    - Collect metrics: batch_id, steps, runtime, total_trim_loss, remaining_stock, used_stocks, avg_used_stock_area, total_reward.
    - After training, plot evaluation graphs and save the results to a CSV file.

    Args:
        num_episodes (int): Number of episodes per batch.
        state_size (int): Size of the state space.
        action_size (int): Size of the action space.
    """
    # List to store metrics for each batch
    results = []

    # Ensure that the directory for storing checkpoints exists
    os.makedirs("checkpoints/q_learning/csv_train", exist_ok=True)

    # Iterate over batch IDs from 1 to 10
    for batch_id in range(1, 11):
        print(f"\n--- Training for Batch {batch_id} ---")
        static_config = STATIC_DATA[batch_id]

        # Initialize environment with fixed dimensions of 50 for both width and height.
        # 'render_mode' is set to "rgb_array" to avoid displaying a graphical window during training.
        env = CuttingStockEnvOptimized(
            render_mode="rgb_array",
            max_w=50,
            max_h=50,
            stock_list=static_config["stocks"],
            product_list=static_config["products"],
            seed=42
        )

        # Initialize Q-Learning agent
        agent = QLearningAgent(
            state_size=state_size,
            action_size=action_size,
            alpha=0.1,
            gamma=0.9,
            epsilon=1.0,
            epsilon_decay=0.995,
            min_epsilon=0.01
        )

        best_reward = -np.inf
        best_metrics = None
        best_Q_table = None
        total_steps = 0

        batch_start_time = time.time()

        # Training loop for each episode
        for episode in range(num_episodes):
            observation, info = env.reset(seed=42)
            state = agent.get_state(observation)
            episode_reward = 0
            episode_steps = 0
            done = False

            while not done:
                action = agent.get_action(state)
                env_action = agent.get_env_action(action, observation)
                observation, reward_terminal, terminated, truncated, info = env.step(env_action)
                reward = get_reward(observation, info)
                next_state = agent.get_state(observation)
                agent.update(state, action, reward, next_state)
                state = next_state
                episode_reward += reward
                episode_steps += 1
                total_steps += 1

                if terminated or truncated:
                    done = True

            print(f"Batch {batch_id} - Episode {episode}: Reward = {episode_reward:.4f}, Steps = {episode_steps}, Epsilon = {agent.epsilon:.4f}")

            # If this episode has the highest reward, save the checkpoint and metrics.
            if episode_reward > best_reward:
                best_reward = episode_reward
                best_metrics = compute_metrics(env, episode_steps, episode_reward)
                best_Q_table = agent.Q_table.copy()

        batch_runtime = time.time() - batch_start_time
        if best_metrics is not None:
            best_metrics["runtime"] = batch_runtime
            best_metrics["batch_id"] = batch_id

        results.append(best_metrics)

        # Save the checkpoint of the batch with the highest reward.
        checkpoint_filename = f"checkpoints/q_learning/csv_train/q_table_checkpoint_batch{batch_id}.pkl"
        with open(checkpoint_filename, "wb") as f:
            pickle.dump(best_Q_table, f)
        print(f"Checkpoint for batch {batch_id} saved to {checkpoint_filename}")
        env.close()

    # Save all metrics to a CSV file.
    df = pd.DataFrame(results)
    df = df[["batch_id", "steps", "runtime", "total_trim_loss", "remaining_stock", "used_stocks", "avg_used_stock_area", "total_reward"]]
    csv_filename = "metrics.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\nMetrics saved to {csv_filename}")

    # Plot evaluation graphs for various metrics.
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    axs[0, 0].plot(df["batch_id"], df["total_reward"], marker='o')
    axs[0, 0].set_title("Total Reward vs Batch ID")
    axs[0, 0].set_xlabel("Batch ID")
    axs[0, 0].set_ylabel("Total Reward")

    axs[0, 1].plot(df["batch_id"], df["runtime"], marker='o')
    axs[0, 1].set_title("Runtime vs Batch ID")
    axs[0, 1].set_xlabel("Batch ID")
    axs[0, 1].set_ylabel("Runtime (s)")

    axs[1, 0].plot(df["batch_id"], df["steps"], marker='o')
    axs[1, 0].set_title("Steps vs Batch ID")
    axs[1, 0].set_xlabel("Batch ID")
    axs[1, 0].set_ylabel("Steps")

    axs[1, 1].plot(df["batch_id"], df["total_trim_loss"], marker='o')
    axs[1, 1].set_title("Total Trim Loss vs Batch ID")
    axs[1, 1].set_xlabel("Batch ID")
    axs[1, 1].set_ylabel("Total Trim Loss")

    plt.tight_layout()
    plt.savefig("metrics.png")
    plt.show()

if __name__ == "__main__":
    train()
