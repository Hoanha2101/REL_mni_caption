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
    filled_ratio = info["filled_ratio"]
    trim_loss = info["trim_loss"]
    total_stocks = len(observation["stocks"])
    num_stocks_used = sum(1 for stock in observation["stocks"] if np.any(stock != -2))
    num_stocks_unused = total_stocks - num_stocks_used
    lambda_bonus = 0.2
    stock_bonus = lambda_bonus * (num_stocks_unused / total_stocks)
    return (filled_ratio - trim_loss) + stock_bonus

# def compute_metrics(env, episode_steps, episode_reward):
#     used_stocks = int(np.sum(env.cutted_stocks))
#     remaining_stock = 0
#     used_areas = []
#     total_trim_loss_val = 0
#     for idx, stock in enumerate(env._stocks):
#         valid_area = np.sum(stock != -2)
#         free_area = np.sum(stock == -1)
#         remaining_stock += free_area
#         if env.cutted_stocks[idx] == 1:
#             used_area = valid_area - free_area
#             used_areas.append(used_area)
#             tl = free_area / valid_area if valid_area > 0 else 0
#             total_trim_loss_val += tl
#     avg_used_stock_area = np.mean(used_areas) if used_areas else 0
#     metrics = {
#         "steps": episode_steps,
#         "total_trim_loss": total_trim_loss_val,
#         "remaining_stock": remaining_stock,
#         "used_stocks": used_stocks,
#         "avg_used_stock_area": avg_used_stock_area,
#         "total_reward": episode_reward
#     }
#     return metrics

def compute_metrics(env, episode_steps, episode_reward):
    """
    Tính các chỉ số dựa trên trạng thái cuối của môi trường:
      - total_trim_loss: Tổng diện tích lãng phí (free area) trong các stock đã được cắt.
      - remaining_stock: Số lượng stock chưa được cắt (tổng stock - used_stocks).
      - used_stocks: Số lượng stock đã được cắt.
      - avg_used_stock_area: Trung bình diện tích đã sử dụng (valid_area - free_area) trên các stock đã cắt.
      - steps: Số bước của episode.
      - total_reward: Tổng reward của episode.
    """
    num_stocks = len(env._stocks)
    used_stocks = int(np.sum(env.cutted_stocks))
    remaining_stock = num_stocks - used_stocks  # Số stock chưa sử dụng

    total_trim_loss = 0  # Tổng diện tích lãng phí (free area) trên các stock đã cắt
    used_areas = []      # Diện tích đã sử dụng của từng stock đã cắt

    for idx, stock in enumerate(env._stocks):
        valid_area = np.sum(stock != -2)  # Số ô hợp lệ trong stock
        free_area = np.sum(stock == -1)     # Số ô trống
        if env.cutted_stocks[idx] == 1:
            used_area = valid_area - free_area
            used_areas.append(used_area)
            total_trim_loss += free_area

    avg_used_stock_area = np.mean(used_areas) if used_areas else 0

    metrics = {
        "steps": episode_steps,
        "total_trim_loss": total_trim_loss,
        "remaining_stock": remaining_stock,
        "used_stocks": used_stocks,
        "avg_used_stock_area": avg_used_stock_area,
        "total_reward": episode_reward
    }
    return metrics



def train(num_episodes=500, state_size=100000, action_size=1000):
    """
    Train the Q-Learning agent for 10 batches using static_data.
    For each batch:
      - The environment is initialized with fixed max_w and max_h equal to 50.
      - Run num_episodes episodes and update the Q-table.
      - Save the checkpoint (dictionary containing Q_table, best_action_list, best_metrics)
        from the episode with the highest reward.
      - Record metrics: batch_id, steps, runtime, total_trim_loss, remaining_stock, used_stocks,
        avg_used_stock_area, total_reward.
    Finally, plot evaluation graphs and save the results to "metrics.csv".
    """
    results = []
    os.makedirs("checkpoints/q_learning/csv_train", exist_ok=True)

    # Iterate over batch IDs from 1 to 10
    for batch_id in range(1, 11):
        print(f"\n--- Training for Batch {batch_id} ---")
        static_config = STATIC_DATA[batch_id]

        # Initialize environment with fixed dimensions (max_w = 50, max_h = 50)
        env = CuttingStockEnvOptimized(
            render_mode="rgb_array",  # Sử dụng rgb_array để train không mở cửa sổ GUI
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
        best_action_list = []  # Lưu các action hợp lệ của episode tốt nhất
        total_steps = 0

        # batch_start_time = time.time()

        # Training loop for each episode
        for episode in range(num_episodes):
            observation, info = env.reset(seed=42)
            state = agent.get_state(observation)
            episode_reward = 0
            episode_steps = 0
            done = False

            # List to record only valid actions in this episode
            action_episode_list = []

            while not done:
                action = agent.get_action(state)
                env_action = agent.get_env_action(action, observation)
                observation, reward_terminal, terminated, truncated, info = env.step(env_action)
                # Chỉ lưu các action hợp lệ (không dummy)
                if env_action["size"] != (0, 0):
                    action_episode_list.append(env_action)
                reward = get_reward(observation, info)
                episode_reward += reward

                next_state = agent.get_state(observation)
                agent.update(state, action, reward, next_state)
                state = next_state

                episode_steps += 1
                total_steps += 1

                if terminated or truncated:
                    done = True

            print(f"\nBatch {batch_id} - Episode {episode}: Reward = {episode_reward:.4f}, Steps = {episode_steps}, Epsilon = {agent.epsilon:.4f}")

            # Cập nhật nếu episode này có tổng reward cao hơn
            if episode_reward > best_reward:
                best_reward = episode_reward
                best_metrics = compute_metrics(env, episode_steps, episode_reward)
                best_Q_table = agent.Q_table.copy()
                best_action_list = action_episode_list.copy()

        print("best_reward:", best_reward)
        print("best_action_list:", best_action_list)
        if best_action_list:
            print("Length of best_action_list:", len(best_action_list))
        else:
            print("No valid actions recorded in best_action_list.")
        
        # Replay best action list for verification
        if best_action_list:
            print("Start replaying best action list for verification...")
            env.reset(seed=42)
            start_time = time.time()
            for act in best_action_list:
                observation, reward_terminal, terminated, truncated, info = env.step(act)
                if terminated or truncated:
                    break
            replay_time = time.time() - start_time
            print("Replay action list length:", len(best_action_list))
            print("Replay time:", replay_time)
        else:
            print("Skipping replay since best_action_list is empty.")

        batch_runtime = time.time() - start_time
        if best_metrics is None:
            best_metrics = {
                "steps": 0,
                "total_trim_loss": 0,
                "remaining_stock": 0,
                "used_stocks": 0,
                "avg_used_stock_area": 0,
                "total_reward": 0
            }
        best_metrics["runtime"] = batch_runtime
        best_metrics["batch_id"] = batch_id

        results.append(best_metrics)

        # Save checkpoint as a dictionary
        checkpoint = {
            "Q_table": best_Q_table,
            "best_action_list": best_action_list,
            "best_metrics": best_metrics
        }
        checkpoint_filename = f"checkpoints/q_learning/csv_train/q_table_checkpoint_batch{batch_id}.pkl"
        with open(checkpoint_filename, "wb") as f:
            pickle.dump(checkpoint, f)
        print(f"Checkpoint for batch {batch_id} saved to {checkpoint_filename}")
        env.close()

    df = pd.DataFrame(results)
    expected_columns = ["batch_id", "steps", "runtime", "total_trim_loss", "remaining_stock",
                        "used_stocks", "avg_used_stock_area", "total_reward"]
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_columns]
    csv_filename = "metrics.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\nMetrics saved to {csv_filename}")

    # Plot evaluation graphs
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