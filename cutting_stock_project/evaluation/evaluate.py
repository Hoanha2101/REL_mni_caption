import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt

from env.CuttingStockEnvOptimized import CuttingStockEnvOptimized
from agents.q_learning_agent import QLearningAgent
from data.static_data import STATIC_DATA

def get_reward(observation, info):
    """
    Tính reward dựa trên filled_ratio, trim_loss và bonus cho stock chưa dùng.
    """
    filled_ratio = info["filled_ratio"]
    trim_loss = info["trim_loss"]
    total_stocks = len(observation["stocks"])
    num_stocks_used = sum(1 for stock in observation["stocks"] if (stock != -2).any())
    num_stocks_unused = total_stocks - num_stocks_used
    lambda_bonus = 0.2
    stock_bonus = lambda_bonus * (num_stocks_unused / total_stocks)
    return (filled_ratio - trim_loss) + stock_bonus

def evaluate_one_batch(batch_id, state_size=100000, action_size=1000, max_steps=10000):
    """
    Đánh giá một batch bằng cách:
      - Load checkpoint của batch từ file pickle.
      - Nếu checkpoint chứa key "best_action_list" (và danh sách không rỗng),
        replay chuỗi action đó.
      - Nếu không, chạy một episode mới dựa trên Q-table (epsilon=0) với giới hạn max_steps.
      - Tính tổng reward và số bước theo cách tính reward giống lúc train.
      - Lấy ảnh cuối của môi trường (render_mode="rgb_array").
    
    Trả về:
      - final_frame (np.ndarray): Ảnh cuối của môi trường.
      - metrics (dict): { "batch_id": batch_id, "steps": steps, "total_reward": total_reward }.
    """
    checkpoint_filename = f"checkpoints/q_learning/csv_train/q_table_checkpoint_batch{batch_id}.pkl"
    if not os.path.exists(checkpoint_filename):
        print(f"[ERROR] Checkpoint file cho batch {batch_id} không tồn tại.")
        return None, {"batch_id": batch_id, "steps": 0, "total_reward": 0.0}
    
    with open(checkpoint_filename, "rb") as f:
        checkpoint = pickle.load(f)
    
    # Nếu checkpoint là dictionary, lấy Q_table và best_action_list nếu có.
    if isinstance(checkpoint, dict):
        Q_table = checkpoint.get("Q_table", None)
        best_action_list = checkpoint.get("best_action_list", None)
    else:
        Q_table = checkpoint
        best_action_list = None
    
    if Q_table is None:
        print(f"[ERROR] Trong checkpoint của batch {batch_id}, key 'Q_table' không tồn tại.")
        return None, {"batch_id": batch_id, "steps": 0, "total_reward": 0.0}
    
    # Khởi tạo môi trường với cấu hình giống lúc train
    static_config = STATIC_DATA[batch_id]
    env = CuttingStockEnvOptimized(
        render_mode="rgb_array",
        max_w=50,
        max_h=50,
        stock_list=static_config["stocks"],
        product_list=static_config["products"],
        seed=42
    )
    
    # Reset môi trường
    observation, info = env.reset(seed=42)
    total_reward = 0.0
    steps = 0

    # Khởi tạo agent với epsilon=0 (tham lam) và gán Q_table đã load
    agent = QLearningAgent(
        state_size=state_size,
        action_size=action_size,
        alpha=0.1,
        gamma=0.9,
        epsilon=0.0,
        epsilon_decay=1.0,
        min_epsilon=0.0
    )
    agent.Q_table = Q_table

    # Nếu có best_action_list (đã lưu từ training) và không rỗng, replay chuỗi action đó.
    if best_action_list and len(best_action_list) > 0:
        print(f"[INFO] Batch {batch_id}: Replay best_action_list (length = {len(best_action_list)})")
        for act in best_action_list:
            observation, reward_terminal, terminated, truncated, info = env.step(act)
            step_reward = get_reward(observation, info)
            total_reward += step_reward
            steps += 1
            if terminated or truncated:
                break
    else:
        # Nếu không có best_action_list, chạy một episode mới với max_steps (mặc định 10,000 bước)
        print(f"[INFO] Batch {batch_id}: Không có best_action_list, chạy episode mới với max_steps={max_steps}")
        done = False
        while not done and steps < max_steps:
            state = agent.get_state(observation)
            action_idx = agent.get_action(state)
            env_action = agent.get_env_action(action_idx, observation)
            observation, reward_terminal, terminated, truncated, info = env.step(env_action)
            step_reward = get_reward(observation, info)
            total_reward += step_reward
            steps += 1
            done = terminated or truncated

    final_frame = env.render()  # Lấy ảnh cuối (RGB array)
    env.close()

    metrics = {
        "batch_id": batch_id,
        "steps": steps,
        "total_reward": total_reward
    }
    return final_frame, metrics

def evaluate_all_batches(state_size=100000, action_size=1000, max_steps=100000):
    """
    Đánh giá 10 batch bằng cách:
      - Load checkpoint cho từng batch từ 1 đến 10.
      - Nếu checkpoint có best_action_list, replay chính xác chuỗi action đó.
      - Nếu không, chạy một episode mới với giới hạn max_steps.
      - Thu thập ảnh cuối và metrics (steps, total_reward) cho mỗi batch.
      - Vẽ một Figure dạng grid 2 hàng x 5 cột hiển thị ảnh cuối và thông tin của mỗi batch.
    """
    frames = []
    metrics_list = []
    
    for batch_id in range(1, 11):
        print(f"\n=== Đang đánh giá Batch {batch_id} ===")
        frame, met = evaluate_one_batch(batch_id, state_size, action_size, max_steps)
        frames.append(frame)
        metrics_list.append(met)
    
    # Vẽ grid 2x5
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle("Final Cutting Layouts (Replay Best Actions) của 10 Batch", fontsize=16)
    
    for i, (frame, met) in enumerate(zip(frames, metrics_list)):
        ax = axs[i // 5, i % 5]
        if frame is None:
            ax.text(0.5, 0.5, f"Batch {met['batch_id']}\nNo checkpoint", ha="center", va="center", fontsize=12)
            ax.axis("off")
        else:
            ax.imshow(frame)
            # ax.set_title(f"Batch {met['batch_id']}\nSteps: {met['steps']}\nReward: {met['total_reward']:.2f}")
            ax.set_title(f"Batch {met['batch_id']}")
            ax.axis("off")
    
    plt.tight_layout()
    plt.savefig("evaluation_all_batches.png")
    plt.show()
    
    # print("\nTóm tắt Metrics:")
    # for met in metrics_list:
    #     print(f"Batch {met['batch_id']}: Steps = {met['steps']}, Total Reward = {met['total_reward']:.2f}")

if __name__ == "__main__":
    evaluate_all_batches()
