# evaluation/evaluate_no_maxstep.py

import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import gymnasium as gym

from env.CuttingStockEnvOptimized import CuttingStockEnvOptimized
from agents.q_learning_agent import QLearningAgent
from data.static_data import STATIC_DATA

def load_q_table(checkpoint_path):
    """Tải Q-table từ file checkpoint."""
    with open(checkpoint_path, "rb") as f:
        q_table = pickle.load(f)
    return q_table

def evaluate_single_batch(agent, batch_id, render_mode=None, show_final_layout=False):
    """
    Đánh giá một episode cho batch_id sử dụng agent đã load Q-table từ checkpoint,
    không giới hạn số bước (chạy cho đến khi environment báo terminated).

    Trả về dict chứa các metric:
      - runtime: thời gian chạy (seconds)
      - total_trim_loss: tổng trim loss (số ô -1 trên các stock đã sử dụng)
      - remaining_stock: số stock chưa bị cắt (tất cả ô bên trong là -1)
      - used_stocks: số stock đã được sử dụng (có ít nhất một ô khác -1)
      - avg_used_stock_area: trung bình diện tích của các stock đã được sử dụng
      - steps: số bước thực hiện trong episode
      - total_reward: tổng reward tích lũy theo env.step
    """
    static_config = STATIC_DATA[batch_id]
    stock_list = static_config["stocks"]
    product_list = static_config["products"]

    # Sử dụng kích thước môi trường cố định (bạn có thể điều chỉnh nếu cần)
    max_w, max_h = 50, 50

    env = CuttingStockEnvOptimized(
        render_mode=render_mode,
        max_w=max_w,
        max_h=max_h,
        seed=42,
        stock_list=stock_list,
        product_list=product_list,
    )

    obs, info = env.reset()
    start_time = time.time()
    total_reward = 0
    steps = 0
    done = False

    # Vòng lặp evaluate không giới hạn số bước (chỉ dừng khi terminated)
    while not done:
        state = agent.get_state(obs)
        action_idx = agent.get_action(state)
        env_action = agent.get_env_action(action_idx, obs)
        obs, reward, done, _, info = env.step(env_action)
        total_reward += reward
        steps += 1

    end_time = time.time()
    runtime = end_time - start_time

    # Tính toán metric:
    total_trim_loss = sum((stock == -1).sum() for stock in env._stocks if np.any(stock[stock != -2] != -1))
    remaining_stock = sum(1 for stock in env._stocks if np.all(stock[stock != -2] == -1))
    used_stocks = len(env._stocks) - remaining_stock
    used_stock_areas = [np.sum(stock != -2) for stock in env._stocks if np.any(stock[stock != -2] != -1)]
    avg_used_stock_area = sum(used_stock_areas) / len(used_stock_areas) if used_stock_areas else 0

    if show_final_layout and render_mode == "human":
        print(f"[INFO] Showing final layout for batch {batch_id}...")
        env.render()
        input("Nhấn Enter để đóng cửa sổ và tiếp tục...")

    env.close()

    return {
        "batch_id": batch_id,
        "steps": steps,
        "runtime": runtime,
        "total_trim_loss": total_trim_loss,
        "remaining_stock": remaining_stock,
        "used_stocks": used_stocks,
        "avg_used_stock_area": avg_used_stock_area,
        "total_reward": total_reward,
    }

def main():
    # Đường dẫn tới file Q-table checkpoint đã huấn luyện
    checkpoint_path = os.path.join("checkpoints", "q_learning", "q_table_checkpoint_new.pkl")
    if not os.path.exists(checkpoint_path):
        print("[ERROR] Checkpoint file không tồn tại. Vui lòng huấn luyện trước hoặc kiểm tra lại đường dẫn.")
        return

    # Khởi tạo agent với các tham số giống lúc training
    agent = QLearningAgent(state_size=100000, action_size=1000,
                           alpha=0.1, gamma=0.9, epsilon=0.0,
                           epsilon_decay=0.995, min_epsilon=0.01)
    agent.Q_table = load_q_table(checkpoint_path)
    print("Loaded Q-table từ checkpoint:", checkpoint_path)

    # Danh sách batch_id cần đánh giá (ví dụ: từ 1 đến 10)
    batch_ids = list(range(1, 11))
    results = []

    # Nếu muốn hiển thị bố trí cuối cùng trên pygame, đặt True; render_mode sẽ được "human"
    show_final_layout = False
    chosen_render_mode = "human" if show_final_layout else None

    for bid in batch_ids:
        print(f"Đánh giá batch {bid}...")
        result = evaluate_single_batch(agent, bid, render_mode=chosen_render_mode, show_final_layout=show_final_layout)
        results.append(result)
    
    # In bảng số liệu chi tiết
    print("\n===== Detailed Results =====")
    header = ("Batch", "Policy", "Steps", "Runtime(s)", "TotalTrimLoss", "Remaining", "UsedStocks", "AvgUsedArea", "TotalReward")
    print("{:<6s} {:<15s} {:<7s} {:<12s} {:<15s} {:<10s} {:<11s} {:<12s} {:<12s}".format(*header))
    for r in results:
        print("{:<6d} {:<15s} {:<7d} {:<12.3f} {:<15d} {:<10d} {:<11d} {:<12.3f} {:<12.3f}".format(
            r["batch_id"], "QLearnAgent", r["steps"], r["runtime"],
            r["total_trim_loss"], r["remaining_stock"], r["used_stocks"],
            r["avg_used_stock_area"], r["total_reward"]
        ))
    
    # Vẽ biểu đồ các metric
    batch_ids_plot = [r["batch_id"] for r in results]
    runtimes = [r["runtime"] for r in results]
    trim_losses = [r["total_trim_loss"] for r in results]
    remainings = [r["remaining_stock"] for r in results]
    useds = [r["used_stocks"] for r in results]
    avg_used_areas = [r["avg_used_stock_area"] for r in results]
    steps_list = [r["steps"] for r in results]
    total_rewards = [r["total_reward"] for r in results]

    fig, axes = plt.subplots(1, 7, figsize=(28, 4))
    
    axes[0].plot(batch_ids_plot, runtimes, marker='o')
    axes[0].set_title("Runtime (s)")
    axes[0].set_xlabel("Batch ID")
    axes[0].set_ylabel("Time (s)")

    axes[1].plot(batch_ids_plot, trim_losses, marker='o', color='orange')
    axes[1].set_title("Total Trim Loss")
    axes[1].set_xlabel("Batch ID")
    axes[1].set_ylabel("Trim Loss (cells)")

    axes[2].plot(batch_ids_plot, remainings, marker='o', color='green')
    axes[2].set_title("Remaining Stocks")
    axes[2].set_xlabel("Batch ID")
    axes[2].set_ylabel("Stocks")

    axes[3].plot(batch_ids_plot, useds, marker='o', color='red')
    axes[3].set_title("Used Stocks")
    axes[3].set_xlabel("Batch ID")
    axes[3].set_ylabel("Stocks")

    axes[4].plot(batch_ids_plot, avg_used_areas, marker='o', color='purple')
    axes[4].set_title("Avg Used Stock Area")
    axes[4].set_xlabel("Batch ID")
    axes[4].set_ylabel("Area (cells)")

    axes[5].plot(batch_ids_plot, steps_list, marker='o', color='brown')
    axes[5].set_title("Steps per Episode")
    axes[5].set_xlabel("Batch ID")
    axes[5].set_ylabel("Steps")

    axes[6].plot(batch_ids_plot, total_rewards, marker='o', color='magenta')
    axes[6].set_title("Total Reward")
    axes[6].set_xlabel("Batch ID")
    axes[6].set_ylabel("Reward")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
