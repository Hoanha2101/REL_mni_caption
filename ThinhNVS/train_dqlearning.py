import gymnasium as gym
import numpy as np
import torch
from env.CuttingStockEnvOptimized import CuttingStockEnvOptimized
from agents.deep_qlearning_agent import DeepQLearningAgent, state_to_tensor
import signal
import sys
import os
import random

# Global agent để cho hàm signal handler có thể truy cập
agent = None

def save_checkpoint_handler(signum, frame):
    """
    Signal handler được gọi khi nhấn Ctrl+C.
    Lưu state dictionary của policy network của agent vào checkpoint và thoát chương trình.
    """
    global agent
    try:
        os.makedirs("checkpoints", exist_ok=True)
        checkpoint_path = os.path.join("checkpoints", "dqn_checkpoint_optimized.pt")
        torch.save(agent.policy_net.state_dict(), checkpoint_path)
        print("\nOptimized DQN checkpoint saved (triggered by signal).")
    except Exception as e:
        print(f"\n[ERROR] Saving optimized DQN checkpoint failed: {e}")
    sys.exit(0)

# Đăng ký signal handler cho SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, save_checkpoint_handler)

def train_deep_q_learning_optimized():
    """
    Huấn luyện model Deep Q-Learning trên môi trường CuttingStockEnvOptimized với phần optimizer cải tiến:
      - Khởi tạo môi trường và agent.
      - Huấn luyện qua nhiều episode (ví dụ 5000 episode).
      - Sau mỗi bước, lưu transition và thực hiện tối ưu mini-batch.
      - Cập nhật target network định kỳ.
      - Sau mỗi checkpoint_interval episode, lưu checkpoint và tính trung bình reward của khoảng đó.
      - Cuối cùng, in ra bảng tóm tắt các checkpoint cùng với trung bình reward.
    """
    global agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Sử dụng render_mode="rgb_array" để tăng tốc huấn luyện (không hiển thị cửa sổ Pygame)
    env = CuttingStockEnvOptimized(render_mode="rgb_array")
    agent = DeepQLearningAgent(env, device)
    
    num_episodes = 10000
    batch_size = 64
    target_update = 10       # Cập nhật target network mỗi 10 episode
    checkpoint_interval = 500 # Lưu checkpoint mỗi episode
    total_rewards = []
    checkpoint_summary = []  # Lưu (episode, avg_reward, checkpoint_filename)
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        state_tensor = state_to_tensor(state, device)
        episode_reward = 0
        done = False
        
        while not done:
            # Chọn hành động dưới dạng discrete index.
            action_index = agent.select_action(state_tensor)
            
            # Chuyển đổi từ chỉ số hành động sang dictionary action theo công thức:
            # action_index = stock_idx * (max_product_type * total_positions) + product * total_positions + pos_index
            total_positions = env.max_w * env.max_h
            stock_idx = action_index // (agent.max_product_type * total_positions)
            remainder = action_index % (agent.max_product_type * total_positions)
            product = remainder // total_positions
            pos_index = remainder % total_positions
            x = pos_index // env.max_h
            y = pos_index % env.max_h
            
            action = {
                "stock_idx": stock_idx,
                "product_idx": product,
                "size": np.array([0, 0]),  # Môi trường sẽ xác định kích thước thực.
                "position": np.array([x, y])
            }
            
            next_state, reward, done, truncated, _ = env.step(action)
            next_state_tensor = state_to_tensor(next_state, device)
            
            agent.memory.push(state, action_index, reward, next_state, done)
            
            state = next_state
            state_tensor = next_state_tensor
            episode_reward += reward
            
            agent.optimize_model(batch_size)
        
        agent.update_epsilon()
        total_rewards.append(episode_reward)
        print(f"Episode {episode+1}: Total Reward = {episode_reward}")
        
        # Lưu checkpoint sau mỗi checkpoint_interval episode.
        if (episode + 1) % checkpoint_interval == 0:
            checkpoint_directory = os.path.join("checkpoints", "Case_optimized")
            os.makedirs(checkpoint_directory, exist_ok=True)
            checkpoint_filename = os.path.join(checkpoint_directory, f"dqn_checkpoint_ep{episode+1}_optimized.pt")
            try:
                torch.save(agent.policy_net.state_dict(), checkpoint_filename)
                print(f"Checkpoint saved at episode {episode+1} to '{checkpoint_filename}'")
            except Exception as e:
                print(f"Error saving checkpoint at episode {episode+1}: {e}")
            
            # Tính trung bình reward của khoảng checkpoint_interval vừa qua.
            avg_reward = np.mean(total_rewards[-checkpoint_interval:])
            checkpoint_summary.append((episode+1, avg_reward, checkpoint_filename))
        
        # Cập nhật target network định kỳ.
        if episode % target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
    
    env.close()
    avg_reward_all = np.mean(total_rewards)
    print("Average Reward over all episodes:", avg_reward_all)
    
    # In bảng tóm tắt các checkpoint với trung bình reward.
    print("\nCheckpoint Summary:")
    print("Episode\tAvg Reward\tCheckpoint File")
    for ep, avg_r, path in checkpoint_summary:
        print(f"{ep}\t{avg_r:.2f}\t{path}")

if __name__ == "__main__":
    train_deep_q_learning_optimized()
