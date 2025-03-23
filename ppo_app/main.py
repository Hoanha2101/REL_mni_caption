import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from cutting_glass_env import CuttingGlassEnv
from ppo_policy import PPOPolicy
from ppo_agent import PPOAgent
import gc

# Cho phép xử lý nhiều thư viện OpenMP, tránh lỗi duplicate
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Giải phóng bộ nhớ trên GPU trước khi chạy (nếu có GPU)
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# Tạo các thư mục lưu mô hình và biểu đồ loss nếu chưa tồn tại
os.makedirs("Model", exist_ok=True)
os.makedirs("Loss_Plots", exist_ok=True)

# Cấu hình tập dữ liệu cần dùng (có thể thay đổi dataset_id nếu cần)
dataset_id = 1

# Khởi tạo môi trường và các đối tượng mô hình, agent PPO
env = CuttingGlassEnv(dataset=dataset_id, csv_path="data_custom.csv")
policy = PPOPolicy(num_stocks=env.num_stocks, num_products=env.num_products).to(device)
agent = PPOAgent(env, policy)

# Thiết lập số lượng episode huấn luyện, khoảng cách lưu mô hình và danh sách lưu lại reward của các episode
num_episodes = 10000
save_interval = 1000
rewards_history = []

# Hàm vẽ biểu đồ loss và reward, sau đó lưu vào file ảnh
def save_loss_plot(actor_losses, critic_losses, rewards_history, episode):
    plt.figure(figsize=(15, 5))
    
    # Vẽ biểu đồ Actor Loss
    plt.subplot(1, 3, 1)
    plt.plot(actor_losses, label="Actor Loss", color='blue')
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Actor Loss")
    plt.legend()
    plt.grid()
    
    # Vẽ biểu đồ Critic Loss
    plt.subplot(1, 3, 2)
    plt.plot(critic_losses, label="Critic Loss", color='green')
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Critic Loss")
    plt.legend()
    plt.grid()
    
    # Vẽ biểu đồ Reward qua thời gian
    plt.subplot(1, 3, 3)
    plt.plot(rewards_history, label="Total Reward", color='red')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward Over Time")
    plt.legend()
    plt.grid()
    
    # Lưu biểu đồ vào thư mục Loss_Plots với tên file bao gồm episode
    loss_plot_path = f"Loss_Plots/loss_plot_ep{episode}.png"
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"📊 Loss plot saved at {loss_plot_path}")

# Vòng lặp huấn luyện qua các episode
for episode in range(1, num_episodes + 1):
    # Reset môi trường tại đầu mỗi episode
    state = env.reset()
    # Chuyển đổi state từ numpy về tensor, thêm chiều batch (unsqueeze(0))
    state = {key: torch.tensor(value, dtype=torch.float32).unsqueeze(0).to(device) for key, value in state.items()}
    done = False
    total_reward = 0

    # Vòng lặp tương tác trong một episode cho đến khi kết thúc (done=True)
    while not done:
        # Chọn hành động từ agent dựa trên state hiện tại
        action, log_probs, value = agent.choose_action(state)
        # Thực hiện hành động trong môi trường và nhận kết quả (next_state, reward, done, info)
        next_state, reward, done, _ = env.step(action)
        # Chuyển đổi next_state từ numpy sang tensor, thêm chiều batch
        for key, val in next_state.items():
            try:
                tensor_val = torch.tensor(val, dtype=torch.float32)
                next_state[key] = tensor_val.unsqueeze(0).to(device)
            except Exception as e:
                print(f" Lỗi khi chuyển đổi {key}: {e}")
                print(f" Giá trị của {key}: {val}")
        # Lưu lại trải nghiệm vào bộ nhớ của agent
        agent.store_memory(state, action, log_probs, value, reward, done)
        # Cập nhật state hiện tại thành next_state cho bước lặp tiếp theo
        state = next_state
        total_reward += reward
    
    # Dọn dẹp bộ nhớ và giải phóng tài nguyên sau mỗi episode
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.8, 0)
    
    # Sau mỗi 10 episode, tiến hành training cho policy
    if episode % 10 == 0:
        agent.train()
        print(f"Model training at episode {episode}")
    # Sau mỗi 200 episode, xóa bộ nhớ của agent (có thể giúp tránh memory overflow)
    if episode % 200 == 0:
        agent.memory.clear()
        print(f"Memory cleared at episode {episode}")
    
    # Lưu lại tổng reward của episode vào danh sách lịch sử
    rewards_history.append(total_reward)
    print(f" Episode {episode}/{num_episodes},  Total Reward: {total_reward}")
    
    # Lưu model và biểu đồ loss sau mỗi save_interval episode
    if episode % save_interval == 0:
        model_path = f"Model/ppo_policy_ep{episode}.pth"
        torch.save(policy, model_path)
        print(f" Model saved at {model_path}")
        save_loss_plot(agent.actor_losses, agent.critic_losses, rewards_history, episode)
