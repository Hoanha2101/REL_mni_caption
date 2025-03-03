import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pickle
from collections import deque
from env.cutting_stock import CuttingStockEnv

# **Khởi tạo môi trường Cutting Stock**
stocks = [
    (100, 100), (80, 60), (90, 50), (120, 80)
]
products = [
    (20, 10), (20, 10), (30, 20), (40, 30), (50, 25)
]

env = CuttingStockEnv(
    render_mode=None,  # hoặc "human" nếu muốn xem trực quan
    max_w=120,
    max_h=120,
    seed=42,
    stock_list=stocks,
    product_list=products,
)

# **Tham số DQN**
state_size = 3  # Số features đầu vào (tổng diện tích trống, số sản phẩm chưa cắt, số stock đã sử dụng)
action_size = 500  # Giới hạn số lượng hành động có thể có
learning_rate = 0.001
gamma = 0.95  # Discount factor
epsilon = 1.0  # Epsilon-greedy
epsilon_decay = 0.995
min_epsilon = 0.01
num_episodes = 5000
batch_size = 32
memory_size = 10000
update_target_frequency = 50  # Mỗi bao nhiêu tập thì cập nhật Target Network

# **Replay Buffer**
memory = deque(maxlen=memory_size)

# **Mô hình Deep Q-Network (DQN)**
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# **Khởi tạo mạng**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(state_size, action_size).to(device)
target_net = DQN(state_size, action_size).to(device)
target_net.load_state_dict(policy_net.state_dict())  # Đồng bộ target_net với policy_net
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# **Hàm chuyển trạng thái thành vector**
def get_state(observation):
    stocks = observation["stocks"]
    products = observation["products"]

    empty_space = sum(np.sum(stock == -1) for stock in stocks)
    remaining_products = sum(prod["quantity"] for prod in products)
    num_stocks_used = sum(1 for stock in stocks if np.any(stock != -2))

    return np.array([empty_space, remaining_products, num_stocks_used], dtype=np.float32)

# **Hàm lấy danh sách hành động hợp lệ**
def get_valid_actions(observation):
    actions = []
    for stock_idx, stock in enumerate(observation["stocks"]):
        stock_w = np.sum(np.any(stock != -2, axis=1))
        stock_h = np.sum(np.any(stock != -2, axis=0))

        for prod_idx, prod in enumerate(observation["products"]):
            if prod["quantity"] <= 0:
                continue
            prod_w, prod_h = prod["size"]

            if prod_w <= stock_w and prod_h <= stock_h:
                actions.append((stock_idx, prod_idx, 0, 0))
    return actions

# **Hàm chọn hành động theo epsilon-greedy**
def get_action(state, valid_actions):
    if np.random.rand() < epsilon:
        return random.choice(valid_actions)  
    else:
        state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
        with torch.no_grad():
            q_values = policy_net(state_tensor)
            return valid_actions[np.argmax(q_values[:len(valid_actions)])]  

# **Hàm tính phần thưởng**
def get_reward(observation, info):
    filled_ratio = info["filled_ratio"]
    trim_loss = info["trim_loss"]
    num_stocks_used = sum(1 for stock in observation["stocks"] if np.any(stock != -2))
    total_stocks = len(observation["stocks"])
    num_stocks_unused = total_stocks - num_stocks_used
    lambda_bonus = 0.2  
    stock_bonus = lambda_bonus * (num_stocks_unused / total_stocks)  
    reward = (filled_ratio - trim_loss) + stock_bonus
    return reward

# **Huấn luyện bằng DQN**
for episode in range(num_episodes):
    observation, info = env.reset()
    state = get_state(observation)
    total_reward = 0
    done = False

    for _ in range(200):  
        valid_actions = get_valid_actions(observation)
        if not valid_actions:
            break

        action = get_action(state, valid_actions)
        env_action = {"stock_idx": action[0], "size": observation["products"][action[1]]["size"], "position": (action[2], action[3])}

        observation, reward, terminated, truncated, info = env.step(env_action)
        next_state = get_state(observation)
        reward = get_reward(observation, info)

        memory.append((state, action, reward, next_state, terminated))
        state = next_state
        total_reward += reward
        done = terminated or truncated

        if done:
            break

    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    if len(memory) > batch_size:
        batch = random.sample(memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)

        Q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_Q_values = target_net(next_states).max(1)[0]
        target_Q_values = rewards + (gamma * next_Q_values * (1 - dones))

        loss = loss_fn(Q_values, target_Q_values.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if episode % update_target_frequency == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if episode % 100 == 0:
        print(f"Episode {episode}, Reward: {total_reward:.4f}, Epsilon: {epsilon:.4f}")

env.close()
