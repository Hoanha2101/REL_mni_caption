import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from env.cutting_stock import CuttingStockEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Danh sách stocks (width, height) - Tấm nguyên liệu có kích thước nhỏ, tối đa 200x200
stocks = [
    (50, 50),   (60, 40),   (70, 50),   (80, 60),   (90, 70),
    (100, 50),  (110, 60),  (120, 80),  (130, 90),  (140, 100),
    (150, 120), (160, 130), (170, 140), (180, 150), (200, 200)
]

# Danh sách products (width, height) - Sản phẩm có kích thước nhỏ, phù hợp với stocks
products = [
    (10, 5),  (15, 10), (20, 10), (25, 15), (30, 20),
    (35, 20), (40, 30), (45, 25), (50, 30), (55, 35),
    (20, 15), (25, 10), (30, 15), (35, 20), (40, 25),
    (45, 30), (50, 35), (55, 40), (60, 45), (65, 50),
    (70, 30), (75, 40), (80, 50), (85, 55), (90, 60),
    (15, 10), (20, 15), (25, 20), (30, 25), (35, 30)
]

env = CuttingStockEnv(render_mode="human", max_w=250, max_h=250, seed=42, stock_list=stocks, product_list=products)

# Hyperparameters
alpha = 0.1 
gamma = 0.9  
epsilon = 1.0  
epsilon_decay = 0.995  
min_epsilon = 0.01 
num_episodes = 100  
update_targetnn_rate = 10
BATCH_SIZE = 32
LR = 1e-4

state_size = 3  # [empty_space, remaining_products, unused_stocks]
action_size = 1000  # Giả sử có 10 hành động (cắt sản phẩm từ stock)

# DQN Model
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(n_observations, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

policy_net = DQN(state_size, action_size).to(device)
target_net = DQN(state_size, action_size).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
replay_buffer = deque(maxlen=10000)

def get_state(observation):
    """
    Chuyển trạng thái từ môi trường thành vector tensor để dùng cho DQN.
    """
    stocks = observation["stocks"]
    products = observation["products"]
    
    empty_space = sum(np.sum(stock == -1) for stock in stocks)
    remaining_products = sum(prod["quantity"] for prod in products)
    unused_stocks = sum(1 for stock in stocks if np.all(stock == -1))
    
    state_vector = torch.tensor([empty_space, remaining_products, unused_stocks], dtype=torch.float32, device=device)
    return state_vector

def get_action(state):
    """
    Chọn hành động theo chính sách epsilon-greedy.
    """
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, action_size - 1)
    else:
        with torch.no_grad():
            return policy_net(state.unsqueeze(0)).argmax(dim=1).item()

def get_env_action(action, observation):
    """
    Chuyển action từ Q-table thành action thực tế cho môi trường Gym.
    """
    list_prods = observation["products"]
    list_stocks = observation["stocks"]

    if not list_prods or not list_stocks:
        return {"stock_idx": 0, "size": (0, 0), "position": (0, 0)}

    # Chọn sản phẩm có thể cắt
    prod_idx = action % len(list_prods)
    prod = list_prods[prod_idx]

    if prod["quantity"] == 0:
        return {"stock_idx": 0, "size": (0, 0), "position": (0, 0)}

    prod_w, prod_h = prod["size"]

    # Chọn stock phù hợp
    suitable_stocks = [i for i, stock in enumerate(list_stocks) if np.sum(stock == -1) >= prod_w * prod_h]
    stock_idx = suitable_stocks[0] if suitable_stocks else 0

    stock = list_stocks[stock_idx]
    stock_w = np.sum(np.any(stock != -2, axis=1))
    stock_h = np.sum(np.any(stock != -2, axis=0))

    # Chọn vị trí trong stock
    for x in range(stock_w - prod_w + 1):
        for y in range(stock_h - prod_h + 1):
            if np.all(stock[x:x + prod_w, y:y + prod_h] == -1):
                return {"stock_idx": stock_idx, "size": (prod_w, prod_h), "position": (x, y)}

    return {"stock_idx": 0, "size": (0, 0), "position": (0, 0)}

for episode in range(num_episodes):
    print("episode:", episode)
    observation, _ = env.reset(seed=42)
    state = get_state(observation)
    
    while True:
        action = get_action(state)
        env_action = get_env_action(action, observation)
        observation, reward, terminated, truncated, _ = env.step(env_action)
        next_state = get_state(observation)
        
        replay_buffer.append((state, action, reward, next_state, terminated))
        
        if len(replay_buffer) > BATCH_SIZE:
            minibatch = random.sample(replay_buffer, BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*minibatch)
            
            states = torch.stack(states).to(device)
            actions = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
            next_states = torch.stack(next_states).to(device)
            dones = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)
            
            q_values = policy_net(states).gather(1, actions)
            with torch.no_grad():
                max_next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
                target_q_values = rewards + (gamma * max_next_q_values * (1 - dones))
            
            loss = F.mse_loss(q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if terminated:
            break
        state = next_state
    
    # Giảm epsilon sau mỗi episode
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    
    if episode % update_targetnn_rate == 0:
        target_net.load_state_dict(policy_net.state_dict())

print("Training Completed!")
