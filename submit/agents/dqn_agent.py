# agents/dqn_agent.py

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_dim, num_stocks, num_products, max_w, max_h,
                 lr=0.001, gamma=0.95, epsilon=1.0, epsilon_decay=0.995,
                 epsilon_min=0.01, batch_size=64, memory_size=10000):
        # Set device: use GPU if available.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.state_dim = state_dim  # For simplicity, we use the product vector (remaining quantities) as the state.
        self.num_stocks = num_stocks
        self.num_products = num_products
        self.max_w = max_w
        self.max_h = max_h
        self.total_positions = max_w * max_h
        self.total_actions = num_stocks * num_products * self.total_positions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        
        self.memory = deque(maxlen=memory_size)
        
        # Initialize networks and move them to the device.
        self.policy_net = DQN(state_dim, self.total_actions).to(self.device)
        self.target_net = DQN(state_dim, self.total_actions).to(self.device)
        self.update_target()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def state_to_tensor(self, state):
        """
        Convert the state (here, we use the product vector, i.e. the remaining quantities)
        into a PyTorch tensor and move it to the device.
        """
        product_vector = state["products"]
        tensor = torch.FloatTensor(product_vector).unsqueeze(0).to(self.device)
        return tensor
    
    def action_to_index(self, action):
        """
        Map the composite action (a dict with keys "stock_idx", "product_idx", and "position")
        to a unique integer index.
        """
        stock = action["stock_idx"]
        product = action["product_idx"]
        x, y = action["position"]
        pos_index = int(x) * self.max_h + int(y)
        index = stock * (self.num_products * self.total_positions) + product * self.total_positions + pos_index
        return index
    
    def index_to_action(self, index):
        """
        Convert an integer index back to a composite action.
        """
        total_positions = self.total_positions
        prod_actions = self.num_products * total_positions
        stock = index // prod_actions
        remainder = index % prod_actions
        product = remainder // total_positions
        pos_index = remainder % total_positions
        x = pos_index // self.max_h
        y = pos_index % self.max_h
        return {"stock_idx": stock, "product_idx": product, "position": np.array([x, y])}
    
    def choose_action(self, state):
        """
        Epsilon-greedy action selection.
        """
        state_tensor = self.state_to_tensor(state)
        if np.random.rand() < self.epsilon:
            action_index = np.random.randint(self.total_actions)
        else:
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                action_index = int(torch.argmax(q_values, dim=1).item())
        return self.index_to_action(action_index)
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store the experience tuple in the replay memory.
        """
        self.memory.append((state, action, reward, next_state, done))
        
    def train_step(self):
        """
        Sample a minibatch from replay memory and perform a gradient descent step on the loss.
        """
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        state_tensors = torch.cat([self.state_to_tensor(s) for s in states], dim=0)
        next_state_tensors = torch.cat([self.state_to_tensor(s) for s in next_states], dim=0)
        
        action_indices = torch.LongTensor([self.action_to_index(a) for a in actions]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Compute current Q values using the policy network.
        current_q = self.policy_net(state_tensors).gather(1, action_indices)
        # Compute the maximum next Q value using the target network.
        with torch.no_grad():
            max_next_q = self.target_net(next_state_tensors).max(1)[0].unsqueeze(1)
        target_q = rewards + self.gamma * max_next_q * (1 - dones)
        
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
