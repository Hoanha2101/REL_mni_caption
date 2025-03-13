import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import namedtuple, deque
import numpy as np

# Define a named tuple to store transitions in the replay buffer.
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    """
    Experience Replay Memory for storing transitions.
    
    This memory holds a fixed number of transitions (state, action, reward, next_state, done)
    and allows for random sampling of batches during training.
    """
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, *args):
        """Store a transition."""
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        """Return a random sample of transitions."""
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class DeepQNetwork(nn.Module):
    """
    A Deep Q-Network that approximates the Q-function.
    
    This network is a simple feed-forward neural network that takes a flattened
    state vector as input and outputs Q-values for each discrete action.
    """
    def __init__(self, input_dim, output_dim):
        super(DeepQNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

def state_to_tensor(state, device):
    """
    Convert the environment state into a flattened torch tensor.
    
    The state is a dictionary with two keys:
      - "stocks": a tuple of 2D numpy arrays representing each stock grid.
      - "products": a 1D numpy array representing the remaining quantity for each product type.
    
    This function flattens all stock grids and concatenates them with the products vector.
    
    Args:
        state (dict): The environment state.
        device (torch.device): The device on which to allocate the tensor.
    
    Returns:
        torch.Tensor: A 1D tensor representing the state.
    """
    stocks = state["stocks"]
    products = state["products"]
    # Flatten each stock grid and then concatenate them.
    stocks_flat = np.concatenate([s.flatten() for s in stocks])
    state_vec = np.concatenate([stocks_flat, products])
    return torch.tensor(state_vec, dtype=torch.float32, device=device)

class DeepQLearningAgent:
    """
    Deep Q-Learning agent that uses a neural network to approximate Q-values.
    
    The agent employs an epsilon-greedy policy for action selection, an experience replay
    buffer for stable training, and a target network that is periodically updated.
    """
    def __init__(self, env, device):
        """
        Initialize the Deep Q-Learning Agent.
        
        Args:
            env: An instance of the environment.
            device (torch.device): The device to run computations on.
        """
        self.env = env
        self.device = device
        self.num_stocks = env.num_stocks
        self.max_w = env.max_w
        self.max_h = env.max_h
        self.max_product_type = env.max_product_type
        
        # Compute the input dimension: all stock grids flattened plus the products vector.
        self.input_dim = self.num_stocks * (self.max_w * self.max_h) + self.max_product_type
        
        # Compute the output dimension:
        # total_actions = num_stocks * max_product_type * (max_w * max_h)
        self.output_dim = self.num_stocks * self.max_product_type * (self.max_w * self.max_h)
        
        self.policy_net = DeepQNetwork(self.input_dim, self.output_dim).to(device)
        self.target_net = DeepQNetwork(self.input_dim, self.output_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.memory = ReplayMemory(10000)
        self.steps_done = 0
        
        # Epsilon-greedy parameters.
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95

    def select_action(self, state):
        """
        Select an action using an epsilon-greedy strategy.
        
        Args:
            state (torch.Tensor): The current state tensor.
        
        Returns:
            int: The selected action index.
        """
        if random.random() < self.epsilon:
            return random.randrange(self.output_dim)
        else:
            with torch.no_grad():
                q_values = self.policy_net(state.unsqueeze(0))
                return q_values.argmax(dim=1).item()
    
    def optimize_model(self, batch_size):
        """
        Sample a batch of transitions and perform a gradient descent step.
        
        Args:
            batch_size (int): The number of transitions to sample.
        """
        if len(self.memory) < batch_size:
            return
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))
        
        state_batch = torch.stack([state_to_tensor(s, self.device) for s in batch.state])
        action_batch = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_state_batch = torch.stack([state_to_tensor(s, self.device) for s in batch.next_state])
        done_batch = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)
        
        # Compute Q(s, a) from the policy network.
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute target Q-values using the target network.
        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
            next_state_values = next_state_values * (1 - done_batch)
        expected_state_action_values = reward_batch + self.gamma * next_state_values
        
        loss = nn.MSELoss()(state_action_values, expected_state_action_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_epsilon(self):
        """
        Decay the exploration rate epsilon.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
