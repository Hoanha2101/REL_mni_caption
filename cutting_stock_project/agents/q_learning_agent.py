# agents/q_learning_agent.py

import numpy as np
import torch

class QLearningAgent:
    """
    A Q-Learning agent for the cutting stock environment.

    This agent uses a Q-table stored as a dictionary mapping state representations
    (as strings) to Q-value tensors. The agent employs an epsilon-greedy strategy
    for action selection.

    Attributes:
        num_stocks (int): Number of stocks.
        num_products (int): Number of product types.
        max_w (int): Maximum width of the stock grid.
        max_h (int): Maximum height of the stock grid.
        total_positions (int): Total possible positions in the grid.
        total_actions (int): Total number of discrete actions.
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        epsilon (float): Exploration rate.
        epsilon_decay (float): Decay factor for epsilon.
        epsilon_min (float): Minimum exploration rate.
        device (torch.device): Device for torch tensors (CPU or GPU).
        q_table (dict): Dictionary mapping state keys to Q-value tensors.
    """

    def __init__(self, num_stocks, num_products, max_w, max_h,
                 learning_rate=0.1, discount_factor=0.95,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """
        Initialize the QLearningAgent.

        Args:
            num_stocks (int): Number of stocks.
            num_products (int): Number of product types.
            max_w (int): Maximum width of the stock grid.
            max_h (int): Maximum height of the stock grid.
            learning_rate (float, optional): Learning rate (alpha). Defaults to 0.1.
            discount_factor (float, optional): Discount factor (gamma). Defaults to 0.95.
            epsilon (float, optional): Initial exploration rate. Defaults to 1.0.
            epsilon_decay (float, optional): Decay factor for epsilon. Defaults to 0.995.
            epsilon_min (float, optional): Minimum exploration rate. Defaults to 0.01.
        """
        self.num_stocks = num_stocks
        self.num_products = num_products
        self.max_w = max_w
        self.max_h = max_h
        self.total_positions = max_w * max_h
        # Total actions = number of stocks * (number of products * total_positions)
        self.total_actions = num_stocks * num_products * self.total_positions

        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Set device for GPU acceleration.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Q-table: key is a string (state representation) and value is a torch tensor of shape (total_actions,)
        self.q_table = {}

    def action_to_index(self, action):
        """
        Convert a composite action (dict) to a unique action index.

        Args:
            action (dict): Contains "stock_idx", "product_idx", and "position".

        Returns:
            int: Unique index corresponding to the composite action.
        """
        stock = action["stock_idx"]
        product = action["product_idx"]
        x, y = action["position"]
        pos_index = int(x) * self.max_h + int(y)
        index = stock * (self.num_products * self.total_positions) + product * self.total_positions + pos_index
        return index

    def index_to_action(self, index):
        """
        Convert a unique action index back to a composite action dictionary.

        Args:
            index (int): The unique action index.

        Returns:
            dict: A dictionary with "stock_idx", "product_idx", and "position".
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

    def state_to_key(self, state):
        """
        Convert a state to a unique key for the Q-table.

        Here, we use the product vector (remaining quantities) as the state key.
        A more complex representation can be used if needed.

        Args:
            state (dict): The state containing "stocks" and "products".

        Returns:
            str: A string representation of the state.
        """
        return str(state["products"])

    def choose_action(self, state):
        """
        Choose an action for the given state using an epsilon-greedy strategy.

        Args:
            state (dict): The current state.

        Returns:
            dict: The selected action as a composite action dictionary.
        """
        key = self.state_to_key(state)
        # Initialize Q-values for unseen state as a torch tensor on the device.
        if key not in self.q_table:
            self.q_table[key] = torch.zeros(self.total_actions, device=self.device)
        # Epsilon-greedy strategy.
        if np.random.rand() < self.epsilon:
            action_index = np.random.randint(self.total_actions)
        else:
            action_index = int(torch.argmax(self.q_table[key]).item())
        return self.index_to_action(action_index)

    def update(self, state, action, reward, next_state, done):
        """
        Update the Q-table based on the transition.

        Q(s,a) ← Q(s,a) + α × [reward + γ × max Q(s',a') - Q(s,a)]

        Args:
            state (dict): The current state.
            action (dict): The action taken.
            reward (float): The reward received.
            next_state (dict): The next state.
            done (bool): Whether the episode has ended.
        """
        key = self.state_to_key(state)
        next_key = self.state_to_key(next_state)
        # Ensure the state keys exist.
        if key not in self.q_table:
            self.q_table[key] = torch.zeros(self.total_actions, device=self.device)
        if next_key not in self.q_table:
            self.q_table[next_key] = torch.zeros(self.total_actions, device=self.device)
        action_index = self.action_to_index(action)
        best_next = torch.max(self.q_table[next_key]).item()
        td_target = reward + self.gamma * best_next * (not done)
        td_error = td_target - self.q_table[key][action_index].item()
        # Update Q-value.
        self.q_table[key][action_index] += self.alpha * td_error
        
        # Decay epsilon when the episode ends.
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

#Q(s,a)←Q(s,a)+α×[reward+γ×maxQ(s',a')-Q(s,a)] 
