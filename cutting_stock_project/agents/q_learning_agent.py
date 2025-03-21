import numpy as np
import random

class QLearningAgent:
    """
    Q-Learning Agent for the Cutting Stock problem.

    This agent learns to optimize the cutting of stock materials by interacting with the environment,
    updating its Q-table, and selecting actions based on the Q-learning algorithm.

    Args:
        state_size (int, optional): The number of possible states (default is 100000).
        action_size (int, optional): The number of possible actions (default is 1000).
        alpha (float, optional): Learning rate (default is 0.1).
        gamma (float, optional): Discount factor for future rewards (default is 0.9).
        epsilon (float, optional): Exploration rate for epsilon-greedy strategy (default is 1.0).
        epsilon_decay (float, optional): Decay factor for epsilon (default is 0.995).
        min_epsilon (float, optional): Minimum epsilon value (default is 0.01).
    """

    def __init__(self, state_size=100000, action_size=1000, alpha=0.1, gamma=0.9,
                 epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.Q_table = np.zeros((state_size, action_size))

    def get_state(self, observation):
        """
        Computes the state representation from the given observation.

        The state is determined based on:
        - The total empty space in all stocks.
        - The total number of remaining products.

        Args:
            observation (dict): The observation from the environment, containing stocks and products.

        Returns:
            int: The computed state index.
        """
        empty_space = sum(np.sum(stock == -1) for stock in observation["stocks"])
        remaining_products = sum(prod["quantity"] for prod in observation["products"])
        state = (empty_space * 1000 + remaining_products) % self.state_size
        return int(state)

    def get_action(self, state):
        """
        Selects an action based on the epsilon-greedy strategy.

        Args:
            state (int): The current state index.

        Returns:
            int: The selected action index.
        """
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_size - 1)  # Exploration
        else:
            return int(np.argmax(self.Q_table[state]))  # Exploitation

    def get_env_action(self, action, observation):
        """
        Converts a Q-table action index into an environment-compatible action.

        The action selection process:
        - Selects a product: action % number of available products.
        - Selects a stock: (action // number of products) % number of available stocks.
        - Finds a valid position within the selected stock.

        Args:
            action (int): The action index from the Q-table.
            observation (dict): The current environment observation.

        Returns:
            dict: The environment action in the format:
                {
                    "stock_idx": int,
                    "size": (width, height),
                    "position": (x, y)
                }
        """
        list_prods = observation["products"]
        list_stocks = observation["stocks"]

        if not list_prods or not list_stocks:
            return {"stock_idx": 0, "size": (0, 0), "position": (0, 0)}

        # Select product based on action
        prod_idx = action % len(list_prods)
        prod = list_prods[prod_idx]

        # If the selected product is unavailable, return a dummy action
        if prod["quantity"] == 0:
            return {"stock_idx": 0, "size": (0, 0), "position": (0, 0)}
        
        prod_w, prod_h = prod["size"]

        # Select stock based on action
        stock_idx = (action // len(list_prods)) % len(list_stocks)
        stock = list_stocks[stock_idx]
        stock_w = int(np.sum(np.any(stock != -2, axis=1)))
        stock_h = int(np.sum(np.any(stock != -2, axis=0)))

        # Find a valid placement position within the selected stock
        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if np.all(stock[x:x+prod_w, y:y+prod_h] == -1):
                    return {"stock_idx": stock_idx, "size": (prod_w, prod_h), "position": (x, y)}

        # Return a dummy action if no valid placement is found
        return {"stock_idx": 0, "size": (0, 0), "position": (0, 0)}

    def update(self, state, action, reward, next_state):
        """
        Updates the Q-table using the Q-learning formula.

        Q(s, a) = (1 - alpha) * Q(s, a) + alpha * (reward + gamma * max(Q(s', a')))

        Args:
            state (int): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (int): The next state after taking the action.
        """
        best_next = np.max(self.Q_table[next_state])  # Get the best possible next Q-value
        self.Q_table[state, action] = (1 - self.alpha) * self.Q_table[state, action] + \
                                      self.alpha * (reward + self.gamma * best_next)

        # Decay epsilon to reduce exploration over time
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
