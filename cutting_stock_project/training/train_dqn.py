# training/train_dqn.py
import gymnasium as gym
from env.CuttingStockEnvOptimized import CuttingStockEnvOptimized
from agents.dqn_agent import DQNAgent
import numpy as np

def main():
    env = CuttingStockEnvOptimized(render_mode="human")
    num_stocks = env.num_stocks
    num_products = env.max_product_type
    max_w = env.max_w
    max_h = env.max_h
    # State dimension is the product vector (length = num_products)
    state_dim = num_products

    agent = DQNAgent(state_dim, num_stocks, num_products, max_w, max_h,
                     lr=0.001, gamma=0.95, epsilon=1.0, epsilon_decay=0.995,
                     epsilon_min=0.01, batch_size=64, memory_size=10000)
    
    num_episodes = 500
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.train_step()
            state = next_state
            total_reward += reward
        if episode % 10 == 0:
            agent.update_target()
        print(f"Episode {episode}: Total Reward = {total_reward}")
    
    env.close()

if __name__ == "__main__":
    main()
