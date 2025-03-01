import gymnasium as gym
from env.CuttingStockEnvOptimized import CuttingStockEnvOptimized
from agents.q_learning_agent import QLearningAgent
import numpy as np
import signal
import pickle
import sys

# Declare agent as a global variable so that the signal handler can access it.
agent = None

def save_q_table_handler(signum, frame):
    global agent
    try:
        with open("q_table_checkpoint.pkl", "wb") as f:
            pickle.dump(agent.q_table, f)
        print("\nQ-table checkpoint saved (triggered by signal).")
    except Exception as e:
        print(f"\n[ERROR] Saving Q-table failed: {e}")
    # Uncomment the next line if you want to exit after saving:
    sys.exit(0)

# Register the signal handler for SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, save_q_table_handler)

def main():
    global agent
    env = CuttingStockEnvOptimized(render_mode="human")
    num_stocks = env.num_stocks
    num_products = env.max_product_type  # using max_product_type as number of product types
    max_w = env.max_w
    max_h = env.max_h

    agent = QLearningAgent(num_stocks, num_products, max_w, max_h,
                           learning_rate=0.1, discount_factor=0.95,
                           epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01)
    
    num_episodes = 500
    checkpoint_interval = 50

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        print(f"Episode {episode}: Total Reward = {total_reward}")
        
        # Save checkpoint every checkpoint_interval episodes.
        if (episode + 1) % checkpoint_interval == 0:
            checkpoint_filename = f"q_table_checkpoint_ep{episode+1}.pkl"
            try:
                with open(checkpoint_filename, "wb") as f:
                    pickle.dump(agent.q_table, f)
                print(f"Checkpoint saved at episode {episode+1} to '{checkpoint_filename}'")
            except Exception as e:
                print(f"Error saving checkpoint at episode {episode+1}: {e}")
    
    env.close()

if __name__ == "__main__":
    main()
