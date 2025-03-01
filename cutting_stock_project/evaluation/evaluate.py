import gymnasium as gym
from env.CuttingStockEnvOptimized import CuttingStockEnvOptimized
from agents.q_learning_agent import QLearningAgent
import pickle
import sys

def evaluate_agent(num_episodes=10):
    """
    Evaluate the Q-learning agent for a specified number of episodes,
    printing only the final reward of each episode and the average reward.
    
    Args:
        num_episodes (int): Number of evaluation episodes to run.
    """
    # Initialize the environment (render_mode can be "human" or "rgb_array")
    env = CuttingStockEnvOptimized(render_mode="human")
    num_stocks = env.num_stocks
    num_products = env.max_product_type  # using max_product_type as number of product types
    max_w = env.max_w
    max_h = env.max_h

    # Initialize the agent with epsilon=0 to always select the best action.
    agent = QLearningAgent(num_stocks, num_products, max_w, max_h,
                           learning_rate=0.1, discount_factor=0.95,
                           epsilon=0.0, epsilon_decay=0.995, epsilon_min=0.01)

    # Load the trained Q-table from the checkpoint file.
    checkpoint_file = "checkpoints/Case_1(1,25,20)/q_table_checkpoint_ep200.pkl"
    try:
        with open(checkpoint_file, "rb") as f:
            agent.q_table = pickle.load(f)
    except Exception as e:
        print(f"Error loading Q-table checkpoint: {e}")
        sys.exit(1)

    total_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            state = next_state
            episode_reward += reward
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Total Reward = {episode_reward}")

    avg_reward = sum(total_rewards) / len(total_rewards)
    print(f"Average Reward over {num_episodes} episodes: {avg_reward}")

    env.close()

if __name__ == "__main__":
    evaluate_agent(num_episodes=10)
