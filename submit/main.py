from policy import FirstFit_Policy, BestFit_Policy, Combination_Policy
from env.cutting_stock import CuttingStockEnv

# Danh sách stocks(width, height)
stocks = [
    (100, 100),  
    (80, 60),       
    (90, 50),    
    (120, 80),       
]

# Danh sách products(width, height)
products = [
    (20, 10),  
    (20, 10),  
    (30, 20),  
    (40, 30),  
    (50, 25),  
]

env = CuttingStockEnv(
    render_mode="human",   # hoặc None nếu không cần hiển thị
    max_w=120,             # Giá trị max_w, max_h nên lớn hơn hoặc bằng kích thước của stocks
    max_h=120,
    seed=42,
    stock_list=stocks,
    product_list=products,
)

if __name__ == "__main__":
    observation, info = env.reset()
    for _ in range(200):
        # action = FirstFit_Policy.first_fit_policy(observation, info)
        # action = BestFit_Policy.best_fit_policy(observation, info)
        action = Combination_Policy.combination_policy(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()
