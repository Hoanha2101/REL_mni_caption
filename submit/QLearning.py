from env.cutting_stock import CuttingStockEnv
import numpy as np
import random
import pickle

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
    render_mode="human",   
    max_w=120,           
    max_h=120,
    seed=42,
    stock_list=stocks,
    product_list=products,
)


alpha = 0.1 
gamma = 0.9  
epsilon = 1.0  
epsilon_decay = 0.995  
min_epsilon = 0.01 
num_episodes = 10  # Số tập huấn luyện

# Kích thước Q-table
# Bởi vì kích thước của Q -Table quá lớn, khi dàn trải action và state,
# ===> Sử dụng các phương pháp Normalization Q-table để dạt được khoảng lưu trữ tối ưu
# trong cho bộ nhớ, đồng tời, vẫn đáp ứng được đầy đủ thông tin cần cho Q-table
 

state_size = 100000  
action_size = 1000  
Q_table = np.zeros((state_size, action_size))


def get_state(observation):
    """
    Chuyển trạng thái từ môi trường thành dạng số nguyên để lưu vào Q-table.
    """
    stocks = observation["stocks"]
    products = observation["products"]
    
    # Lấy tổng diện tích còn trống của tất cả stock
    empty_space = sum(np.sum(stock == -1) for stock in stocks)
    
    # Tổng số sản phẩm chưa cắt
    remaining_products = sum(prod["quantity"] for prod in products)

    state = (empty_space * 1000 + remaining_products) % state_size
    
    # Tạo một số nguyên state bằng cách kết hợp:
    # empty_space * 1000: Làm cho các trạng thái có diện tích khác nhau cách xa nhau trong Q-table.
    # + remaining_products: Thêm thông tin về số lượng sản phẩm chưa cắt.
    # % state_size: Đảm bảo state không vượt quá kích thước Q-table (state_size).
    
    return state


def get_action(state):
    """
    Chọn hành động sử dụng epsilon-greedy.
    """
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, action_size - 1)
    else:
        return np.argmax(Q_table[state])


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

    # Chọn stock
    stock_idx = (action // len(list_prods)) % len(list_stocks)
    stock = list_stocks[stock_idx]

    stock_w = np.sum(np.any(stock != -2, axis=1))
    stock_h = np.sum(np.any(stock != -2, axis=0))

    # Chọn vị trí trong stock
    for x in range(stock_w - prod_w + 1):
        for y in range(stock_h - prod_h + 1):
            if np.all(stock[x:x + prod_w, y:y + prod_h] == -1):
                return {"stock_idx": stock_idx, "size": (prod_w, prod_h), "position": (x, y)}

    return {"stock_idx": 0, "size": (0, 0), "position": (0, 0)}


def get_reward(observation, info):
    """
    Tính toán phần thưởng (reward) cho agent dựa trên:
    1. filled_ratio: Tỷ lệ diện tích đã được sử dụng trong tất cả các stock.
    2. trim_loss: Phần diện tích bị lãng phí sau khi cắt.
    3. Số stock chưa bị cắt (càng nhiều càng tốt).
    """

    filled_ratio = info["filled_ratio"] 
    trim_loss = info["trim_loss"]  

    # Tính số lượng stock đã sử dụng
    # Tổng số stock có trong môi trường
    # Tính số stock còn nguyên (chưa bị cắt)
    num_stocks_used = sum(1 for stock in observation["stocks"] if np.any(stock != -2))
    total_stocks = len(observation["stocks"])
    num_stocks_unused = total_stocks - num_stocks_used

    lambda_bonus = 0.2  # Hệ số điều chỉnh mức độ thưởng
    stock_bonus = lambda_bonus * (num_stocks_unused / total_stocks)  # Thưởng theo tỷ lệ stock chưa cắt

    # Tính tổng phần thưởng
    reward = (filled_ratio - trim_loss) + stock_bonus

    return reward


# Biến theo dõi phần thưởng cao nhất đạt được
max_ep_reward = -999  # Giá trị phần thưởng lớn nhất tìm thấy
max_ep_action_list = []  # Danh sách hành động tương ứng với phần thưởng cao nhất
max_start_state = None  # Trạng thái bắt đầu tương ứng với phần thưởng cao nhất


# Train -------------------------------------
for episode in range(num_episodes):
    observation, info = env.reset(seed=42)
    state = get_state(observation)
    
    ep_reward = 0  # Khởi tạo phần thưởng của episode
    ep_start_state = state  # Lưu trạng thái bắt đầu
    action_list = []

    done = False

    while not done:
        action = get_action(state)
        env_action = get_env_action(action, observation)
        
        observation, reward, terminated, truncated, info = env.step(env_action)
        
        done = terminated
        
        if done:
            if ep_reward > max_ep_reward:
                max_ep_reward = ep_reward
                max_ep_action_list = action_list
                max_start_state = ep_start_state
        
        action_list.append(env_action)
        ep_reward += reward

        next_state = get_state(observation)
        reward = get_reward(observation, info)

        # Cập nhật Q-table
        Q_table[state, action] = (1 - alpha) * Q_table[state, action] + alpha * (
            reward + gamma * np.max(Q_table[next_state])
        )

        state = next_state
        

    # Update epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)


    print(f"Episode {episode}, Epsilon: {epsilon:.4f}")
        
# Hiển thị kết quả tốt nhất tìm được
print("Max reward = ", max_ep_reward)
print("Max action list = ", max_ep_action_list)



# # Lưu Q-table để sử dụng sau này
# with open("q_table.pkl", "wb") as f:
#     pickle.dump(Q_table, f)

# env.close()

        
# Phát lại tập tốt nhất tìm được
env.reset()
env.state = max_start_state  # Đặt trạng thái ban đầu thành trạng thái tốt nhất
for action in max_ep_action_list:
    env.step(action)  # Thực hiện hành động
    env.render()  # Hiển thị môi trường

done = False
while not done:
    # _, _, done, _ = env.step(0)  #
    env.render()