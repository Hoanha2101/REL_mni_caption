
# from env.cutting_stock import CuttingStockEnv
# import random
# import numpy as np
# from itertools import product
# import pandas as pd
# import os

# # Danh sách stocks(width, height)
# stocks = [
#     (100, 100),  
#     (80, 60),       
#     (90, 50),    
#     (120, 80),       
# ]

# # Danh sách products(width, height)
# products = [
#     (20, 10),  
#     (20, 10),  
#     (30, 20),  
#     (40, 30),  
#     (50, 25),  
# ]

# # Sắp xếp theo diện tích giảm dần
# stocks = sorted(stocks, key=lambda x: x[0] * x[1], reverse=True)
# products = sorted(products, key=lambda x: x[0] * x[1], reverse=True)

# c_no_of_eps = 5000
# v_epsilon = 0.9
# c_learning_rate = 0.1  # Tốc độ học (alpha)
# c_discount_value = 0.9  # Hệ số chiết khấu (gamma)

# c_start_ep_epsilon_decay = 1  # Episode bắt đầu giảm epsilon
# c_end_ep_epsilon_decay = c_no_of_eps // 2  # Episode kết thúc giảm epsilon
# v_epsilon_decay = v_epsilon / (c_end_ep_epsilon_decay - c_start_ep_epsilon_decay)  # Tốc độ giảm epsilon

# max_ep_reward = 0
# max_ep_action_list = []
# max_start_state = None

# w_max_product = 0
# h_max_product = 0
# w_max_stock = 0
# h_max_stock = 0

# for item_product in products:
#     if item_product[0] > w_max_product:
#         w_max_product = item_product[0]
        
#     if item_product[1] > h_max_product:
#         h_max_product = item_product[1]
    
# for item_stock in stocks:
#     if item_stock[0] > w_max_stock:
#         w_max_stock = item_stock[0]
        
#     if item_stock[1] > h_max_stock:
#         h_max_stock = item_stock[1]

# print(f"w_max_stock = {w_max_stock}, h_max_stock = {h_max_stock}")
# print(f"w_max_product = {w_max_product}, h_max_product = {h_max_product}")


# ############ State note ##############
# #-----distance--------Note------------
# #  0 < x <= 25%   |     0
# #  25 < x <= 50%  |     1
# #  50 < x <= 75%  |     2
# #  75 < x <= 100% |     3

# RANGE = [range(0, 4) for i in range(len(stocks))]

# state_title_list = list(product(*RANGE))

# action_title_list = []

# for stock_id, stock_value in enumerate(stocks):
#     for product_id, product_value in enumerate(products):
#         # Duyệt qua các giá trị x, y có thể từ 0 đến w_max_product, h_max_product
#         for x_id in range((w_max_stock) + 1):
#             for y_id in range((h_max_stock) + 1):
#                 action_key = f"{stock_id}-{product_id}-{x_id}-{y_id}"
#                 action_title_list.append((stock_id, product_id, x_id,y_id))
                
# # Q-table: key là (state_key, action) và value là Q-value
# Q_table = np.random.uniform(low=-2, high=0, size=(len(state_title_list) , len(action_title_list)))

# print("Shape of Q_table: ", Q_table.shape)

# env = CuttingStockEnv(
#     render_mode="human",   # hoặc None nếu không cần hiển thị
#     max_w=120,             # Giá trị max_w, max_h nên lớn hơn hoặc bằng kích thước của stocks
#     max_h=120,
#     seed=42,
#     stock_list=stocks,
#     product_list=products,
# )

# def get_valid_actions(state, env):
#     """
#     Sinh ra danh sách các hành động khả thi theo một số quy tắc:
#       - Với mỗi tấm vật liệu (stock) và với mỗi sản phẩm còn lượng > 0,
#         nếu kích thước sản phẩm có thể vừa với kích thước tấm vật liệu, thêm một vài lựa chọn vị trí.
#     Mỗi hành động được biểu diễn dưới dạng tuple:
#       (stock_idx, product_idx, x_ngẫu nhiên, y_ngẫu nhiên)  -------------> Chúng ta sẽ phạt hay thưởng trong hàm step của env
#     """
#     actions = []
#     for stock_idx in range(env.num_stocks):
#         stock = env._stocks[stock_idx]
#         # Tính kích thước thực của tấm vật liệu (stock)
#         stock_width = int(np.sum(np.any(stock != -2, axis=1)))
#         stock_height = int(np.sum(np.any(stock != -2, axis=0)))
#         for prod_idx, product in enumerate(state["products"]):
#             if product["quantity"] <= 0:
#                 continue
#             prod_width, prod_height = product["size"]
#             # Kiểm tra nếu sản phẩm có thể vừa với tấm vật liệu
#             if prod_width <= stock_width and prod_height <= stock_height:
#                 # Lựa chọn vị trí cố định (ví dụ: góc trái)
#                 actions.append((stock_idx, prod_idx, 0, 0))
#                 # Thêm thêm một hành động với vị trí ngẫu nhiên (nếu có không gian)
#                 max_x = stock_width - prod_width
#                 max_y = stock_height - prod_height
#                 if max_x > 0 and max_y > 0:
#                     rand_x = np.random.randint(0, max_x + 1)
#                     rand_y = np.random.randint(0, max_y + 1)
#                     actions.append((stock_idx, prod_idx, rand_x, rand_y))
#     return actions

# def state_to_key(state):
#     """
#     Trả lại tuple các ghi chú cho từng stock dựa trên diện tích còn lại.
    
#     Với mỗi stock trong state["stocks"] (là ma trận biểu diễn tấm vật liệu):
#       - stock_width: số hàng trong vùng hợp lệ (các hàng có giá trị khác -2)
#       - stock_height: số cột trong vùng hợp lệ (các cột có giá trị khác -2)
#       - total_area = stock_width * stock_height
#       - free_area = số ô trong vùng này có giá trị -1
#       - fraction = free_area / total_area (nếu total_area > 0, ngược lại = 0)
    
#     Gán note theo:
#       0 <= fraction <= 0.25  -> note = 0
#       0.25 < fraction <= 0.50 -> note = 1
#       0.50 < fraction <= 0.75 -> note = 2
#       0.75 < fraction <= 1.0  -> note = 3
#     """
#     notes = []
#     for stock in state["stocks"]:
#         # Xác định kích thước thực của stock (vùng có giá trị khác -2)
#         stock_width = int(np.sum(np.any(stock != -2, axis=1)))
#         stock_height = int(np.sum(np.any(stock != -2, axis=0)))
#         total_area = stock_width * stock_height
#         if total_area == 0:
#             fraction = 0
#         else:
#             free_area = np.count_nonzero(stock[:stock_width, :stock_height] == -1)
#             fraction = free_area / total_area
#         # Gán note theo fraction:
#         if fraction <= 0.25:
#             note = 0
#         elif fraction <= 0.50:
#             note = 1
#         elif fraction <= 0.75:
#             note = 2
#         else:
#             note = 3
#         notes.append(note)
#     return tuple(notes)


# def choose_action(state_key, valid_actions, state_title_list, action_title_list, v_epsilon ,env ):
#     row_idx = 0
    
#     for idx, value in enumerate(state_title_list):
#         if value == state_key:
#             row_idx = idx
#             break
        
#     column_idx = np.argmax(Q_table[row_idx])
#     max_q_value = np.max(Q_table[row_idx])
   
#     # Quyết định dùng explore hay exploit
#     if random.random() > v_epsilon:
#         chosen_action = random.choice(valid_actions)
#         return [{
#         "stock_idx": chosen_action[0],
#         "size": env._products[chosen_action[1]]["size"],
#         "position": (chosen_action[2], chosen_action[3])
#         }, chosen_action, row_idx, column_idx]
#     else: 
#         chosen_action = action_title_list[column_idx]
#         return [{
#         "stock_idx": chosen_action[0],
#         "size": env._products[chosen_action[1]]["size"],
#         "position": (chosen_action[2], chosen_action[3])
#         },chosen_action, row_idx, column_idx]      

# env.reset()

# for ep in range(c_no_of_eps):
#     done = False
#     current_state, _ = env.reset()
    
#     # Bên trong ô chứa mảnh stock
#     # -2: Biểu thị các ô không thuộc về vùng của tấm stock, tức là phần "ngoài" giới hạn kích thước thực của tấm vật liệu.
#     # -1: Biểu thị các ô bên trong vùng của tấm stock nhưng chưa được sử dụng (chưa có sản phẩm nào cắt vào đó).

#     ep_reward = 0 # Khởi tạo phần thưởng của episode
#     ep_start_state = current_state  # Lưu trạng thái bắt đầu
#     action_list = []  # Lưu danh sách hành động của episode
    
#     action_valid = get_valid_actions(current_state, env) # ===> Trả lại list danh sách ------ [ (stock_idx, prod_idx, rand_x, rand_y),.... ]-------- khả thi.
    
#     state_key = state_to_key(current_state)

#     while not done:
#         action = choose_action(state_key, action_valid, state_title_list, action_title_list, v_epsilon, env) 
#         action_list.append(action[0])
        
#         next_state, reward, done, _, info = env.step(action[0])
#         ep_reward += reward
        
#         if done:
#             print("Đã xong ep = {}, reward = {}".format(ep, ep_reward))
#             if ep_reward > max_ep_reward:
#                 max_ep_reward = ep_reward
#                 max_ep_action_list = action_list
#                 max_start_state = ep_start_state
#         else:
#             next_state_key = state_to_key(next_state)
#             for idx_next_st, value_next_st in enumerate(state_title_list):
#                 if value_next_st == next_state_key:
#                     break
#             current_q_value = Q_table[action[2]][action[3]]
#             new_q_value = (1 - c_learning_rate) * current_q_value + c_learning_rate * (reward + c_discount_value * np.max(Q_table[idx_next_st]))
#             Q_table[action[2]][action[3]] = new_q_value
#             current_state = next_state
    
#     # Giảm dần epsilon trong khoảng xác định
#     if c_end_ep_epsilon_decay >= ep > c_start_ep_epsilon_decay:
#         v_epsilon = v_epsilon - v_epsilon_decay

# # Hiển thị kết quả tốt nhất tìm được
# print("Max reward = ", max_ep_reward)
# print("Max action list = ", max_ep_action_list)

# # Phát lại tập tốt nhất tìm được
# env.reset()
# env.state = max_start_state  # Đặt trạng thái ban đầu thành trạng thái tốt nhất
# for action in max_ep_action_list:
#     env.step(action)  # Thực hiện hành động
#     env.render()  # Hiển thị môi trường

# done = False
# while not done:
#     # _, _, done, _ = env.step(0)  #
#     env.render()


from env.cutting_stock import CuttingStockEnv
import random
import numpy as np
from itertools import product

# Danh sách stocks (width, height)
stocks = [
    (100, 100),  
    (80, 60),       
    (90, 50),    
    (120, 80),       
]

# Danh sách products (width, height)
products = [
    (20, 10),  
    (20, 10),  
    (30, 20),  
    (40, 30),  
    (50, 25),  
]

# Sắp xếp theo diện tích giảm dần
stocks = sorted(stocks, key=lambda x: x[0] * x[1], reverse=True)
products = sorted(products, key=lambda x: x[0] * x[1], reverse=True)

# Tham số Q-learning
num_episodes = 5000
learning_rate = 0.1  # Alpha
discount_factor = 0.9  # Gamma
epsilon = 1.0  # Epsilon-greedy
epsilon_decay = 0.995  # Giảm epsilon dần
min_epsilon = 0.01
max_actions_per_episode = 200

# Khởi tạo môi trường
env = CuttingStockEnv(render_mode="human", max_w=120, max_h=120, seed=42, stock_list=stocks, product_list=products)

# Xác định số lượng trạng thái & hành động
num_states = 1000  # Giới hạn số lượng trạng thái
num_actions = 500  # Giới hạn số lượng hành động

# Q-table
Q_table = np.random.uniform(low=-1, high=1, size=(num_states, num_actions))

### ✅ Hàm chuyển đổi `state` thành một số nguyên để lưu vào Q-table ###
def get_state(observation):
    stocks = observation["stocks"]
    products = observation["products"]
    
    empty_space = sum(np.sum(stock == -1) for stock in stocks)
    remaining_products = sum(prod["quantity"] for prod in products)
    num_stocks_used = sum(1 for stock in stocks if np.any(stock != -2))

    # Biểu diễn state bằng hash (giới hạn num_states)
    state = (empty_space * 1000 + remaining_products + num_stocks_used * 100) % num_states
    return state

### ✅ Hàm lấy danh sách hành động hợp lệ ###
def get_valid_actions(observation):
    actions = []
    for stock_idx, stock in enumerate(observation["stocks"]):
        stock_w = np.sum(np.any(stock != -2, axis=1))
        stock_h = np.sum(np.any(stock != -2, axis=0))

        for prod_idx, prod in enumerate(observation["products"]):
            if prod["quantity"] <= 0:
                continue
            prod_w, prod_h = prod["size"]

            if prod_w <= stock_w and prod_h <= stock_h:
                actions.append((stock_idx, prod_idx, 0, 0))  # Cắt từ góc trên
    return actions

### ✅ Hàm chọn hành động theo epsilon-greedy ###
def get_action(state, valid_actions):
    if np.random.rand() < epsilon:
        return random.choice(valid_actions)  # Chọn hành động ngẫu nhiên
    else:
        return valid_actions[np.argmax(Q_table[state][:len(valid_actions)])]  # Chọn hành động tốt nhất

### ✅ Hàm tính phần thưởng ###
def get_reward(observation, info):
    filled_ratio = info["filled_ratio"]
    trim_loss = info["trim_loss"]
    return filled_ratio - trim_loss  # Khuyến khích giảm lãng phí vật liệu

### ✅ Huấn luyện bằng Q-learning ###
for episode in range(num_episodes):
    observation, info = env.reset()
    state = get_state(observation)
    total_reward = 0
    done = False

    for _ in range(max_actions_per_episode):
        valid_actions = get_valid_actions(observation)
        if not valid_actions:
            break

        action = get_action(state, valid_actions)
        env_action = {"stock_idx": action[0], "size": observation["products"][action[1]]["size"], "position": (action[2], action[3])}

        observation, reward, terminated, truncated, info = env.step(env_action)
        next_state = get_state(observation)
        reward = get_reward(observation, info)

        # Cập nhật Q-table
        best_next_action = np.max(Q_table[next_state]) if next_state in range(num_states) else 0
        Q_table[state][valid_actions.index(action)] = (1 - learning_rate) * Q_table[state][valid_actions.index(action)] + learning_rate * (reward + discount_factor * best_next_action)

        state = next_state
        total_reward += reward
        done = terminated or truncated

        if done:
            break

    # Giảm epsilon để chuyển từ khám phá sang khai thác
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    if episode % 100 == 0:
        print(f"Episode {episode}, Reward: {total_reward:.4f}, Epsilon: {epsilon:.4f}")

env.close()
