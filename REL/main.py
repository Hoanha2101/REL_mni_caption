
from env.cutting_stock import CuttingStockEnv
import random
import numpy as np
import itertools

# Danh sách stocks: mỗi stock được xác định bởi (width, height)
stocks = [
    (100, 100),  # Stock 1
    (80, 60),    # Stock 2
    (100, 70),   # Stock 3
    (90, 50),    # Stock 4
    (30, 30),    # Stock 5
    (120, 80),   # Stock 6
    (75, 75),    # Stock 7
    (50, 100)    # Stock 8
]

# Danh sách products: mỗi product được xác định bởi (width, height)
products = [
    (20, 10),    # Product 1
    (20, 10),    # Product 2
    (30, 20),    # Product 3
    (40, 30),    # Product 4
    (10, 10),    # Product 5
    (15, 15),    # Product 6
    (25, 20),    # Product 7
    (35, 40)     # Product 8
]


c_no_of_eps = 100
v_epsilon = 0.1
c_learning_rate = 0.1  # Tốc độ học (alpha)
c_discount_value = 0.9  # Hệ số chiết khấu (gamma)

max_ep_reward = 0
max_ep_action_list = []
max_start_state = None

w_max_product = 0
h_max_product = 0

for item_product in products:
    if item_product[0] > w_max_product:
        w_max_product = item_product[0]
        
    if item_product[1] > h_max_product:
        h_max_product = item_product[1]
        
w_max_stock = 0
h_max_stock = 0

for item_stock in stocks:
    if item_stock[0] > w_max_stock:
        w_max_stock = item_stock[0]
        
    if item_stock[1] > h_max_stock:
        h_max_stock = item_stock[1]

print(f"w_max_stock = {w_max_stock}, h_max_stock = {h_max_stock}")
print(f"w_max_product = {w_max_product}, h_max_product = {h_max_product}")

state_title_list = list(itertools.product([0, 1], repeat=len(products)))

action_title_list = []


for stock_id, stock_value in enumerate(stocks):
    for product_id, product_value in enumerate(products):
        # Duyệt qua các giá trị x, y có thể từ 0 đến w_max_product, h_max_product
        for x_id in range((w_max_stock) + 1):
            for y_id in range((h_max_stock) + 1):
                action_key = f"{stock_id}-{product_id}-{x_id}-{y_id}"
                action_title_list.append((stock_id, product_id, x_id,y_id))
                
# Q-table: key là (state_key, action) và value là Q-value
Q_table = np.random.uniform(low=-2, high=0, size=(len(state_title_list) , len(action_title_list)))

env = CuttingStockEnv(
    render_mode="human",   # hoặc None nếu không cần hiển thị
    max_w=120,             # Giá trị max_w, max_h nên lớn hơn hoặc bằng kích thước của stocks
    max_h=120,
    seed=42,
    stock_list=stocks,
    product_list=products,
)

def get_valid_actions(state, env):
    """
    Sinh ra danh sách các hành động khả thi theo một số quy tắc:
      - Với mỗi tấm vật liệu (stock) và với mỗi sản phẩm còn lượng > 0,
        nếu kích thước sản phẩm có thể vừa với kích thước tấm vật liệu, thêm một vài lựa chọn vị trí.
    Mỗi hành động được biểu diễn dưới dạng tuple:
      (stock_idx, product_idx, x_ngẫu nhiên, y_ngẫu nhiên)  -------------> Chúng ta sẽ phạt hay thưởng trong hàm step của env
    """
    actions = []
    for stock_idx in range(env.num_stocks):
        stock = env._stocks[stock_idx]
        # Tính kích thước thực của tấm vật liệu (stock)
        stock_width = int(np.sum(np.any(stock != -2, axis=1)))
        stock_height = int(np.sum(np.any(stock != -2, axis=0)))
        for prod_idx, product in enumerate(state["products"]):
            if product["quantity"] <= 0:
                continue
            prod_width, prod_height = product["size"]
            # Kiểm tra nếu sản phẩm có thể vừa với tấm vật liệu
            if prod_width <= stock_width and prod_height <= stock_height:
                # Lựa chọn vị trí cố định (ví dụ: góc trái)
                actions.append((stock_idx, prod_idx, 0, 0))
                # Thêm thêm một hành động với vị trí ngẫu nhiên (nếu có không gian)
                max_x = stock_width - prod_width
                max_y = stock_height - prod_height
                if max_x > 0 and max_y > 0:
                    rand_x = np.random.randint(0, max_x + 1)
                    rand_y = np.random.randint(0, max_y + 1)
                    actions.append((stock_idx, prod_idx, rand_x, rand_y))
    return actions

def state_to_key(state):
    """
    Rút gọn trạng thái bằng cách chỉ lấy thông tin số lượng của các sản phẩm.
    Vì state là dict gồm "stocks" và "products", ta sẽ lấy tuple các quantity của từng sản phẩm.
    """
    return tuple(product["quantity"] for product in state["products"])

def choose_action(state_key, valid_actions, state_title_list, action_title_list, env):

    max_q_value = -99999
    row_idx = 0
    column_idx = 0
    
    for idx, value in enumerate(state_title_list):
        if value == state_key:
            row_idx = idx
            break

    
    for idx_max, value in enumerate(Q_table[row_idx]):
        if value > max_q_value:
            max_q_value = value
            column_idx = idx_max
    
    # Quyết định dùng explore hay exploit
    if random.random() < v_epsilon:
        chosen_action = random.choice(valid_actions)
        return [{
        "stock_idx": chosen_action[0],
        "size": env._products[chosen_action[1]]["size"],
        "position": (chosen_action[2], chosen_action[3])
        }, chosen_action, row_idx, column_idx]
    else: 
        chosen_action = action_title_list[column_idx]
        return [{
        "stock_idx": chosen_action[0],
        "size": env._products[chosen_action[1]]["size"],
        "position": (chosen_action[2], chosen_action[3])
        },chosen_action, row_idx, column_idx]
        
def max_Q_next_state(state_key, state_title_list):
    max_q_value = -99999
    row_idx = 0
    
    for idx, value in enumerate(state_title_list):
        if value == state_key:
            row_idx = idx
            break

    
    for idx_max, value in enumerate(Q_table[row_idx]):
        if value > max_q_value:
            max_q_value = value

    return max_q_value

env.reset()

for ep in range(c_no_of_eps):
    done = False
    current_state, _ = env.reset()
    
    # Bên trong ô chứa mảnh stock
    # -2: Biểu thị các ô không thuộc về vùng của tấm stock, tức là phần "ngoài" giới hạn kích thước thực của tấm vật liệu.
    # -1: Biểu thị các ô bên trong vùng của tấm stock nhưng chưa được sử dụng (chưa có sản phẩm nào cắt vào đó).

    ep_reward = 0 # Khởi tạo phần thưởng của episode
    ep_start_state = current_state  # Lưu trạng thái bắt đầu
    action_list = []  # Lưu danh sách hành động của episode
    
    while not done:
        
        action_valid = get_valid_actions(current_state, env) # ===> Trả lại list danh sách ------ [ (stock_idx, prod_idx, rand_x, rand_y),.... ]-------- khả thi.
        state_key = state_to_key(current_state) # ===> Trả lại số lượng của product theo đúng index của từng product trung danh sách product (1,0,1,1,1,...)
        action = choose_action(state_key, action_valid, state_title_list, action_title_list, env) 
        action_list.append(action[0])
        

        next_state, reward, done, _, info = env.step(action[0])
        ep_reward += reward
        
        
        next_action_valid = get_valid_actions(next_state, env)
        next_state_key = state_to_key(next_state)
        
        
        if done:
            print("Đã xong ep = {}, reward = {}".format(ep, ep_reward))
            if ep_reward > max_ep_reward:
                max_ep_reward = ep_reward
                max_ep_action_list = action_list
                max_start_state = ep_start_state
        
        else:
            current_q_value = Q_table[action[2]][action[3]]
            new_q_value = (1 - c_learning_rate) * current_q_value + c_learning_rate * (reward + c_discount_value * max_Q_next_state(next_state_key, state_title_list))
            Q_table[action[2]][action[3]] = new_q_value
        
            current_state = next_state
        
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