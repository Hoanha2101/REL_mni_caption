import pygame
import torch
import numpy as np
import random
from cutting_glass_env import CuttingGlassEnv
from ppo_policy import PPOPolicy

# Khởi tạo Pygame
pygame.init()

# Cấu hình cửa sổ hiển thị
WINDOW_WIDTH = 1500           # Chiều rộng cửa sổ
WINDOW_HEIGHT = 700           # Chiều cao cửa sổ
GRID_SIZE = 280               # Kích thước mỗi ô hiển thị cho một stock (dựa trên scale)
MARGIN = 15                   # Khoảng cách giữa các stock khi hiển thị
ROWS, COLS = 2, 5             # Hiển thị 10 stock (2 hàng, 5 cột)

# Tạo cửa sổ Pygame
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Stock Cutting Visualization")

# Hàm tạo bảng màu ngẫu nhiên cho các product
def generate_colors(num_colors):
    colors = {}
    # Duyệt từ 1 đến num_colors để tạo màu cho từng product
    for i in range(1, num_colors + 1):
        # Chọn màu ngẫu nhiên với giá trị mỗi kênh từ 50 đến 255
        colors[i] = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
    return colors

# Load môi trường và mô hình
dataset_id = 1
env = CuttingGlassEnv(dataset=dataset_id, csv_path="data_custom.csv")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tạo đối tượng policy và chuyển sang device (GPU/CPU)
policy = PPOPolicy(num_stocks=env.num_stocks, num_products=env.num_products).to(device)

# Tải model đã huấn luyện từ file đã lưu
model_path = "Kaggle_Model\ppo_policy_ep8000_2.pth"
policy = torch.load(model_path, map_location=device)
policy.eval()  # Đưa model vào chế độ evaluation để không cập nhật trọng số

# Chạy thử nghiệm mô phỏng trên môi trường
state = env.reset()
# Chuyển đổi state thành tensor với batch dimension và đưa lên device
state = {key: torch.tensor(value, dtype=torch.float32).unsqueeze(0).to(device) for key, value in state.items()}
done = False
total_reward = 0

# Vòng lặp mô phỏng cho đến khi episode kết thúc (done = True)
while not done:
    # Gọi policy để lấy hành động, các phân phối xác suất và giá trị V(s)
    (stock_idx, x, y, rot, prod), (pos_dist, rot_dist, prod_dist), value = policy(
        state["stocks"], state["valid_mask"], state["products"]
    )

    # Tạo action dưới dạng tuple với các giá trị scalar
    action = (stock_idx.item(), x.item(), y.item(), rot.item(), prod.item())

    # Thực hiện bước trong môi trường với hành động được chọn
    next_state, reward, done, new_value = env.step(action)
    # Chuyển đổi next_state từ numpy sang tensor với batch dimension
    state = {key: torch.tensor(value, dtype=torch.float32).unsqueeze(0).to(device) for key, value in next_state.items()}
    total_reward += reward
print("Total reward:", total_reward)

# Lấy trạng thái cuối cùng của stocks sau khi mô phỏng
final_stocks = env.stocks.copy()
# Lưu lại kích thước thực tế của mỗi stock để hiển thị đúng tỷ lệ
stock_sizes = [(s.shape[0], s.shape[1]) for s in final_stocks]

# Tạo màu sắc cho từng product dựa trên ID
# Nối tất cả stocks lại để lấy các giá trị product duy nhất (loại trừ 0, tức vùng trống)
unique_products = np.unique(np.concatenate([stock.flatten() for stock in final_stocks]))
unique_products = unique_products[unique_products > 0]
product_colors = generate_colors(len(unique_products))

# Hàm vẽ các stock với kích thước thực tế dựa trên thông tin đã lưu
def draw_stocks():
    screen.fill((30, 30, 30))  # Đặt background màu tối (xám đậm)
    
    # Duyệt qua tối đa 10 stock để hiển thị
    for i in range(min(10, len(final_stocks))):
        # Tính toán vị trí hiển thị của stock dựa trên hàng và cột
        row, col = divmod(i, COLS)
        x, y = col * (GRID_SIZE + MARGIN), row * (GRID_SIZE + MARGIN)
        
        stock = final_stocks[i]
        stock_width, stock_height = stock_sizes[i]  # Lấy kích thước thật của stock
        scale_x = GRID_SIZE / 50  # Tỷ lệ scale theo chiều ngang (vì stock đã được padding về 50x50)
        scale_y = GRID_SIZE / 50  # Tỷ lệ scale theo chiều dọc
        
        # Vẽ nền cho stock: hình chữ nhật trắng kích thước thật của stock
        pygame.draw.rect(screen, (255, 255, 255), (x, y, stock_width * scale_x, stock_height * scale_y))
        
        # Duyệt qua từng cell trong stock để vẽ product nếu có
        for r in range(stock.shape[0]):
            for c in range(stock.shape[1]):
                product_id = stock[r, c]
                if product_id > 0:
                    # Lấy màu tương ứng với product_id, nếu không có thì dùng màu mặc định
                    color = product_colors.get(product_id, (200, 200, 200))
                    # Vẽ cell với màu tương ứng, điều chỉnh vị trí và kích thước theo scale
                    pygame.draw.rect(screen, color, (x + c * scale_x, y + r * scale_y, scale_x, scale_y))
                        
        # Vẽ viền cho stock để phân biệt rõ ràng giữa các stock
        pygame.draw.rect(screen, (255, 255, 255), (x, y, GRID_SIZE, GRID_SIZE), 2)
    
    # Cập nhật hiển thị
    pygame.display.flip()

# Vòng lặp chính của Pygame để hiển thị cửa sổ
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    # Gọi hàm vẽ stocks lên màn hình
    draw_stocks()

# Thoát Pygame khi vòng lặp kết thúc
pygame.quit()
