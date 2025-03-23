import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from gym import Env
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ----------------------- Định nghĩa môi trường CuttingGlassEnv -----------------------
class CuttingGlassEnv(Env):
    def __init__(self, dataset, csv_path):
        super().__init__()
        self.dataset = dataset      # ID của dataset (batch_id trong CSV)
        self.csv_path = csv_path    # Đường dẫn tới file CSV chứa dữ liệu
        self.height = 50            # Chiều cao của mỗi stock (sau khi padding: 50)
        self.width = 50             # Chiều rộng của mỗi stock
        self.num_stocks = 10        # Số lượng stock có sẵn trong môi trường
        # Khởi tạo mảng stocks với giá trị ban đầu là 0 (chưa cắt sản phẩm nào)
        self.stocks = np.zeros((self.num_stocks, self.height, self.width), dtype=np.int32)
        # valid_mask cho biết các vị trí có thể cắt, khởi tạo với True (được phép cắt)
        self.valid_mask = np.ones((self.num_stocks, self.height, self.width), dtype=bool)
        self.load_data()            # Load dữ liệu từ CSV (các thông tin về sản phẩm)
        self.reset()                # Reset môi trường về trạng thái ban đầu

    def load_data(self):
        # Đọc dữ liệu từ file CSV
        df = pd.read_csv(self.csv_path)
        # Lọc dữ liệu theo batch_id tương ứng với dataset được chọn
        batch_data = df[df['batch_id'] == self.dataset]
        # Lọc những dòng có type là 'product'
        product_data = batch_data[batch_data['type'] == 'product']
        self.num_products = len(product_data)
        # Tạo mảng products với 3 cột: width (là length theo CSV), height (là width theo CSV) và số lượng
        self.products = np.zeros((self.num_products, 3), dtype=np.int32)
        self.products[:, 0] = product_data['width'].values   # width trong CSV được xem là length
        self.products[:, 1] = product_data['height'].values  # height trong CSV được xem là width
        # Khởi tạo số lượng của mỗi sản phẩm ngẫu nhiên trong khoảng từ 1 đến 4 (có thể điều chỉnh)
        self.products[:, 2] = np.random.randint(1, 5, size=(self.num_products))

    def reset(self):
        # Reset lại trạng thái của môi trường
        self.stocks = np.zeros((self.num_stocks, self.height, self.width), dtype=np.int32)
        self.valid_mask = np.ones((self.num_stocks, self.height, self.width), dtype=bool)
        self.current_step = 0
        self.load_data()  # Tải lại dữ liệu để đảm bảo cập nhật số lượng sản phẩm mới
        return self.get_state()

    def get_state(self):
        # Trả về trạng thái hiện tại của môi trường dưới dạng dictionary chứa:
        # stocks, valid_mask, và products (chuyển thành tensor để dễ xử lý với PyTorch)
        return {
            "stocks": torch.tensor(self.stocks, dtype=torch.float32),
            "valid_mask": torch.tensor(self.valid_mask, dtype=torch.float32),
            "products": torch.tensor(self.products, dtype=torch.float32)
        }

    def step(self, action):
        # Giải mã action: (stock_idx, x, y, rotate, product_idx)
        stock_idx, x, y, rotate, product_idx = action
        # Lấy thông tin kích thước và số lượng của sản phẩm được chọn
        length, width, quantity = self.products[product_idx]
        # Nếu rotate = 1, hoán đổi chiều dài và chiều rộng
        if rotate:
            length, width = width, length

        # Nếu sản phẩm đã hết số lượng, phạt mạnh và không thực hiện cắt
        if quantity <= 0:
            return self.get_state(), -100, False, {}

        # Kiểm tra giới hạn: nếu vị trí cắt vượt quá giới hạn stock, phạt
        if (stock_idx < 0 or stock_idx >= self.num_stocks or
            x < 0 or x + length > self.height or
            y < 0 or y + width > self.width):
            return self.get_state(), -100, False, {}

        # Kiểm tra xem vùng cắt đã có sản phẩm nào chưa (không được trùng)
        if np.any(self.stocks[stock_idx, x:x+length, y:y+width] != 0):
            return self.get_state(), -100, False, {}

        # Thực hiện cắt: gán giá trị (product_idx + 1) vào vùng cắt của stock
        self.stocks[stock_idx, x:x+length, y:y+width] = product_idx + 1
        # Cập nhật valid_mask: đánh dấu vùng vừa cắt là không hợp lệ cho cắt tiếp
        self.valid_mask[stock_idx, x:x+length, y:y+width] = 0
        # Giảm số lượng sản phẩm sau khi cắt thành công
        self.products[product_idx, 2] -= 1

        # Tính số lượng stock đã sử dụng
        used_stocks = np.sum(np.any(self.stocks != 0, axis=(1, 2)))
        # Tính chu vi của union các sản phẩm trên stock được cắt (dùng để tính trim loss)
        union_perimeter = self.calculate_union_perimeter(stock_idx)
        # Tính reward: cố gắng tối ưu sử dụng stock (reward cao nếu sử dụng ít stock và trim loss thấp)
        reward = 10 - 0.5 * used_stocks - 0.1 * union_perimeter

        # Kết thúc episode nếu tất cả sản phẩm đã cắt hết (số lượng bằng 0)
        done = np.all(self.products[:, 2] == 0)
        return self.get_state(), reward, done, {}

    def calculate_union_perimeter(self, stock_idx):
        # Tính chu vi của union các phần đã được cắt trên stock
        stock = self.stocks[stock_idx]
        perimeter = 0
        for i in range(self.height):
            for j in range(self.width):
                if stock[i, j] != 0:
                    # Kiểm tra 4 hướng: phải, dưới, trái, trên
                    for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        ni, nj = i + di, j + dj
                        # Nếu ô kề ngoài biên hoặc chưa được cắt (giá trị 0) thì tăng chu vi
                        if ni < 0 or ni >= self.height or nj < 0 or nj >= self.width or stock[ni, nj] == 0:
                            perimeter += 1
        return perimeter

# ----------------------- Định nghĩa PPOPolicy -----------------------
class PPOPolicy(nn.Module):
    def __init__(self, num_stocks, num_products, learning_rate=0.0001):
        super(PPOPolicy, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_stocks = num_stocks
        self.num_products = num_products

        # Mạng CNN để xử lý input từ stocks và valid_mask (2 kênh)
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((5, 5)),
            nn.Flatten()
        )

        # Mạng MLP để xử lý thông tin từ products
        self.mlp_products = nn.Sequential(
            nn.Linear(num_products * 3, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )

        # Kích thước đầu ra của CNN
        cnn_output_size = 16 * 5 * 5
        # Kích thước đầu vào cho các fully-connected layer là tổng của đặc trưng từ CNN và MLP
        input_size = cnn_output_size + 8

        # Tính số lượng vị trí khả dụng: num_stocks x 50 x 50
        self.num_positions = num_stocks * 50 * 50
        # Tổng số đầu ra của actor: gồm vị trí (num_positions), 2 cho rotation, và num_products cho lựa chọn sản phẩm
        self.output_dim = self.num_positions + 2 + num_products

        # Mạng actor: Fully-connected layers sinh ra logits cho các hành động
        self.actor_fc = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, self.output_dim)
        )

        # Mạng critic: Dự đoán giá trị của state
        self.critic_fc = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Sử dụng Adam optimizer cho toàn bộ các tham số của model
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.to(self.device)

    def forward(self, stocks, valid_mask, products):
        # stocks: (batch, num_stocks, height, width)
        # valid_mask: (batch, num_stocks, height, width)
        # products: (batch, num_products, 3)
        batch_size, num_stocks, height, width = stocks.shape

        # Ghép stocks và valid_mask theo kênh, tạo tensor có 2 kênh cho mỗi stock
        stocks_valid = torch.stack([stocks, valid_mask.float()], dim=2)
        # Reshape để đưa vào CNN: (batch*num_stocks, 2, height, width)
        cnn_input = stocks_valid.view(-1, 2, height, width)
        # Trích xuất đặc trưng từ CNN
        stock_features = self.cnn(cnn_input)
        # Tái cấu trúc và tính trung bình các đặc trưng trên từng stock của mỗi batch
        stock_features = stock_features.view(batch_size, num_stocks, -1).mean(dim=1)

        # Xử lý thông tin sản phẩm qua MLP
        product_flat = products.view(batch_size, -1)
        product_features = self.mlp_products(product_flat)

        # Ghép các đặc trưng lại thành vector đặc trưng chung
        features = torch.cat([stock_features, product_features], dim=-1)

        # Tính toán logits cho actor và giá trị của critic
        logits = self.actor_fc(features)
        logits = torch.log_softmax(logits, dim=-1)
        value = self.critic_fc(features).squeeze(-1)

        # Tách logits thành 3 phần: vị trí, rotation, và lựa chọn sản phẩm
        pos_logits = logits[:, :self.num_positions]
        rot_logits = logits[:, self.num_positions:self.num_positions+2]
        prod_logits = logits[:, self.num_positions+2:]

        # Áp dụng valid_mask: giảm xác suất các vị trí không hợp lệ bằng cách cộng giá trị rất nhỏ
        valid_mask_flat = valid_mask.view(batch_size, -1).float()
        pos_logits = pos_logits + (1 - valid_mask_flat) * (-1e6)

        # Tạo các phân phối xác suất cho vị trí, rotation, và sản phẩm
        pos_dist = Categorical(logits=pos_logits)
        rot_dist = Categorical(logits=rot_logits)

        # Áp dụng mask cho sản phẩm nếu số lượng <= 0, giảm xác suất chọn
        remaining_products = products[:, :, 2]
        prod_logits = prod_logits - ((remaining_products <= 0).float() * 1e6)
        prod_dist = Categorical(logits=prod_logits)

        # Sample hành động từ các phân phối
        pos_index = pos_dist.sample()
        rot = rot_dist.sample()
        prod = prod_dist.sample()

        # Giải mã vị trí: tính ra (stock_idx, x, y) từ pos_index
        total_positions = 50 * 50
        stock_idx = pos_index // total_positions
        pos_in_stock = pos_index % total_positions
        x = pos_in_stock // 50
        y = pos_in_stock % 50

        return (stock_idx, x, y, rot, prod), (pos_dist, rot_dist, prod_dist), value

# ----------------------- Hàm chạy PPO trên 1 bộ dữ liệu -----------------------
def run_ppo(env, policy, device):
    policy.eval()  # Đưa model vào chế độ evaluation (không cập nhật trọng số)
    state = env.reset()
    start_time = time.time()
    done = False
    # Vòng lặp mô phỏng cho đến khi episode kết thúc
    while not done:
        # Chuyển state về device (GPU/CPU)
        state = {key: val.to(device) for key, val in state.items()}
        with torch.no_grad():
            # Unsqueeze state để tạo batch có kích thước 1
            (stock_idx, x, y, rot, prod), dists, _ = policy(
                state["stocks"].unsqueeze(0),
                state["valid_mask"].unsqueeze(0),
                state["products"].unsqueeze(0)
            )
        # Tạo action dưới dạng tuple các giá trị scalar
        action = (stock_idx.item(), x.item(), y.item(), rot.item(), prod.item())
        state, reward, done, _ = env.step(action)
    runtime = time.time() - start_time
    return env, runtime

# ----------------------- Hàm đánh giá kết quả -----------------------
def evaluate(env):
    # Đếm số lượng stock đã sử dụng (có ít nhất 1 cell != 0)
    used_stocks = np.sum(np.any(env.stocks > 0, axis=(1, 2)))
    remaining_stocks = env.num_stocks - used_stocks
    total_trim_loss = 0
    avg_used_stock_area = 0
    stock_area = env.height * env.width  # Tổng diện tích của 1 stock (50*50 = 2500)
    if used_stocks > 0:
        for stock_idx in range(env.num_stocks):
            if np.any(env.stocks[stock_idx] != 0):
                used_area = np.sum(env.stocks[stock_idx] != 0)
                total_trim_loss += (stock_area - used_area)
                avg_used_stock_area += used_area
        avg_used_stock_area /= used_stocks
    else:
        total_trim_loss = 0
        avg_used_stock_area = 0
    return {
        "used_stocks": used_stocks,
        "remaining_stocks": remaining_stocks,
        "total_trim_loss": total_trim_loss,
        "avg_used_stock_area": avg_used_stock_area
    }

# ----------------------- Chạy PPO trên 10 bộ dữ liệu và thu thập kết quả -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Số lượng sản phẩm cho từng batch (dựa trên dữ liệu từ data_custom.csv)
num_products_per_batch = {
    1: 18, 2: 15, 3: 14, 4: 16, 5: 11,
    6: 11, 7: 10, 8: 13, 9: 16, 10: 12
}

# Khởi tạo cấu trúc để lưu kết quả đánh giá của mô hình PPO
results = {
    "PPO": {
        "batch_id": [],
        "steps": [],
        "runtime": [],
        "total_trim_loss": [],
        "remaining_stocks": [],
        "used_stocks": [],
        "avg_used_stock_area": []
    }
}

# Tải mô hình PPO đã huấn luyện (toàn bộ mô hình, không chỉ trọng số)
episode = 9000
model_path = f"Model_kaggle/ppo_policy_ep{episode}.pth"  # Sửa lại đường dẫn nếu cần

# Lặp qua các batch_id từ 1 đến 10
for batch_id in range(1, 11):
    print(f"Processing batch_id {batch_id}...")

    # Tạo môi trường với batch_id hiện tại
    env = CuttingGlassEnv(dataset=batch_id, csv_path="data_custom.csv")

    # Lấy số lượng sản phẩm cho batch hiện tại
    num_products = num_products_per_batch[batch_id]
    # Tải mô hình PPO từ file đã lưu, chuyển sang device
    policy = torch.load(model_path, map_location=device)
    # Cập nhật num_products trong policy
    policy.num_products = num_products
    # Cập nhật lại mlp_products của policy để phù hợp với số lượng sản phẩm mới
    policy.mlp_products = nn.Sequential(
        nn.Linear(num_products * 3, 16),
        nn.ReLU(),
        nn.Linear(16, 8)
    ).to(device)
    # Cập nhật lại output_dim của policy: vị trí + 2 cho rotation + num_products cho sản phẩm
    policy.output_dim = policy.num_positions + 2 + num_products
    # Cập nhật lại actor_fc của policy
    policy.actor_fc = nn.Sequential(
        nn.Linear(16 * 5 * 5 + 8, 32),
        nn.ReLU(),
        nn.Linear(32, policy.output_dim)
    ).to(device)
    policy.to(device)

    # Chạy mô phỏng PPO trên batch hiện tại
    env, runtime = run_ppo(env, policy, device)
    metrics = evaluate(env)

    # Lưu kết quả đánh giá cho batch hiện tại vào dictionary results
    results["PPO"]["batch_id"].append(batch_id)
    results["PPO"]["steps"].append(num_products)  # Số lượng sản phẩm được coi như số bước (steps)
    results["PPO"]["runtime"].append(runtime)
    results["PPO"]["total_trim_loss"].append(metrics["total_trim_loss"])
    results["PPO"]["remaining_stocks"].append(metrics["remaining_stocks"])
    results["PPO"]["used_stocks"].append(metrics["used_stocks"])
    results["PPO"]["avg_used_stock_area"].append(metrics["avg_used_stock_area"])

# ----------------------- Tạo bảng đánh giá và lưu kết quả -----------------------
data = []
for i in range(10):
    data.append({
        "batch_id": results["PPO"]["batch_id"][i],
        "policy": "PPO",  # Đổi tên thành "PPO" hoặc tên khác nếu cần
        "steps": results["PPO"]["steps"][i],
        "runtime": results["PPO"]["runtime"][i],
        "total_trim_loss": results["PPO"]["total_trim_loss"][i],
        "remaining_stocks": results["PPO"]["remaining_stocks"][i],
        "used_stocks": results["PPO"]["used_stocks"][i],
        "avg_used_stock_area": results["PPO"]["avg_used_stock_area"][i]
    })

df = pd.DataFrame(data)
print("\nEvaluation Table:")
print(df)

# Lưu bảng đánh giá vào file CSV
df.to_csv("ppo_evaluation_results.csv", index=False)

# ----------------------- Vẽ biểu đồ kết quả -----------------------
# Tạo figure với 6 subplot (3 hàng, 2 cột) với kích thước tổng thể được giảm nhẹ
fig, axes = plt.subplots(3, 2, figsize=(10, 8))
fig.suptitle("Performance of PPO", fontsize=12)

# Định dạng chung cho các biểu đồ: giảm kích thước font của nhãn và trục
for ax in axes.flatten():
    ax.tick_params(axis='both', labelsize=6)
    ax.set_xlabel("Batch ID", fontsize=6)

# Biểu đồ Runtime
axes[0, 0].plot(range(1, 11), results["PPO"]["runtime"], marker="o", color="g", label="PPO")
axes[0, 0].set_title("Runtime", fontsize=8)
axes[0, 0].set_ylabel("Runtime (s)", fontsize=6)
axes[0, 0].legend(fontsize=6)

# Biểu đồ Total Trim Loss
axes[0, 1].plot(range(1, 11), results["PPO"]["total_trim_loss"], marker="o", color="g", label="PPO")
axes[0, 1].set_title("Total Trim Loss", fontsize=8)
axes[0, 1].set_ylabel("Total Trim Loss", fontsize=6)
axes[0, 1].legend(fontsize=6)

# Biểu đồ Remaining Stocks
axes[1, 0].plot(range(1, 11), results["PPO"]["remaining_stocks"], marker="o", color="g", label="PPO")
axes[1, 0].set_title("Remaining Stocks", fontsize=8)
axes[1, 0].set_ylabel("Remaining Stocks", fontsize=6)
axes[1, 0].legend(fontsize=6)

# Biểu đồ Used Stocks
axes[1, 1].plot(range(1, 11), results["PPO"]["used_stocks"], marker="o", color="g", label="PPO")
axes[1, 1].set_title("Used Stocks", fontsize=8)
axes[1, 1].set_ylabel("Used Stocks", fontsize=6)
axes[1, 1].legend(fontsize=6)

# Biểu đồ Avg Used Stock Area
axes[2, 0].plot(range(1, 11), results["PPO"]["avg_used_stock_area"], marker="o", color="g", label="PPO")
axes[2, 0].set_title("Avg Used Stock Area", fontsize=8)
axes[2, 0].set_ylabel("Avg Used Stock Area", fontsize=6)
axes[2, 0].legend(fontsize=6)

# Ẩn subplot cuối cùng (không sử dụng)
axes[2, 1].axis("off")

# Điều chỉnh khoảng cách giữa các subplot
plt.subplots_adjust(wspace=0.4, hspace=0.5)
plt.tight_layout(rect=[0, 0, 1, 0.95])
# Lưu biểu đồ kết quả với độ phân giải cao
plt.savefig("ppo_performance_plot.png", dpi=300)
plt.show()
