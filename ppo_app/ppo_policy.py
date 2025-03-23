import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PPOPolicy(nn.Module):
    def __init__(self, num_stocks, num_products, learning_rate=0.0001):
        super(PPOPolicy, self).__init__()
        # Xác định device để chạy (GPU nếu có, ngược lại là CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_stocks = num_stocks
        self.num_products = num_products

        # Mạng CNN xử lý input từ stocks, kết hợp với valid_mask thành 2 kênh
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((5, 5)),  # Giảm kích thước feature map về 5x5
            nn.Flatten()  # Làm phẳng tensor để đưa vào fully-connected layers sau này
        )

        # Mạng MLP xử lý thông tin từ products, chuyển đổi đầu vào là vector
        self.mlp_products = nn.Sequential(
            nn.Linear(num_products * 3, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )

        # Xác định kích thước đầu ra của CNN (16 kênh, mỗi kênh 5x5)
        cnn_output_size = 16 * 5 * 5
        # Tổng hợp đặc trưng từ CNN và MLP của products
        input_size = cnn_output_size + 8

        # Số lượng vị trí khả dụng: num_stocks x (50x50)
        self.num_positions = num_stocks * 50 * 50
        # Tổng số đầu ra của actor: gồm logits cho vị trí, 2 cho rotation và num_products cho lựa chọn sản phẩm
        self.output_dim = self.num_positions + 2 + num_products

        # Mạng actor: Fully-connected layers để sinh ra logits cho các hành động
        self.actor_fc = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, self.output_dim)
        )

        # Mạng critic: Dự đoán giá trị V(s)
        self.critic_fc = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Sử dụng Adam optimizer với learning_rate được định nghĩa
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        # Chuyển model về đúng device (GPU hoặc CPU)
        self.to(self.device)

    def forward(self, stocks, valid_mask, products):
        """
        Hàm forward của mô hình PPOPolicy.
        
        Tham số:
        - stocks: Tensor có kích thước (batch, num_stocks, height, width)
        - valid_mask: Tensor có kích thước (batch, num_stocks, height, width)
        - products: Tensor có kích thước (batch, num_products, 3)
        
        Quy trình:
        1. Ghép stocks và valid_mask thành input cho CNN.
        2. Trích xuất đặc trưng từ stocks thông qua CNN.
        3. Xử lý thông tin sản phẩm thông qua MLP.
        4. Ghép các đặc trưng lại để tạo thành đầu vào cho actor và critic.
        5. Sinh ra logits cho actor và giá trị dự đoán từ critic.
        6. Tách logits của actor thành 3 phần: cho vị trí, rotation và lựa chọn sản phẩm.
        7. Áp dụng bias cho vị trí và mask các vị trí không hợp lệ.
        8. Sinh ra các phân phối xác suất (Categorical) và sample hành động từ đó.
        9. Giải mã index vị trí thành (stock_idx, x, y).
        10. Trả về tuple hành động, tuple các phân phối và giá trị V(s).
        """
        batch_size, num_stocks, height, width = stocks.shape

        # Ghép stocks và valid_mask theo kênh, tạo ra tensor có 2 kênh
        stocks_valid = torch.stack([stocks, valid_mask.float()], dim=2)
        # Chuyển đổi kích thước: gộp batch và num_stocks để xử lý độc lập với CNN
        cnn_input = stocks_valid.view(-1, 2, height, width)  # (batch*num_stocks, 2, height, width)

        # Trích xuất đặc trưng từ CNN
        stock_features = self.cnn(cnn_input)  # Kết quả: (batch*num_stocks, feature_dim)
        # Tái sắp xếp và tính trung bình đặc trưng trên từng stock trong batch
        stock_features = stock_features.view(batch_size, num_stocks, -1).mean(dim=1)  # (batch, feature_dim)

        # Xử lý thông tin sản phẩm: flatten tensor sản phẩm để đưa qua MLP
        product_flat = products.view(batch_size, -1)  # (batch, num_products*3)
        product_features = self.mlp_products(product_flat)  # (batch, 8)

        # Ghép các đặc trưng của stocks và products lại thành vector đặc trưng chung
        features = torch.cat([stock_features, product_features], dim=-1)  # (batch, feature_dim+8)

        # Tính toán logits cho actor và giá trị cho critic dựa trên đặc trưng chung
        logits = self.actor_fc(features)  # (batch, output_dim)
        # Áp dụng hàm log_softmax để chuẩn hóa logits
        logits = torch.log_softmax(logits, dim=-1)
        value = self.critic_fc(features).squeeze(-1)  # (batch,)

        # Tách logits thành 3 phần:
        # - pos_logits: Logits cho việc chọn vị trí trên stocks (batch, num_positions)
        pos_logits = logits[:, :self.num_positions]
        # - rot_logits: Logits cho hành động xoay (batch, 2)
        rot_logits = logits[:, self.num_positions:self.num_positions+2]
        # - prod_logits: Logits cho việc chọn sản phẩm (batch, num_products)
        prod_logits = logits[:, self.num_positions+2:]

        # ---------------------- Phần tạo bias cho pos_logits ---------------------- #
        # Các hệ số bias được dùng để ưu tiên chọn từ trên xuống dưới, từ trái sang phải
        stock_bias_factor = 1
        position_bias_factor = 0.1

        # Tạo bias cho từng stock dựa vào chỉ số của stock, nhân với hệ số
        stock_bias = torch.arange(self.num_stocks, device=self.device).unsqueeze(1).unsqueeze(2)  # (num_stocks, 1, 1)
        stock_bias = stock_bias.repeat(1, 50, 50)  # (num_stocks, 50, 50)
        stock_bias = -stock_bias * stock_bias_factor

        # Tạo bias cho từng vị trí trong stock: giảm dần theo tổng của tọa độ (i, j)
        position_bias = torch.zeros((50, 50), device=self.device)
        for i in range(50):
            for j in range(50):
                position_bias[i, j] = -(i + j) * position_bias_factor
        # Lặp lại bias cho từng stock
        position_bias = position_bias.repeat(self.num_stocks, 1, 1)  # (num_stocks, 50, 50)

        # Tổng hợp bias từ stock và vị trí
        total_bias = stock_bias + position_bias  # (num_stocks, 50, 50)
        total_bias = total_bias.view(-1)  # Flatten thành vector (num_stocks * 50 * 50)
        total_bias = total_bias.unsqueeze(0).repeat(batch_size, 1)  # (batch, num_positions)

        # Cộng bias vào pos_logits để điều chỉnh xác suất chọn vị trí
        pos_logits = pos_logits + total_bias

        # Áp dụng valid_mask: các vị trí không hợp lệ sẽ bị giảm xác suất rất mạnh
        valid_mask_flat = valid_mask.view(batch_size, -1).float()  # (batch, num_positions)
        pos_logits = pos_logits + (1 - valid_mask_flat) * (-1e6)

        # ---------------------- Tạo phân phối và sample hành động ---------------------- #
        # Tạo phân phối cho các phần của hành động
        pos_dist = Categorical(logits=pos_logits)
        rot_dist = Categorical(logits=rot_logits)
        # Áp dụng mask cho sản phẩm: nếu số lượng sản phẩm <= 0 thì giảm xác suất chọn về rất thấp
        remaining_products = products[:, :, 2]  # (batch, num_products)
        prod_logits = prod_logits - ((remaining_products <= 0).float() * 1e6)
        prod_dist = Categorical(logits=prod_logits)

        # Sample hành động từ các phân phối đã tạo
        pos_index = pos_dist.sample()  # (batch,)
        rot = rot_dist.sample()        # (batch,)
        prod = prod_dist.sample()      # (batch,)

        # Giải mã pos_index thành các chỉ số (stock_idx, x, y)
        total_positions = 50 * 50
        stock_idx = pos_index // total_positions  # Chỉ số stock: phần nguyên của phép chia
        pos_in_stock = pos_index % total_positions  # Vị trí trong stock: phần dư
        x = pos_in_stock // 50  # Hàng trong stock
        y = pos_in_stock % 50   # Cột trong stock

        # Trả về tuple hành động, tuple các phân phối và giá trị của critic (V(s))
        return (stock_idx, x, y, rot, prod), (pos_dist, rot_dist, prod_dist), value
