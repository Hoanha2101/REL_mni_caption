# Cutting Stock Problem với Deep Q-Learning

## Tổng quan
Dự án này triển khai giải pháp cho bài toán Cutting Stock (Cắt vật liệu) sử dụng phương pháp Deep Q-Learning. Hệ thống được thiết kế để tối ưu hóa việc cắt các sản phẩm từ các tấm vật liệu có sẵn, giảm thiểu lãng phí và tối đa hóa hiệu quả sử dụng.

## Cấu trúc dự án 
```bash
├── agents/
│ └── deep_qlearning_agent.py
├── env/
│ └── CuttingStockEnvOptimized.py
├── train_dqlearning.py
└── README.md
```

## Các thành phần chính

### 1. Môi trường (CuttingStockEnvOptimized)
- Môi trường mô phỏng quá trình cắt vật liệu với các đặc điểm:
  - Sử dụng lưới 2D để biểu diễn tấm vật liệu
  - Hỗ trợ nhiều loại sản phẩm với kích thước khác nhau
  - Áp dụng chính sách đặt sản phẩm có cấu trúc:
    - Sản phẩm đầu tiên luôn được đặt ở góc trên bên trái (0,0)
    - Sản phẩm thứ hai được đặt ngay bên phải sản phẩm đầu
    - Các sản phẩm tiếp theo được đặt ở vị trí khả dụng đầu tiên

### 2. Deep Q-Learning Agent
- Sử dụng mạng neural để học chiến lược tối ưu:
  - Policy Network: Dự đoán Q-values cho các hành động
  - Target Network: Ổn định quá trình học
  - Experience Replay: Lưu trữ và tái sử dụng kinh nghiệm
- Kiến trúc mạng:
  ```python
  DeepQNetwork(
      Linear(input_dim, 512)
      ReLU()
      Linear(512, 256)
      ReLU()
      Linear(256, output_dim)
  )
  ```

### 3. Hệ thống phần thưởng (Reward System)
- Phần thưởng được thiết kế để khuyến khích:
  - Đặt sản phẩm đầu tiên ở vị trí (0,0)
  - Tối ưu hóa không gian sử dụng
  - Giảm thiểu diện tích lãng phí
- Công thức tính phần thưởng:
  ```
  reward = base_reward + bonus
  base_reward = width * height
  bonus = (1 - bbox_area / total_area) * alpha * base_reward
  ```

## Quá trình huấn luyện
- Số episode: 10,000
- Batch size: 64
- Target network update: mỗi 10 episode
- Checkpoint lưu: mỗi 500 episode
- Epsilon-greedy exploration:
  - Epsilon ban đầu: 1.0
  - Epsilon minimum: 0.01
  - Epsilon decay: 0.995

## Cách sử dụng

### Huấn luyện mô hình
```bash
python train_dqlearning.py
```

### Tham số môi trường có thể điều chỉnh
```python
env = CuttingStockEnvOptimized(
min_w=20, # Chiều rộng tối thiểu
min_h=20, # Chiều cao tối thiểu
max_w=20, # Chiều rộng tối đa
max_h=20, # Chiều cao tối đa
num_stocks=1, # Số lượng tấm vật liệu
max_product_type=3, # Số loại sản phẩm tối đa
max_product_per_type=5 # Số lượng tối đa mỗi loại
)
```

## Kết quả trực quan
- Quá trình huấn luyện được lưu dưới dạng GIF trong thư mục `demo/`
- Ba loại GIF được tạo ra:
  - combine.gif: Tổng hợp quá trình
  - bestfit.gif: Kết quả tốt nhất
  - firstfit.gif: Kết quả đầu tiên đạt được

## Lưu ý
- Checkpoint được lưu tự động mỗi 500 episode
- Có thể dừng huấn luyện bằng Ctrl+C, checkpoint sẽ được lưu
- Sử dụng GPU nếu có để tăng tốc quá trình huấn luyện
