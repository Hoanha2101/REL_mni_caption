import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from torch.distributions import Categorical
import gc 

class PPOAgent:
    def __init__(self, env, policy, gamma=0.99, lr=1e-4, batch_size=16, epochs=10, clip_epsilon=0.2, entropy_bonus=0.1):
        # Khởi tạo các tham số và biến cần thiết
        self.env = env  # Môi trường tương tác
        self.policy = policy  # Mô hình chính sách (actor-critic)
        self.gamma = gamma  # Hệ số giảm giá cho reward tương lai
        self.lr = lr  # Learning rate
        self.batch_size = batch_size  # Kích thước batch cho quá trình huấn luyện
        self.epochs = epochs  # Số epoch huấn luyện cho mỗi lần cập nhật
        self.clip_epsilon = clip_epsilon  # Hệ số clipping trong PPO
        self.entropy_bonus = entropy_bonus  # Hệ số entropy bonus để khuyến khích exploration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Bộ nhớ để lưu trải nghiệm, giới hạn kích thước để tránh quá tải bộ nhớ
        self.memory = deque(maxlen=150000)
        # Optimizer dùng để cập nhật các tham số của policy
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.actor_losses = []  # Lưu lại các giá trị loss của actor
        self.critic_losses = []  # Lưu lại các giá trị loss của critic

    def store_memory(self, state, action, log_probs, value, reward, done):
        # Lưu lại các trải nghiệm dưới dạng tuple vào bộ nhớ
        if not hasattr(self, "memory"):
            self.memory = deque(maxlen=150000)

        # Clone các tensor state về CPU để lưu giữ, tránh lưu tham chiếu đến GPU
        state_cpu = {k: v.cpu().clone().detach() for k, v in state.items()}
        # Nếu action, log_probs hay value là tensor, chuyển về CPU và tách rời (detach)
        action = action.cpu().detach() if isinstance(action, torch.Tensor) else action
        log_probs = log_probs.cpu().detach() if isinstance(log_probs, torch.Tensor) else log_probs
        value = value.cpu().clone().detach() if isinstance(value, torch.Tensor) else value
        
        # Lưu tuple (state, action, log_probs, value, reward, done) vào bộ nhớ
        self.memory.append((state_cpu, action, log_probs, value, reward, done))

    def choose_action(self, state):
        """
        Chọn hành động từ policy dựa trên state hiện tại.
        Quy trình:
        - Chuyển state từ CPU sang device (GPU nếu có).
        - Dùng policy để tính toán các phân phối xác suất và giá trị của state.
        - Tính log probability tổng của hành động (vị trí, rotation, sản phẩm).
        - Trả về hành động dưới dạng tuple và các giá trị liên quan.
        """
        # Đưa các thành phần của state về đúng device
        state_stocks = state["stocks"].clone().detach().to(self.device)
        state_valid_mask = state["valid_mask"].clone().detach().to(self.device)
        state_products = state["products"].clone().detach().to(self.device)

        with torch.no_grad():
            # Lấy hành động và các phân phối từ mô hình policy
            (stock_idx, x, y, rot, prod), dists, value = self.policy(state_stocks, state_valid_mask, state_products)
        
        # Tính chỉ số vị trí duy nhất: pos_index = stock_idx * (50x50) + (x*50 + y)
        pos_index = stock_idx * (50 * 50) + (x * 50 + y)
       
        # Tổng hợp log probability của các thành phần hành động
        log_prob = dists[0].log_prob(pos_index) + dists[1].log_prob(rot) + dists[2].log_prob(prod)
        
        # Chuyển các tensor thành giá trị scalar để dễ lưu trữ
        action = (stock_idx.item(), x.item(), y.item(), rot.item(), prod.item())
        return action, log_prob, value

    def train(self):
        # Nếu bộ nhớ chưa đủ dữ liệu cho một batch, không thực hiện training
        if len(self.memory) < self.batch_size:
            return

        batch_size = 16  # Có thể dùng self.batch_size, nhưng ở đây được cứng là 16
       
        # Lấy ngẫu nhiên một mini-batch từ bộ nhớ
        mini_batch = random.sample(self.memory, min(len(self.memory), batch_size))
       
        # Tách các thành phần trong mini-batch
        states, actions, log_probs_old, values, rewards, dones = zip(*mini_batch)
        
        # Ghép các thành phần của state lại theo từng key và chuyển về device
        states = {key: torch.cat([s[key].cpu() for s in states], dim=0).to(self.device) for key in states[0]}
       
        # Chuyển đổi các giá trị thành tensor
        values = torch.tensor([v if isinstance(v, (int, float)) else v.item() for v in values],
                              dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.long, device=self.device)
        log_probs_old = torch.tensor(np.array([lp.cpu().numpy() if isinstance(lp, torch.Tensor) else lp for lp in log_probs_old]),
                                     dtype=torch.float32, device=self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32, device=self.device)

        # Tính toán các giá trị returns cho từng bước bằng cách dùng discounted sum
        returns = []
        discounted_sum = 0
       
        # Lặp ngược qua rewards và dones
        for reward, done in zip(reversed(rewards), reversed(dones)):
            discounted_sum = reward + self.gamma * discounted_sum * (1 - done)
            returns.insert(0, discounted_sum)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        # Tính advantage: hiệu giữa return và value đã dự đoán
        advantages = returns - values
       
        # Chuẩn hóa advantage để ổn định quá trình huấn luyện
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Cập nhật policy trong số epoch được chỉ định
        for _ in range(self.epochs):
            # Sử dụng autocast cho CUDA để tăng hiệu năng (nếu sử dụng GPU)
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                # Forward pass trên mini-batch với state đã lưu
                (stock_idx_new, x_new, y_new, rot_new, prod_new), dists_new, new_values = self.policy(
                    states["stocks"], states["valid_mask"], states["products"]
                )
          
            # Các hành động đã lưu có dạng tuple: (stock_idx, x, y, rot, prod)
            stored_stock = actions[:, 0]
            stored_x = actions[:, 1]
            stored_y = actions[:, 2]
           
            # Tính chỉ số vị trí mới từ các giá trị đã lưu: pos_index_new = stock_idx * (50x50) + (x*50 + y)
            pos_index_new = stored_stock * (50 * 50) + (stored_x * 50 + stored_y)
           
            # Tính log probability mới cho các hành động đã lưu
            log_prob_new = (dists_new[0].log_prob(pos_index_new) +
                            # Sử dụng tensor cho các giá trị rot và prod từ actions
                            dists_new[1].log_prob(torch.tensor([a[3] for a in actions], dtype=torch.long, device=self.device)) +
                            dists_new[2].log_prob(torch.tensor([a[4] for a in actions], dtype=torch.long, device=self.device)))
           
            # Tính ratio giữa log_prob mới và log_prob cũ (exp(log_new - log_old))
            ratio = torch.exp(log_prob_new - log_probs_old)
          
            # Tính surrogate loss theo PPO
            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surrogate1, surrogate2).mean()
          
            # Tính entropy bonus từ các phân phối, nhằm tăng tính exploration
            entropy = dists_new[0].entropy().mean() + dists_new[1].entropy().mean() + dists_new[2].entropy().mean()
            actor_loss = actor_loss - self.entropy_bonus * entropy
          
            # Tính critic loss sử dụng hàm MSE giữa new_values và returns
            critic_loss = nn.MSELoss()(new_values.squeeze(), returns)
           
            # Tổng hợp loss của actor và critic (critic được nhân hệ số 0.5)
            loss = actor_loss + 0.5 * critic_loss
           
            # Reset gradients của optimizer
            self.optimizer.zero_grad()
            torch.autograd.set_detect_anomaly(True) 
          
            # Lan truyền gradient ngược
            loss.backward()
           
            # Cập nhật các tham số của policy
            self.optimizer.step()
           
            # Ghi lại các giá trị loss để tiện theo dõi
            self.actor_losses.append(actor_loss.item())
            self.critic_losses.append(critic_loss.item())
        
