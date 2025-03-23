import gym
import numpy as np
import pandas as pd
from gym import spaces

class CuttingGlassEnv(gym.Env):
    def __init__(self, dataset=1, csv_path="/kaggle/input/data-5050/data_custom.csv"):
        super(CuttingGlassEnv, self).__init__()
        
        # Đọc dữ liệu từ file CSV, đây là nguồn dữ liệu chính cho stocks và products
        self.df = pd.read_csv(csv_path)
        self.dataset = dataset
        
        # Khởi tạo danh sách stocks và products dựa trên dataset đã chọn
        self.stocks, self.products = self.load_dataset(dataset)

        # Đặt số lượng stocks và products dựa vào kích thước của danh sách sau khi load
        self.num_stocks = len(self.stocks)
        self.num_products = len(self.products)

        # Padding các stocks về kích thước chuẩn 50x50 (điều này giúp việc xử lý trở nên nhất quán)
        self.stocks = self.pad_stocks(self.stocks)

        # Tạo valid mask, dùng để xác định các vị trí hợp lệ để cắt sản phẩm
        self.valid_mask = self.create_valid_mask()
        
        # Biến đếm số lần không thực hiện được hành động cắt liên tiếp
        self.no_cut_counter = 0
        self.max_no_cut = 500  # Số lần không cắt được tối đa trước khi episode kết thúc

        # Biến tích lũy reward trong episode
        self.episode_reward = 0

        # Định nghĩa không gian hành động của môi trường:
        # Hành động gồm 5 phần: (stock_idx, x, y, rotate, product_idx)
        # - stock_idx: chỉ số của stock được chọn
        # - x, y: vị trí bắt đầu cắt trên stock (trong không gian 50x50)
        # - rotate: biến nhị phân để xác định có xoay sản phẩm hay không (0 hoặc 1)
        # - product_idx: chỉ số của sản phẩm được chọn
        self.action_space = spaces.MultiDiscrete([self.num_stocks, 50, 50, 2, self.num_products])
        
        # Định nghĩa không gian quan sát của môi trường
        # - stocks: ma trận stocks đã được padding, giá trị từ -1 đến 1
        # - products: ma trận thông tin sản phẩm với kích thước (width, height, số lượng)
        # - valid_mask: mặt nạ cho biết các vị trí hợp lệ trên mỗi stock (0 hoặc 1)
        self.observation_space = spaces.Dict({
            "stocks": spaces.Box(low=-1, high=1, shape=(self.num_stocks, 50, 50), dtype=np.float16),
            "products": spaces.Box(low=0, high=100, shape=(self.num_products, 3), dtype=np.float16),
            "valid_mask": spaces.Box(low=0, high=1, shape=(self.num_stocks, 50, 50), dtype=np.float16)
        })

    def get_state(self):
        """Trả về trạng thái hiện tại của môi trường dưới dạng dictionary gồm stocks, valid_mask và products."""
        padded_stocks = self.pad_stocks(self.stocks)  # Đảm bảo stocks luôn ở kích thước chuẩn
        return {
            "stocks": padded_stocks,
            "valid_mask": self.valid_mask.copy(),
            "products": self.products.copy()
        }

    def load_dataset(self, dataset):
        """Tải stocks và products từ CSV dựa trên dataset chỉ định.
        
        Cụ thể:
        - Lọc dataframe theo batch_id bằng dataset.
        - Với các dòng có type 'stock', khởi tạo một ma trận zeros có kích thước (width x height).
        - Với các dòng có type 'product', lưu thông tin width, height và số lượng mặc định là 1.
        """
        df_filtered = self.df[self.df["batch_id"] == dataset]
        
        # Lấy danh sách stocks
        stocks = []
        stock_data = df_filtered[df_filtered["type"] == "stock"]
        for _, row in stock_data.iterrows():
            stocks.append(np.zeros((row["width"], row["height"]), dtype=int))
        
        # Lấy danh sách products
        products = []
        product_data = df_filtered[df_filtered["type"] == "product"]
        for _, row in product_data.iterrows():
            products.append([row["width"], row["height"], 1])  # Mặc định số lượng mỗi sản phẩm là 1
        
        return stocks, np.array(products, dtype=np.float16)

    def pad_stocks(self, stocks):
        """
        Đưa tất cả các stocks về kích thước cố định 50x50 bằng cách padding.
        Các vùng ngoài kích thước của stock ban đầu sẽ được điền giá trị -1.
        Điều này giúp đồng nhất kích thước của tất cả stocks, thuận tiện cho việc xử lý sau này.
        """
        padded_stocks = np.full((self.num_stocks, 50, 50), fill_value=-1, dtype=np.int8)
        for i, stock in enumerate(stocks):
            w, h = stock.shape
            # Copy nội dung của stock vào góc trên bên trái của ma trận padded
            padded_stocks[i, :w, :h] = stock
        return padded_stocks

    def create_valid_mask(self):
        """Khởi tạo valid mask cho từng stock.
        
        Với mỗi stock:
        - Nếu stock đã có sản phẩm (giá trị > 0) thì xác định các vị trí hợp lệ dựa vào các vị trí kề cạnh.
        - Nếu stock trống, chỉ cho phép cắt ở vị trí (0,0).
        """
        valid_mask = np.zeros((self.num_stocks, 50, 50), dtype=np.uint8)
        for i, stock in enumerate(self.stocks):
            if np.any(stock > 0):
                valid_positions = self.get_valid_positions(stock)
                for x, y in valid_positions:
                    valid_mask[i, x, y] = 1
            else:
                valid_mask[i, 0, 0] = 1  # Stock rỗng: cho phép duy nhất vị trí (0,0)
        return valid_mask

    def update_valid_mask(self, stock_idx):
        """Cập nhật valid mask của một stock cụ thể sau mỗi bước cắt.
        
        Quá trình:
        - Đặt lại valid mask của stock thành mảng zeros.
        - Nếu stock có chứa sản phẩm (giá trị > 0), cập nhật các vị trí hợp lệ theo get_valid_positions.
        - Nếu không, giữ vị trí (0,0) làm hợp lệ.
        """
        stock = self.stocks[stock_idx]
        self.valid_mask[stock_idx] = np.zeros((50, 50), dtype=np.uint8)
        if np.any(stock > 0):
            valid_positions = self.get_valid_positions(stock)
            for x, y in valid_positions:
                self.valid_mask[stock_idx, x, y] = 1
        else:
            self.valid_mask[stock_idx, 0, 0] = 1  # Nếu stock trống, giữ vị trí (0,0)

    def get_valid_positions(self, stock):
        """
        Xác định danh sách các vị trí hợp lệ để cắt sản phẩm tiếp theo trong stock.
        
        Tiêu chí:
        - Vị trí hợp lệ là những ô trống (giá trị 0) nằm cạnh một ô đã có sản phẩm (giá trị > 0).
        - Phạm vi kiểm tra bao gồm 4 hướng: trên, dưới, trái, phải.
        """
        valid_positions = set()
        rows, cols = stock.shape
        for x in range(rows):
            for y in range(cols):
                if stock[x, y] > 0:
                    # Xét các ô kề cạnh của ô hiện tại
                    neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
                    for nx, ny in neighbors:
                        if 0 <= nx < rows and 0 <= ny < cols and stock[nx, ny] == 0:
                            valid_positions.add((nx, ny))
        return valid_positions

    def step(self, action):
        """
        Thực hiện hành động cắt sản phẩm trên stock dựa trên action được cung cấp.
        
        Các bước thực hiện:
        1. Giải nén action: xác định stock, vị trí (x, y), có xoay sản phẩm hay không và sản phẩm được chọn.
        2. Kiểm tra giới hạn số lần không cắt thành công, nếu vượt quá thì kết thúc episode.
        3. Kiểm tra các ràng buộc của action (chỉ số nằm ngoài phạm vi, sản phẩm đã hết, vị trí không hợp lệ, không thể cắt vì vị trí bị chặn, …).
        4. Nếu hợp lệ, thực hiện cắt, cập nhật stock, valid mask và giảm số lượng của sản phẩm.
        5. Tính reward cho hành động và cập nhật reward tích lũy.
        6. Kiểm tra nếu tất cả sản phẩm đã được cắt hết, kết thúc episode và tính reward cuối cùng.
        """
        stock_idx, x, y, rotate, product_idx = action
        stock_idx, x, y, product_idx = map(int, [stock_idx, x, y, product_idx])

        # Khởi tạo reward cho bước này
        reward = 0

        # Nếu không cắt được liên tiếp quá số lần cho phép thì kết thúc episode
        if self.no_cut_counter >= self.max_no_cut:
            print("Quá 500 lần chọn action nhưng không cắt được!")
            # Tính reward tổng kết dựa trên stock và perimeter khi episode kết thúc
            final_reward = -300 + self.calculated_reward()
            self.episode_reward = 0  # Đặt lại reward tích lũy
            return self.get_state(), final_reward, True, {}
        
        # Kiểm tra nếu chỉ số stock hoặc sản phẩm nằm ngoài phạm vi
        if stock_idx < 0 or stock_idx >= self.num_stocks or product_idx < 0 or product_idx >= len(self.products):
            print("Chọn stock hoặc product ngoài phạm vi")
            self.no_cut_counter += 1
            self.episode_reward += -0.1  # Phạt hành động không hợp lệ
            return self.get_state(), -0.1, False, {}
        
        # Kiểm tra nếu sản phẩm được chọn đã hết số lượng
        if self.products[product_idx, 2] <= 0:
            print("Sản phẩm được chọn đã hết")
            self.no_cut_counter += 1
            self.episode_reward += -0.1  # Phạt hành động không hợp lệ
            return self.get_state(), -0.1, False, {}
        
        stock = self.stocks[stock_idx]
        stock_h, stock_w = stock.shape
        # Lấy kích thước sản phẩm (width, height) theo thông tin sản phẩm
        w, h = map(int, self.products[product_idx, :2])
        # Nếu rotate bằng 1, hoán đổi kích thước (xoay sản phẩm)
        if rotate == 1:
            w, h = h, w

        # Kiểm tra xem vị trí (x, y) có nằm trong valid mask không
        if self.valid_mask[stock_idx, x, y] == 0:
            self.no_cut_counter += 1
            self.episode_reward += -0.1  # Phạt vì chọn vị trí không hợp lệ
            return self.get_state(), -0.1, False, {}
        
        # Kiểm tra khả năng cắt: đảm bảo vùng cắt không vượt quá giới hạn của stock
        if x + w <= stock_w and y + h <= stock_h:
            # Kiểm tra vùng cắt có trùng với các sản phẩm đã cắt hay không (các ô khác 0)
            if np.any(stock[x:x+w, y:y+h] != 0):
                self.no_cut_counter += 1
                self.episode_reward += -0.1  # Phạt vì vùng cắt không trống
                return self.get_state(), -0.1, False, {}
        else:
            # Nếu vùng cắt vượt quá biên của stock, phạt hành động
            self.no_cut_counter += 1
            self.episode_reward += -0.1  # Phạt vì vượt quá kích thước stock
            return self.get_state(), -0.1, False, {}
        
        # Thực hiện cắt: đánh dấu các ô trong vùng cắt với product_idx + 1 (để phân biệt với 0)
        stock[x:x+w, y:y+h] = product_idx + 1
        self.no_cut_counter = 0  # Reset lại counter sau khi cắt thành công
        self.products[product_idx, 2] -= 1  # Giảm số lượng của sản phẩm vừa cắt
        self.update_valid_mask(stock_idx)  # Cập nhật lại valid mask cho stock
        reward += 5  # Cắt thành công được thưởng +5
        self.episode_reward += reward  # Cập nhật reward tích lũy

        # Kiểm tra nếu tất cả sản phẩm đã cắt hết (số lượng bằng 0), kết thúc episode
        done = np.all(self.products[:, 2] == 0)
        if done:
            # Tính reward tổng kết khi episode kết thúc dựa trên số stock sử dụng và perimeter
            reward = self.episode_reward + self.calculated_reward()
            self.episode_reward = 0  # Reset lại reward tích lũy
        
        # Trả về state mới, reward, trạng thái done và một dict rỗng (info)
        return {
            "stocks": self.pad_stocks(self.stocks),
            "products": self.products,
            "valid_mask": self.valid_mask
        }, reward, done, {}

    def calculate_union_perimeter(self, mask):
        """
        Tính chu vi của union các cell True trong mask.
        
        Với mỗi cell có giá trị True, nếu các ô kề (trên, dưới, trái, phải)
        hoặc không tồn tại (nằm ngoài biên) hoặc có giá trị False thì cộng thêm 1 vào chu vi.
        Phương pháp này giúp đánh giá mức độ rời rạc của các vùng cắt.
        """
        perimeter = 0
        rows, cols = mask.shape
        for i in range(rows):
            for j in range(cols):
                if mask[i, j]:
                    # Kiểm tra cạnh trên
                    if i == 0 or not mask[i - 1, j]:
                        perimeter += 1
                    # Kiểm tra cạnh dưới
                    if i == rows - 1 or not mask[i + 1, j]:
                        perimeter += 1
                    # Kiểm tra cạnh trái
                    if j == 0 or not mask[i, j - 1]:
                        perimeter += 1
                    # Kiểm tra cạnh phải
                    if j == cols - 1 or not mask[i, j + 1]:
                        perimeter += 1
        return perimeter

    def calculated_reward(self):
        """
        Tính tổng reward dựa trên:
        - Số lượng stock đã sử dụng: reward_stock được tính dựa trên công thức 100/(số stock sử dụng)
        - Chu vi của union các sản phẩm trên các stock: reward_perimeter được tính và nhân với hệ số âm
        Sau đó, tổng hợp lại reward từ cả hai phần này.
        """
        # Đếm số stock có chứa ít nhất 1 sản phẩm (giá trị > 0)
        used_stocks = sum(np.any(stock > 0) for stock in self.stocks)
        total_reward = 0
        reward_perimeter = 0
        for stock in self.stocks:
            # Nếu stock không chứa sản phẩm nào, bỏ qua
            if not np.any(stock > 0):
                continue
            
            # Tạo union mask: tất cả các ô chứa sản phẩm
            union_mask = (stock > 0)
            # Tính chu vi của union_mask
            union_perimeter = self.calculate_union_perimeter(union_mask)

            reward_perimeter += union_perimeter
        # Tính reward từ số stock sử dụng, công thức: 100 / (số stock đã dùng)
        reward_stock = 100 / (used_stocks + 1e-8)
        # Tính reward từ perimeter với hệ số phạt
        reward_perimeter = -0.05 * reward_perimeter
        # Tổng hợp lại reward theo tỉ lệ trọng số cho từng phần
        total_reward += 10 * reward_stock + 5 * reward_perimeter
            
        print("reward_stock:", reward_stock)
        print("union_perimeter:", union_perimeter)
        print("reward_perimeter:", reward_perimeter)
            
        return total_reward

    def reset(self):
        """Khởi tạo lại môi trường cho một episode mới.
        
        Quá trình reset:
        - Tải lại danh sách stocks và products từ dataset.
        - Tạo lại valid mask cho các stocks.
        - Reset lại biến đếm không cắt liên tiếp.
        - Trả về state khởi tạo ban đầu.
        """
        self.stocks, self.products = self.load_dataset(self.dataset)
        self.valid_mask = self.create_valid_mask()
        self.no_cut_counter = 0
        return {
            "stocks": self.pad_stocks(self.stocks),
            "products": self.products,
            "valid_mask": self.valid_mask
        }
