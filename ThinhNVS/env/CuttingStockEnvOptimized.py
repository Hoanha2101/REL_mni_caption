import gymnasium as gym
import numpy as np
import pygame
from pygame.locals import QUIT
from gymnasium import spaces
from matplotlib import colormaps, colors
from PIL import Image

class CuttingStockEnvOptimized(gym.Env):
    """
    An optimized cutting stock environment for reinforcement learning.

    This environment simulates a stock-cutting process in which products are
    placed onto stock grids. It enforces a structured placement policy:
      - The first product is always placed at the top-left corner (0,0).
      - The second product is forced to be placed immediately to the right of the first,
        regardless of size differences.
      - Subsequent products are placed in the first available position (scanning
        left-to-right, top-to-bottom) that is adjacent to an already placed product
        or touches a stock border, thereby minimizing unused space.

    The reward function has been improved so that:
      - For the very first product, if the agent chooses (0,0), it receives a high reward (e.g., double the product area).
      - Otherwise, a strong penalty is applied.
      - For subsequent placements, the reward equals the product area plus a bonus based on how compact
        the overall layout is (smaller bounding box compared to the total available area yields a higher bonus).

    The episode terminates when all products have been placed or when a maximum
    number of steps is reached.
    
    The state also includes a flag "first_product_placed" to help the agent know the current stage.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        render_mode=None,
        min_w=20,
        min_h=20,
        max_w=20,
        max_h=20,
        num_stocks=1,
        max_product_type=3,
        max_product_per_type=5,
        seed=42,
        max_steps=200,
    ):
        """
        Initialize the CuttingStockEnvOptimized environment.

        Args:
            render_mode (str, optional): The rendering mode ("human" or "rgb_array").
            min_w, min_h (int): Minimum width and height for the stock grids.
            max_w, max_h (int): Maximum width and height for the stock grids.
            num_stocks (int): Number of stock grids.
            max_product_type (int): Maximum number of product types.
            max_product_per_type (int): Maximum quantity per product type.
            seed (int): Seed for the random number generator.
            max_steps (int): Maximum steps allowed per episode.
        """
        super().__init__()
        self.render_mode = render_mode
        self.seed_val = seed
        self.rng = np.random.default_rng(seed)

        self.min_w, self.min_h = min_w, min_h
        self.max_w, self.max_h = max_w, max_h
        self.num_stocks = num_stocks
        self.max_product_type = max_product_type
        self.max_product_per_type = max_product_per_type
        self.max_steps = max_steps

        # Array indicating whether each stock has been cut.
        self.cutted_stocks = np.zeros(self.num_stocks, dtype=int)

        # Variables to support forced placement:
        # - For the first product, force placement at (0,0).
        # - For the second product, if only the first product is placed, force placement immediately to its right.
        self.first_product_placed = False
        self.first_product_size = None  # (width, height) of the first product

        # Define observation space.
        stock_low = -2 * np.ones((self.max_w, self.max_h), dtype=int)
        stock_high = np.full((self.max_w, self.max_h), self.max_product_type + 2, dtype=int)
        self.observation_space = spaces.Dict({
            "stocks": spaces.Tuple([
                spaces.Box(low=stock_low, high=stock_high, shape=(self.max_w, self.max_h), dtype=int)
                for _ in range(self.num_stocks)
            ]),
            "products": spaces.Box(low=0, high=100, shape=(self.max_product_type,), dtype=int),
            "first_product_placed": spaces.Discrete(2)  # 0 (False) or 1 (True)
        })

        # Define the action space.
        self.action_space = spaces.Dict({
            "stock_idx": spaces.Discrete(self.num_stocks),
            "size": spaces.Box(
                low=np.array([1, 1]),
                high=np.array([self.max_w, self.max_h]),
                shape=(2,),
                dtype=int
            ),
            "position": spaces.Box(
                low=np.array([0, 0]),
                high=np.array([self.max_w - 1, self.max_h - 1]),
                shape=(2,),
                dtype=int
            )
        })

        self._stocks = None
        self._products = None

        self.current_step = 0

        # Rendering variables.
        self.window = None
        self.clock = None
        self.frames = []
        self.gif_names = ["demo/combine.gif", "demo/bestfit.gif", "demo/firstfit.gif"]

        self.reset()

    def _get_obs(self):
        """
        Get the current observation.

        Returns:
            dict: Contains "stocks", "products", and "first_product_placed" flag.
        """
        prod_vector = np.array([p["quantity"] for p in self._products], dtype=int)
        return {
            "stocks": self._stocks,
            "products": prod_vector,
            "first_product_placed": int(self.first_product_placed)
        }

    def _get_info(self):
        """
        Get supplementary information about the current state.

        Returns:
            dict: Contains the filled ratio, total used cells, waste rate, and total waste.
        """
        total_used = 0
        total_waste = 0
        for stock in self._stocks:
            if stock[0, 0] >= 0:
                total_waste += np.count_nonzero(stock == -1)
                total_used += np.count_nonzero(stock > -2)
        filled_ratio = float(np.mean(self.cutted_stocks))
        waste_rate = round(total_waste / (total_used + 1e-7), 2)
        return {
            "filled_ratio": filled_ratio,
            "total used": int(total_used),
            "wasted rate": waste_rate,
            "wasted total": int(total_waste)
        }

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.

        Returns:
            tuple: (observation, info)
        """
        self.frames = []
        self.current_step = 0
        self.cutted_stocks = np.zeros(self.num_stocks, dtype=int)
        self.first_product_placed = False
        self.first_product_size = None

        # Create new stocks.
        self._stocks = tuple(self._create_stock() for _ in range(self.num_stocks))

        # Create new products with random sizes and quantities.
        num_types = self.rng.integers(1, self.max_product_type + 1)
        products = []
        for _ in range(num_types):
            w = self.rng.integers(1, max(2, self.min_w // 2) + 1)
            h = self.rng.integers(1, max(2, self.min_h // 2) + 1)
            quantity = self.rng.integers(1, self.max_product_per_type + 1)
            products.append({"size": (w, h), "quantity": quantity})
        while len(products) < self.max_product_type:
            products.append({"size": (1, 1), "quantity": 0})
        self._products = tuple(products)

        return self._get_obs(), self._get_info()

    def _create_stock(self):
        """
        Create a single stock grid.

        Returns:
            np.ndarray: A 2D array representing the stock grid.
        """
        width = self.rng.integers(self.min_w, self.max_w + 1)
        height = self.rng.integers(self.min_h, self.max_h + 1)
        stock = -2 * np.ones((self.max_w, self.max_h), dtype=int)
        stock[:width, :height] = -1
        return stock

    def _get_filled_bounding_box(self, stock):
        """
        Compute the bounding box for cells that have been filled (>= 0).

        Returns:
            tuple or None: (min_row, min_col, max_row, max_col) if any filled cell exists; otherwise None.
        """
        rows, cols = np.where(stock >= 0)
        if len(rows) == 0 or len(cols) == 0:
            return None
        return (rows.min(), cols.min(), rows.max(), cols.max())

    def _find_forced_position(self, stock, width, height):
        """
        Search for the first available empty region of size (width x height) that is adjacent
        to an already placed product or touches a stock border.

        Returns:
            tuple or None: (x, y) if found; else None.
        """
        valid_w = np.count_nonzero(np.any(stock != -2, axis=1))
        valid_h = np.count_nonzero(np.any(stock != -2, axis=0))
        for i in range(valid_w - width + 1):
            for j in range(valid_h - height + 1):
                region = stock[i:i+width, j:j+height]
                if np.all(region == -1) and self._is_valid_placement(stock, i, j, width, height):
                    return i, j
        return None

    def _force_position(self, stock, width, height):
        """
        Enforce the structured placement policy:
          - If no product is placed, return (0,0).
          - If only the first product is placed, force placement immediately to its right.
          - Otherwise, search for the first valid available position.

        Returns:
            tuple or None: The forced (x, y) position.
        """
        valid_w = np.count_nonzero(np.any(stock != -2, axis=1))
        valid_h = np.count_nonzero(np.any(stock != -2, axis=0))
        if not np.any(stock >= 0):
            return 0, 0

        total_filled = np.count_nonzero(stock >= 0)
        first_area = None
        if self.first_product_size is not None:
            first_area = self.first_product_size[0] * self.first_product_size[1]
        # Nếu chỉ có sản phẩm đầu tiên, ép đặt ngay bên phải của nó.
        if first_area is not None and total_filled == first_area:
            bbox = self._get_filled_bounding_box(stock)
            if bbox is not None:
                forced_x = bbox[0]       # Giữ nguyên hàng của sản phẩm đầu tiên.
                forced_y = bbox[3] + 1   # Ngay bên phải của sản phẩm đầu tiên.
                if forced_x + width <= valid_w and forced_y + height <= valid_h:
                    return forced_x, forced_y
        pos = self._find_forced_position(stock, width, height)
        return pos

    def _is_valid_placement(self, stock, x, y, width, height):
        """
        Check if placing a product at (x, y) with size (width, height) is valid.
        A placement is valid if it touches a border or is adjacent to at least one placed product.

        Returns:
            bool: True if valid, False otherwise.
        """
        valid_w = np.count_nonzero(np.any(stock != -2, axis=1))
        valid_h = np.count_nonzero(np.any(stock != -2, axis=0))
        if x == 0 or y == 0 or (x + width) == valid_w or (y + height) == valid_h:
            return True
        adjacent = False
        if x > 0 and np.any(stock[x - 1, y:y+height] >= 0):
            adjacent = True
        if not adjacent and (x + width) < valid_w and np.any(stock[x + width, y:y+height] >= 0):
            adjacent = True
        if not adjacent and y > 0 and np.any(stock[x:x+width, y - 1] >= 0):
            adjacent = True
        if not adjacent and (y + height) < valid_h and np.any(stock[x:x+width, y+height] >= 0):
            adjacent = True
        return adjacent

    def step(self, action):
        """
        Execute one step in the environment.

        Enforces the structured placement policy:
          - Overrides the agent's chosen position with a forced position.
          - If placement is successful, reward = product_area + bonus.
            Bonus is computed as: bonus = (1 - bbox_area / total_area) * alpha.
          - Additionally, for the very first product, if the agent selects (0,0),
            reward is multiplied (e.g., doubled).
          - Otherwise, a penalty (-5) is returned.

        Returns:
            tuple: (observation, reward, done, truncated, info)
        """
        self.current_step += 1
        stock_idx = action["stock_idx"]

        # Xác định kích thước sản phẩm từ action.
        if "size" in action:
            size = action["size"]
        elif "product_idx" in action:
            product_idx = action["product_idx"]
            size = self._products[product_idx]["size"]
        else:
            raise KeyError("Action must include either 'size' or 'product_idx'")

        position = action["position"]
        width, height = int(size[0]), int(size[1])
        x, y = int(position[0]), int(position[1])
        reward = 0

        print(f"[DEBUG] step={self.current_step}, stock_idx={stock_idx}, size=({width},{height}), pos=({x},{y})")

        product_idx = None
        if "product_idx" in action:
            product_idx = action["product_idx"]
            if self._products[product_idx]["quantity"] <= 0:
                product_idx = None
        else:
            for i, prod in enumerate(self._products):
                if np.array_equal(prod["size"], (width, height)) and prod["quantity"] > 0:
                    product_idx = i
                    break

        if product_idx is not None and 0 <= stock_idx < self.num_stocks:
            stock = self._stocks[stock_idx]
            valid_w = np.count_nonzero(np.any(stock != -2, axis=1))
            valid_h = np.count_nonzero(np.any(stock != -2, axis=0))
            if x + width > valid_w or y + height > valid_h:
                reward = -5
                print("[DEBUG] Product doesn't fit -> reward = -5")
            else:
                forced_pos = self._force_position(stock, width, height)
                if forced_pos is None:
                    reward = -5
                    print("[DEBUG] No forced position found -> reward = -5")
                else:
                    x, y = forced_pos
                    print(f"[DEBUG] Forced placement at ({x},{y})")
                
                region = stock[x:x+width, y:y+height]
                if reward == 0 and np.all(region == -1):
                    # Nếu chưa có sản phẩm nào, kiểm tra:
                    if not np.any(stock >= 0):
                        if (x, y) == (0, 0):
                            # Nếu chọn đúng (0,0), thưởng gấp đôi.
                            base_reward = width * height
                            reward = 2 * base_reward
                        else:
                            reward = -10
                            print("[DEBUG] First product must be at (0,0) -> reward = -10")
                            return self._get_obs(), reward, False, False, self._get_info()
                    else:
                        base_reward = width * height

                    # Place the product.
                    self._stocks = list(self._stocks)
                    new_stock = self._stocks[stock_idx].copy()
                    new_stock[x:x+width, y:y+height] = product_idx
                    self._stocks[stock_idx] = new_stock
                    self._stocks = tuple(self._stocks)

                    self._products = list(self._products)
                    self._products[product_idx]["quantity"] -= 1
                    self._products = tuple(self._products)

                    # Tính bonus cho layout gọn gàng.
                    bbox_after = self._get_filled_bounding_box(new_stock)
                    if bbox_after is not None:
                        bbox_area = (bbox_after[2] - bbox_after[0] + 1) * (bbox_after[3] - bbox_after[1] + 1)
                    else:
                        bbox_area = base_reward
                    total_area = valid_w * valid_h
                    bonus = (1 - bbox_area / total_area)
                    alpha = 1.0  # Hệ số điều chỉnh bonus.
                    reward = base_reward + alpha * bonus * base_reward

                    self.cutted_stocks[stock_idx] = 1
                    print(f"[DEBUG] Placed product {product_idx} at ({x},{y}), reward={reward}")

                    if not self.first_product_placed:
                        self.first_product_placed = True
                        self.first_product_size = (width, height)
                else:
                    reward = -5
                    print("[DEBUG] Region not empty -> reward = -5")
        else:
            reward = -5
            print("[DEBUG] Invalid product or stock -> reward = -5")

        done = all(prod["quantity"] <= 0 for prod in self._products)
        if self.current_step >= self.max_steps:
            print(f"[DEBUG] Reached max_steps={self.max_steps}, force done.")
            done = True

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
            if reward > 0 and done:
                if self.frames:
                    for _ in range(100):
                        self.frames.append(self.frames[-1])
                    self.frames[0].save(self.gif_names.pop(0),
                                          save_all=True,
                                          append_images=self.frames[1:],
                                          duration=100,
                                          loop=0)
                    self.gif_names.append("demo/newgif.gif")
                    self.frames.clear()

        return obs, reward, done, False, info

    def render(self):
        """
        Render the current state of the environment.

        Returns:
            np.ndarray or None: An RGB array if render_mode is "rgb_array", otherwise None.
        """
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _get_window_size(self):
        """
        Calculate the window size for rendering based on the number of stocks.

        Returns:
            tuple: (width, height) of the display window.
        """
        cols = int(np.ceil(np.sqrt(self.num_stocks)))
        rows = int(np.ceil(self.num_stocks / cols))
        return (cols * self.max_w * 20, rows * self.max_h * 20)

    def _render_frame(self):
        """
        Render a frame of the environment for visualization.

        Returns:
            np.ndarray or None: An RGB array if render_mode is "rgb_array", otherwise None.
        """
        window_size = self._get_window_size()
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption("Cutting Stock Env - Optimized")
            self.window = pygame.display.set_mode(window_size)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(window_size)
        canvas.fill((0, 0, 0))
        pix_square_size = 20

        cmap = colormaps.get_cmap("hsv")
        norm = colors.Normalize(vmin=0, vmax=self.max_product_type - 1)
        list_colors = [cmap(norm(i)) for i in range(self.max_product_type + 1)]
        list_colors[0] = (1, 1, 1, 1)

        cols = window_size[0] // (self.max_w * pix_square_size)
        for i, stock in enumerate(self._stocks):
            pos_x = (i % cols) * self.max_w * pix_square_size
            pos_y = (i // cols) * self.max_h * pix_square_size

            stock_w = int(np.count_nonzero(np.any(stock != -2, axis=1)))
            stock_h = int(np.count_nonzero(np.any(stock != -2, axis=0)))

            pygame.draw.rect(canvas, (128, 128, 128),
                             pygame.Rect(pos_x, pos_y, stock_w * pix_square_size, stock_h * pix_square_size))

            for x in range(stock.shape[0]):
                for y in range(stock.shape[1]):
                    cell = stock[x, y]
                    if cell == -2:
                        continue
                    elif cell == -1:
                        color = (200, 200, 200)
                    else:
                        color_rgb = list_colors[cell][0:3]
                        color = (
                            int(color_rgb[0] * 255),
                            int(color_rgb[1] * 255),
                            int(color_rgb[2] * 255)
                        )
                    rect = pygame.Rect(pos_x + x * pix_square_size,
                                       pos_y + y * pix_square_size,
                                       pix_square_size, pix_square_size)
                    pygame.draw.rect(canvas, color, rect)

            pygame.draw.rect(canvas, (255, 255, 255),
                             pygame.Rect(pos_x, pos_y, self.max_w * pix_square_size, self.max_h * pix_square_size), 1)
        
        for i in range((window_size[0] // pix_square_size) + 1):
            pygame.draw.line(canvas, (255, 255, 255),
                             (i * pix_square_size, 0),
                             (i * pix_square_size, window_size[1]))
        for i in range((window_size[1] // pix_square_size) + 1):
            pygame.draw.line(canvas, (255, 255, 255),
                             (0, i * pix_square_size),
                             (window_size[0], i * pix_square_size))

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(pygame.surfarray.array3d(canvas), (1, 0, 2))
        
        frame_data = np.transpose(pygame.surfarray.array3d(canvas), (1, 0, 2))
        self.frames.append(Image.fromarray(frame_data))

    def close(self):
        """
        Clean up rendering resources and close the display window.
        """
        if self.window is not None:
            pygame.display.quit()
            pygame.font.quit()
