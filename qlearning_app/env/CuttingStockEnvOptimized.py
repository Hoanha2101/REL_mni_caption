import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
import matplotlib as mpl
from matplotlib import colormaps
from PIL import Image

def set_seed(seed):
    """
    Sets the random seed for numpy to ensure reproducibility.
    
    Args:
        seed (int): The seed value for random number generation.
    """
    np.random.seed(seed)

class CuttingStockEnvOptimized(gym.Env):
    """
    An optimized environment for the Cutting Stock problem. This environment follows the logic from 'cutting_stock.py'.
    It takes user inputs for stock and product lists and simulates the cutting stock problem.

    Args:
        render_mode (str, optional): The rendering mode. Options include "human" or "rgb_array".
        min_w (int, optional): The minimum width of a stock (default is 50).
        min_h (int, optional): The minimum height of a stock (default is 50).
        max_w (int, optional): The maximum width of a stock (default is 120).
        max_h (int, optional): The maximum height of a stock (default is 120).
        num_stocks (int, optional): The number of stocks to generate (default is 10).
        max_product_type (int, optional): The maximum number of different product types (default is 25).
        max_product_per_type (int, optional): The maximum number of products per type (default is 20).
        seed (int, optional): The random seed (default is 42).
        stock_list (list, optional): List of stocks provided by the user (default is None).
        product_list (list, optional): List of products provided by the user (default is None).
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 500}

    def __init__(self, render_mode=None, 
                 min_w=50, 
                 min_h=50, 
                 max_w=120, 
                 max_h=120,
                 num_stocks=10, 
                 max_product_type=25, max_product_per_type=20,
                 seed=42, stock_list=None, product_list=None):
        self.seed = seed
        set_seed(seed)
        self.render_mode = render_mode
        self.min_w = min_w
        self.min_h = min_h
        self.max_w = max_w
        self.max_h = max_h
        
        if stock_list is not None:
            self.stock_list = stock_list
            self.num_stocks = len(stock_list)
        else:
            self.stock_list = None
            self.num_stocks = num_stocks

        self.max_product_type = max_product_type
        self.max_product_per_type = max_product_per_type
        self.product_list = product_list

        self.cutted_stocks = np.zeros(self.num_stocks, dtype=int)

        # Define observation space: stocks and products
        stock_low = -2 * np.ones((self.max_w, self.max_h), dtype=int)
        stock_high = np.full((self.max_w, self.max_h), max_product_type + 2, dtype=int)
        self.observation_space = spaces.Dict({
            "stocks": spaces.Tuple([spaces.Box(low=stock_low, high=stock_high, shape=(self.max_w, self.max_h), dtype=int)
                                    for _ in range(self.num_stocks)]),
            "products": spaces.Sequence(
                spaces.Dict({
                    "size": spaces.Box(low=np.array([1,1]), high=np.array([self.max_w, self.max_h]), shape=(2,), dtype=int),
                    "quantity": spaces.Discrete(max_product_per_type+1)
                })
            )
        })

        self.action_space = spaces.Dict({
            "stock_idx": spaces.Discrete(self.num_stocks),
            "size": spaces.Box(low=np.array([1,1]), high=np.array([self.max_w, self.max_h]), shape=(2,), dtype=int),
            "position": spaces.Box(low=np.array([0,0]), high=np.array([self.max_w-1, self.max_h-1]), shape=(2,), dtype=int)
        })

        self._stocks = []
        self._products = []

        self.window = None
        self.clock = None

    def _get_obs(self):
        """
        Returns the current observation of stocks and products.
        
        Returns:
            dict: The current observation, which includes the stocks and products.
        """
        return {"stocks": self._stocks, "products": self._products}

    def _get_info(self):
        """
        Returns additional information about the environment such as filled ratio and trim loss.
        
        Returns:
            dict: Information about the current environment state.
        """
        filled_ratio = np.mean(self.cutted_stocks).item()
        trim_loss = []
        for sid, stock in enumerate(self._stocks):
            if self.cutted_stocks[sid] == 0:
                continue
            tl = (stock == -1).sum() / (stock != -2).sum()
            trim_loss.append(tl)
        trim_loss = np.mean(trim_loss).item() if trim_loss else 1
        return {"filled_ratio": filled_ratio, "trim_loss": trim_loss}

    def reset(self, seed=None, options=None):
        """
        Resets the environment and initializes the stocks and products.
        
        Args:
            seed (int, optional): The random seed for the reset (default is None).
            options (dict, optional): Additional options (default is None).
        
        Returns:
            tuple: The initial observation and information.
        """
        if seed is not None:
            self.seed = seed
            set_seed(seed)
        self.cutted_stocks = np.zeros(self.num_stocks, dtype=int)
        self._stocks = []
        
        # Initialize stocks:
        if self.stock_list is not None:
            for (w, h) in self.stock_list:
                stock = -2 * np.ones((self.max_w, self.max_h), dtype=int)
                stock[:w, :h] = -1
                self._stocks.append(stock)
        else:
            for _ in range(self.num_stocks):
                w = np.random.randint(self.min_w, self.max_w+1)
                h = np.random.randint(self.min_h, self.max_h+1)
                stock = -2 * np.ones((self.max_w, self.max_h), dtype=int)
                stock[:w, :h] = -1
                self._stocks.append(stock)
        self._stocks = tuple(self._stocks)

        # Initialize products:
        self._products = []
        if self.product_list is not None:
            for (w, h) in self.product_list:
                product = {"size": np.array([w, h]), "quantity": 1}
                self._products.append(product)
        else:
            num_products = np.random.randint(1, self.max_product_type+1)
            for _ in range(num_products):
                w = np.random.randint(1, self.min_w+1)
                h = np.random.randint(1, self.min_h+1)
                quantity = np.random.randint(1, self.max_product_per_type+1)
                product = {"size": np.array([w, h]), "quantity": quantity}
                self._products.append(product)
        self._products = tuple(self._products)

        obs = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_frame()
        return obs, info

    def step(self, action):
        stock_idx = action["stock_idx"]
        size = action["size"]
        position = action["position"]
        width, height = size
        x, y = position

        successful_cut = False  # <- Đánh dấu mặc định

        prod_idx = None
        for i, product in enumerate(self._products):
            if (np.array_equal(product["size"], np.array(size)) or 
                np.array_equal(product["size"], np.array(size)[::-1])) and product["quantity"] > 0:
                prod_idx = i
                break

        if prod_idx is not None and 0 <= stock_idx < self.num_stocks:
            stock = self._stocks[stock_idx]
            stock_width = int(np.sum(np.any(stock != -2, axis=1)))
            stock_height = int(np.sum(np.any(stock != -2, axis=0)))
            if x >= 0 and y >= 0 and x + width <= stock_width and y + height <= stock_height:
                if np.all(stock[x:x+width, y:y+height] == -1):
                    self.cutted_stocks[stock_idx] = 1
                    stock[x:x+width, y:y+height] = prod_idx
                    self._products[prod_idx]["quantity"] -= 1
                    successful_cut = True  # <- Ghi nhận thành công

        terminated = all(product["quantity"] == 0 for product in self._products)
        reward = 1 if terminated else 0

        obs = self._get_obs()
        info = self._get_info()
        info["successful_cut"] = successful_cut  # <- Trả về trạng thái này

        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, terminated, False, info


    def render(self):
        """
        Renders the current frame based on the render mode.
        
        Returns:
            np.ndarray: The rendered frame as an RGB array.
        """
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _get_window_size(self):
        """
        Calculates the window size for rendering the environment.
        
        Returns:
            tuple: The window size (width, height).
        """
        cols = int(np.ceil(np.sqrt(self.num_stocks)))
        rows = int(np.ceil(self.num_stocks / cols))
        return (cols * self.max_w, rows * self.max_h)

    def _render_frame(self):
        """
        Renders the environment frame, drawing the stocks and their products.
        """
        window_size = self._get_window_size()
        # scale = 3
        # scaled_size = (window_size[0] * scale, window_size[1] * scale)
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption("Cutting Stock Environment")
            self.window = pygame.display.set_mode(window_size)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(window_size)
        canvas.fill((0, 0, 0))
        pix_square_size = 1

        cmap = colormaps.get_cmap("hsv")
        norms = mpl.colors.Normalize(vmin=0, vmax=self.max_product_type - 1)
        list_colors = [cmap(norms(i))[:3] for i in range(self.max_product_type)]
        list_colors.extend([(1, 1, 1)] * 10)

        for i, stock in enumerate(self._stocks):
            stock_width = int(np.sum(np.any(stock != -2, axis=1)))
            stock_height = int(np.sum(np.any(stock != -2, axis=0)))
            offset_x = (i % (window_size[0] // self.max_w)) * self.max_w
            offset_y = (i // (window_size[0] // self.max_w)) * self.max_h
            pygame.draw.rect(canvas, (128, 128, 128),
                             pygame.Rect(offset_x, offset_y, stock_width, stock_height))
            for x in range(stock.shape[0]):
                for y in range(stock.shape[1]):
                    if stock[x, y] > -1:
                        idx = int(stock[x, y])
                        if 0 <= idx < len(list_colors):
                            color = list_colors[idx]
                        else:
                            color = (1, 1, 1)
                        color = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
                        pygame.draw.rect(canvas, color,
                                         pygame.Rect(offset_x + x, offset_y + y, pix_square_size, pix_square_size))
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        """
        Closes the rendering window and cleans up resources.
        """
        if self.window is not None:
            pygame.display.quit()
            pygame.font.quit()
