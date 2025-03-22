# env/CuttingStockEnv.py

import gymnasium as gym
import numpy as np
import pygame
from pygame.locals import QUIT
from gymnasium import spaces
from matplotlib import colormaps, colors
from PIL import Image

class CuttingStockEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, seed=42):
        super().__init__()
        self.render_mode = render_mode
        self.seed_val = seed
        self.rng = np.random.default_rng(seed)
        
        # Define stock sizes (from Table 1)
        # Each tuple: (Length, Width) in inches
        self.stock_sizes = [
            (24, 14),
            (24, 13),
            (18, 10),
            (13, 10)
        ]
        
        # Define product demands (from Table 2)
        # Each tuple: (Length, Width, Quantity)
        self.product_demands = [
            (2, 1, 5),
            (4, 2, 5),
            (5, 3, 2),
            (7, 4, 3),
            (8, 5, 2)
        ]
        
        # For grid representation, use the maximum dimensions among stocks
        self.max_w = max(s[0] for s in self.stock_sizes)
        self.max_h = max(s[1] for s in self.stock_sizes)
        self.num_stocks = len(self.stock_sizes)
        
        # Observation space:
        # "stocks": a Tuple of stock grids (each of shape (max_w, max_h))
        #   - Cells: -2 = out-of-bound, -1 = empty, 0,1,2,... = product placements
        # "products": a vector of remaining quantities for each product
        stock_low = -2 * np.ones((self.max_w, self.max_h), dtype=int)
        stock_high = np.full((self.max_w, self.max_h), len(self.product_demands))
        self.observation_space = spaces.Dict({
            "stocks": spaces.Tuple(
                [spaces.Box(low=stock_low, high=stock_high, shape=(self.max_w, self.max_h), dtype=int)
                 for _ in range(self.num_stocks)]
            ),
            "products": spaces.Box(low=0, high=100, shape=(len(self.product_demands),), dtype=int)
        })
        
        # Action space:
        # A dictionary with:
        #   - "stock_idx": which stock (Discrete(num_stocks))
        #   - "product_idx": which product (Discrete(number of products))
        #   - "position": where to place it (Box with shape (2,) for x, y)
        self.action_space = spaces.Dict({
            "stock_idx": spaces.Discrete(self.num_stocks),
            "product_idx": spaces.Discrete(len(self.product_demands)),
            "position": spaces.Box(low=np.array([0, 0]),
                                   high=np.array([self.max_w - 1, self.max_h - 1]),
                                   shape=(2,),
                                   dtype=int)
        })
        
        # Internal states
        self._stocks = None
        self._products = None
        
        # Rendering variables
        self.window = None
        self.clock = None
        self.frames = []
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        # Initialize stocks based on defined stock sizes.
        self._stocks = []
        for (length, width) in self.stock_sizes:
            # Create a grid of size (max_w, max_h)
            # Cells with -2 represent areas outside the stock;
            # cells within the stock (up to the defined length & width) are initialized to -1 (empty).
            grid = -2 * np.ones((self.max_w, self.max_h), dtype=int)
            grid[:length, :width] = -1
            self._stocks.append(grid)
        self._stocks = tuple(self._stocks)
        
        # Initialize products: each product is a dictionary with its "size" and remaining "quantity".
        products = []
        for (length, width, quantity) in self.product_demands:
            products.append({"size": (length, width), "quantity": quantity})
        self._products = tuple(products)
        
        # Build observation: products vector contains remaining quantities.
        product_obs = np.array([prod["quantity"] for prod in self._products], dtype=int)
        obs = {"stocks": self._stocks, "products": product_obs}
        return obs, {}
    
    def step(self, action):
        # Unpack the action dictionary.
        stock_idx = action["stock_idx"]
        product_idx = action["product_idx"]
        pos = action["position"]
        x, y = int(pos[0]), int(pos[1])
        
        # Get the chosen stock grid and product details.
        stock = self._stocks[stock_idx].copy()  # work on a copy
        product = self._products[product_idx]
        p_length, p_width = product["size"]
        
        reward = 0
        
        # Determine the valid region size of the stock (cells != -2).
        valid_w = np.count_nonzero(np.any(stock != -2, axis=1))
        valid_h = np.count_nonzero(np.any(stock != -2, axis=0))
        if x + p_length > valid_w or y + p_width > valid_h:
            reward = -5  # penalty if product does not fit within stock boundaries
        else:
            # Check if the target region is empty (all cells equal -1).
            region = stock[x:x+p_length, y:y+p_width]
            if np.all(region == -1):
                # Place the product: mark the region with product index (0-indexed)
                self._stocks = list(self._stocks)
                new_stock = self._stocks[stock_idx].copy()
                new_stock[x:x+p_length, y:y+p_width] = product_idx
                self._stocks[stock_idx] = new_stock
                self._stocks = tuple(self._stocks)
                
                # Decrease the product's remaining quantity.
                self._products = list(self._products)
                self._products[product_idx]["quantity"] -= 1
                self._products = tuple(self._products)
                
                reward = p_length * p_width  # reward proportional to the area placed
            else:
                reward = -5  # penalty for attempting to place on an occupied area
        
        # Check termination: when all product quantities are zero.
        done = all(prod["quantity"] <= 0 for prod in self._products)
        product_obs = np.array([prod["quantity"] for prod in self._products], dtype=int)
        obs = {"stocks": self._stocks, "products": product_obs}
        info = {}  # Additional info can be added here
        
        return obs, reward, done, False, info
    
    def render(self):
        # Rendering via pygame.
        window_size = self._get_window_size()
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption("Cutting Stock Environment")
            self.window = pygame.display.set_mode(window_size)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface(window_size)
        canvas.fill((0, 0, 0))
        pix_square_size = 5  # scale factor for visibility
        
        # Use a colormap to assign colors to products.
        cmap = colormaps.get_cmap("tab20")
        cols = int(np.ceil(np.sqrt(self.num_stocks)))
        for i, stock in enumerate(self._stocks):
            pos_x = (i % cols) * self.max_w * pix_square_size
            pos_y = (i // cols) * self.max_h * pix_square_size
            # Draw each cell of the stock grid.
            for x in range(self.max_w):
                for y in range(self.max_h):
                    cell = stock[x, y]
                    if cell == -2:
                        continue  # out-of-bound area
                    elif cell == -1:
                        color = (200, 200, 200)  # empty cell: light gray
                    else:
                        # Use the product index to pick a color.
                        color_rgb = cmap(cell / (len(self._products)))[0:3]
                        color = (int(color_rgb[0] * 255), int(color_rgb[1] * 255), int(color_rgb[2] * 255))
                    rect = pygame.Rect(pos_x + x * pix_square_size,
                                       pos_y + y * pix_square_size,
                                       pix_square_size, pix_square_size)
                    pygame.draw.rect(canvas, color, rect)
            # Draw a white border around each stock.
            pygame.draw.rect(canvas, (255, 255, 255),
                             pygame.Rect(pos_x, pos_y, self.max_w * pix_square_size, self.max_h * pix_square_size), 1)
        
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.array(pygame.surfarray.pixels3d(canvas)).transpose(1, 0, 2)
        
        # Save frame for GIF creation.
        frame = pygame.surfarray.array3d(canvas).transpose(1, 0, 2)
        self.frames.append(Image.fromarray(frame))
    
    def _get_window_size(self):
        # Arrange stocks in a grid (nearly square)
        cols = int(np.ceil(np.sqrt(self.num_stocks)))
        rows = int(np.ceil(self.num_stocks / cols))
        return (cols * self.max_w * 5, rows * self.max_h * 5)
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.font.quit()
