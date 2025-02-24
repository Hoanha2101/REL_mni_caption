from modules import FirstFit, BestFit, CombinationAlgorithm
import pandas as pd
from IPython.display import display

stocks = [(100, 100), (80, 80), (120, 120)]  # Các kích thước tấm vải khác nhau
products = [(50, 100),(40, 30), (30, 30), (60, 60), (20, 20), (40, 50), (50, 50)]  # Các mảnh vải cần cắt

# cutting_problem = FirstFit.CuttingStockProblem2D(stocks, products)
# cutting_problem.solve()
# images = cutting_problem.visualize()

best_fit_cutting = BestFit.BestFitCuttingStock(stocks, products)
best_fit_cutting.solve()
images = best_fit_cutting.visualize()

# combination_cutting = CombinationAlgorithm.CombinationCuttingStock(stocks, products)
# combination_cutting.solve()
# images = combination_cutting.visualize()

# Hiển thị từng ảnh trong danh sách images
for img in images:
    display(img)

# Hoặc lưu ảnh ra file
for idx, img in enumerate(images):
    img.save(f"stock_visual_{idx + 1}.png")