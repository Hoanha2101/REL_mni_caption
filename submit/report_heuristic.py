import time
import pandas as pd
import matplotlib.pyplot as plt
from policy.FirstFit_Policy import first_fit_policy
from policy.BestFit_Policy import best_fit_policy
from policy.Combination_Policy import combination_policy
from env.cutting_stock import CuttingStockEnv
import numpy as np

# Đọc dữ liệu từ file CSV
df = pd.read_csv("data/data_custom.csv")

# Lấy danh sách batch_id
batch_ids = df["batch_id"].unique()

# Lưu kết quả
results = []

# Chạy từng batch với từng policy
for batch_id in batch_ids:
    print(f"-----batch_id: {batch_id}-----")
    df_batch = df[df["batch_id"] == batch_id]

    # Lấy danh sách stocks và products
    stock_list = df_batch[df_batch["type"] == "stock"][["width", "height"]].to_records(index=False).tolist()
    product_list = df_batch[df_batch["type"] == "product"][["width", "height"]].to_records(index=False).tolist()

    for policy_name, policy_fn in [("First-Fit", first_fit_policy), 
                                   ("Best-Fit", best_fit_policy), 
                                   ("Combination", combination_policy)]:
        print(f"Policy: {policy_name}")
        
        # Khởi tạo môi trường
        env = CuttingStockEnv(
            render_mode="human",
            max_w=200,
            max_h=200,
            seed=42,
            stock_list=stock_list,
            product_list=product_list,
        )

        # Reset môi trường
        obs, info = env.reset()

        # Đo thời gian chạy
        start_time = time.time()

        total_reward, steps = 0, 0
        done = False

        while not done:
            action = policy_fn(obs, info)
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
            steps += 1
        end_time = time.time()
        runtime = end_time - start_time

        # Tính số lượng stock chưa sử dụng
        remaining_stocks = sum(1 for stock in env._stocks if np.all(stock[stock != -2] == -1))

        # Tính số lượng stock đã được sử dụng
        used_stocks = len(env._stocks) - remaining_stocks

        # Tính tổng diện tích phần thừa của các stock đã sử dụng
        total_trim_loss = sum((stock == -1).sum() for stock in env._stocks if np.any(stock[stock != -2] != -1))

        # Tính trung bình diện tích của các stock đã được sử dụng
        used_stock_areas = [np.sum(stock != -2) for stock in env._stocks if np.any(stock[stock != -2] != -1)]
        avg_used_stock_area = sum(used_stock_areas) / len(used_stock_areas) if used_stock_areas else 0

        # Lưu kết quả
        results.append({
            "batch_id": batch_id,
            "policy": policy_name,
            "steps": steps,
            "runtime": runtime,
            "total_trim_loss": total_trim_loss, 
            "remaining_stocks": remaining_stocks,
            "used_stocks": used_stocks,
            "avg_used_stock_area": avg_used_stock_area
        })

        env.close()

# Chuyển kết quả thành DataFrame
df_results = pd.DataFrame(results)

# Lưu kết quả vào file CSV
csv_filename = "data/policy_comparison_results.csv"
df_results.to_csv(csv_filename, index=False)
