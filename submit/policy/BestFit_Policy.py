################### ý tưởng ###############
# Duyệt qua từng sản phẩm theo thứ tự ban đầu.
# Tìm stock phù hợp nhất:
#     + Kiểm tra từng stock có thể chứa sản phẩm.
#     + Chọn stock để lại ít không gian dư nhất sau khi đặt sản phẩm.
# Nếu không có stock nào phù hợp, tạo một stock mới.
# Lặp lại cho đến khi tất cả sản phẩm được đặt xong.


import numpy as np



def best_fit_policy(observation, info):
    """
    Best-Fit Algorithm for 2D Cutting-Stock Problem.
    - Tìm stock để lại ít diện tích dư thừa nhất sau khi đặt sản phẩm.
    - Nếu không có stock phù hợp, mở một stock mới.
    """
    list_prods = observation["products"]
    list_stocks = observation["stocks"]

    best_stock_idx, best_pos_x, best_pos_y = None, None, None
    min_remaining_area = float("inf")

    for prod in list_prods:
        if prod["quantity"] <= 0:
            continue

        prod_w, prod_h = prod["size"]

        # Duyệt qua từng stock để tìm stock tốt nhất
        for idx, stock in enumerate(list_stocks):
            stock_w = np.sum(np.any(stock != -2, axis=1))
            stock_h = np.sum(np.any(stock != -2, axis=0))

            if stock_w < prod_w or stock_h < prod_h:
                continue  # Nếu stock không đủ lớn, bỏ qua

            # Tìm vị trí đặt sản phẩm sao cho diện tích dư thừa nhỏ nhất
            for x in range(stock_w - prod_w + 1):
                for y in range(stock_h - prod_h + 1):
                    if np.all(stock[x:x + prod_w, y:y + prod_h] == -1):
                        # Tính diện tích dư thừa sau khi đặt sản phẩm
                        remaining_area = (stock_w * stock_h) - (prod_w * prod_h)
                        if remaining_area < min_remaining_area:
                            best_stock_idx = idx
                            best_pos_x, best_pos_y = x, y
                            min_remaining_area = remaining_area

        if best_stock_idx is None:
            # Nếu không tìm thấy stock phù hợp, mở một stock mới
            new_stock_idx = len(list_stocks)
            return {
                "stock_idx": new_stock_idx,
                "size": (prod_w, prod_h),
                "position": (0, 0)
            }

        # Trả về hành động tốt nhất tìm được
        return {
            "stock_idx": best_stock_idx,
            "size": (prod_w, prod_h),
            "position": (best_pos_x, best_pos_y)
        }

    return {
        "stock_idx": 0,
        "size": (0, 0),
        "position": (0, 0)
    }  # Nếu không còn sản phẩm nào cần cắt

