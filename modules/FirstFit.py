import sys
sys.path.append("../")
from utils.GenerateColors import generate_colors
from libs.libs import *


class StockSheet:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.placements = []  # Lưu vị trí đặt sản phẩm

    def can_place(self, x, y, w, h):
        """Kiểm tra xem sản phẩm có thể đặt tại (x, y) không?"""
        if x + w > self.width or y + h > self.height:
            return False  # Vượt ra khỏi stock
        for px, py, pw, ph in self.placements:
            if not (x + w <= px or x >= px + pw or y + h <= py or y >= py + ph):
                return False  # Bị chồng lên sản phẩm khác
        return True

    def place_product(self, w, h):
        """Tìm vị trí trống và đặt sản phẩm vào"""
        for y in range(0, self.height - h + 1, 1):  # Duyệt theo từng dòng (bước nhảy 1)
            for x in range(0, self.width - w + 1, 1):  # Duyệt từng cột
                if self.can_place(x, y, w, h):
                    self.placements.append((x, y, w, h))
                    return True
        return False  # Không thể đặt

class CuttingStockProblem2D:
    def __init__(self, stock_sizes, products):
        self.stocks = sorted(stock_sizes, key=lambda x: x[0] * x[1], reverse=True)
        self.products = sorted(products, key=lambda x: x[0] * x[1], reverse=True)
        self.stock_sheets = {size: [] for size in self.stocks}

    def solve(self):
        for w, h in self.products:
            placed = False
            for size in self.stocks:  
                if len(self.stock_sheets[size]) == 0:
                    new_stock = StockSheet(size[0], size[1])
                    new_stock.place_product(w, h)
                    self.stock_sheets[size].append(new_stock)
                    placed = True
                    break 
                for stock in self.stock_sheets[size]:
                    if stock.place_product(w, h):
                        placed = True
                        break
                if placed:
                    break
                
    def visualize(self):
        """Tạo danh sách hình ảnh minh họa các tấm stock sau khi cắt."""
        unique_sizes = list(set((w, h) for w, h in self.products))  # Lấy danh sách các kích thước sản phẩm duy nhất
        colors = generate_colors(len(unique_sizes))  # Sinh màu sắc cho mỗi loại
        color_map = {size: colors[i] for i, size in enumerate(unique_sizes)}  # Ánh xạ kích thước vào màu sắc
        
        images = []  # Danh sách ảnh để return
        for i,(size, stock_list) in enumerate(self.stock_sheets.items()):
            for _, stock in enumerate(stock_list, 1):
                fig, ax = plt.subplots()
                ax.set_xlim(0, stock.width)
                ax.set_ylim(0, stock.height)
                ax.set_title(f"Stock {i + 1} ({size[0]}x{size[1]})")

                ax.add_patch(patches.Rectangle((0, 0), stock.width, stock.height, linewidth=2, edgecolor='black', facecolor='none'))

                for x, y, w, h in stock.placements:
                    color = color_map.get((w, h), "gray")  # Lấy màu theo kích thước sản phẩm
                    ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='black', facecolor=color))
                    ax.text(x + w / 2, y + h / 2, f"{w}x{h}", color='black', ha='center', va='center')

                plt.gca().set_aspect('equal', adjustable='box')

                # Lưu vào buffer
                buf = BytesIO()
                fig.canvas.print_png(buf)
                buf.seek(0)
                img = Image.open(buf)
                images.append(img)

                plt.close(fig)  # Đóng figure để tránh chiếm bộ nhớ

        return images  # Trả về danh sách ảnh

    

