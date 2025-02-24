
import sys
sys.path.append("../")
from utils.GenerateColors import generate_colors
from libs.libs import *

class StockSheet:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.placements = []  # Danh sách các sản phẩm đã đặt trên tấm này

    def can_place(self, x, y, w, h):
        """Kiểm tra xem có thể đặt sản phẩm tại vị trí (x, y) hay không"""
        if x + w > self.width or y + h > self.height:
            return False  # Vượt quá giới hạn tấm stock
        for px, py, pw, ph in self.placements:
            if not (x + w <= px or x >= px + pw or y + h <= py or y >= py + ph):
                return False  # Bị chồng lên sản phẩm khác
        return True

    def first_fit_place_product(self, w, h):
        """Áp dụng First-Fit: Đặt sản phẩm vào vị trí trống đầu tiên"""
        for y in range(0, self.height - h + 1):
            for x in range(0, self.width - w + 1):
                if self.can_place(x, y, w, h):
                    self.placements.append((x, y, w, h))
                    return True
        return False

    def best_fit_refine(self):
        """Áp dụng Best-Fit để tối ưu vị trí các sản phẩm đã đặt"""
        sorted_placements = sorted(self.placements, key=lambda p: p[2] * p[3], reverse=True)  # Sắp xếp theo diện tích
        self.placements = []
        for w, h in [(p[2], p[3]) for p in sorted_placements]:
            for y in range(0, self.height - h + 1):
                for x in range(0, self.width - w + 1):
                    if self.can_place(x, y, w, h):
                        self.placements.append((x, y, w, h))
                        break
                else:
                    continue
                break

class CombinationCuttingStock:
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
                    new_stock.first_fit_place_product(w, h)
                    self.stock_sheets[size].append(new_stock)
                    break
                for stock in self.stock_sheets[size]:
                    if stock.first_fit_place_product(w, h):
                        placed = True
                        break
                if placed:
                    break
            # if not placed:
            #     for size in self.stocks:
            #         if w <= size[0] and h <= size[1]:
            #             new_stock = StockSheet(size[0], size[1])
            #             new_stock.first_fit_place_product(w, h)
            #             self.stock_sheets[size].append(new_stock)
            #             break
        for size in self.stocks:
            for stock in self.stock_sheets[size]:
                stock.best_fit_refine()

    def visualize(self):
        """Hiển thị các tấm stock sau khi cắt, với mỗi loại kích thước sản phẩm có một màu khác nhau"""
        unique_sizes = list(set((w, h) for w, h in self.products))  # Lấy danh sách các kích thước sản phẩm duy nhất
        colors = generate_colors(len(unique_sizes))  # Sinh màu sắc cho mỗi loại
        color_map = {size: colors[i] for i, size in enumerate(unique_sizes)}  # Ánh xạ kích thước vào màu sắc
        
        images = []
        
        for i, (size, stock_list) in enumerate(self.stock_sheets.items()):
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



