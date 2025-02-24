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

    def best_fit_place_product(self, w, h):
        """Tìm vị trí đặt sản phẩm sao cho diện tích dư thừa là ít nhất"""
        best_x, best_y, min_remaining_area = None, None, float('inf')

        for y in range(0, self.height - h + 1, 1):  # Quét từng dòng
            for x in range(0, self.width - w + 1, 1):  # Quét từng cột
                if self.can_place(x, y, w, h):
                    remaining_area = (self.width * self.height) - sum(pw * ph for _, _, pw, ph in self.placements) - (w * h)
                    if remaining_area < min_remaining_area:
                        best_x, best_y, min_remaining_area = x, y, remaining_area

        if best_x is not None:
            self.placements.append((best_x, best_y, w, h))
            return True
        return False

class BestFitCuttingStock:
    def __init__(self, stock_sizes, products):
        self.stocks = sorted(stock_sizes, key=lambda x: x[0] * x[1], reverse=True)
        self.products = sorted(products, key=lambda x: x[0] * x[1], reverse=True)
        self.stock_sheets = {size: [] for size in self.stocks}

    def solve(self):
        for w, h in self.products:
            best_stock = None
            best_stock_remaining_area = float('inf')
            placed = False
            # Tìm tấm stock phù hợp nhất
            for size in self.stocks:

                if len(self.stock_sheets[size]) == 0:
                    new_stock = StockSheet(size[0], size[1])
                    new_stock.best_fit_place_product(w, h)
                    self.stock_sheets[size].append(new_stock)
                    placed = True
                    break
                for stock in self.stock_sheets[size]:
                    before_area = (stock.width * stock.height) - sum(pw * ph for _, _, pw, ph in stock.placements)
                    if stock.best_fit_place_product(w, h):
                        after_area = (stock.width * stock.height) - sum(pw * ph for _, _, pw, ph in stock.placements)
                        remaining_area = after_area - before_area
                        if remaining_area < best_stock_remaining_area:
                            best_stock = stock
                            best_stock_remaining_area = remaining_area
                            placed = True
                if placed:
                    break


    def visualize(self):
        
        """Hiển thị các tấm stock sau khi cắt, với mỗi loại kích thước sản phẩm có một màu khác nhau"""
        unique_sizes = list(set((w, h) for w, h in self.products))  # Lấy danh sách các kích thước sản phẩm duy nhất
        colors = generate_colors(len(unique_sizes))  # Sinh màu sắc cho mỗi loại
        color_map = {size: colors[i] for i, size in enumerate(unique_sizes)}  # Ánh xạ kích thước vào màu sắc
        
        images = []
        
        """Hiển thị các tấm stock sau khi cắt"""
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