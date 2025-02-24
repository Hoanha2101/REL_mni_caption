import matplotlib.pyplot as plt

def generate_colors(n):
    """Tạo danh sách màu sắc ngẫu nhiên"""
    cmap = plt.get_cmap("tab10") 
    colors = [cmap(i % 10) for i in range(n)]
    return colors