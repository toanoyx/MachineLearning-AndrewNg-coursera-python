import matplotlib.pyplot as plt


def plot_data(x, y):
    plt.scatter(x, y, marker='o', s=50, cmap='Blues', alpha=0.5)  # 绘制散点图
    plt.xlabel('population')  # 设置x轴标题
    plt.ylabel('profits')  # 设置y轴标题
    plt.show()
