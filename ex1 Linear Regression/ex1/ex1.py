from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import axes3d, Axes3D
from gradient_descent import *
from plot_data import *


'''第1部分 可视化训练集'''

print('Plotting Data...')
data = np.loadtxt('ex1data1.txt', delimiter=',', usecols=(0, 1))  # 加载txt格式的数据集 每一行以","分隔
X = data[:, 0]  # 输入变量 第一列
y = data[:, 1]  # 输出变量 第二列
m = y.size  # 样本数

plt.ion()
plt.figure(0)
plot_data(X, y)  # 可视化数据集

'''第2部分 梯度下降法'''

print('Running Gradient Descent...')

X = np.c_[np.ones(m), X]  # 输入特征矩阵 前面增加一列1 方便矩阵运算 每行代表一个样本的特征向量
theta = np.zeros(2).reshape(-1, 1)  # 初始化两个参数为0 用2维数组表示列向量 避免不必要的麻烦

# 设置梯度下降迭代次数和学习率
iterations = 1500
alpha = 0.01

# 计算最开始的代价函数值  并与期望值比较 验证程序正确性
print('Initial cost : ' + str(compute_cost(X, y, theta)) + ' (This value should be about 32.07)')

# 使用梯度下降法求解线性回归 返回最优参数 以及每一步迭代后的代价函数值
theta, J_history = gradient_descent(X, y, theta, alpha, iterations)
print(theta.shape)

print('Theta found by gradient descent: ' + str(theta))

# 在数据集上绘制出拟合的直线
plt.figure(0)

line1, = plt.plot(X[:, 1], np.dot(X, theta), label='Linear Regression')
plot_data(X[:, 1], y)  # 可视化数据集
plt.legend(handles=[line1])

input('Program paused. Press ENTER to continue')

# 用训练好的参数 预测人口为3.5*1000时 收益为多少  并与期望值比较 验证程序正确性
predict1 = np.dot(np.array([1, 3.5]).reshape(1, -1), theta)
print('For population = 35,000, we predict a profit of {:0.3f} (This value should be about 4519.77)'.format(
    predict1.squeeze() * 10000))
# 用训练好的参数 预测人口为7*1000时 收益为多少  并与期望值比较 验证程序正确性
predict2 = np.dot(np.array([1, 7]).reshape(1, -1), theta)
print('For population = 70,000, we predict a profit of {:0.3f} (This value should be about 45342.45)'.format(
    predict2.squeeze() * 10000))

'''第3部分 可视化代价函数'''

print('Visualizing J(theta0, theta1) ...')

theta0_vals = np.linspace(-10, 10, 100)  # 参数1的取值
theta1_vals = np.linspace(-1, 4, 100)  # 参数2的取值

xs, ys = np.meshgrid(theta0_vals, theta1_vals)  # 生成网格
J_vals = np.zeros(xs.shape)

for i in range(0, theta0_vals.size):
    for j in range(0, theta1_vals.size):
        t = np.array([theta0_vals[i], theta1_vals[j]]).reshape(-1, 1)
        J_vals[i][j] = compute_cost(X, y, t)  # 计算每个网格点的代价函数值

J_vals = np.transpose(J_vals)

fig1 = plt.figure(1)  # 绘制3d图形
ax = fig1.gca(projection='3d')
ax.plot_surface(xs, ys, J_vals)
plt.xlabel(r'$\theta_0$')  # Python可以识别LaTex语法
plt.ylabel(r'$\theta_1$')

# 绘制等高线图 相当于3d图形的投影
plt.figure(2)
lvls = np.logspace(-2, 3, 20)
plt.contour(xs, ys, J_vals, levels=lvls, norm=LogNorm())
plt.plot(theta[0, 0], theta[1, 0], c='r', marker="x")
plt.show()
