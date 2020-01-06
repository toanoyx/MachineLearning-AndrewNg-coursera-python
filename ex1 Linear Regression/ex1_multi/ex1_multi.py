import numpy as np
from feature_normalize import *
from gradient_descent_multi import *
from normal_eqn import *
import matplotlib.pyplot as plt

'''第1部分 特征缩放'''

print('Loading Data...')
data = np.loadtxt('ex1data2.txt', delimiter=',', dtype=np.int64)  # 加载txt格式数据集
X = data[:, 0:2]  # 得到输入变量矩阵,每个输入变量有两个输入特征
y = data[:, 2]  # 输出变量
m = y.size  # 样本数

# 打印前10个训练样本
print('First 10 examples from the dataset: ')
for i in range(0, 10):
    print('x = {}, y = {}'.format(X[i], y[i]))

input('Program paused. Press ENTER to continue')

# 特征缩放  不同特征的取值范围差异很大  通过特征缩放 使其在一个相近的范围内
print('Normalizing Features ...')

X, mu, sigma = feature_normalize(X)
X = np.c_[np.ones(m), X]  # 得到缩放后的特征矩阵  前面加一列1  方便矩阵运算

'''第2部分 梯度下降法'''

print('Running gradient descent ...')

alpha = 0.03  # 学习率
num_iters = 400  # 迭代次数

theta = np.zeros(3).reshape(-1, 1)  # 初始化参数 转变为列向量 用2维数组表示 避免不必要的麻烦
theta, J_history = gradient_descent_multi(X, y, theta, alpha, num_iters)  # 梯度下降求解参数

# 绘制代价函数值随迭代次数的变化曲线
plt.figure()
plt.plot(np.arange(J_history.size), J_history)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')

# 打印求解的最优的参数
print('Theta computed from gradient descent : \n{}'.format(theta))

# 预测面积是1650 卧室数是3 的房子的价格

x1 = np.array([1650, 3]).reshape(1, -1)

x1 = (x1 - mu) / sigma  # 对预测样例进行特征缩放

x1 = np.c_[1, x1]  # 前面增加一个1
price = h(x1, theta)  # 带入假设函数 求解预测值

print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent) : {:0.3f}'.format(price.squeeze()))

'''第3部分 正规方程法求解多元线性回归'''

# 正规方程法不用进行特征缩放

print('Solving with normal equations ...')

# Load data
data = np.loadtxt('ex1data2.txt', delimiter=',', dtype=np.int64)
X = data[:, 0:2]
y = data[:, 2]
m = y.size

# 增加一列特征1
X = np.c_[np.ones(m), X]

theta = normal_eqn(X, y)  # 正规方程法

# 打印求解的最优参数
print('Theta computed from the normal equations : \n{}'.format(theta))

# 预测面积是1650 卧室数是3 的房子的价格

x2 = np.array([1, 1650, 3]).reshape(1, -1)
price = h(x2, theta)  # 带入假设函数 求解预测值

print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations) : {:0.3f}'.format(price.squeeze()))
