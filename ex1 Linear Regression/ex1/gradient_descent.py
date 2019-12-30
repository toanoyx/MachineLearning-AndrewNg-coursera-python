import numpy as np
from compute_cost import *


def gradient_descent(X, y, theta, alpha, num_iters):
    m = y.size  # 样本数
    J_history = np.zeros(num_iters)  # 每一次迭代都有一个代价函数值
    y = y.reshape(-1, 1)  # 用2维数组表示列向量 (m,1)

    for i in range(0, num_iters):  # num_iters次迭代优化
        theta = theta - (alpha / m) * X.T.dot(h(X, theta) - y)
        J_history[i] = compute_cost(X, y, theta)  # 用每一次迭代产生的参数 来计算代价函数值

    return theta, J_history
