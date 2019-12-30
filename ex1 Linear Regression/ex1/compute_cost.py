
def h(X, theta):  # 线性回归假设函数
    return X.dot(theta)


def compute_cost(X, y, theta):

    m = y.size  # 样本数
    y = y.reshape(-1, 1)  # 转变为列向量 用二维数组表示(m,1)
    cost = 0  # 代价函数值
    myh = h(X, theta)  # 得到假设函数值  (m,1)

    cost = 1 / (2 * m) * (myh - y).T.dot(myh - y)

    return cost
