from feedForward import *


def nnCostFunction(theta, X, y):
    m = X.shape[0]
    _, _, _, _, h = feedForward(theta, X)

    pair_computation = -np.multiply(y, np.log(h)) - np.multiply((1 - y), np.log(1 - h))

    return pair_computation.sum() / m
