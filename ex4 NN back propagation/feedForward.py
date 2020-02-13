import numpy as np
from sigmoid import *


def feedForward(theta, X):
    t1, t2 = deserialize(theta)
    m = X.shape[0]
    a1 = X

    z2 = a1 @ t1.T
    a2 = np.insert(sigmoid(z2), 0, np.ones(m), axis=1)

    z3 = a2 @ t2.T
    h = sigmoid(z3)

    return a1, z2, a2, z3, h


def deserialize(seq):
    return seq[:25 * 401].reshape(25, 401), seq[25 * 401:].reshape(10, 26)
