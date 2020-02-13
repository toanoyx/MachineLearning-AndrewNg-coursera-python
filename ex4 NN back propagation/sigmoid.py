import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), 1 - sigmoid(z))
