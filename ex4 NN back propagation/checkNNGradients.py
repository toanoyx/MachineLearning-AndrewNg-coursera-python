from nnCostFunction import *
from computeNumericalGradient import *


def checkNNGradients(theta, X, y, epsilon, regularized=False):
    def a_numeric_grad(plus, minus, regularized=False):
        if regularized:
            return (regularized_cost(plus, X, y) - regularized_cost(minus, X, y)) / (epsilon * 2)
        else:
            return (nnCostFunction(plus, X, y) - nnCostFunction(minus, X, y)) / (epsilon * 2)

    theta_matrix = expand_array(theta)  # expand to (10285, 10285)
    epsilon_matrix = np.identity(len(theta)) * epsilon

    plus_matrix = theta_matrix + epsilon_matrix
    minus_matrix = theta_matrix - epsilon_matrix

    numeric_grad = np.array([a_numeric_grad(plus_matrix[i], minus_matrix[i], regularized)
                             for i in range(len(theta))])

    analytic_grad = regularized_gradient(theta, X, y) if regularized else computeNumericalGradient(theta, X, y)
    diff = np.linalg.norm(numeric_grad - analytic_grad) / np.linalg.norm(numeric_grad + analytic_grad)

    print(
        'If your backpropagation implementation is correct,\nthe relative difference will be smaller than 10e-9 (assume epsilon=0.0001).\nRelative Difference: {}\n'.format(
            diff))


def regularized_cost(theta, X, y, l=1):
    t1, t2 = deserialize(theta)
    m = X.shape[0]
    reg_t1 = (l / (2 * m)) * np.power(t1[:, 1:], 2).sum()  # this is how you ignore first col
    reg_t2 = (l / (2 * m)) * np.power(t2[:, 1:], 2).sum()
    return nnCostFunction(theta, X, y) + reg_t1 + reg_t2


def expand_array(arr):
    return np.array(np.matrix(np.ones(arr.shape[0])).T @ np.matrix(arr))


def regularized_gradient(theta, X, y, l=1):
    m = X.shape[0]
    delta1, delta2 = deserialize(computeNumericalGradient(theta, X, y))
    t1, t2 = deserialize(theta)

    t1[:, 0] = 0
    reg_term_d1 = (l / m) * t1
    delta1 = delta1 + reg_term_d1

    t2[:, 0] = 0
    reg_term_d2 = (l / m) * t2
    delta2 = delta2 + reg_term_d2

    return np.concatenate((np.ravel(delta1), np.ravel(delta2)))
