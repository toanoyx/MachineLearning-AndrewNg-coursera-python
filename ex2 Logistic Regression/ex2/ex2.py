"""第1部分 可视化训练集"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import seaborn as sns
from sklearn.metrics import classification_report
from scipy import optimize

from costFunction import *
from gradientDescent import *
from predict import *

print('Plotting Data...')
data = pd.read_csv('ex2data1.txt', header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
positive = data[data['Admitted'].isin([1])]
negative = data[data['Admitted'].isin([0])]

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='black', marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='yellow', marker='x', label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
plt.show()

"""第2部分 逻辑回归"""
data.insert(0, 'Ones', 1)
cols = data.shape[1]
X = data.iloc[:, 0:cols-1]
y = data.iloc[:, cols-1:cols]
X = np.array(X.values)
y = np.array(y.values)
theta = np.zeros(3)
print('initial cost : ' + str(costFunction(theta, X, y)) + ' (This value should be about 0.693)')

"""第3部分 梯度下降"""
print('gradient descent : ' + str(gradientDescent(theta, X, y)))
result = opt.fmin_tnc(func=costFunction, x0=theta, fprime=gradientDescent, args=(X, y))
print('result : ' + str(result))
print('cost : ' + str(costFunction(result[0], X, y)) + ' (This value should be about 0.203)')

"""第4部分 用训练集预测和验证"""
params = np.zeros((X.shape[1], 1)).ravel()
args = (X, y)


def f(params, *args):
    X_train, y_train = args
    m, n = X_train.shape
    J = 0
    theta = params.reshape((n, 1))
    h = sigmoid(np.dot(X_train, theta))
    J = -1 * np.sum(y_train * np.log(h) + (1 - y_train) * np.log((1 - h))) / m

    return J


def gradf(params, *args):
    X_train, y_train = args
    m, n = X_train.shape
    theta = params.reshape(-1, 1)
    h = sigmoid(np.dot(X_train, theta))
    grad = np.zeros((X_train.shape[1], 1))
    grad = X_train.T.dot((h - y_train)) / m
    g = grad.ravel()
    return g


res = optimize.fmin_cg(f,x0=params,fprime=gradf,args=args,maxiter=500)
theta_min = np.matrix(result[0])
predictions = predict(theta_min, X)
print('classification report : ')
print(classification_report(y, predictions))
prob = sigmoid(np.dot(np.array([[1,45,85]]),res))
print("For a student with scores 45 and 85, we predict an admission :" + str(prob))
print("Expected value: 0.775 +/- 0.002")
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))
print('accuracy = {0}%'.format(accuracy))

"""第5部分 寻找决策边界"""
label = np.array(y)
index_0 = np.where(label.ravel() == 0)
plt.scatter(X[index_0, 1], X[index_0, 2], marker='x', color='yellow', label='Not admitted', s=30)
index_1 = np.where(label.ravel() == 1)
plt.scatter(X[index_1, 1], X[index_1, 2], marker='o', color='black', label='Admitted', s=30)
x1 = np.arange(20, 100, 0.5)
x2 = (- res[0] - res[1]*x1) / res[2]
plt.plot(x1, x2, color='black')
plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')
plt.legend(loc='upper left')
plt.show()
