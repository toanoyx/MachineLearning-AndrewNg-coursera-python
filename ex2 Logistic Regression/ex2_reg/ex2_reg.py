import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from costFunctionReg import *
from gradientReg import *
from predict import *

""" 第1部分 可视化数据集 """

path = 'ex2data2.txt'
data2 = pd.read_csv(path, header=None, names=['Test 1', 'Test 2', 'Accepted'])

positive = data2[data2['Accepted'].isin([1])]
negative = data2[data2['Accepted'].isin([0])]

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive['Test 1'], positive['Test 2'], s=50, c='black', marker='+', label='y = 1')
ax.scatter(negative['Test 1'], negative['Test 2'], s=50, c='yellow', marker='o', label='y = 0')
ax.legend()
ax.set_xlabel('Microchip Test 1')
ax.set_ylabel('Microchip Test 2')
plt.show()

degree = 5
x1 = data2['Test 1']
x2 = data2['Test 2']

data2.insert(3, 'Ones', 1)

for i in range(1, degree):
    for j in range(0, i):
        data2['F' + str(i) + str(j)] = np.power(x1, i-j) * np.power(x2, j)

data2.drop('Test 1', axis=1, inplace=True)
data2.drop('Test 2', axis=1, inplace=True)

""" 第2部分 正则化代价函数 """

cols = data2.shape[1]
X2 = data2.iloc[:,1:cols]
y2 = data2.iloc[:,0:1]

X2 = np.array(X2.values)
y2 = np.array(y2.values)
theta2 = np.zeros(11)

learningRate = 1

print("cost = " + str(costFunctionReg(theta2, X2, y2, learningRate)) + "(cost should be 0.693)")
print(str(gradientReg(theta2, X2, y2, learningRate)))

result2 = opt.fmin_tnc(func=costFunctionReg, x0=theta2, fprime=gradientReg, args=(X2, y2, learningRate))
print(result2)

theta_min = np.matrix(result2[0])
predictions = predict(theta_min, X2)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y2)]
accuracy = (sum(map(int, correct)) % len(correct))
print('accuracy = {0}%'.format(accuracy))

