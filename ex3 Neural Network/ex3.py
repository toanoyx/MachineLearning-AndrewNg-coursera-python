from loadData import *
from displayData import *
from logisticRegression import *
from predict import *
from loadWeight import *
from sklearn.metrics import classification_report


""" 第1部分 加载数据集 """
X, y = loadData('ex3data1.mat')


""" 第2部分 可视化 """
displayData(X)
plt.show()

raw_X, raw_y = loadData('ex3data1.mat')


""" 第3部分 向量化逻辑回归 """
X = np.insert(raw_X, 0, values=np.ones(raw_X.shape[0]), axis=1)
y_matrix = []

for k in range(1, 11):
    y_matrix.append((raw_y == k).astype(int))

y_matrix = [y_matrix[-1]] + y_matrix[:-1]
y = np.array(y_matrix)

t0 = logisticRegression(X, y[0])
print(t0.shape)
y_pred = predict(X, t0)
print('Accuracy={}'.format(np.mean(y[0] == y_pred)))

k_theta = np.array([logisticRegression(X, y[k]) for k in range(10)])
print(k_theta.shape)

prob_matrix = sigmoid(X @ k_theta.T)
np.set_printoptions(suppress=True)
y_pred = np.argmax(prob_matrix, axis=1)
y_answer = raw_y.copy()
y_answer[y_answer==10] = 0
print(classification_report(y_answer, y_pred))

""" 第4部分 神经网络模型 """
theta1, theta2 = loadWeight('ex3weights.mat')
X, y = loadData('ex3data1.mat', transpose=False)
X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)
a1 = X
z2 = a1 @ theta1.T
z2 = np.insert(z2, 0, values=np.ones(z2.shape[0]), axis=1)
a2 = sigmoid(z2)
z3 = a2 @ theta2.T
a3 = sigmoid(z3)
y_pred = np.argmax(a3, axis=1) + 1
print(classification_report(y, y_pred))