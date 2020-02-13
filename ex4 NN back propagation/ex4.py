from tensorflow.contrib import opt
from sklearn.metrics import classification_report
from loadData import *
from displayData import *
from feedForward import *
from nnCostFunction import *
from computeNumericalGradient import *
from checkNNGradients import *


""" 第1部分 可视化数据集 """
X, _ = loadData('ex4data1.mat')
displayData(X)
plt.show()

""" 第2部分 模型表示 """
X_raw, y_raw = loadData('ex4data1.mat', transpose=False)
X = np.insert(X_raw, 0, np.ones(X_raw.shape[0]), axis=1)


def expand_y(y):
    res = []
    for i in y:
        y_array = np.zeros(10)
        y_array[i - 1] = 1
        res.append(y_array)
    return np.array(res)


y = expand_y(y_raw)


def load_weight(path):
    data = sio.loadmat(path)
    return data['Theta1'], data['Theta2']


t1, t2 = load_weight('ex4weights.mat')

""" 第3部分 前向传播和代价函数 """
theta = np.concatenate((np.ravel(t1), np.ravel(t2)))
_, _, _, _, h = feedForward(theta, X)
print("cost function: " + str(nnCostFunction(theta, X, y)) + "(this should be 0.287629)")

""" 第4部分 正则化代价函数 """
t1, t2 = deserialize(theta)
m = X.shape[0]
l = 1
reg_t1 = (l / (2 * m)) * np.power(t1[:, 1:], 2).sum()
reg_t2 = (l / (2 * m)) * np.power(t2[:, 1:], 2).sum()
regularizedCost = nnCostFunction(theta, X, y) + reg_t1 + reg_t2
print("regularized cost function: " + str(regularizedCost) + "(this should be 0.383770)")

""" 第5部分 反向传播 """
sigmoid_gradient(0)
d1, d2 = deserialize(computeNumericalGradient(theta, X, y))

checkNNGradients(theta, X, y, epsilon= 0.0001)

checkNNGradients(theta, X, y, epsilon=0.0001, regularized=True)

""" 第6部分 """


def random_init(size):
    return np.random.uniform(-0.12, 0.12, size)


def nn_training(X, y):
    init_theta = random_init(10285)  # 25*401 + 10*26

    res = opt.minimize(fun=regularized_cost,
                       x0=init_theta,
                       args=(X, y, 1),
                       method='TNC',
                       jac=regularized_gradient,
                       options={'maxiter': 400})
    return res


res = nn_training(X, y)
print(str(res))
_, y_answer = loadData('ex4data1.mat')
print(str(y_answer[:20]))
final_theta = res.x


def show_accuracy(theta, X, y):
    _, _, _, _, h = feedForward(theta, X)
    y_pred = np.argmax(h, axis=1) + 1
    print(classification_report(y, y_pred))


def plot_hidden_layer(theta):
    final_theta1, _ = deserialize(theta)
    hidden_layer = final_theta1[:, 1:]  # ger rid of bias term theta

    fig, ax_array = plt.subplots(nrows=5, ncols=5, sharey=True, sharex=True, figsize=(5, 5))

    for r in range(5):
        for c in range(5):
            ax_array[r, c].matshow(hidden_layer[5 * r + c].reshape((20, 20)),
                                   cmap=matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))


plot_hidden_layer(final_theta)
plt.show()
