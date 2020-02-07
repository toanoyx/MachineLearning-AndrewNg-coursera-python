import scipy.io as sio


def loadWeight(path):
    data = sio.loadmat(path)
    return data['Theta1'], data['Theta2']
