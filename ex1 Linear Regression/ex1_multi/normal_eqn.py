from numpy.dual import pinv

def normal_eqn(X, y):
    y = y.reshape(-1, 1)
    # theta=inv(X.T.dot(X)).dot(X.T).dot(y)
    theta = pinv(X).dot(y)

    return theta
