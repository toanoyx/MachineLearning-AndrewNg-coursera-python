import numpy as np
import scipy.optimize as opt
from lrCostFunction import *
from lrGradient import *


def logisticRegression(X, y, l=1):
    theta = np.zeros(X.shape[1])

    res = opt.minimize(fun=lrCostFunction,
                       x0=theta,
                       args=(X, y, l),
                       method='TNC',
                       jac=lrGradient,
                       options={'disp': True})
    final_theta = res.x

    return final_theta
