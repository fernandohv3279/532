import numpy as np
from scipy.optimize import linprog


def solve_svm(X_train, y_train, C):
    m, n = X_train.shape
    Y = np.diag(y_train)
    XY = Y @ X_train
    Im = np.identity(m)
    y_col = y_train.reshape(-1, 1)

    Row1 = np.concatenate([ np.identity(n), -np.identity(n), np.zeros((n, m)), np.zeros((n, 1))], axis=1)
    Row2 = np.concatenate([-np.identity(n), -np.identity(n), np.zeros((n, m)), np.zeros((n, 1))], axis=1)
    Row3 = -np.concatenate([XY, -XY, Im, y_col], axis=1)

    A    = np.vstack([Row1, Row2, Row3])
    h    = np.hstack([np.zeros(n), np.zeros(n), -np.ones(m)])
    cost = np.hstack([np.ones(n), np.ones(n), C * np.ones(m), 0])

    bounds = [(0, None)] * len(cost)
    result = linprog(cost, A_ub=A, b_ub=h, bounds=bounds)

    w = result.x[:n] - result.x[n:2*n]
    t = result.x[-1]
    return w, t
