import numpy as np
from scipy.optimize import linprog


def solve_svm(X_train, y_train, C):
    m, n = X_train.shape
    Y = np.diag(y_train)
    YX = Y @ X_train
    Im = np.identity(m)
    y_col = y_train.reshape(-1, 1)

    # Variables: [w, t, z, b]  (sizes: n, n, m, 1)
    # Constraints:
    #   w - t <= 0   (w_i <= t_i)
    #  -w - t <= 0   (-w_i <= t_i)
    #  -YXw - z - y*b <= -1  (SVM margin)

    Row1 = np.concatenate([ np.eye(n), -np.eye(n), np.zeros((n, m)), np.zeros((n, 1))], axis=1)
    Row2 = np.concatenate([-np.eye(n), -np.eye(n), np.zeros((n, m)), np.zeros((n, 1))], axis=1)
    Row3 = np.concatenate([-YX, np.zeros((m, n)), -Im, -y_col], axis=1)

    A = np.vstack([Row1, Row2, Row3])
    h = np.hstack([np.zeros(n), np.zeros(n), -np.ones(m)])

    # Minimize: sum(t) + C*sum(z)
    cost = np.hstack([np.zeros(n), np.ones(n), C * np.ones(m), 0])

    # w unbounded, t >= 0, z >= 0, b unbounded
    bounds = [(None, None)] * n + [(0, None)] * n + [(0, None)] * m + [(None, None)]

    result = linprog(cost, A_ub=A, b_ub=h, bounds=bounds)

    w = result.x[:n]
    t = result.x[n:2*n]
    b = result.x[-1]
    return w, b
