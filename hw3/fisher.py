import numpy as np


def fisher(X_train, y_train):
    # Split into C+ and C- (columns are samples)
    X = X_train[y_train ==  1].T   # n x nx
    Y = X_train[y_train == -1].T   # n x ny

    nx = X.shape[1]
    ny = Y.shape[1]

    x_bar = X.mean(axis=1, keepdims=True)  # class mean of C+
    y_bar = Y.mean(axis=1, keepdims=True)  # class mean of C-

    X_tilde = X - x_bar  # mean-subtracted C+
    Y_tilde = Y - y_bar  # mean-subtracted C-

    # Total within-class scatter matrix (eq. 7.27)
    S = (1/nx) * X_tilde @ X_tilde.T + (1/ny) * Y_tilde @ Y_tilde.T

    # Solve Sw = x_bar - y_bar (eq. 7.31), then normalize (step 5)
    w = np.linalg.solve(S, (x_bar - y_bar).flatten())
    w = w / np.linalg.norm(w)

    return w


def best_threshold(w, X_val, y_val):
    proj = (X_val @ w).flatten()
    thresholds = np.linspace(proj.min(), proj.max(), 500)
    best_t, best_acc = 0, -1
    for t in thresholds:
        preds = np.where(proj >= t, 1, -1)
        acc = np.mean(preds == y_val)
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_t, best_acc


def classify(w, threshold, X):
    proj = (X @ w).flatten()
    return np.where(proj >= threshold, 1, -1)
