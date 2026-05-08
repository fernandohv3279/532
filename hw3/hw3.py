import scipy.io
import numpy as np
from svm import solve_svm

data = scipy.io.loadmat('wisconsin_breast_cancer.mat')

X = data['X']                                          # (569, 30)
y = data['y']                                          # (569, 1)
y = np.where(y == 0, -1, 1).flatten()                 # convert 0/1 → -1/+1

# Split into train (70%), val (15%), test (15%)
np.random.seed(0)
idx = np.random.permutation(len(y))
n_train = int(0.70 * len(idx))
n_val   = int(0.15 * len(idx))

X_train, y_train = X[idx[:n_train]],                y[idx[:n_train]]
X_val,   y_val   = X[idx[n_train:n_train+n_val]],   y[idx[n_train:n_train+n_val]]
X_test,  y_test  = X[idx[n_train+n_val:]],           y[idx[n_train+n_val:]]

# Tune C on validation set
C_values = np.linspace(0.5, 5.5, 100)
best_C, best_acc = None, -1

for C in C_values:
    w, b = solve_svm(X_train, y_train, C)
    preds = np.sign(X_val @ w + b)
    acc = np.mean(preds == y_val)
    print(f"C={C:6.2f}  val_acc={acc:.4f}  nonzero_w={np.sum(w != 0)}")
    if acc > best_acc:
        best_acc, best_C = acc, C

print(f"\nBest C: {best_C}  (val accuracy: {best_acc:.4f})")

# Evaluate on test set
w, b = solve_svm(X_train, y_train, best_C)
test_acc = np.mean(np.sign(X_test @ w + b) == y_test)
print(f"Test accuracy: {test_acc:.4f}")
print(f"Nonzero weights: {np.sum(w != 0)}/30")
