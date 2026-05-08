import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from svm import solve_svm
from fisher import fisher, best_threshold, classify

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

# Part b: find best c on validation set
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

# Part c: plot weights ordered by absolute magnitude

order = np.argsort(np.abs(w))[::-1]
plt.bar(range(len(w)), np.abs(w[order]))
plt.xlabel("Feature index (sorted by |w|)")
plt.ylabel("|w|")
plt.title(f"Feature weights ordered by magnitude (C={best_C:.2f})")
plt.tight_layout()
plt.savefig("weights.png", dpi=150)
plt.show()

plt.figure()
plt.bar(range(len(w)), np.abs(w[order]))
plt.yscale("log")
plt.xlabel("Feature index (sorted by |w|)")
plt.ylabel("|w| (log scale)")
plt.title(f"Feature weights ordered by magnitude - log scale (C={best_C:.2f})")
plt.tight_layout()
plt.savefig("weights_log.png", dpi=150)
plt.show()

print(f"Features discarded: {np.sum(np.abs(w) < 0.001)}/30")
print(f"|w| ordered: {np.abs(w[order])}")

# Part d: retrain and evaluate using only selected features
mask = np.abs(w) >= 0.001
w_sf, b_sf = solve_svm(X_train[:, mask], y_train, best_C)
test_acc_sf = np.mean(np.sign(X_test[:, mask] @ w_sf + b_sf) == y_test)
print(f"\nTest accuracy with selected features ({mask.sum()}/30): {test_acc_sf:.4f}")

# Part e: confusion matrix
preds = np.sign(X_test[:, mask] @ w_sf + b_sf)

TP = np.sum((preds ==  1) & (y_test ==  1))  # correct malignant
FN = np.sum((preds == -1) & (y_test ==  1))  # missed malignant
FP = np.sum((preds ==  1) & (y_test == -1))  # false alarm
TN = np.sum((preds == -1) & (y_test == -1))  # correct benign

print(f"\nConfusion Matrix:")
print(f"                 | Pred Malignant  Pred Benign")
print(f"-----------------+-----------------------------")
print(f"Actual Malignant | {TP:^14}  {FN:^11}")
print(f"Actual Benign    | {FP:^14}  {TN:^11}")

# Problem 2: Fisher discriminant analysis
print("\n--- Problem 2: Fisher Discriminant Analysis ---")

# Full feature set
w_f = fisher(X_train, y_train)
t_f, val_acc_f = best_threshold(w_f, X_val, y_val)
preds_f = classify(w_f, t_f, X_test)
test_acc_f = np.mean(preds_f == y_test)
print(f"Full features  — threshold: {t_f:.4f}  val_acc: {val_acc_f:.4f}  test_acc: {test_acc_f:.4f}")

# Reduced feature set
w_r = fisher(X_train[:, mask], y_train)
t_r, val_acc_r = best_threshold(w_r, X_val[:, mask], y_val)
preds_r = classify(w_r, t_r, X_test[:, mask])
test_acc_r = np.mean(preds_r == y_test)
print(f"Reduced features — threshold: {t_r:.4f}  val_acc: {val_acc_r:.4f}  test_acc: {test_acc_r:.4f}")
