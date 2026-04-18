import numpy as np
import scipy.io
import matplotlib.pyplot as plt

# Load data
corn = scipy.io.loadmat('IPcornnotilC2.mat')
woods = scipy.io.loadmat('IPwoodsC14.mat')

X = corn['X'].astype(float).T   # (1428, 200)
Y = woods['Y'].astype(float).T  # (1265, 200)

# truncate to minimum number of rows
min_rows = min(X.shape[0], Y.shape[0])
X = X[:min_rows, :]   # (1265, 200)
Y = Y[:min_rows, :]   # (1265, 200)

# Row mean subtraction
XX = X - X.mean(axis=0)
YY = Y - Y.mean(axis=0)

# QR decomposition
Qx, Rx = np.linalg.qr(XX, mode='reduced')   # Qx: (1265, 200), Rx: (200, 200)
Qy, Ry = np.linalg.qr(YY, mode='reduced')   # Qy: (1265, 200), Ry: (200, 200)

# SVD of Qx^T Qy
R, D, St = np.linalg.svd(Qx.T @ Qy)
S = St.T

# Part a: canonical correlation
z_star = D[0]
print(f"z* = {z_star:.6f}")

# Canonical correlation vectors
a = np.linalg.solve(Rx, R[:, 0])
b = np.linalg.solve(Ry, S[:, 0])

# Canonical correlation variables
alpha = XX @ a   # (1265,)
beta = YY @ b    # (1265,)

# Part b: scatter plot
plt.figure()
plt.plot(alpha, beta, '.')
x = np.linspace(alpha.min(), alpha.max(), 100)
plt.plot(x, x, 'k')
plt.xlabel(r"$\alpha_k$")
plt.ylabel(r"$\beta_k$")
plt.title(f"Canonical correlation variables, $z^* = {z_star:.4f}$")
plt.tight_layout()
plt.savefig("fig_b_scatter.png", dpi=150)

# Part c: canonical correlation vectors
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(a)
axes[0].set_title("$\\mathbf{a}$ (Corn-notill)")
axes[0].set_xlabel("Wavelength index")
axes[1].plot(b)
axes[1].set_title("$\\mathbf{b}$ (Woods)")
axes[1].set_xlabel("Wavelength index")
plt.suptitle("Canonical correlation vectors", fontsize=14)
plt.tight_layout()
plt.savefig("fig_c_ccvectors.png", dpi=150)
