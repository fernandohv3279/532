import matplotlib.pyplot as plt
import numpy as np
import scipy.io

# load matrices
data = scipy.io.loadmat('datamatrix-1.mat')
X=data['Y'][:, :99] #X.shape=(4096,99)
Y=data['Y'][:, 99:] #X.shape=(4096,99)

# get sizes
m, nx = X.shape
ny = Y.shape[1]

# subtract mean of row vectors
XX = X - X.mean(axis=0)
YY = Y - Y.mean(axis=0)

# compute QR
Qx, Rx = np.linalg.qr(XX)
Qy, Ry = np.linalg.qr(YY)

# SVD
R, D, St = np.linalg.svd(Qx.T @ Qy)
S = St.T

# correlation coefficient
z = D[0]

# Part a

print(z)

# Part b

print(np.arccos(D[0]))

# Part c

a = np.linalg.inv(Rx) @ R[:, 0]
b = np.linalg.inv(Ry) @ S[:, 0]

# correlation variables
alpha = XX @ a
beta = YY @ b

# plot
plt.figure()
plt.plot(alpha, beta, '.')

x = np.linspace(alpha.min(), alpha.max(), 100)
plt.plot(x, x, 'k')

plt.xlabel(r"$\alpha_k$")
plt.ylabel(r"$\beta_k$")
plt.title("Canonical correlation variables")
plt.tight_layout()
plt.savefig("fig_c_scatter.png", dpi=150)

# Part d

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

img_alpha = alpha.reshape(64, 64).T
img_beta = beta.reshape(64, 64).T

axes[0].imshow(img_alpha, cmap='gray')
axes[0].set_title(r"$\alpha$")
axes[0].axis('off')

axes[1].imshow(img_beta, cmap='gray')
axes[1].set_title(r"$\beta$")
axes[1].axis('off')

plt.suptitle("Canonical correlation variables as images", fontsize=14)
plt.tight_layout()
plt.savefig("fig_d_images.png", dpi=150)

# Part e

plt.figure()
gamma = alpha * beta
img_gamma = gamma.reshape(64, 64).T

plt.imshow(np.abs(img_gamma), cmap='hot')
plt.title(r"$\gamma_k = \alpha_k \beta_k$")
plt.axis('off')
plt.tight_layout()
plt.savefig("fig_e_heatmap.png", dpi=150)

# Part f

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(a)
axes[0].set_title("$\\mathbf{a}$")
axes[0].set_xlabel("Index")

axes[1].plot(b)
axes[1].set_title("$\\mathbf{b}$")
axes[1].set_xlabel("Index")

plt.suptitle("Canonical correlation vectors", fontsize=14)
plt.tight_layout()
plt.savefig("fig_f_ccvectors.png", dpi=150)
plt.show()
