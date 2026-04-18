import numpy as np
import scipy.io
import matplotlib.pyplot as plt

def csgen(Q11, Q21):
    m, k = Q11.shape
    p = Q21.shape[0]
    U, c, Xt = np.linalg.svd(Q11, full_matrices=False)
    X = Xt.T
    s = np.sqrt(1 - c**2)
    Vp = (Q21 @ X) / s
    F = np.random.randn(p, p - k)
    Vtemp = (np.eye(p) - Vp @ Vp.T) @ F
    Vorth, _ = np.linalg.qr(Vtemp)
    V = np.hstack([Vorth, Vp])
    # C = np.zeros((m, k))
    C = np.zeros((k, k))
    np.fill_diagonal(C, c)
    S = np.zeros((p, k))
    S[p-k:, :] = np.diag(s)
    return U, V, C, S, X

def gsvd(A, B):
    m, n = A.shape
    p = B.shape[0]
    M = np.vstack([A, B])
    Qhat, Sigmahat, Wt = np.linalg.svd(M, full_matrices=False)
    W = Wt.T
    Q11 = Qhat[:m, :]
    Q21 = Qhat[m:, :]
    U, V, C, S, X = csgen(Q11, Q21)
    G = W @ np.diag(Sigmahat) @ X
    return U, V, C, S, G

data = scipy.io.loadmat('trig_dataset_526x4.mat')
A = data['X'].astype(float)   # (526, 4)
t = data['t'].astype(float).flatten()   # (526,)

# Compute B = 1/sqrt(2) * dA
dA = A - np.roll(A, 1, axis=0)
B = dA / np.sqrt(2)

# GSVD
U, V, C, S, G = gsvd(A, B)

# Estimate T1 = Uhat (first 4 columns of U)
T1 = U[:, :4]

# Estimate M = C @ G.T
n = C.shape[0]
M = C @ G.T

# Plot columns of T1
fig, axes = plt.subplots(4, 1, figsize=(10, 8))
for i in range(4):
    axes[i].plot(t, T1[:, i])
    axes[i].set_ylabel(f"$T_{{1,{i+1}}}$")
axes[-1].set_xlabel("$t$")
plt.suptitle("Estimated signals $T_1 = \\hat{U}$", fontsize=14)
plt.tight_layout()
plt.savefig("fig_T1.png", dpi=150)
plt.show()

# Compare with SVD of A
U_svd, s_svd, Vt_svd = np.linalg.svd(A, full_matrices=False)

fig, axes = plt.subplots(4, 2, figsize=(14, 8))
for i in range(4):
    axes[i, 0].plot(t, T1[:, i])
    axes[i, 0].set_ylabel(f"$T_{{1,{i+1}}}$")
    axes[i, 0].set_title("GSVD" if i == 0 else "")
    axes[i, 1].plot(t, U_svd[:, i])
    axes[i, 1].set_title("SVD" if i == 0 else "")
axes[-1, 0].set_xlabel("$t$")
axes[-1, 1].set_xlabel("$t$")
plt.suptitle("GSVD $T_1$ vs SVD of $A$", fontsize=14)
plt.tight_layout()
plt.savefig("fig_T1_vs_SVD.png", dpi=150)
plt.show()
