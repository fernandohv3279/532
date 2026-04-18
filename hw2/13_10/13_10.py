import numpy as np
import scipy.io
import matplotlib.pyplot as plt

def csgen_fast(Q11, Q21):
    m, k = Q11.shape
    p = Q21.shape[0]
    U, c, Xt = np.linalg.svd(Q11, full_matrices=False)
    X = Xt.T
    s = np.sqrt(1 - c**2)
    C = np.zeros((k, k))
    np.fill_diagonal(C, c)
    S = np.zeros((p, k))
    S[p-k:, :] = np.diag(s)
    return U, C, S, X

def gsvd_fast(A, B):
    m, n = A.shape
    p = B.shape[0]
    M = np.vstack([A, B])
    Qhat, Sigmahat, Wt = np.linalg.svd(M, full_matrices=False)
    W = Wt.T
    Q11 = Qhat[:m, :]
    Q21 = Qhat[m:, :]
    U, C, S, X = csgen_fast(Q11, Q21)
    G = W @ np.diag(Sigmahat) @ X
    return U, C, S, G

data = scipy.io.loadmat('Indian_pines_corrected.mat')
X = data['indian_pines_corrected']

X2d = X.reshape(-1, 200).astype(float)
Xs = np.roll(X, 1, axis=0).reshape(-1, 200).astype(float)  # roll X before reshaping
dX = X2d - Xs

# Part A
N = dX / np.sqrt(2)

# Part B

# GSVD
U, C, S, G = gsvd_fast(X2d, N)

# Extract c and s
n = C.shape[0]
c = np.diag(C)
s = np.diag(S[S.shape[0]-n:, :])

# Estimate i*
istar = np.argmin(np.abs(c**2 - s**2))
print(f"i* = {istar}, c^2 = {c[istar]**2:.4f}, s^2 = {s[istar]**2:.4f}")

plt.figure()
plt.plot(c**2, label="$c_i^2$")
plt.plot(s**2, '--', label="$s_i^2$")
plt.axvline(x=istar, color='k', linestyle=':', label=f"$i^* = {istar}$")
plt.xlabel("Index $i$")
plt.ylabel("Generalized singular value pairs squared")
plt.title("GSVD singular value pairs")
plt.legend()
plt.tight_layout()
plt.savefig("fig_gsvd_pairs.png", dpi=150)

# Part C

# Rank i* approximation of band 25
e25 = np.zeros((200, 1))
e25[24] = 1.0

# keep only first istar columns of U, c, G
Ui = U[:, :istar]
ci = c[:istar]
Gi = G[:, :istar]

# denoised band 25
band25_denoised = (Ui * ci) @ (Gi.T @ e25)
band25_denoised_img = band25_denoised.reshape(145, 145)

# original band 25
band25_original = X2d[:, 24].reshape(145, 145)

# plot side by side
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(band25_original, cmap='gray')
axes[0].set_title("Original band 25")
axes[0].axis('off')

axes[1].imshow(band25_denoised_img, cmap='gray')
axes[1].set_title(f"Denoised band 25 (rank $i^*={istar}$)")
axes[1].axis('off')

plt.suptitle("GSVD denoising of band 25", fontsize=14)
plt.tight_layout()
plt.savefig("fig_band25_denoised.png", dpi=150)
