import numpy as np
import scipy.io
import matplotlib.pyplot as plt

data = scipy.io.loadmat('datamatrix-1.mat')
cats = data['Y'][:, :99]   # (4096, 99)
dogs = data['Y'][:, 99:]   # (4096, 99)

# one dog image (same for all three sets)
dog = dogs[:, 0:1]   # (4096, 1)

# three sets of cats + same dog
A1 = np.hstack([cats[:, :6], dog])    # (4096, 7)
A2 = np.hstack([cats[:, 6:13], dog])  # (4096, 8)
A3 = np.hstack([cats[:, 13:21], dog]) # (4096, 9) -- 8+1=9

# compute orthonormal bases
X1, _ = np.linalg.qr(A1, mode='reduced')   # (4096, 7)
X2, _ = np.linalg.qr(A2, mode='reduced')   # (4096, 8)
X3, _ = np.linalg.qr(A3, mode='reduced')   # (4096, 9)

# concatenate to form X
X = np.hstack([X1, X2, X3])   # (4096, 24)

# SVD of X - left singular vectors are the flag mean
U, s, Vt = np.linalg.svd(X, full_matrices=False)

# plot first 3 dimensions of flag mean
fig, axes = plt.subplots(1, 3, figsize=(10, 4))
for i in range(3):
    img = -U[:, i].reshape(64, 64).T
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f"$\\mathbf{{u}}_{i+1}$")
    axes[i].axis('off')

plt.suptitle("First 3 dimensions of flag mean", fontsize=14)
plt.tight_layout()
plt.savefig("fig_flag_mean.png", dpi=150)
plt.show()
