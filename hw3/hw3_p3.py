import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ksvd import ksvd, sparse_code

np.random.seed(0)

# --- Load image and convert to grayscale ---
img = np.array(Image.open('dante.jpeg').convert('L'), dtype=float)
H, W = img.shape
print(f"Image size: {H}x{W}")

# Part a: Extract non-overlapping 8x8 patches
patch_size = 8
positions = []
patches = []
for i in range(0, H - patch_size + 1, patch_size):
    for j in range(0, W - patch_size + 1, patch_size):
        patches.append(img[i:i+patch_size, j:j+patch_size].flatten())
        positions.append((i, j))

Y = np.array(patches).T.astype(float)   # shape: (64, N)
N = Y.shape[1]
print(f"Number of patches: {N}")

# Part b: Initialize dictionary (K=256) from random patches
K = 256
init_idx = np.random.choice(N, K, replace=False)
D = Y[:, init_idx].copy()
for k in range(K):
    norm = np.linalg.norm(D[:, k])
    if norm > 0:
        D[:, k] /= norm

# Part c: Run K-SVD for 10 iterations with T=8
T = 8
n_iter = 10
D, X, errors = ksvd(D, Y, T, n_iter)

plt.figure()
plt.plot(range(1, n_iter + 1), errors, 'o-')
plt.xlabel('Iteration')
plt.ylabel(r'$\|Y - DX\|_F$')
plt.title(f'K-SVD convergence (K={K}, T={T})')
plt.tight_layout()
plt.savefig('ksvd_error.png', dpi=150)
plt.show()

# Part d: Order atoms by pixel variance, plot top 12
variances = np.var(D, axis=0)   # variance of pixel values within each atom
order = np.argsort(variances)[::-1]

fig, axes = plt.subplots(3, 4, figsize=(8, 6))
for i, ax in enumerate(axes.flat):
    atom = D[:, order[i]].reshape(patch_size, patch_size)
    ax.imshow(atom, cmap='gray')
    ax.axis('off')
plt.suptitle(f'Top 12 dictionary atoms by variance (K={K}, T={T})')
plt.tight_layout()
plt.savefig('ksvd_atoms.png', dpi=150)
plt.show()

# Part e: Reconstruct image by reassembling patches
recon = np.zeros((H, W))
for idx, (i, j) in enumerate(positions):
    patch = (D @ X[:, idx]).reshape(patch_size, patch_size)
    recon[i:i+patch_size, j:j+patch_size] = patch

recon = np.clip(recon, 0, 255)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.imshow(img, cmap='gray', vmin=0, vmax=255)
ax1.set_title('Original')
ax1.axis('off')
ax2.imshow(recon, cmap='gray', vmin=0, vmax=255)
ax2.set_title(f'Reconstructed (K={K}, T={T})')
ax2.axis('off')
plt.tight_layout()
plt.savefig('ksvd_reconstruction.png', dpi=150)
plt.show()

print(f"\nReconstruction error: {np.linalg.norm(img - recon):.2f}")
