from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Part A
img = Image.open("dante.jpeg").convert("L")

A = np.array(img, dtype=float)

fig, ax = plt.subplots()
ax.imshow(A, cmap="gray")
ax.axis("off")
ax.set_title("Plot of $\\mathbf{A}$",fontsize=15)
plt.savefig("fig_a_original.png", dpi=150)

# Part B

U, s, Vt = np.linalg.svd(A, full_matrices=False)

fig, ax = plt.subplots(figsize=(6, 3))
ax.semilogy(s, "k.", markersize=3)
ax.set_xlabel("Index $i$")
ax.set_ylabel("$\\sigma_i$")
ax.set_title("Singular values of $\\mathbf{A}$",fontsize=15)
plt.tight_layout()
plt.savefig("fig_b_singular_values.png", dpi=150)
# plt.show()

# Part C
fig, axes = plt.subplots(1, 4, figsize=(14, 4))

for i in range(4):
    outer = np.outer(U[:, i], Vt[i, :])
    axes[i].imshow(outer, cmap="gray")
    axes[i].set_title(f"$\\mathbf{{u}}_{i+1}\\mathbf{{v}}_{i+1}^T$",fontsize=26)
    axes[i].axis("off")

plt.suptitle("Rank-1 outer products", fontsize=36)
plt.tight_layout()
plt.savefig("fig_c_outer_products.png", dpi=150)

# Part D
fig, axes = plt.subplots(1, 4, figsize=(14, 4))

for i, k in enumerate([1, 2, 3, 4]):
    Ak = (U[:, :k] * s[:k]) @ Vt[:k, :]
    axes[i].imshow(np.clip(Ak, 0, 255), cmap="gray")
    axes[i].set_title(f"$\\mathbf{{A}}_{k}$",fontsize=26)
    axes[i].axis("off")

plt.suptitle("Rank-$k$ approximations, $k=1,2,3,4$", fontsize=36)
plt.tight_layout()
plt.savefig("fig_d_approx_1to4.png", dpi=150)

# Part E
fig, axes = plt.subplots(1, 4, figsize=(14, 4))

for i, k in enumerate([10, 20, 30, 40]):
    Ak = (U[:, :k] * s[:k]) @ Vt[:k, :]
    axes[i].imshow(np.clip(Ak, 0, 255), cmap="gray")
    axes[i].set_title(f"$\\mathbf{{A}}_{{{k}}}$",fontsize=26)
    axes[i].axis("off")

plt.suptitle("Rank-$k$ approximations, $k=10,20,30,40$", fontsize=36)
plt.tight_layout()
plt.savefig("fig_e_approx_10to40.png", dpi=150)

# Part F
A100 = (U[:, :100] * s[:100]) @ Vt[:100, :]

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

axes[0].imshow(A, cmap="gray")
axes[0].set_title("Original $\\mathbf{A}$",fontsize=26)
axes[0].axis("off")

axes[1].imshow(np.clip(A100, 0, 255), cmap="gray")
axes[1].set_title("$\\mathbf{A}_{100}$",fontsize=26)
axes[1].axis("off")

residual = np.log(np.abs(A - A100) + 1e-10)
# residual = np.log(np.abs(A - A100))
im = axes[2].imshow(residual, cmap="inferno")
axes[2].set_title("$\\log|\\mathbf{A} - \\mathbf{A}_{100}|$",fontsize=26)
axes[2].axis("off")
fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

plt.suptitle("Original, rank-100 approximation, and log-residual", fontsize=36)
plt.tight_layout()
plt.savefig("fig_f_residual.png", dpi=150)

# plt.show()
