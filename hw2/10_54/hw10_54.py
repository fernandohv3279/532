import numpy as np
import scipy.io
import matplotlib.pyplot as plt

data = scipy.io.loadmat('Kingrynormalized.mat')
X = data['Kingrynorm']

# Part a: thin SVD
U, s, Vt = np.linalg.svd(X, full_matrices=False)

# Plot singular values
plt.figure()
plt.semilogy(s, 'k.', markersize=4)
plt.xlabel("Index $i$")
plt.ylabel("$\\sigma_i$")
plt.title("Singular Values")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig("fig_a_singular_values.png", dpi=150)

# Part b: fraction of energy captured by k singular values
energy = np.cumsum(s**2) / np.sum(s**2)

# How much energy is captured by 3 dimensions?
print(f"Energy captured by 3 dimensions: {energy[2]:.4f}")

# How many dimensions to capture 95% of energy?
k95 = np.argmax(energy >= 0.95) + 1
print(f"Dimensions required for 95% energy: {k95}")

# Plot
plt.figure()
plt.plot(energy, 'k.')
plt.axhline(y=0.95, color='r', linestyle='--', label="95%")
plt.axvline(x=k95-1, color='b', linestyle=':', label=f"$k={k95}$")
plt.xlabel("$k$")
plt.ylabel("$E_k$")
plt.title("Fraction of energy captured by $k$ singular values")
plt.legend()
plt.tight_layout()
plt.savefig("fig_b_energy.png", dpi=150)

# Part c: rank 3 approximation as heatmap
X3 = (U[:, :3] * s[:3]) @ Vt[:3, :]

plt.figure(figsize=(10, 6))
plt.imshow(X3, cmap='hot', aspect='auto')
plt.colorbar()
plt.xlabel("Sample index")
plt.ylabel("Gene index")
plt.title("Rank-3 approximation")
plt.tight_layout()
plt.savefig("fig_c_rank3.png", dpi=150)

# Part d: 3D PCA plot
coords = (Vt[:3, :] * s[:3, np.newaxis]).T  # shape (108, 3)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c='k', s=10)
ax.set_xlabel("$a_1$")
ax.set_ylabel("$a_2$")
ax.set_zlabel("$a_3$")
ax.set_title("3D PCA plot of F. Tularensis data")
plt.tight_layout()
plt.savefig("fig_d_pca3d.png", dpi=150)

# Part e: separate SVD/PCA on each group
groups = {
    'Schu4 lung':   X[:, 6:30],
    'LVS lung':     X[:, 30:54],
    'Schu4 spleen': X[:, 60:84],
    'LVS spleen':   X[:, 84:108],
}

plt.figure(figsize=(8, 5))
for name, Xg in groups.items():
    _, sg, _ = np.linalg.svd(Xg, full_matrices=False)
    Ek = np.cumsum(sg**2) / np.sum(sg**2)
    plt.plot(Ek, label=name)

plt.axhline(y=0.95, color='k', linestyle='--', label="95%")
plt.xlabel("$k$")
plt.ylabel("$E_k$")
plt.title("Energy curves by group")
plt.legend()
plt.tight_layout()
plt.savefig("fig_e_energy_curves.png", dpi=150)
