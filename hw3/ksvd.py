import numpy as np


def omp(D, y, T):
    """Orthogonal Matching Pursuit: find T-sparse x such that D @ x ≈ y."""
    r = y.copy()
    selected = []
    x_S = None
    for _ in range(T):
        k = int(np.argmax(np.abs(D.T @ r)))
        if k not in selected:
            selected.append(k)
        D_S = D[:, selected]
        x_S = np.linalg.lstsq(D_S, y, rcond=None)[0]
        r = y - D_S @ x_S
    x = np.zeros(D.shape[1])
    if selected and x_S is not None:
        for i, idx in enumerate(selected):
            x[idx] = x_S[i]
    return x


def sparse_code(D, Y, T):
    """Apply OMP to every column of Y. Returns X of shape (K, N)."""
    N = Y.shape[1]
    X = np.zeros((D.shape[1], N))
    for i in range(N):
        X[:, i] = omp(D, Y[:, i], T)
        if (i + 1) % 1000 == 0:
            print(f"    OMP {i+1}/{N}")
    return X


def ksvd(D, Y, T, n_iter):
    """
    Run K-SVD for n_iter iterations.
    Phase I: sparse coding via OMP.
    Phase II: dictionary update via SVD.
    Returns D, X, list of Frobenius errors after each Phase II.
    """
    errors = []
    for it in range(n_iter):
        print(f"Iteration {it+1}/{n_iter}")

        # Phase I: sparse coding
        X = sparse_code(D, Y, T)

        # Phase II: dictionary update
        n_atoms = D.shape[1]
        for k in range(n_atoms):
            omega = np.where(np.abs(X[k, :]) > 1e-10)[0]
            if len(omega) == 0:
                continue
            # Residual with atom k removed
            E_k = Y[:, omega] - D @ X[:, omega] + np.outer(D[:, k], X[k, omega])
            U, s, Vt = np.linalg.svd(E_k, full_matrices=False)
            D[:, k] = U[:, 0]
            X[k, omega] = s[0] * Vt[0, :]

        err = np.linalg.norm(Y - D @ X, 'fro')
        errors.append(err)
        print(f"  error: {err:.2f}")

    return D, X, errors
