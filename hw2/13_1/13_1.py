import numpy as np

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

A = np.array([[1, 2],
              [0, 1],
              [2, 1],
              [0, -1],
              [1, 1]], dtype=float)

B = np.array([[1, 1],
              [0, 1],
              [2, 0],
              [0, 1]], dtype=float)

U, V, C, S, G = gsvd(A, B)

print("C:\n", C)
print("S:\n", S)
print("G:\n", G)
print("Reconstruction A:", np.linalg.norm(A - U @ C @ G.T))
print("Reconstruction B:", np.linalg.norm(B - V @ S @ G.T))

c = np.diag(C)
s = np.diag(S[S.shape[0]-2:, :])
print("c^2 + s^2:", c**2 + s**2)
