import numpy as np

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
