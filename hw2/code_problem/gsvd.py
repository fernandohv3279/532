import numpy as np

def csgen(Q11, Q21):
    m, k = Q11.shape
    p = Q21.shape[0]

    # Step 1: full SVD of Q11
    U, c, Xt = np.linalg.svd(Q11, full_matrices=True)
    X = Xt.T

    # Step 2: compute s and V+
    s = np.sqrt(1 - c**2)
    Vp = (Q21 @ X) / s

    # Step 3: extend to full V via random projection + QR
    F = np.random.randn(p, p - k)
    Vtemp = (np.eye(p) - Vp @ Vp.T) @ F
    Vorth, _ = np.linalg.qr(Vtemp)

    # Step 4: assemble V, C, S
    V = np.hstack([Vorth, Vp])

    C = np.zeros((m, k))
    np.fill_diagonal(C, c)

    S = np.zeros((p, k))
    S[p-k:, :] = np.diag(s)

    return U, V, C, S, X

#Test csdecomp
# Generate a valid test input
n = 5
m = 3
p = 4
k = 2

# Random orthogonal matrix of size (m+p) x (m+p)
Q, _ = np.linalg.qr(np.random.randn(m+p, m+p))

# Extract block column
Q11 = Q[:m, :k]
Q21 = Q[m:, :k]

# Run CS decomposition
U, V, C, S, X = csgen(Q11, Q21)

# Test 1: reconstruction
print("Reconstruction error Q11:", np.linalg.norm(Q11 - U @ C @ X.T))
print("Reconstruction error Q21:", np.linalg.norm(Q21 - V @ S @ X.T))

# Test 2: CS property
c = np.diag(C[:k, :])
s = np.diag(S[p-k:, :])   # extract from bottom k rows
print("c^2 + s^2:", c**2 + s**2)

# Test 3: orthogonality
print("U^T U error:", np.linalg.norm(U.T @ U - np.eye(U.shape[1])))
print("V^T V error:", np.linalg.norm(V.T @ V - np.eye(V.shape[0])))
print("X^T X error:", np.linalg.norm(X.T @ X - np.eye(k)))


def gsvd(A, B):
    m, n = A.shape
    p = B.shape[0]

    # Step 1-2: stack and thin SVD
    M = np.vstack([A, B])
    Qhat, Sigmahat, Wt = np.linalg.svd(M, full_matrices=False)
    W = Wt.T

    # Step 3: partition Qhat
    Q11 = Qhat[:m, :]
    Q21 = Qhat[m:, :]

    # Step 4: CS decomposition
    U, V, C, S, X = csgen(Q11, Q21)

    # Step 5: compute G
    G = W @ np.diag(Sigmahat) @ X

    return U, V, C, S, G

# Test gsvd

A = np.random.randn(5, 3)
B = np.random.randn(4, 3)

U, V, C, S, G = gsvd(A, B)

print("Reconstruction error A:", np.linalg.norm(A - U @ C @ G.T))
print("Reconstruction error B:", np.linalg.norm(B - V @ S @ G.T))
print("c^2 + s^2:", np.diag(C[:3,:3])**2 + np.diag(S[1:,:3])**2)
