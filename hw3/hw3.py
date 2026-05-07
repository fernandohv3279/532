import scipy.io
import numpy as np
from scipy.optimize import linprog

data = scipy.io.loadmat('wisconsin_breast_cancer.mat')

X = data['X']                                          # (569, 30)
y = data['y']                                          # (569, 1)
y = np.where(y == 0, -1, 1)                           # convert 0/1 → -1/+1
Y=np.diag(y.flatten())
m = X.shape[0]
n = X.shape[1]
feature_names = data['feature_names'].flatten()
target_names = data['target_names'].flatten()


# Make "row" one of the big matrix
Row1=np.concatenate([np.identity(n),-np.identity(n),np.zeros((n,m)),np.zeros((n,1))], axis=1)


# Make "row" 2 of the big matrix
Row2=np.concatenate([-np.identity(n),-np.identity(n),np.zeros((n,m)),np.zeros((n,1))], axis=1)

# Make "row" 3 of the big matrix
XY= Y @ X
Zmn=np.zeros((m,n))
Im=np.identity(m)
Row3=-np.concatenate([XY,-XY,Im,y], axis=1)

# Assemble big matrix
A=np.vstack([Row1,Row2,Row3])

# Make cost vector
c=1 # Change this later
cost=np.hstack([np.ones(n),np.ones(n),c*np.ones(m),0])

# Make h vector

h=np.hstack([np.zeros(n),np.zeros(n),-np.ones(m)])
# print(A.shape)     #(629,630)
# print(cost.shape)  #(630,)
# print(h.shape)     #(629,)

# Solve linear program

  # Minimize: c @ x
  # Subject to: A_ub @ x <= b_ub
  #             A_eq @ x == b_eq
  #             bounds on x

bounds = [(0, None)] * len(cost)

result = linprog(cost, A_ub=A, b_ub=h, bounds=bounds)

# print(result.x)      # optimal solution
# print(len(result.x))      # optimal solution
# print(result.fun)    # optimal objective value

w = result.x[:n] - result.x[n:2*n]
z = result.x[2*n:2*n+m]
t = result.x[-1]

print("w:", w)
print("nonzero weights:", np.sum(w != 0))
print("bias t:", t)
print("misclassified:", np.sum(z > 0))
