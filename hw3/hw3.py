import scipy.io
import numpy as np
from scipy.optimize import linprog

data = scipy.io.loadmat('wisconsin_breast_cancer.mat')

X = data['X']                                          # (569, 30)
y = data['y']                                          # (569, 1)
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
Row3=-np.concatenate([XY,Zmn,Im,y], axis=1)

# Assemble big matrix
A=np.vstack([Row1,Row2,Row3])

# Make cost vector
c=1 # Change this later
cost=np.hstack([np.zeros(n),np.ones(n),c*np.ones(m),0])

# Make h vector

h=np.hstack([np.zeros(n),np.zeros(n),np.ones(m)])
# print(A.shape)     #(629,630)
# print(cost.shape)  #(630,)
# print(h.shape)     #(629,)

# Add more inequalities

Row4=np.hstack([np.zeros((n,n)),-np.identity(n),np.zeros((n,m+1))])
# print(Row4.shape)

Row5=np.hstack([np.zeros((m,2*n)),-np.identity(m),np.zeros((m,1))])
# print(Row5.shape)

A_big=np.vstack([A,Row4,Row5])
print(A_big.shape)
print(A.shape)

h_big=np.hstack([h,np.zeros(n),np.zeros(m)])

# Solve linear program

  # Minimize: c @ x
  # Subject to: A_ub @ x <= b_ub
  #             A_eq @ x == b_eq
  #             bounds on x

bounds = [(0, None)] * len(cost)

result = linprog(cost, A_ub=A_big, b_ub=h_big, bounds=bounds)

print(result.x)      # optimal solution
# print(len(result.x))      # optimal solution
# print(result.fun)    # optimal objective value
