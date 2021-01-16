import numpy as np
from omp import omp
from lp import lp
import matplotlib.pyplot as plt
import random as rdm

n = 50
m = 100
min_coeff_val = 1
max_coeff_val = 3

s = 10

# Set the optimality tolerance of the linear programing solver
tol_lp = 1e-4

# TODO: Create a random matrix A of size (n x m)
A = np.random.randn(n,m)

# TODO: Normalize the columns of the matrix to have a unit norm
col_norm = np.linalg.norm(A,axis=0)
A_normalized = np.divide(A,np.transpose(col_norm))

x = np.zeros(m)
        
# TODO: Draw at random a true_supp vector
true_supp = np.array(rdm.sample(range(m), s))

# Draw random signs
sign = np.random.randint(2, size=s)
sign[sign == 0] = -1

# TODO: Draw at random the coefficients of x in true_supp locations
x[true_supp] = sign*(min_coeff_val + (max_coeff_val-min_coeff_val)*np.random.rand(1,s))     

print("x",x)

# TODO: Create the signal b
b = A_normalized @ x

print("###################### OMP ###############################")        
# TODO: Run OMP
#x_omp = omp(A_normalized, b, s)

print("###################### BP ###############################")
# TODO: Run BP
x_lp = lp(A_normalized, b, tol_lp)

print(x_lp)

x_lp[np.absolute(x_lp)<=1e-4] = 0

print(x_lp)

plt.figure(1)
plt.plot(x)
plt.figure(2)
plt.plot(x_lp,color='red')
plt.show()