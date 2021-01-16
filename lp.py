import numpy as np
from scipy.optimize import linprog

def lp(A, b, tol):
# LP Solve Basis Pursuit via linear programing
#
# Solves the following problem:
#   min_x || x ||_1 s.t. b = Ax
#
# The solution is returned in the vector x.

    # number of columns
    m = A.shape[1]

    # Set the options to be used by the linprog solver
    opt = {"tol": tol, "disp": False}

    # TODO: Use scipy.optimize linprog function to solve the BP problem
    # function to minimize
    c = np.ones((2*m,1))
    
    # Equality constraint matrix
    A_eq = np.concatenate((A,-A),axis=1)
   
    # Solve the linear programming problem
    result = linprog(c,A_eq = A_eq, b_eq = b, options=opt)
    
    x = result.x[0:m]-result.x[m:2*m]

    return x