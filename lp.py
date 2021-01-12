import numpy as np
from scipy.optimize import linprog

def lp(A, b, tol):
# LP Solve Basis Pursuit via linear programing
#
# Solves the following problem:
#   min_x || x ||_1 s.t. b = Ax
#
# The solution is returned in the vector x.

    # Set the options to be used by the linprog solver
    opt = {"tol": tol, "disp": False}

    # TODO: Use scipy.optimize linprog function to solve the BP problem
    # Write your code here ... x = ????



    return x