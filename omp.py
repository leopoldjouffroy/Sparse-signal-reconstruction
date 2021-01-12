import numpy as np

def omp(A, b, k):
# OMP Solve the P0 problem via OMP
#
# Solves the following problem:
#   min_x ||b - Ax||_2^2 s.t. ||x||_0 <= k
#
# The solution is returned in the vector x

    # Initialize the vector x
    x = np.zeros(np.shape(A)[1])

    # TODO: Implement the OMP algorithm
    

    # return the obtained x
    return x
