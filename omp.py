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
    # residual
    r = b
    # support
    sup = []

    for it in range(k):
        
        #print("ITERATION",it)

        # Determine the atom to choose
        atom = np.argmax(np.absolute(np.transpose(A) @ r))
        #print("Atome le plus corrélé",atom)
        
        # Update the support
        if atom not in sup:
            sup.append(atom)

        # Update x
        coefi, _, _, _ = np.linalg.lstsq(A[:, sup], b,rcond=None)
        x[sup] = coefi
        #print("x",x)

        # Update the residual r
        r = b - np.dot(A[:,sup],coefi)
        #print("Update de r",r)

        #print("\n\n\n")
    # return the obtained x
    return x
