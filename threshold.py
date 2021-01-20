import numpy as np

def threshold(A, b, k):
# OMP Solve the P0 problem via OMP
#
# Solves the following problem:
#   min_x ||b - Ax||_2^2 s.t. ||x||_0 <= k
#
# The solution is returned in the vector x

    # Initialize Beta
    x = np.zeros(np.shape(A)[1])
    Beta = np.abs(np.dot(np.transpose(A),b))
    #print(Beta)
    #indices = (-Beta[:,0]).argsort()
    indices = (-Beta).argsort()
    #print(indices)

    # TODO: Implement the OMP algorithm
    # residual
    r = b
    norm_r = np.linalg.norm(r)
    # support
    sup = []

    it = 0
    while it <= k and norm_r >=1e-4:

        #print("ITERATION",it)

        # Update the support
        if indices[it] not in sup:
            sup.append(indices[it])



        # Update x
        coefi, _, _, _ = np.linalg.lstsq(A[:, sup], b,rcond=None)
        x[sup] = coefi
        #print("x",x)

        # Update the residual r
        r = b - np.dot(A[:,sup],coefi)
        norm_r = np.linalg.norm(r)
        #print("Update de r",norm_r)

        #print("\n\n\n")
        it = it+1
    # return the obtained x

    return x
