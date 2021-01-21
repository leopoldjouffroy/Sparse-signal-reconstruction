# In this project we demonstrate the OMP and BP algorithms, by running them 
# on a set of signals and checking whether they provide the desired outcome
import numpy as np
from omp import omp
from lp import lp
from threshold import threshold
import matplotlib.pyplot as plt
 
#%% Parameters
 
# TODO: Set the length of the signal
n = 50

# TODO: Set the number of atoms in the dictionary
m = 100


# TODO: Set the maximum number of non-zeros in the generated vector
s_max = 15


# TODO: Set the minimal entry value
min_coeff_val = 1


# TODO: Set the maximal entry value
max_coeff_val = 3


# Number of realizations
num_realizations = 200

# Base seed: A non-negative integer used to reproduce the results
# TODO: Set an arbitrary value for base seed
base_seed = 115

 
#%% Create the dictionary
 
# TODO: Create a random matrix A of size (n x m)
A = np.random.randn(n,m)

# TODO: Normalize the columns of the matrix to have a unit norm
col_norm = np.linalg.norm(A,axis=0)
A_normalized = np.divide(A,np.transpose(col_norm))

#%% Create data and run OMP and BP
 
# Nullify the entries in the estimated vector that are smaller than eps
eps_coeff = 1e-4
# Set the optimality tolerance of the linear programing solver
tol_lp = 1e-4
 
# Allocate a matrix to save the L2 error of the obtained solutions
L2_error = np.zeros((s_max,num_realizations,3))
# Allocate a matrix to save the support recovery score
support_error = np.zeros((s_max,num_realizations,3))
           
# Loop over the sparsity level
for s in range(s_max):

    s = s+1
    # Use the same random seed in order to reproduce the results if needed
    np.random.seed(s + base_seed)
    
    # Loop over the number of realizations
    for experiment in range(num_realizations):
   
        # In this part we will generate a test signal b = A_normalized @ x by 
        # drawing at random a sparse vector x with s non-zeros entries in 
        # true_supp locations with values in the range of [min_coeff_val, max_coeff_val]
        
        x = np.zeros(m)
        
        # TODO: Draw at random a true_supp vector
        true_supp = np.random.randint(0,high=m,size=s)
        
        # Draw random signs
        sign = np.random.randint(2, size=s)
        sign[sign == 0] = -1

        # TODO: Draw at random the coefficients of x in true_supp locations
        x[true_supp] = sign*(min_coeff_val + (max_coeff_val-min_coeff_val)*np.random.rand(1,s))     
        
        # TODO: Create the signal b
        b = A_normalized @ x
        
        # TODO: Run OMP
        x_omp = omp(A_normalized, b, s)
        x_omp[np.absolute(x_omp)<=eps_coeff] = 0
                
        # TODO: Compute the relative L2 error
        L2_error[s-1,experiment,0] = (np.linalg.norm(x_omp-x)**2)/(np.linalg.norm(x)**2)
        
        # TODO: Get the indices of the estimated support
        estimated_supp = np.nonzero(x_omp)
        
        
        # TODO: Compute the support recovery error
        norm_true_supp = np.linalg.norm(true_supp)
        norm_estimated_supp = np.linalg.norm(estimated_supp)
        norm_intersection = np.linalg.norm(np.intersect1d(true_supp,estimated_supp))
        max_norm = np.maximum(norm_true_supp,norm_estimated_supp)
        
        if(max_norm != 0):
            support_error[s-1,experiment,0] = 1 - (norm_intersection/max_norm)
        else:
            support_error[s-1,experiment,0] = 0

        # TODO: Run BP
        x_lp = lp(A_normalized, b, tol_lp)
        x_lp[np.absolute(x_lp)<=eps_coeff] = 0
        
        # TODO: Compute the relative L2 error
        L2_error[s-1,experiment,1] = (np.linalg.norm(x_lp-x)**2)/(np.linalg.norm(x)**2)
        
        
        # TODO: Get the indices of the estimated support, where the
        # coeffecients are larger (in absolute value) than eps_coeff
        estimated_supp = np.nonzero(x_lp)
                
        # TODO: Compute the support recovery score
        norm_estimated_supp = np.linalg.norm(estimated_supp)
        norm_intersection = np.linalg.norm(np.intersect1d(true_supp,estimated_supp))
        max_norm = np.maximum(norm_true_supp,norm_estimated_supp)
        
        if(max_norm != 0):
            support_error[s-1,experiment,1] = 1 - (norm_intersection/max_norm)
        else:
            support_error[s-1,experiment,1] = 0



        # TODO: Run Threshold
        x_threshold = threshold(A_normalized, b, s)
        x_threshold[np.absolute(x_omp)<=eps_coeff] = 0

        # TODO: Compute the relative L2 error
        L2_error[s-1,experiment,2] = (np.linalg.norm(x_threshold-x)**2)/(np.linalg.norm(x)**2)

        # TODO: Get the indices of the estimated support
        estimated_supp = np.nonzero(x_threshold)


        # TODO: Compute the support recovery error
        norm_true_supp = np.linalg.norm(true_supp)
        norm_estimated_supp = np.linalg.norm(estimated_supp)
        norm_intersection = np.linalg.norm(np.intersect1d(true_supp,estimated_supp))
        max_norm = np.maximum(norm_true_supp,norm_estimated_supp)

        if(max_norm != 0):
            support_error[s-1,experiment,2] = 1 - (norm_intersection/max_norm)
        else:
            support_error[s-1,experiment,2] = 0


 
#%% Display the results 
plt.rcParams.update({'font.size': 14})
# Plot the average relative L2 error, obtained by the OMP and BP versus the cardinality
plt.figure()
plt.plot(np.arange(s_max)+1, np.mean(L2_error[:s_max,:,0],axis=1),color='red')
plt.plot(np.arange(s_max)+1, np.mean(L2_error[:s_max,:,1],axis=1),color='green')
plt.plot(np.arange(s_max)+1, np.mean(L2_error[:s_max,:,2],axis=1),color='blue')
plt.xlabel('Cardinality of the true solution')
plt.ylabel('Average and Relative L2-Error')
plt.axis((0,s_max,0,1))
plt.legend(['OMP','LP','Threshold'])
plt.show()

# Plot the average support recovery score, obtained by the OMP and BP versus the cardinality
plt.figure()
plt.plot(np.arange(s_max)+1, np.mean(support_error[:s_max,:,0],axis=1),color='red')
plt.plot(np.arange(s_max)+1, np.mean(support_error[:s_max,:,1],axis=1),color='green')
plt.plot(np.arange(s_max)+1, np.mean(support_error[:s_max,:,2],axis=1),color='blue')
plt.xlabel('Cardinality of the true solution')
plt.ylabel('Probability of Error in Support')
plt.axis((0,s_max,0,1))
plt.legend(['OMP','LP','Threshold'])
plt.show()
