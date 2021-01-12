# In this project we demonstrate the OMP and BP algorithms, by running them 
# on a set of signals and checking whether they provide the desired outcome
import numpy as np
from omp import omp
from lp import lp
import matplotlib.pyplot as plt
 
#%% Parameters
 
# TODO: Set the length of the signal
n = 50

# TODO: Set the number of atoms in the dictionary
m = 100


# TODO: Set the maximum number of non-zeros in the generated vector
s_max = 10


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
print(np.linalg.norm(A_normalized,axis=0))
#%% Create data and run OMP and BP
 
# Nullify the entries in the estimated vector that are smaller than eps
eps_coeff = 1e-4
# Set the optimality tolerance of the linear programing solver
tol_lp = 1e-4
 
# Allocate a matrix to save the L2 error of the obtained solutions
L2_error = np.zeros((s_max,num_realizations,2))
# Allocate a matrix to save the support recovery score
support_error = np.zeros((s_max,num_realizations,2))
           
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
        # Write your code here... true_supp = ????
        
        
        # TODO: Draw at random the coefficients of x in true_supp locations
        # Write your code here... x = ????        
        
        
        # TODO: Create the signal b
        # Write your code here... b = ????
        
        
        # TODO: Run OMP
        # Write your code here... x_omp = omp(????, ????, ????)
        
                
        # TODO: Compute the relative L2 error
        # Write your code here... L2_error[s-1,experiment,0] = ????
        
        
        # TODO: Get the indices of the estimated support
        # Write your code here... estimated_supp = ????
        
        
        # TODO: Compute the support recovery error
        # Write your code here... support_error[s-1,experiment,0] = ????
        
    
        # TODO: Run BP
        # Write your code here... x_lp = lp(????, ????, ????)
        
        
        # TODO: Compute the relative L2 error
        # Write your code here... L2_error[s-1,experiment,1] = ????
        
        
        # TODO: Get the indices of the estimated support, where the
        # coeffecients are larger (in absolute value) than eps_coeff
        # Write your code here... estimated_supp = ????
        
        
        # TODO: Compute the support recovery score
        # Write your code here... support_error[s-1,experiment,1] = ????
        
 
#%% Display the results 
plt.rcParams.update({'font.size': 14})
# Plot the average relative L2 error, obtained by the OMP and BP versus the cardinality
plt.figure()
plt.plot(np.arange(s_max)+1, np.mean(L2_error[:s_max,:,0],axis=1),color='red')
plt.plot(np.arange(s_max)+1, np.mean(L2_error[:s_max,:,1],axis=1),color='green')
plt.xlabel('Cardinality of the true solution')
plt.ylabel('Average and Relative L2-Error')
plt.axis((0,s_max,0,1))
plt.legend(['OMP','LP'])
plt.show()

# Plot the average support recovery score, obtained by the OMP and BP versus the cardinality
plt.figure()
plt.plot(np.arange(s_max)+1, np.mean(support_error[:s_max,:,0],axis=1),color='red')
plt.plot(np.arange(s_max)+1, np.mean(support_error[:s_max,:,1],axis=1),color='green')
plt.xlabel('Cardinality of the true solution')
plt.ylabel('Probability of Error in Support')
plt.axis((0,s_max,0,1))
plt.legend(['OMP','LP'])
plt.show()