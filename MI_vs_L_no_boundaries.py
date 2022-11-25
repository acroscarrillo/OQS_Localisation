# import model.py methods (A, C, MI, ...) 
# which contains some main packages (numpy, numba,..)
from src import *
import pickle #to save python objects


# import plotting stuff
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

##########################
# Data generation method #
##########################
@njit(nogil=True, parallel=False)
def MI_vs_L(L_array,run_array,p_array,V,lamb):
    plotting_data = []
    for p in p_array:
        MI_array = np.zeros(len(L_array))
        MI_err_array = np.zeros(len(L_array))
        for L_ind, L in enumerate(L_array):
            realisations_array = np.zeros(run_array[L_ind])
            for k in range(run_array[L_ind]):
                A_mat = A(L,p)
                subsys_A = np.arange(0,L//3,1)
                subsys_B = np.arange(2*L//3,L,1)
                realisations_array[k] = L*MI_by_L_NESS(V,lamb,A_mat,subsys_A,subsys_B)
            MI_array[L_ind] = np.sum(realisations_array)/len(realisations_array)
            MI_err_array[L_ind] =stnd_err(realisations_array)
        plotting_data.append(  (L_array, MI_array, MI_err_array, p)  )
    return plotting_data

#################
# Plotting code #
#################

# parameter space
L_array = np.arange(100,1000,100)
run_array = np.arange(500,50,-50)
# L_array = np.array([400,800,1200,1600,2000])
# run_array = np.array([500,250,100,50,10])
p_array = np.array([0.6,1.13,1.6])
V, lamb = 0, 0

plotting_data = MI_vs_L(L_array,run_array,p_array,V,lamb)
with open("data/MI_vs_L_no_boundaries.ob", 'wb') as fp:
    pickle.dump(plotting_data, fp)

with open ("data/MI_vs_L_no_boundaries.ob", 'rb') as fp:
    plotting_data = pickle.load(fp)

for data in plotting_data:
    L_array, MI_array, MI_err_array, p = data
    plt.errorbar(L_array,MI_array,MI_err_array,label="p="+str(p),ms=2,lw=1)
    
plt.legend()
plt.show()