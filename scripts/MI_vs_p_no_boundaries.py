# import model.py methods (A, C, MI, ...) 
# which contains some main packages (numpy, numba,..)
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join('..')))
from src import *
import pickle #to save python objects

# import plotting stuff
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

##########################
# Data generation method #
##########################
@njit(nogil=True, parallel=False)
def MI_vs_p(L_array,run_array,p_array,V,lamb): 
    plotting_data = []
    for L_ind, L in enumerate(L_array):
        print(L_ind/len(L_array))
        MI_array = np.zeros(len(p_array))
        MI_err_array = np.zeros(len(p_array))
        for p_ind, p in enumerate(p_array):
            realisations_array = np.zeros(run_array[L_ind])
            for k in range(run_array[L_ind]):
                A_mat = A(L,p)
                subsys_A = np.arange(0,L//3,1)
                subsys_B = np.arange(2*L//3,L,1)
                realisations_array[k] = L*MI_by_L_NESS(V,lamb,A_mat,subsys_A,subsys_B)
            MI_array[p_ind] = np.sum(realisations_array)/len(realisations_array)
            MI_err_array[p_ind] = stnd_err(realisations_array)
        plotting_data.append(  (p_array, MI_array, MI_err_array, L)  )
    return plotting_data

#################
# Plotting code #
#################

# parameter space
# L_array = np.arange(100,1000,100)
# run_array = np.arange(500,50,-50)
# L_array = np.array([10,20,30])
# run_array= np.array([300,200,100])
L_array = np.array([400,800,1200,1600,2000])
run_array = np.array([500,200,200,200,100])
p_array = np.arange(0.9,1.1,0.005)
V, lamb = 0, 0

# plotting_data = MI_vs_p(L_array,run_array,p_array,V,lamb)
# with open("data/MI_vs_p_no_boundaries.ob", 'wb') as fp:
#     pickle.dump(plotting_data, fp)

with open ("data/MI_vs_p_no_boundaries.ob", 'rb') as fp:
    plotting_data = pickle.load(fp)

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
pt = 0.0138889 
fig, ax  = plt.subplots(1,1,figsize = (246*pt,150*pt))
for data in plotting_data:
    p_array, MI_array, MI_err_array, L = data
    ax.errorbar(p_array,MI_array,yerr=MI_err_array,label="L="+str(L),ms=1.5,marker="o",lw=1)
    

# ax.set_yscale("log")
# ax.set_xscale("log")
ax.set_ylabel(r"$\mathcal{I}$")
ax.set_xlabel(r"$p$")
plt.legend(fontsize=5,loc="upper right")
plt.tight_layout()
plt.show()