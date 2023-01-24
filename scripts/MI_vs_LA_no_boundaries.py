# import model.py methods (A, C, MI, ...) 
# which contains some main packages (numpy, numba,..)
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import *
import pickle #to save python objects

# import plotting stuff
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

##########################
# Data generation method #
##########################
@njit(nogil=True, parallel=False)
def MI_vs_LA(L,LA_steps,runs,p_array,V,lamb):
    plotting_data = []
    lenght = len(range(1,2*L//3+1,LA_steps))
    for p in p_array:
        MI_array = np.zeros(lenght)
        MI_err_array = np.zeros(lenght)
        for LA_ind, LA in enumerate(range(1,2*L//3+1,LA_steps)):
            print(LA_ind/lenght)
            subsys_A = np.arange(0,LA,1)
            subsys_B = np.arange(LA+L//3,L,1)
            realisations_array = np.zeros(runs)
            for k in range(runs):
                A_mat = A(L,p)
                realisations_array[k] = L*MI_by_L_NESS(V,lamb,A_mat,subsys_A,subsys_B)
            MI_array[LA_ind] = np.sum(realisations_array)/len(realisations_array)
            MI_err_array[LA_ind] =stnd_err(realisations_array)
        
        LA_array = np.arange(1,2*L//3+1,LA_steps)
        plotting_data.append(  (LA_array, MI_array, MI_err_array, p)  )
    return plotting_data

#################
# Plotting code #
#################
# parameter space
L = 1000
LA_steps = 60
runs = 20
p_array = np.array([0, 0.01, 0.1, 0.2, 0.6, 0.97, 1.6, 4])
V, lamb = 0, 0

# plotting_data = MI_vs_LA(L,LA_steps,runs,p_array,V,lamb)
# with open("data/MI_vs_LA_no_boundaries.ob", 'wb') as fp:
#     pickle.dump(plotting_data, fp)

with open ("data/MI_vs_LA_no_boundaries.ob", 'rb') as fp:
    plotting_data = pickle.load(fp)

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
pt = 0.0138889 
fig, ax  = plt.subplots(1,1,figsize = (246*pt,150*pt))
for data in plotting_data:
    LA_array, MI_array, MI_err_array, p = data
    if np.round(p,2) !=4.0: 
        # gradient =  np.round((np.log(MI_array[-1])- np.log(MI_array[0]))/(np.log(LA_array[-1])-np.log(LA_array[0])),2)
        ax.errorbar(LA_array[1:len(LA_array)//2+1],MI_array[1:len(LA_array)//2+1],yerr=MI_err_array[1:len(LA_array)//2+1],label="p="+str(p),ms=2,marker="o",lw=1)
        # ax.errorbar(LA_array,MI_array,yerr=MI_err_array,label="p="+str(p),ms=2,marker="o",lw=1)

# ax.set_yscale("log")
ax.set_xscale("log")
ax.set_ylabel(r"$\mathcal{I}$")
ax.set_xlabel(r"$L_A$")
plt.legend(fontsize=6,loc="upper left")
plt.tight_layout()
plt.show()