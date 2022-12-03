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
def MI_vs_LA(L,runs,p_array,V,lamb):
    plotting_data = []
    for p in p_array:
        MI_array = np.zeros(2*L//3)
        MI_err_array = np.zeros(2*L//3)
        for LA_ind, LA in enumerate(range(1,2*L//3+1)):
            subsys_A = np.arange(0,LA,1)
            subsys_B = np.arange(LA+L//3,L,1)
            realisations_array = np.zeros(runs)
            for k in range(runs):
                A_mat = A(L,p)
                realisations_array[k] = L*MI_by_L_NESS(V,lamb,A_mat,subsys_A,subsys_B)
            MI_array[LA_ind] = np.sum(realisations_array)/len(realisations_array)
            MI_err_array[LA_ind] =stnd_err(realisations_array)
        
        LA_array = np.arange(1,2*L//3+1,1)
        plotting_data.append(  (LA_array, MI_array, MI_err_array, p)  )
    return plotting_data

#################
# Plotting code #
#################
# parameter space
L = 100
runs = 100
p_array = np.array([0.6,0.97,1.6])
V, lamb = 0, 0

# plotting_data = MI_vs_LA(L,runs,p_array,V,lamb)
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
    # gradient =  np.round((np.log(MI_array[-1])- np.log(MI_array[0]))/(np.log(LA_array[-1])-np.log(LA_array[0])),2)
    ax.errorbar(LA_array[1:len(LA_array)//2],MI_array[1:len(LA_array)//2],yerr=MI_err_array[1:len(LA_array)//2],label="p="+str(p),ms=2,marker="o",lw=1)
    

ax.set_yscale("log")
# ax.set_xscale("log")
ax.set_ylabel(r"$\mathcal{I}$")
ax.set_xlabel(r"$L_A$")
plt.legend(fontsize=7,loc="upper left")
plt.tight_layout()
plt.show()