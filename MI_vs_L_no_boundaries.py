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
            subsys_A = np.arange(0,L//3,1)
            subsys_B = np.arange(2*L//3,L,1)
            for k in range(run_array[L_ind]):
                A_mat = A(L,p)
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
# run_array = np.array([500,100,50,50,10])
# p_array = np.array([0.6,0.9,1,1.13,1.6])
p_array = np.array([0,0.01,0.1,0.2])
V, lamb = 0, 0

plotting_data = MI_vs_L(L_array,run_array,p_array,V,lamb)
# with open("data/MI_vs_L_no_boundaries.ob", 'wb') as fp:
#     pickle.dump(plotting_data, fp)

# with open ("data/MI_vs_L_no_boundaries.ob", 'rb') as fp:
#     plotting_data = pickle.load(fp)

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
pt = 0.0138889 
fig, ax  = plt.subplots(1,1,figsize = (246*pt,150*pt))
for data in plotting_data:
    L_array, MI_array, MI_err_array, p = data
    # gradient = np.around( np.log(MI_array[-1])-np.log(MI_array[0]),2 )
    # gradient =  np.round((np.log(MI_array[-1])- np.log(MI_array[0]))/(np.log(L_array[-1])-np.log(L_array[0])),2)
    g =  (y[-1]-y[0])/(x[-1]-x[0])
    ax.errorbar(L_array,MI_array,yerr=MI_err_array,label="p="+str(p)+", g (log)="+str(g),ms=2,marker="o",lw=1)
    

# ax.set_yscale("log")
# ax.set_xscale("log")
ax.set_ylabel(r"$\mathcal{I}$")
ax.set_xlabel(r"$L$")
plt.legend(fontsize=5,loc="lower right")
plt.tight_layout()
plt.show()