# import model.py methods (A, C, MI, ...) 
# which contains some main packages (numpy, numba,..)
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import *
from tqdm import tqdm

# import plotting stuff
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"


#####################
# cluster MI method #
#####################
@jit
def cluster_MI(L,p,subsys_A,subsys_B):
    V,lamb = 0,0
    A_mat = A(L,p)
    return MI_by_L_NESS(V,lamb,A_mat,subsys_A,subsys_B)

#trigger the jit
cluster_MI(4,10,np.array([0]),np.array([1]))

#################
# generate data #
#################
L=1000
p_array = np.arange(0,1.5+0.1,0.1)
runs = 200

c_size = 10
subsys_A = np.arange( L//3, L//3 + c_size, 1 )
subsys_B = np.arange( 2*L//3, 2*L//3 + c_size, 1 )

MI_data = np.zeros((len(p_array),runs))
for p_ind, p in tqdm(enumerate(p_array)):
    for i in range(runs):
        MI_data[p_ind, i] = cluster_MI(L,p,subsys_A,subsys_B)

MI_avg = np.average(MI_data, axis=1)
with open('MI_cluster_avg.npy', 'wb') as f:
    np.save(f, MI_avg)

MI_err = stats.sem(MI_data, axis=1)
with open('MI_cluster_err.npy', 'wb') as f:
    np.save(f, MI_err)


#################
# Plotting code #
#################
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
pt = 0.0138889 
fig, ax  = plt.subplots(1,1,figsize = (246*pt,150*pt))
color_ls = ["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9"]



#load data
MI_avg = np.load('MI_cluster_avg.npy')
MI_err = np.load('MI_cluster_err.npy')

ax.errorbar(p_array,MI_avg,yerr=MI_err)
ax.set_ylabel(r"$\mathcal{I}$")
ax.set_xlabel(r"$p$")
plt.tight_layout()

plt.show()