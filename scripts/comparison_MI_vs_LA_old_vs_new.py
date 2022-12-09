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



plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
pt = 0.0138889 
fig, ax  = plt.subplots(1,1,figsize = (246*pt,150*pt))

p_array = [0,0.2,0.97,1.6]
color_array = ["C0","C1","C2","C3"]

with open ("data/MI_vs_LA_no_boundaries.ob", 'rb') as fp:
    plotting_data = pickle.load(fp)

j=0
for i, data in enumerate(plotting_data):
    LA_array, MI_array, MI_err_array, p = data
    if p in p_array:
        # gradient =  np.round((np.log(MI_array[-1])- np.log(MI_array[0]))/(np.log(LA_array[-1])-np.log(LA_array[0])),2)
        ax.errorbar(LA_array[0:len(LA_array)//2+1],MI_array[0:len(LA_array)//2+1],yerr=MI_err_array[0:len(LA_array)//2+1],label="p="+str(p),ms=2,marker="o",lw=1,color=color_array[j])
        # ax.errorbar(LA_array,MI_array,yerr=MI_err_array,label="p="+str(p),ms=2,marker="o",lw=1)
        j += 1


with open ("data/MI_vs_LA_with_boundaries.ob", 'rb') as fp:
    plotting_data = pickle.load(fp)

j=0
for i, data in enumerate(plotting_data):
    LA_array, MI_array, MI_err_array, p = data
    if p in p_array:
        # gradient =  np.round((np.log(MI_array[-1])- np.log(MI_array[0]))/(np.log(LA_array[-1])-np.log(LA_array[0])),2)
        ax.errorbar(LA_array[0:len(LA_array)//2+1],MI_array[0:len(LA_array)//2+1],yerr=MI_err_array[0:len(LA_array)//2+1],ms=2,marker="o",lw=1,color=color_array[j],ls="dashed")
        # ax.errorbar(LA_array,MI_array,yerr=MI_err_array,label="p="+str(p),ms=2,marker="o",lw=1)
        j += 1
# ax.set_yscale("log")
# ax.set_xscale("log")
ax.set_ylabel(r"$\mathcal{I}$")
ax.set_xlabel(r"$L_A$")
plt.legend(fontsize=6,loc="upper left")
plt.tight_layout()
plt.show()