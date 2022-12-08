# import model.py methods (A, C, MI, ...) 
# which contains some main packages (numpy, numba,..)
from src import *
import pickle #to save python objects
from scipy.optimize import curve_fit, minimize #to do the fitting


# import plotting stuff
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

###############
# import data #
###############
with open ("data/MI_vs_L_no_boundaries.ob", 'rb') as fp:
    mi_vs_L_n_b = pickle.load(fp), "MI vs L no boundaries", "MI", "L"
with open ("data/MI_vs_LA_no_boundaries.ob", 'rb') as fp:
    mi_vs_LA_n_b = pickle.load(fp), "MI vs L_A no boundaries", "MI", "L_A"
with open ("data/MI_vs_p_no_boundaries.ob", 'rb') as fp:
    mi_vs_p_n_b = pickle.load(fp), "MI vs p no boundaries", "MI", "p"
with open ("data/MI_vs_LA_with_boundaries.ob", 'rb') as fp:
    mi_vs_LA_w_b = pickle.load(fp), "MI vs L_A with boundaries", "MI", "L_A"

data2plot = [mi_vs_L_n_b, mi_vs_LA_n_b, mi_vs_p_n_b, mi_vs_LA_w_b]

#############
# plot data #
#############
pt = 0.0138889 
t = (4,4)
fig, ax  = plt.subplots(ncols=t[0],nrows=t[1], figsize = (246*pt,200*pt),constrained_layout=True)

plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 200


r,c=0,0
for data in data2plot:
    for d in data[0]:
        L_array, MI_array, MI_err_array, lab = d
        ax[r,c].errorbar(L_array,MI_array,yerr=MI_err_array,label="lab="+str(lab),ms=2,marker="o",lw=1)
        
    ax[r,c].set_yscale("log")
    ax[r,c].set_xscale("log")
    ax[r,c].set_title(data[1])
    ax[r,c].set_ylabel(data[2])
    ax[r,c].set_xlabel(data[3])
    ax[r,c].legend(fontsize=5)#,loc="lower right")
    r,c = cycle_table(r, c, t)

    for d in data[0]:
        L_array, MI_array, MI_err_array, lab = d
        ax[r,c].errorbar(L_array,MI_array,yerr=MI_err_array,label="lab="+str(lab),ms=2,marker="o",lw=1)
        
    ax[r,c].set_yscale("log")
    ax[r,c].set_title(data[1])
    ax[r,c].set_ylabel(data[2])
    ax[r,c].set_xlabel(data[3])
    ax[r,c].legend(fontsize=5)#,loc="lower right")
    r,c = cycle_table(r, c, t)    

    for d in data[0]:
        L_array, MI_array, MI_err_array, lab = d
        ax[r,c].errorbar(L_array,MI_array,yerr=MI_err_array,label="lab="+str(lab),ms=2,marker="o",lw=1)
        
    ax[r,c].set_xscale("log")
    ax[r,c].set_title(data[1])
    ax[r,c].set_ylabel(data[2])
    ax[r,c].set_xlabel(data[3])
    ax[r,c].legend(fontsize=5)#,loc="lower right")
    r,c = cycle_table(r, c, t)      

    for d in data[0]:
        L_array, MI_array, MI_err_array, lab = d
        ax[r,c].errorbar(L_array,MI_array,yerr=MI_err_array,label="lab="+str(lab),ms=2,marker="o",lw=1)
        
    ax[r,c].set_title(data[1])
    ax[r,c].set_ylabel(data[2])
    ax[r,c].set_xlabel(data[3])
    ax[r,c].legend(fontsize=5)#,loc="lower right")
    r,c = cycle_table(r, c, t)  

# fig.tight_layout()
plt.show()