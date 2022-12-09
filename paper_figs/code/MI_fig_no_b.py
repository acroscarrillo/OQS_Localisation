# import model.py methods (A, C, MI, ...) 
# which contains some main packages (numpy, numba,..)
import sys
import os

dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(dir,'..',"..")))
from src import *
import pickle #to save python objects

# import plotting stuff
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

###############
# import data #
###############
with open ("data/MI_vs_L_no_boundaries.ob", 'rb') as fp:
    plotting_data = pickle.load(fp)

#############
# plot data #
#############
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
pt = 0.0138889 
fig, ax  = plt.subplots(1,1,figsize = (246*pt,150*pt))
for data in plotting_data:
    L_array, MI_array, MI_err_array, p = data

    g, y_intercept = fit_log_log( L_array, MI_array)
    ax.scatter(x,y,label="p="+str( np.round(y_label,2) )+", g log="+str(np.round(g,2)),s=1)

    ax.errorbar(L_array,MI_array,yerr=MI_err_array,label="p="+str(p)+", g (log)="+str(g),ms=2,marker="o",lw=1)
    

ax.set_yscale("log")
ax.set_xscale("log")
ax.set_ylabel(r"$\mathcal{I}$")
ax.set_xlabel(r"$L$")
plt.legend(fontsize=5,loc="lower right")
plt.tight_layout()
plt.show()