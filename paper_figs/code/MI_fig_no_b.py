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
# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=5)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=5)    # fontsize of the tick labels
plt.rc('ytick', labelsize=5)    # fontsize of the tick labels
plt.rc('legend', fontsize=5)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

###############
# import data #
###############
with open ("data/MI_vs_L_no_boundaries.ob", 'rb') as fp:
    MI_vs_L_n_b = pickle.load(fp)
with open ("data/MI_vs_LA_no_boundaries.ob", 'rb') as fp:
    MI_vs_LA_n_b = pickle.load(fp)
with open ("data/MI_vs_p_no_boundaries.ob", 'rb') as fp:
    MI_vs_p_n_b = pickle.load(fp)
#############
# plot data #
#############
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
pt = 0.0138889 
fig = plt.figure(figsize = (246*pt,250*pt))
gs = fig.add_gridspec(2,2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, :])
left, bottom, width, height = [0.73, 0.25, 0.2, 0.2]
ax3_inset = fig.add_axes([left, bottom, width, height])

# MI vs L
for data in MI_vs_L_n_b:
    L_array, MI_array, MI_err_array, p = data
    g, y_intercept = fit_log_log( L_array, MI_array )
    ax1.plot(L_array, np.exp(y_intercept)*(L_array**g),lw=0.6,c="black")
    ax1.errorbar(L_array,MI_array,yerr=MI_err_array,label="p="+str(p)+", "+r'$\Delta$'+"="+str(np.round(g,2)),ms=0.6,marker="o",lw=0.6)

ax1.legend(fontsize=4,loc="lower right")
ax1.set_yscale("log")
ax1.set_xscale("log")
ax1.set_ylabel(r"$\mathcal{I}$")
ax1.set_xlabel(r"$L$")

# MI vs L_A
for data in MI_vs_LA_n_b:
    LA_array, MI_array, MI_err_array, p = data
    ax2.errorbar(LA_array[1:len(LA_array)//2+1],MI_array[1:len(LA_array)//2+1],yerr=MI_err_array[1:len(LA_array)//2+1],label="p="+str(p),ms=0.6,marker="o",lw=0.6)

ax2.legend(fontsize=4,loc="upper left")
ax2.set_xscale("log")
ax2.set_ylabel(r"$\mathcal{I}$")
ax2.set_xlabel(r"$L_A$")

# MI vs p
for data in MI_vs_p_n_b:
    p_array, MI_array, MI_err_array, L = data
    ax3.errorbar(p_array,MI_array,yerr=MI_err_array,label="L="+str(L),ms=0.6,marker="o",lw=0.6)
    
ax3.set_ylabel(r"$\mathcal{I}$")
ax3.set_xlabel(r"$p$")
ax3.legend(fontsize=4,loc="lower left")

# Collapse inset




plt.tight_layout()
plt.show()