# import model.py methods (A, C, MI, ...) 
# which contains some main packages (numpy, numba,..)
import sys
import os
os.environ["PATH"] += os.pathsep + "C:/Users/Meli/AppData/Local/Programs/MiKTeX/miktex/bin/x64"

dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(dir,'..',"..")))
from src import *
import pickle #to save python objects

# import plotting stuff
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, NullLocator, ScalarFormatter, LogFormatter

plt.rcParams["font.family"] = "Times New Roman"
# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=10)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=10)    # fontsize of the tick labels
plt.rc('ytick', labelsize=10)    # fontsize of the tick labels
plt.rc('legend', fontsize=10)    # legend fontsize
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
with open ("data/collapse_data.ob", 'rb') as fp:
    collapse_data = pickle.load(fp)
#############
# plot data #
#############
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 600
plt.rcParams["text.usetex"] = True
pt = 0.0138889 
fig = plt.figure(figsize = (246*pt,250*pt))
gs = fig.add_gridspec(2,2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, :])
left, bottom, width, height = [0.5625, 0.268, 0.355, 0.2]
ax3_inset = fig.add_axes([left, bottom, width, height])

color_ls = ["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9"]

# MI vs L
for i, data in enumerate(MI_vs_L_n_b):
    L_array, MI_array, MI_err_array, p = data
    if np.round(p,2) !=4.0: 
        g, y_intercept = fit_log_log( L_array, MI_array )
        ax1.plot(L_array, np.exp(y_intercept)*(L_array**g),lw=0.6,c=color_ls[i],ls="dashed")
        ax1.errorbar(L_array,MI_array,yerr=MI_err_array,label=r'$\Delta$'+"="+str(np.round(g,2)),ms=0.8,marker="o", fmt=".",lw=0.6,color=color_ls[i])

ax1.legend(fontsize=4,loc="lower right",framealpha=0.5)
ax1.set_yscale("log")
ax1.set_xscale("log")
ax1.set_ylabel(r"$\mathcal{I}$",rotation=180)
ax1.set_yticks([10,0.001])
ax1.yaxis.set_label_coords(-0.2,0.6)
ax1.xaxis.set_label_coords(-0.2,-0.15)
plt.text(-2, 3.3, '(b)')
ax1.set_xlabel(r"$L$")

# MI vs L_A
for i, data in enumerate(MI_vs_LA_n_b):
    LA_array, MI_array, MI_err_array, p = data
    if np.round(p,2) !=4.0: 
        g, y_intercept = fit_log_log( LA_array[1:len(LA_array)//4+1],MI_array[1:len(LA_array)//4+1] )
        ax2.plot(LA_array[1:len(LA_array)//2+1], np.exp(y_intercept)*(LA_array[1:len(LA_array)//2+1]**g),lw=0.6,c=color_ls[i],ls="dashed",label=r"$p=$"+str(round(p,2))+", "+ r'$\Delta$'+"="+str(np.round(g,2)))
        ax2.errorbar(LA_array[1:len(LA_array)//2+1],MI_array[1:len(LA_array)//2+1],fmt=".",yerr=MI_err_array[1:len(LA_array)//2+1],c=color_ls[i],alpha=0.4,ms=0.8,marker="o",lw=1)

ax2.legend(fontsize=4,loc="lower left",framealpha=0.5)
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_ylabel(r"$\mathcal{I}$",rotation=180)
ax2.set_xlabel(r"$L_A$")
# ax2.xaxis.set_minor_formatter(LogFormatter(),minor_thresholds=None)
# minor_thresholds=(2, 0.5)
# ax2.yaxis.set_minor_formatter(LogFormatter(),minor_thresholds=None)

# ax2.xaxis.set_major_formatter(LogFormatter(labelOnlyBase=False))
# ax2.yaxis.set_major_formatter(LogFormatter(la3belOnlyBase=True))



# ax2.yaxis.set_minor_formatter(NullFormatter())
#ax2.xaxis.set_major_locator(NullLocator())
#ax2.xaxis.set_minor_locator(NullLocator())
ax2.yaxis.set_ticks([10**(-6),20],[r"$10^{-6}$", r"$2\times 10^1$"])
ax2.xaxis.set_ticks([10.0,100, 300], [r"$10$", "", r"$3\times 10^2$"])
ax2.yaxis.set_label_coords(-0.15,0.6)
ax2.xaxis.set_label_coords(0.5,-0.15)
plt.text(-0.8, 3.3, '(c)')
# MI vs p
for data in MI_vs_p_n_b:
    p_array, MI_array, MI_err_array, L = data
    ax3.errorbar(p_array,MI_array,yerr=MI_err_array,label=r"$L=$"+str(L),ms=0.6,marker="o",lw=0.6)
    
ax3.set_ylabel(r"$\mathcal{I}$",rotation=180)
ax3.set_xlabel(r"$p$")
ax3.set_yticks([0.2,0.0])
ax3.set_xticks([0.9,0.95,1.05,1.1])
ax3.yaxis.set_label_coords(-0.09,0.6)
ax3.xaxis.set_label_coords(0.5,-0.1)
ax3.legend(fontsize=4.5,loc="lower left")
plt.text(-2, 1.2, '(d)')

# Collapse inset
L_list = [400,800,1200,1600,2000]
for i in range(len(L_list)):
    ax3_inset.scatter(collapse_data[i][:,1],collapse_data[i][:,0],s=0.6,marker="o")#label=r"$L=$"+str(L_list[i]),s=3)

# ax.set_yscale('log')
ax3_inset.set_ylabel(r"$\mathcal{I}$",fontsize=8,rotation=180)
ax3_inset.set_xlabel(r"$L^{1/\nu}(p-p_c)$",fontsize=7)
ax3_inset.set_xticks([])
ax3_inset.set_yticks([])
ax3_inset.xaxis.set_label_coords(0.5, -0.05)
ax3_inset.yaxis.set_label_coords(-0.075,0.6)

# send it!
plt.tight_layout()
plt.show()
plt.savefig("paper_figs/images/MI_no_boundaries_fig.pdf")
