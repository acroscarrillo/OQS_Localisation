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
plt.rcParams['text.usetex'] = True
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



runs = 1000
L = 1000
    
# aux = []
# for i in tqdm_notebook(range(runs)):
#     xi = diagonal(A(L,0.4))[L//2-L//6:L//2+L//6]
#     aux.append(np.log(1/xi-1))
# beta_p_04_array = np.array(aux, dtype=np.float64).flatten()
# np.save('beta_p_04_array.npy', beta_p_04_array)

# aux = []
# for i in tqdm_notebook(range(runs)):
#     xi = diagonal(A(L,4))[L//2-L//6:L//2+L//6]
#     aux.append(np.log(1/xi-1))
# beta_p_04_array = np.array(aux, dtype=np.float64).flatten()
# np.save('beta_p_4_array.npy', beta_p_04_array)

#SELECT WHICH DATA TO PLOT#
beta_area_law_array = np.load('data/beta_p_4_array.npy')
beta_vol_law_array = np.load('data/beta_p_04_array.npy')

#############
# plot data #
#############
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 600
pt = 0.0138889 
fig,ax = plt.subplots(1,1,figsize = (246*pt,125*pt))
# fig,ax = plt.subplots(1,1,figsize = (246*pt,150*pt))

ax.hist(beta_area_law_array,label=r"$p=4$",color="C0",bins="auto",density=True,alpha=0.8)
ax.hist(beta_vol_law_array,label=r"$p=0.4$",color="C1",bins="auto",density=True,alpha=0.8)


# blue_patch = mpatches.Patch(color=(66/255, 135/255, 245/255), label='$p=0.4$')
# orange_patch = mpatches.Patch(color='orange', label='$p=4$')

# ax.legend(handles=[ blue_patch , orange_patch], ncol=1,prop={'size': 6}, loc = "upper left")

ax.legend(loc="upper left",fontsize=7,framealpha=0.5)
# ax.legend(loc="lower right",fontsize=8)


ax.set_ylabel(r"$\rho(\tilde{n})$",rotation=0)
ax.set_xlabel(r"$\tilde{n}$")
ax.set_yticks([0.0,5])
ax.set_xlim([0.0,10])
ax.yaxis.set_label_coords(-0.08,0.5)
ax.xaxis.set_label_coords(0.5,-0.08)

######################
# standard dev inset #
######################

L_array = np.arange(50,1000,50)

area_dev = np.zeros(len(L_array))
vol_dev = np.zeros(len(L_array))
counter = 0
for L in L_array:
    name_2_load = "data/temps/temp_area_4_L_"+str(L)
    area_array = np.load(name_2_load + ".npy")
    area_dev[counter] = np.std(area_array)#/(np.sum(area_array)/len(area_array))
    
    name_2_load = "data/temps/temp_vol_04_L_"+str(L)
    vol_array = np.load(name_2_load + ".npy")
    vol_dev[counter] = np.std(vol_array)#/(np.sum(vol_array)/len(vol_array))
    
    counter += 1


# These are in unitless percentages of the figure size. (0,0 is bottom left)
left, bottom, width, height = [0.49, 0.445, 0.41, 0.44]
ax_inset = fig.add_axes([left, bottom, width, height])

g_vol, y_intercept_vol = fit_log_log( L_array, vol_dev )
ax_inset.plot(L_array, vol_dev, label="Volume",lw=1,color="C1")
g_area, y_intercept_area = fit_log_log( L_array, area_dev )
ax_inset.plot(L_array, area_dev, label="Area",lw=1,color="C0")

print(g_vol, y_intercept_vol)
print(g_area, y_intercept_area)

ax_inset.set_ylabel(r"$\sigma[\tilde{n}]$",size=10,rotation=0)
ax_inset.set_xlabel(r"$L$",size=10)

ax_inset.set_facecolor('white')

ax_inset.xaxis.set_tick_params(labelsize=8)
ax_inset.yaxis.set_tick_params(labelsize=8)

# ax_inset.set_yticks([])
ax_inset.set_xticks([100,1000])

ax_inset.set_xticklabels(ax_inset.get_xticks())

ax_inset.set_yscale("log")
ax_inset.set_xscale("log")

# ax_inset.yaxis.set_minor_formatter(NullFormatter())
# ax_inset.set_xticks([100,1000])
ax_inset.yaxis.set_label_coords(-0.13,0.5)
ax_inset.xaxis.set_label_coords(0.5,-0.13)
ax_inset.set_xlim([47,1000])

plt.tight_layout()
plt.savefig("paper_figs/images/temp_fig.png")
plt.show()