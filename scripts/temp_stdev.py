# import model.py methods (A, C, MI, ...) 
# which contains some main packages (numpy, numba,..)
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import *
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
plt.rcParams["font.family"] = "Times New Roman"
# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=10)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=10)    # fontsize of the tick labels
plt.rc('ytick', labelsize=10)    # fontsize of the tick labels
plt.rc('legend', fontsize=10)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rc('font', size = 10)


####################
# useful functions #
####################
@njit
def temps(L,p,runs):
    temp_batch = np.zeros(runs,dtype=np.float64)
    for i in range(runs):
            xi = np.real(A(L,p)[L//2,L//2])
            temp_batch[i] = np.log(1/xi-1)
    return temp_batch


###################
# data generation #
###################
L_array = np.arange(50,1000,50)
runs_array = np.arange(2000,100,-100)
p_array = np.arange(0.5,1.5,0.1)

# generate batch of temperatures at each L and p
for p_ind, p in enumerate(p_array):
    print("p%="+str(100*p_ind/len(p_array)))
    for L_ind, L in enumerate(L_array):
        print("L%="+str(100*L_ind/len(L_array)))
        temp_batch = temps(L,p,runs_array[L_ind])

        name_2_save = "data/temps/temp_p_"+str(round(p,2))+"_L_"+str(L) + ".npy"
        np.save(name_2_save, temp_batch)

# from the generated batch of temperatures 
# get the standard dev at each L and p
temp_dev = np.zeros((len(L_array),len(p_array)))

for p_ind, p in enumerate(p_array):
    for L_ind, L in enumerate(L_array):
        name_2_load = "data/temps/temp_p_"+str(round(p,2))+"_L_"+str(L) + ".npy"
        temp_array = np.load(name_2_load)
        temp_dev[L_ind, p_ind] = np.std(temp_array)#/(np.sum(area_array)/len(area_array))
        

###################
# data generation #
###################
pt = 0.0138889 
fig, ax = plt.subplots(figsize = (246*pt,250*pt))
color_ls = ["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9"]

for p_ind, p in enumerate(p_array):
    std_dev = temp_dev[:, p_ind]
    g, y_intercept = fit_log_log( L_array, std_dev )
    ax.plot(L_array, std_dev, label=r"$p=$"+str(round(p,2))+r" $\Delta=$"+str(round(g,3)),c=color_ls[p_ind])
    ax.plot(L_array, np.exp(y_intercept)*(L_array**g),lw=0.6,c=color_ls[p_ind],ls="dashed")   


ax.set_ylabel(r"$\sigma[n]$",size=8)
ax.set_xlabel(r"$L$",size=8)
ax.legend(fontsize=6)
ax.set_yscale('log')
ax.set_xscale('log')

plt.tight_layout()
plt.show()