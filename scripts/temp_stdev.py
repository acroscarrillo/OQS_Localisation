# import model.py methods (A, C, MI, ...) 
# which contains some main packages (numpy, numba,..)
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import *
from tqdm import tqdm


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
runs_array = np.arange(1000,50,-50)
p_array = np.arange(0.5,1.5,0.1)

area_dev = np.zeros((len(L_array),len(p_array)))
vol_dev = np.zeros((len(L_array),len(p_array)))

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

# name_2_load = "temp_area_4_L_"+str(L)
# area_array = np.load(name_2_load + ".npy")
# area_dev[counter] = np.std(area_array)#/(np.sum(area_array)/len(area_array))

# counter += 1

# %matplotlib notebook
# plt.rcParams['figure.dpi'] = 100
# plt.rcParams['savefig.dpi'] = 100

# plt.rc('font', size = 10)

# pt = 0.0138889 

# fig, ax = plt.subplots()
# ax.plot(L_array, area_dev, label="Area")
# ax.plot(L_array, vol_dev, label="Volume")

# ax.set_ylabel(r"$\frac{\sigma[\beta]}{\mathbb{E}[\beta]}$",size=16)
# ax.set_xlabel(r"$L$",size=16)
# plt.rcParams['axes.facecolor'] = 'white'
# ax.legend()
# # ax.set_yscale('log')
# plt.rcParams['axes.facecolor'] = 'white'
# fig.set_facecolor("white")
# ax.set_facecolor('white')
# plt.tight_layout()
# plt.show()