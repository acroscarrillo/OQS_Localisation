# import model.py methods (A, C, MI, ...) 
# which contains some main packages (numpy, numba,..)
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import *

# import plotting stuff
import pandas as pd
# import plotting stuff
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
plt.rcParams["font.family"] = "Times New Roman"
# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=7)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=7)    # fontsize of the tick labels
plt.rc('ytick', labelsize=7)    # fontsize of the tick labels
plt.rc('legend', fontsize=7)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
#####################
@njit(nogil=True, parallel=False)
def A_non_unit(L,b):
    aux_mat = np.zeros((L,L),dtype=np.complex128)
    for i in range(L): 
        for j in range(L):
            if j+i<=L-1:
                aux_mat[j,j+i]= ( np.random.normal(0,1) + 1j*np.random.normal(0,1) )/(i+1)**b#np.exp(i*b/L)#(i+1)**b#(i/L+1)**b
                aux_mat[j+i,j]= ( np.random.normal(0,1) + 1j*np.random.normal(0,1) )/(i+1)**b#np.exp(i*b/L)#(i+1)**b#(i/L+1)**b
#                 aux_mat[j,j+i]= np.random.normal(0,1) #ATENTION!!! MATRIX "A" IS RESTRICTED TO REAL ENTRIES, CHANGE THIS IF NEEDED!!! 
    return np.dot(aux_mat, aux_mat.conj().T)

#####################
#assumes V, lamb = 0, 0
@njit
def lamb_vs_L(L_min,L_max,runs, p_array):
    # data structure lamb, L, p
    data = np.zeros((runs*len(p_array),3),dtype=np.float64)
    counter = 0
    for p_ind, p in enumerate(p_array):
        print(p_ind/len(p_array))
        for k in range(runs):
            # print(k/runs)
            L = np.random.randint(L_min,L_max)
            A_mat = A_non_unit(L,p)
            lambs = LA.eigvalsh(A_mat)
            data[counter,0] = max(lambs)
            data[counter,1] = L
            data[counter,2] = p
            counter += 1
    return data


##############
# parameters #
##############
# L_array = np.arange(100,1000+100,100)
L_min, L_max = 100, 1000
runs = 100000
p_array = np.array([0.4,0.97,4])

# data = lamb_vs_L(L_min, L_max, runs, p_array) 
# with open('data/lamb_vs_L.npy', 'wb') as f:
#     np.save(f, data)

data = np.load("data/lamb_vs_L.npy")

df = pd.DataFrame(data, columns = ['lamb', 'L', 'p'])
ps = np.sort(pd.unique(df['p']))
Ls = np.array(np.sort(pd.unique(df['L'])), dtype = int)

###############
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
pt = 0.0138889 
fig, ax  = plt.subplots(1,1,figsize = (246*pt,150*pt))

df[df["p"]==0.4].plot(kind='scatter', x='L', y='lamb', color='r',ax=ax,s=0.1,marker="x",alpha=0.2)    
df[df["p"]==0.97].plot(kind='scatter', x='L', y='lamb', color='g',ax=ax,s=0.1,marker="x",alpha=0.2)
df[df["p"]==4].plot(kind='scatter', x='L', y='lamb', color='b',ax=ax,s=0.1,marker="x",alpha=0.2)

################
plotting_mean_df = df.groupby(['L',"p"], as_index=False)['lamb'].mean()

plotting_mean_df[plotting_mean_df["p"]==0.4].plot(kind='line', x='L', y='lamb', color='yellow',label=r"$p=0.4$",lw=0.5,ax=ax)    
plotting_mean_df[plotting_mean_df["p"]==0.97].plot(kind='line', x='L', y='lamb', color='magenta', label=r"$p=1.13$",lw=0.5,ax=ax)  
plotting_mean_df[plotting_mean_df["p"]==4].plot(kind='line', x='L', y='lamb', color='white', label=r"$p=4$",lw=0.5,ax=ax)  

################
# plotting_std_df = df.groupby(['L',"p"], as_index=False)['lamb'].std()

ax.set_xscale("log")
ax.set_ylabel(r"$\lambda$")
ax.set_xlabel(r"$L$")

legend_elements = [Patch(facecolor='yellow', edgecolor='red', label=r"$p=0.4$"),
                  Patch(facecolor='magenta', edgecolor='green',label=r"$p=1.13$"),
                   Patch(facecolor='white', edgecolor='blue',label=r"$p=4$")]

ax.legend(handles=legend_elements, loc='upper left')

# ax.fill_between

fig.tight_layout()

plt.show()