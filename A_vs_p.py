# import model.py methods (A, C, MI, ...) 
# which contains some main packages (numpy, numba,..)
from src import *
from tqdm import tqdm
import pickle #to save python objects

# import plotting stuff
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

#####################
def A_antidiag_vs_p(L,p_array,runs):
    data_2_plot = []
    for p in tqdm(p_array):
        A_ant_diag_avg = np.zeros(L//2-1,dtype=np.float64)
        for i in range(runs):
            A_ant_diag_avg +=  np.absolute(np.fliplr(A(L,p)).diagonal()[L//2+1:L]) / runs   
        data_2_plot.append((np.arange(1,L//2,1), A_ant_diag_avg, p))
    return data_2_plot


#######################
# PARAMETER LANDSCAPE #
#######################
L = 1000
runs = 1000
p_array = np.arange(0,2.4,0.4)

# data_2_plot = A_antidiag_vs_p(L,p_array,runs)
# with open("data/A_vs_d.ob", 'wb') as fp:
#     pickle.dump(data_2_plot, fp)

with open ("data/A_vs_d.ob", 'rb') as fp:
    data_2_plot = pickle.load(fp)

#############
# plot data #
#############
pt = 0.0138889 
fig, ax  = plt.subplots(figsize = (246*pt,200*pt))

plt.rcParams['axes.facecolor'] = 'white'
fig.set_facecolor("white")
ax.set_facecolor('white')
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600

for data in data_2_plot:
    x, y, y_label = data
    g = (np.log(y[-1])-np.log(y[-L//4]))/(np.log(x[-1])-np.log(x[-L//4]))
    ax.plot(x,y,label="p="+str( np.round(y_label,2) )+", g log="+str(np.round(g,1)))

ax.set_ylabel(r'$\langle A_{L/2,L/2+d}\rangle$')
ax.set_xlabel(r'$d$')

ax.legend(fontsize=7)
ax.set_yscale("log")
ax.set_xscale("log")

plt.tight_layout()

plt.show()