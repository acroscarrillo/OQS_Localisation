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
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, NullLocator, ScalarFormatter, LogFormatter

plt.rcParams["font.family"] = "Times New Roman"
# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=10)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=10)    # fontsize of the tick labels
plt.rc('ytick', labelsize=10)    # fontsize of the tick labels
plt.rc('legend', fontsize=10)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams["text.usetex"] = True

#############
# plot data #
#############
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 600
pt = 0.0138889 
fig = plt.figure(figsize = (246*pt,250*pt))
gs = fig.add_gridspec(2,2)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[1, :])


color_list = ["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9"]
ls_list = ['solid', 'dashed', 'dotted']

fig.set_facecolor("white")

#################
# Dynamics plot #
#################

t_i = 0 #kind of has to be zero.
t_f = 10
dt = 0.1
t_array = np.arange(t_i,t_f,dt)

runs = 100
L = 100
subsys_A = np.array([i for i in range(L//2)])

# # plot lamb 5 #

data = [ np.load("data/MI_vs_t_p_06_lamb_5_L_100.npy"),
         np.load("data/MI_vs_t_p_113_lamb_5_L_100.npy"),
          np.load("data/MI_vs_t_p_16_lamb_5_L_100.npy") ]
data_ness = [ np.load("data/MI_NESS_p_06_lamb_5_L_100.npy"), 
            np.load("data/MI_NESS_p_113_lamb_5_L_100.npy"), 
            np.load("data/MI_NESS_p_16_lamb_5_L_100.npy") ]


p_str=["0.6","1.13","1.6"]
for i,d in enumerate(data):
    avg = np.average(d,axis=1)
    err = stats.sem(d,axis=1)
    ax0.errorbar(t_array,avg,yerr=err,label=r"$p=$"+p_str[i],ms=1,lw=1,color= color_list[i])
    ax0.axhline(y=data_ness[i],lw=1, color= color_list[i],linestyle="--")

ax0.set_ylabel(r"$\mathcal{I}$",rotation=180)
ax0.set_xlabel(r"$t$")
ax0.set_yticks([0.0,0.3])
ax0.yaxis.set_label_coords(-0.2,0.6)
ax0.xaxis.set_label_coords(0.5,-0.13)
ax0.legend(loc="center right",fontsize=8,framealpha=0.5)
plt.text(-1.1, 100000, '(a)')

################################
# Bump height against L instet #
################################

t_i = 0 #kind of has to be zero.
t_f = 4 #seems reasonable that bump lies before that
dt = 0.1
t_array = np.arange(t_i,t_f,dt)

runs = 500
L_array = np.arange(100,1000+100,100)
   
# plot bump #
data = [ np.load("data/bump_vs_L_p_06_lamb_5.npy"), 
        np.load("data/bump_vs_L_p_113_lamb_5.npy"), 
        np.load("data/bump_vs_L_p_16_lamb_5.npy") ]
data_ness = [np.load("data/NESS_p_06_lamb_5_at_Ls.npy"), 
            np.load("data/NESS_p_113_lamb_5_at_Ls.npy"),
             np.load("data/NESS_p_16_lamb_5_at_Ls.npy")]



p_str=["0.6","1.13","1.6"]
for i,d in enumerate(data):
    avg = np.average(d,axis=1)/data_ness[i]
    err = stats.sem(d,axis=1)/data_ness[i]
    ax1.errorbar(L_array,avg,yerr=err,label=r"$p=$"+p_str[i],ms=1.2,marker="o",lw=0.8,color= color_list[i],ls="dashed")

ax1.set_ylabel(r"$\frac{\Delta B}{\mathcal{I}_{NESS}}$",rotation=0)
ax1.set_xlabel(r"$L$")

# ax1.legend()

ax1.set_facecolor('white')
ax1.set_yscale("log")
ax1.set_xscale("log")
ax1.set_xticks([100,1000])
ax1.yaxis.set_minor_formatter(NullFormatter())
ax1.set_yticks([1,0.01])
ax1.yaxis.set_label_coords(-0.2,0.4)
ax1.xaxis.set_label_coords(0.5,-0.13)
plt.text(2.0, 100000, '(b)')

# ax1.set_yticks([])
    
###################
# MI against lamb #
###################
lamb_array = np.arange(0,5,0.2)
L_array = np.array([400,800,1200])
# L_array = np.array([10,20,30])
run_array = np.array([2000,1000,500])

with open ("data/MI_vs_lamb_at_Ls_at_p_06.ob", 'rb') as fp:
    p_06_data, p_06_err = pickle.load(fp)
    
for i,L in enumerate(L_array):
    ax2.errorbar(lamb_array,p_06_data[:,i],yerr=p_06_err[:,i],label="L="+str(L),ms=2,lw=0.8,color= "C0",ls=ls_list[i])

with open ("data/MI_vs_lamb_at_Ls_at_p_113.ob", 'rb') as fp:
    p_113_data, p_113_err = pickle.load(fp)
    
for i,L in enumerate(L_array):
    ax2.errorbar(lamb_array,p_113_data[:,i],yerr=p_113_err[:,i],label="L="+str(L),ms=2,lw=0.8,color= "C1",ls=ls_list[i])

with open ("data/MI_vs_lamb_at_Ls_at_p_16.ob", 'rb') as fp:
    p_16_data, p_16_err = pickle.load(fp)
    
for i,L in enumerate(L_array):
    ax2.errorbar(lamb_array,p_16_data[:,i],yerr=p_16_err[:,i],label="L="+str(L),ms=2,lw=0.8,color= "C2",ls=ls_list[i])

    
lines = [Line2D([0], [0], color="black", linewidth=1, linestyle='solid'),Line2D([0], [0], color="black", linewidth=1, linestyle='dashed'),
        Line2D([0], [0], color="black", linewidth=1, linestyle='dotted')]
labels = [r"$L=$"+str(L_array[0]), r"$L=$"+str(L_array[1]), r"$L=$"+str(L_array[2])]
ax2.legend(lines, labels,loc="upper right",fontsize=8,framealpha=0.5)

ax2.set_ylabel(r"$\mathcal{I}_{NESS}$",rotation=0)
ax2.set_xlabel(r"$\lambda$")
ax2.set_yscale("log")
ax2.set_xticks([0,5])
ax2.yaxis.set_minor_formatter(NullFormatter())
ax2.set_yticks([10,0.1])
ax2.yaxis.set_label_coords(-0.1,0.5)
ax2.xaxis.set_label_coords(0.5,-0.13)
plt.text(-1.1, 38, '(c)')



plt.tight_layout(w_pad = 1, h_pad=-11)
plt.subplot_tool()


plt.show()