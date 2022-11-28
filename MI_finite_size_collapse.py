# import model.py methods (A, C, MI, ...) 
# which contains some main packages (numpy, numba,..)
from src import *
import pickle #to save python objects
from scipy.optimize import curve_fit, minimize #to do the fitting


# import plotting stuff
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"


##########################
# Data generation method #
##########################
@njit(nogil=True, parallel=False)
def O(params,data_in=np.load("zoomed_crit_reg_data.npy")):
    
    p_c, nu, zeta = params[0], params[1], params[2]
    
    # data_in structure MI|MI_ERR|L|p|L_A
    # data    structure y|x|d
    data  = np.zeros((len(data_in),3), dtype=np.float64)
    for i in range(len(data_in)):
        data[i,0] = data_in[i,0]*data_in[i,2]                       # y_i
        data[i,1] = ( data_in[i,3] - p_c )*(data_in[i,2]**(1/nu))   # x_i
        data[i,2] = data_in[i,1]*np.sqrt(data_in[i,2])              # d_i
    
    data_sorted = data[data[:, 1].argsort()] # sort in ascending x_i

    O_val = 0
    for i in range(1,len(data_sorted)-1): # as each loop requires n.n.
        y_m, y, y_p = data_sorted[i-1,0], data_sorted[i,0], data_sorted[i+1,0]
        x_m, x, x_p = data_sorted[i-1,1], data_sorted[i,1], data_sorted[i+1,1]
        d_m, d, d_p = data_sorted[i-1,2], data_sorted[i,2], data_sorted[i+1,2]
        
        ###################
        #CAREFUL WITH THIS#   this just handles the p=p_c situation
        ###################
        try:
            frac_p = (x_p-x)/(x_p-x_m)
        except:
            frac_p = 1/2
        try:
            frac_m = (x-x_m)/(x_p-x_m)
        except: 
            frac_p = 1/2
        
        if np.isnan(frac_p): 
            frac_p = 1/2
        
        if np.isnan(frac_m): 
            frac_m = 1/2
        
        ####################
        
        y_bar = y_m*frac_p + y_p*frac_m
        Delta_sqrd = d**2 + (d_m*frac_p)**2 + (d_p*frac_m)**2
                
        O_val += (y-y_bar)**2 #/Delta_sqrd
    
    return O_val 

##############################
# call fitter, generate data #
##############################
res = minimize(O, np.array([1,1,1]), method='Powell',options={'disp': True, "maxiter": 1000}, bounds=((0,None),(0,None),(0,None)),tol=1e-20)
p_c, nu, zeta = res.x

data_in=np.load("zoomed_crit_reg_data.npy")
data  = np.zeros((len(data_in),4), dtype=np.float64)
for i in range(len(data_in)):
    data[i,0] = (data_in[i,2]**(zeta/nu))*data_in[i,0]*data_in[i,2]   # y_i
    data[i,1] = ( data_in[i,3] - p_c )*(data_in[i,2]**(1/nu))         # x_i
    data[i,2] = data_in[i,1]                                          # d_i
    data[i,3] = data_in[i,2]                                          # L

data_sorted = data[data[:, 1].argsort()] # sort in ascending x_i

L_list = [400,800,1200,1600,2000]
data_2_plot = []
for L in L_list:
    aux_array = np.zeros((int(len(data_sorted)/len(L_list)),3))
    counter = 0
    for d in data_sorted:
        if d[3] == L:
            aux_array[counter,0] = d[0]
            aux_array[counter,1] = d[1]
            aux_array[counter,2] = d[2]
            counter += 1
    data_2_plot.append(aux_array)


#############
# plot data #
#############
pt = 0.0138889 
fig, ax  = plt.subplots(figsize = (246*pt,150*pt))

plt.rcParams['axes.facecolor'] = 'white'
fig.set_facecolor("white")
ax.set_facecolor('white')
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 200

for i in range(len(L_list)):
    ax.scatter(data_2_plot[i][:,1],data_2_plot[i][:,0],label=r"$L=$"+str(L_list[i]),s=2)

# ax.set_yscale('log')
ax.set_ylabel(r"$\mathcal{I}$")
ax.set_xlabel(r"$L^{1/\nu}(p-p_c)$")
plt.title(r"$p_c=$"+str(p_c)+", "+r"$\nu=$"+str(nu),size=10)
ax.legend(fontsize=8);
plt.tight_layout()

plt.show()