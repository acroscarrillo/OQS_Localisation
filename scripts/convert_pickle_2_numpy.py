import numpy as np
import pickle #to save python objects


with open ("data/MI_vs_p_no_boundaries.ob", 'rb') as fp:
    data = pickle.load(fp)

data2save = np.array([0,0,0,0,0])
for d in data:
    p_array, MI_array, MI_err_array, L = d 
    aux = np.zeros((len(p_array),5)) # data_in structure MI|MI_ERR|L|p|L_A
    for i in range(len(p_array)):
        aux[i,0] = MI_array[i]
        aux[i,1] = MI_err_array[i]
        aux[i,2] = L
        aux[i,3] = p_array[i]
        aux[i,4] = L//2
    data2save = np.vstack((data2save,aux))

data2save = np.delete(data2save,(0),axis=0)

# with open("data/zoomed_crit_reg_data.ob", 'wb') as fp:
#     pickle.dump(data2save, fp)

with open('data/MI_vs_p_no_boundaries.npy', 'wb') as f:
    np.save(f, data2save)