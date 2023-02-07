# this is an (Nx5) matrix where each row has the following structure
# row 1: IPR_avg_1 | IPR_err_1 | L_1 | p_1 | lamb_1
# row 2: IPR_avg_2 | IPR_err_2 | L_2 | p_2 | lamb_2
# and so on...
import numpy as np
import os

directory_name = os.path.dirname(__file__)
file_name = os.path.join(directory_name, "..", "data", "AVG_IPR_joined_data.npy")
ipr_data = np.load(file_name)
  
L = ipr_data[0,2] #value of L for the first row
IPR = ipr_data[0,0] #value of IPR for the first row

L_array = []
for data in ipr_data:
    y, y_err, L, p, lamb = data
    if L not in L_array:
        L_array.append(L)

print(L_array)
#and so on..
