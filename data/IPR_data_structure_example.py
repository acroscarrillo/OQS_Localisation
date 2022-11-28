# this is an (Nx5) matrix where each row has the following structure
# row 1: IPR_avg_1 | IPR_err_1 | L_1 | p_1 | lamb_1
# row 2: IPR_avg_2 | IPR_err_2 | L_2 | p_2 | lamb_2
# and so on...

ipr_data = np.load("AVG_IPR_joined_data.npy")   
L = ipr_data[0,2] #value of L for the first row
IPR = ipr_data[0,0] #value of IPR for the first row
#and so on..
