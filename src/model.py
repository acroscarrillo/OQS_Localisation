import os

from numba import jit, njit, types, vectorize, prange
import numpy as np
from numpy import linalg as LA
from numpy.linalg import inv

@njit(nogil=True, parallel=False)
def delta(j,k):
    if j==k:
        return 1
    else:
        return 0

@njit(nogil=True, parallel=False)
def stnd_err(data):
    avg = np.sum(data)/len(data)
    tot = 0
    for i in data:
        tot += (i - avg)**2
    return np.sqrt(tot)/len(data)

@njit(nogil=True, parallel=False)   
def A_diag(L, seed = None):
    if seed is not None:
            np.random.seed(seed)

    aux_mat = np.zeros((L,L),dtype=np.complex128)
    for i in range(L):
        aux_mat[i,i]= ( np.random.normal(0,1) + 1j*np.random.normal(0,1) )

    A = np.dot(aux_mat, aux_mat.conj().T)

    lamb = LA.eigvalsh(A)

    return A/(2*max(lamb)) #divide by 2*lamb to keep eigenvals of A and B=1-A between 0 and 1/2. 
                               #This is to avoid problems in C_ij. This also keeps temperature range in particle/hole sector.
    
@njit(nogil=True, parallel=False)
def A(L,b, seed = None):
    if b == "diag":
        if seed is not None:
            np.random.seed(seed)

        aux_mat = np.zeros((L,L),dtype=np.complex128)
        for i in range(L):
            aux_mat[i,i]= ( np.random.normal(0,1) + 1j*np.random.normal(0,1) )

        A = np.dot(aux_mat, aux_mat.conj().T)

        lamb = LA.eigvalsh(A)

        return A/(2*max(lamb)) #divide by 2*lamb to keep eigenvals of A and B=1-A between 0 and 1/2. 
                                   #This is to avoid problems in C_ij. This also keeps temperature range in particle/hole sector.
    
    elif b == "$\infty$":
        if seed is not None:
            np.random.seed(seed)

        aux_mat = np.zeros((L,L),dtype=np.complex128)
        for i in range(L):
            aux_mat[i,i]= ( np.random.normal(0,1) + 1j*np.random.normal(0,1) )
            if i+1<L:
                aux_mat[i,i+1]= ( np.random.normal(0,1) + 1j*np.random.normal(0,1) )
                aux_mat[i+1,i]= ( np.random.normal(0,1) + 1j*np.random.normal(0,1) )
            
        A = np.dot(aux_mat, aux_mat.conj().T)

        lamb = LA.eigvalsh(A)

        return A/(2*max(lamb)) #divide by 2*lamb to keep eigenvals of A and B=1-A between 0 and 1/2. 
                                   #This is to avoid problems in C_ij. This also keeps temperature range in particle/hole sector.

    else:
        if seed is not None:
            np.random.seed(seed)

        aux_mat = np.zeros((L,L),dtype=np.complex128)
        for i in range(L): 
            for j in range(L):
                if j+i<=L-1:
                    aux_mat[j,j+i]= ( np.random.normal(0,1) + 1j*np.random.normal(0,1) )/(i+1)**b#np.exp(i*b/L)#(i+1)**b#(i/L+1)**b
                    aux_mat[j+i,j]= ( np.random.normal(0,1) + 1j*np.random.normal(0,1) )/(i+1)**b#np.exp(i*b/L)#(i+1)**b#(i/L+1)**b
    #                 aux_mat[j,j+i]= np.random.normal(0,1) #ATENTION!!! MATRIX "A" IS RESTRICTED TO REAL ENTRIES, CHANGE THIS IF NEEDED!!! 
        A = np.dot(aux_mat, aux_mat.conj().T)

        lamb = LA.eigvalsh(A)

        return A/(2*max(lamb)) #divide by 2*lamb to keep eigenvals of A and B=1-A between 0 and 1/2. 
                                   #This is to avoid problems in C_ij. This also keeps temperature range in particle/hole sector.
@njit(nogil=True, parallel=False)
def H_hop(L,V,lamb):
    return V*np.eye(L,dtype=np.complex128)+np.diag(lamb*np.ones(L-1),1)+np.diag(lamb*np.ones(L-1),-1) #the L-1 is because it will fit the L-1 in the off diagonal, thus the diagonal will be L-1+1=L

@njit(nogil=True, parallel=False)
def H_hop_eig(L,V,lamb):
    """Returns E,V. The column V[:, i] is the normalized eigenvector corresponding to the eigenvalue E[i] of H_hop(L,V,lamb)."""
    L_array = np.arange(1,L+1,1)
    E = V + 2*lamb*np.cos(L_array*np.pi/(L+1))    
    V_array = np.sqrt(2/(L+1))*np.sin(np.expand_dims(L_array, 1)*np.expand_dims(L_array,0)*np.pi/(L+1))
    return E,V_array.astype(np.complex128)

def H_hop_nn(L,V,lamb):
    H = np.zeros((L,L),dtype=np.complex128)
    for j in range(L):
        for k in range(L):
            H[j,k] = V*delta(j,k)+lamb*(delta(j,k+2)+delta(j+2,k))
    return H

@njit(nogil=True, parallel=False)
# def C(H_mat,A_mat,C_0,t):
def C(V,lamb,A_mat,C_0,t):
    if lamb != 0:
        L = len(A_mat)
    #     if len(H_mat) != len(A_mat) or len(H_mat)!=len(C_0) or len(C_0)!=len(A_mat):
    #         raise ValueError("Matrices H, A and C_0 must have the same dimensions.")

        # THIS IS COMMENTED OUT CAUSE WE USE A HOPPING HAMILTONIAN
    #     E, V_mat = LA.eigh(H_mat) #v[:, i] ith eigenvect

        E, V_mat = H_hop_eig(L,V,lamb) # SO DELETE THIS FOR GENERIC H!!!!!!!!!!

        #Project A to H's basis
        A_tilde = np.dot(np.conjugate(np.transpose(V_mat)), np.dot(np.transpose(A_mat), V_mat))

        #Project C_0 to H's basis
        C_0_tilde = np.dot(np.conjugate(np.transpose(V_mat)), np.dot(C_0, V_mat))

        #Get C in H's basis
    #     exponent = np.exp(1j*np.subtract.outer(E, E)*t-t)
        exponent = np.exp(1j*(np.expand_dims(E, 1) - np.expand_dims(E,0))*t-t)

    #     quotient = np.subtract.outer(E, E)+1j
        quotient = np.expand_dims(E, 1) - np.expand_dims(E,0)+1j
        C_h_basis = (C_0_tilde-1j*A_tilde/quotient)*exponent  +  1j*A_tilde/quotient

        #Project C back 
        C_tot = np.dot(V_mat, np.dot(C_h_basis, np.conjugate(np.transpose(V_mat))))

        return C_tot
    if lamb==0:
        return A_mat*(1-np.exp(-t)) + C_0*np.exp(-t)

@njit(nogil=True, parallel=False)
# def C_NESS(H_mat,A_mat):
def C_NESS(V,lamb,A_mat):
    if lamb != 0:
        L = len(A_mat)
    #     if len(H_mat) != len(A_mat):
    #         raise ValueError("Matrices H, A and C_0 must have the same dimensions.")

        # THIS IS COMMENTED OUT CAUSE WE USE A HOPPING HAMILTONIAN
    #     E, V_mat = LA.eigh(H_mat) #v[:, i] ith eigenvect

        E, V_mat = H_hop_eig(L,V,lamb) # SO DELETE THIS FOR GENERIC H!!!!!!!!!!

        #Project A to H's basis
        A_tilde = np.dot(np.conjugate(np.transpose(V_mat)), np.dot(np.transpose(A_mat), V_mat))

        #Get C in H's basis
    #     quotient = np.subtract.outer(E, E,dtype=np.complex128)+1j  #unsupported by numba
        quotient = np.expand_dims(E, 1) - np.expand_dims(E,0)+1j
        C_h_basis = 1j*(A_tilde/quotient)

        #Project C back 
        C_tot = np.dot(V_mat, np.dot(C_h_basis, np.conjugate(np.transpose(V_mat))))

        return C_tot
    if lamb==0:
        return A_mat

@njit(nogil=True, parallel=False)
# def C_sub(subsys,H_mat,A_mat,C_0,t):
def C_sub(subsys,V,lamb,A_mat,C_0,t):
#     C_tot = C(H_mat,A_mat,C_0,t)
    C_tot = C(V,lamb,A_mat,C_0,t)
    L_sub=len(subsys)
    C_sub = np.zeros((L_sub,L_sub),dtype=np.complex128)
    for j in range(len(subsys)):
        for k in range(len(subsys)):
            C_sub[j,k] = C_tot[subsys[j],subsys[k]]
    return C_sub

@njit(nogil=True, parallel=False)
# def C_sub_NESS(subsys,H_mat,A_mat):
def C_sub_NESS(subsys,V,lamb,A_mat):
#     C_tot = C_NESS(H_mat,A_mat)
    C_tot = C_NESS(V,lamb,A_mat)
    L_sub=len(subsys)
    C_sub = np.zeros((L_sub,L_sub),dtype=np.complex128)
    for j in range(len(subsys)):
        for k in range(len(subsys)):
            C_sub[j,k] = C_tot[subsys[j],subsys[k]]
    return C_sub

@njit(nogil=True, parallel=False)
# def entropy(H_mat,A_mat,subsys,C_0,t):                           #THIS USES LOG_2 !! AND NOT LOG_E!!!
def entropy(V,lamb,A_mat,subsys,C_0,t):
    if len(subsys) == 0:
        raise ValueError("Cannot calculate the entropy of an empty subsystem! Subsystem has to have at least lenght 1.")
    eigenvals = LA.eigvalsh(C_sub(subsys,V,lamb,A_mat,C_0,t))  
#     eigenvals = LA.eigvalsh(C_sub(subsys,H_mat,A_mat,C_0,t))
    entropy = 0
    for i in range(len(eigenvals)):
        if 0<eigenvals[i]<1:
            entropy += -(1-eigenvals[i])*np.log2(1-eigenvals[i])-eigenvals[i]*np.log2(eigenvals[i])
    return entropy 

@njit(nogil=True, parallel=False)
# def entropy_NESS(H_mat,A_mat,subsys):                           #THIS USES LOG_2 !! AND NOT LOG_E!!!
def entropy_NESS(V,lamb,A_mat,subsys):
    if len(subsys) == 0:
        raise ValueError("Cannot calculate the entropy of an empty subsystem! Subsystem has to have at least lenght 1.")
    eigenvals = LA.eigvalsh(C_sub_NESS(subsys,V,lamb,A_mat))
    entropy = 0
    for i in prange(len(eigenvals)):
        if 0<eigenvals[i]<1:
            entropy += -(1-eigenvals[i])*np.log2(1-eigenvals[i])-eigenvals[i]*np.log2(eigenvals[i])
    return entropy 

@njit(nogil=True, parallel=False)
# def MI_by_L(H_mat,A_mat,subsys_A,C_0,t,subsys_B=np.array([12345])):              #THIS OBVIOUSLY ALSO USES LOG_2 !! AND NOT LOG_E!!!
def MI_by_L(V,lamb,A_mat,subsys_A,C_0,t,subsys_B=np.array([12345])):
    L = len(A_mat)
    if subsys_B[0] == np.array([12345])[0]:
        subsys_B = np.zeros(L-len(subsys_A),dtype=np.int64)
        counter = 0
        for i in range(L):
            aux = 0
            for j in range(len(subsys_A)):
                if i != subsys_A[j]:
                    aux += 1
            if aux == len(subsys_A):
                subsys_B[counter] = i
                counter += 1
        return ( entropy(V,lamb,A_mat,subsys_A,C_0,t) + entropy(V,lamb,A_mat,subsys_B,C_0,t) - entropy(V,lamb,A_mat,np.array([i for i in range(L)]),C_0,t) )/L
    
    else:
        tot_sub_sys = np.append(subsys_A,subsys_B)
        return ( entropy(V,lamb,A_mat,subsys_A,C_0,t) + entropy(V,lamb,A_mat,subsys_B,C_0,t) - entropy(V,lamb,A_mat,tot_sub_sys,C_0,t) )/L

@njit(nogil=True, parallel=False)
# def MI_by_L_NESS(H_mat,A_mat,subsys_A,subsys_B=np.array([12345])):              #THIS OBVIOUSLY ALSO USES LOG_2 !! AND NOT LOG_E!!!
def MI_by_L_NESS(V,lamb,A_mat,subsys_A,subsys_B=np.array([12345])):  
    L = len(A_mat)
    if subsys_B[0] == np.array([12345])[0]:
        subsys_B = np.zeros(L-len(subsys_A),dtype=np.int64)
        counter = 0
        for i in range(L):
            aux = 0
            for j in range(len(subsys_A)):
                if i != subsys_A[j]:
                    aux += 1
            if aux == len(subsys_A):
                subsys_B[counter] = i
                counter += 1
        return ( entropy_NESS(V,lamb,A_mat,subsys_A) + entropy_NESS(V,lamb,A_mat,subsys_B) - entropy_NESS(V,lamb,A_mat,np.array([i for i in range(L)])) )/L
    
    else:
        tot_sub_sys = np.append(subsys_A,subsys_B)
        return ( entropy_NESS(V,lamb,A_mat,subsys_A) + entropy_NESS(V,lamb,A_mat,subsys_B) - entropy_NESS(V,lamb,A_mat,tot_sub_sys) )/L
    
#     L = len(A_mat)
#     subsys_B = np.zeros(L-len(subsys_A),dtype=np.int32)
#     counter = 0
#     for i in range(L):
#         aux = 0
#         for j in range(len(subsys_A)):
#             if i != subsys_A[j]:
#                 aux += 1
#         if aux == len(subsys_A):
#             subsys_B[counter] = i
#             counter += 1

#     return (entropy_NESS(A_mat,subsys_A) + entropy_NESS(A_mat,subsys_B) - entropy_NESS(A_mat,np.array([i for i in range(L)])))/L

@njit(nogil=True, parallel=False)
def MI_plus_tot_entropy_NESS(H_mat,A_mat,subsys_A,subsys_B=np.array([12345])):      #YOU NEED TO SUBSTRACT THE TOTAL ENTROPY TO GET M.I.!!!! (this is to avoid calculating the total entropy too many times)
    L = len(A_mat)
    if subsys_B[0] == np.array([12345])[0]:
        subsys_B = np.zeros(L-len(subsys_A),dtype=np.int64)
        counter = 0
        for i in range(L):
            aux = 0
            for j in range(len(subsys_A)):
                if i != subsys_A[j]:
                    aux += 1
            if aux == len(subsys_A):
                subsys_B[counter] = i
                counter += 1
        return  entropy_NESS(H_mat,A_mat,subsys_A) + entropy_NESS(H_mat,A_mat,subsys_B)
    
    else:
        tot_sub_sys = np.append(subsys_A,subsys_B)
        return  entropy_NESS(H_mat,A_mat,subsys_A) + entropy_NESS(H_mat,A_mat,subsys_B) 



@njit(nogil=True, parallel=False)
def O(params, data_in=None):
    """Calculates the value of the finite size scalling collapse cost function. 

    Calculates the cost function for a given critical point, exponent and zeta and fitting data.

    Args:
        params: Tuple or array containing the critical point, critical exponent and zeta, in that order. 
        data_in:  2D array of floats of structure Nx5 where each row has MI|MI_ERR|L|p|L_A.

    Returns:
        A single float for the cost.

    Raises:
        N/A.
    """
    p_c, nu, zeta = params[0], params[1], params[2]
    
    if data_in == None:
        directory_name = os.path.dirname(__file__)
        file_name = os.path.join(directory_name, "..", "data", "zoomed_crit_reg_data.npy")
        data_in = np.load(file_name)

    # data_in structure MI|MI_ERR|L|p|L_A
    # data    structure y|x|d
    data  = np.zeros((len(data_in),3), dtype=np.float64)
    for i in range(len(data_in)):
        data[i,0] = (data_in[i,2]**(zeta/nu))*data_in[i,0]          # y_i
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
                
        O_val += (y-y_bar)**2#/Delta_sqrd
    
    return O_val 


@njit(nogil=True, parallel=False)
def O_alt(params,data_in=np.load("data/zoomed_crit_reg_data.npy")):
    """Calculates the value of the finite size scalling collapse cost function with alternative ansatz. 

    Calculates the cost function for a given critical point, exponent and zeta and fitting data.

    Args:
        params: Tuple or array containing the critical point, critical exponent and zeta, in that order. 
        data_in:  2D array of floats of structure Nx5 where each row has MI|MI_ERR|L|p|L_A.

    Returns:
        A single float for the cost.

    Raises:
        N/A.
    """
    p_c, nu, C = params[0], params[1], params[2]
    
    # data_in structure MI|MI_ERR|L|p|L_A
    # data    structure y|x|d
    data  = np.zeros((len(data_in),3), dtype=np.float64)
    for i in range(len(data_in)):
        data[i,0] = (data_in[i,0]-C)*(data_in[i,2]**(1/nu))/np.log(data_in[i,2])         # y_i
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
                
        O_val += (y-y_bar)**2#/Delta_sqrd
    
    return O_val 