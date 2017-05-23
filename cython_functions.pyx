#!python
# cython: profile=False, language_level=3, boundscheck=False, wraparound=False
# cython --compile-args=-fopenmp --link-args=-fopenmp --force -a
## for compilation run: python setup.py build_ext --inplace

# %%cython --compile-args=-fopenmp --link-args=-fopenmp
cimport cython
from cython.parallel import prange, parallel
from libc.math cimport exp
from libc.stdlib cimport rand, RAND_MAX
cimport numpy as np
import numpy as np
data_type = np.int8
ctypedef np.int8_t data_type_t


# @cython.wraparound(False)
# @cython.boundscheck(False)
def draw_unified_noparents(data_type_t[:,:] x,  # N x D
                         data_type_t[:,:] sibling, # D x Lc
                         data_type_t[:,:] child, # N x Lc
                         float lbda,
                         float prior,
                         data_type_t[:,:] sampling_indicator): # N x D
    
    cdef float p, acc_child
    cdef int n, d, N = x.shape[0], D=x.shape[1]
    for n in prange(N, schedule=dynamic, nogil=True): # parallelise
        for d in range(D):
            if sampling_indicator[n,d] is True:  
                acc_child = lbda*score_no_parents_unified(child[n,:], x[n,:], sibling, d)
#                 print(lbda, score_no_parents_unified(child[n,:], x[n,:], sibling, d))
                p = sigmoid(acc_child + prior)
                x[n, d] = swap_metropolised_gibbs_unified(p, x[n,d])       

def draw_unified_nochild(data_type_t[:,:] x,  # N x D
                         data_type_t[:,:,:] z_pa, # K x N x Lp 
                         data_type_t[:,:,:] u_pa, # K x D x Lp
                         double[:] lbda_pa, # K
                         float prior,
                         data_type_t[:,:] sampling_indicator): # N x D

    cdef float p, acc_par
    cdef int n, d, K = len(lbda_pa), N=x.shape[0], D=x.shape[1]
    for n in range(N): # parallelise
        for d in range(D):
            if sampling_indicator[n,d] is True:
                # accumulate over all parents
                acc_par = 0
                for k in range(K):
                    acc_par += lbda_pa[k]*compute_g_alt_tilde_unified(u_pa[k,d,:], z_pa[k,n,:]) 
                p = sigmoid(acc_par + prior)
                x[n, d] = swap_metropolised_gibbs_unified(p, x[n,d]) 

def draw_unified(data_type_t[:,:] x,  # N x D
                 data_type_t[:,:,:] z_pa, # K x N x Lp 
                 data_type_t[:,:,:] u_pa, # K x D x Lp
                 double[:] lbda_pa, # K
                 data_type_t[:,:] sibling, # D x Lc
                 data_type_t[:,:] child, # N x Lc
                 float lbda,
                 float prior,
                 data_type_t[:,:] sampling_indicator): # N x D
    
    cdef float p, acc_par, acc_child
    cdef int n, d, K = len(lbda_pa), N = x.shape[0], D=x.shape[1]
    for n in range(N): # parallelise
        for d in range(D):
            if sampling_indicator[n,d] is True:
                # accumulate over all parents
                acc_par = 0
                for k in range(K):
                    acc_par += lbda_pa[k]*compute_g_alt_tilde_unified(u_pa[k,d,:], z_pa[k,n,:])
                    
                acc_child = lbda*score_no_parents_unified(child[n,:], x[n,:], sibling, d)  
                
                p = sigmoid(acc_par + acc_child + prior)
                x[n, d] = swap_metropolised_gibbs_unified(p, x[n,d])            
                

# @cython.wraparound(False)
# @cython.boundscheck(False)
cpdef inline int compute_g_alt_tilde_unified(data_type_t[:] u,
                                            data_type_t[:] z) nogil:
    """
    for two vectors of same length, u and z, comput
    2*np.min(1, u^T z)-1
    """
    cdef int i
    for i in xrange(u.shape[0]):
        if u[i] == 1 and z[i] == 1:
            return 1
    return -1
    
# @cython.boundscheck(False)
# @cython.wraparound(False)
cpdef inline float score_no_parents_unified(
    data_type_t[:] x, # (N x) D
    data_type_t[:] z, # (N x) L 
    data_type_t[:,:] u, # D x L
    int l) nogil: # feature index
    """
    This is essentially algorithm 1 from the ICML paper
    """
    
    cdef int L = u.shape[1]
    cdef int D = u.shape[0]
    cdef bint alrdy_active
    cdef int score = 0
    
    for d in range(D):
        if u[d, l] != 1:
            continue

        alrdy_active = False
        for l_prime in range(L):
            if (z[l_prime] == 1 and
                u[d, l_prime] == 1 and
                l_prime != l):
                alrdy_active = True
                break

        if (alrdy_active is False):
            score += x[d]

    return score


# @cython.boundscheck(False)
# @cython.wraparound(False)
@cython.cdivision(True)
cdef inline int swap_metropolised_gibbs_unified(float p, data_type_t x) nogil:
    cdef float alpha
    if x == 1:
        if p <= .5:
            alpha = 1
        else:
            alpha = (1-p)/p
    else:
        if p >= .5:
            alpha = 1
        else:
            alpha = p/(1-p)
    if rand()/float(RAND_MAX) < alpha:
        return -x
    else:
        return x
    
            
cdef inline float sigmoid(float x) nogil:
    cdef float p
    p = 1/(1+exp(-x))
    return p


cpdef inline long compute_P_parallel(data_type_t[:,:] x,
                                     data_type_t[:,:] u,
                                     data_type_t[:,:] z) nogil:
    """ parallelised over n (not d). """
    cdef long P = 0
    cdef int d, n

    for n in prange(x.shape[0], schedule=dynamic, nogil=True):
        for d in xrange(x.shape[1]):
            if compute_g_alt_tilde_unified(u[d,:], z[n,:]) == x[n, d]:
                P += 1
    return P
