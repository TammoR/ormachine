#!python
#cython: profile=False, language_level=3, boundscheck=False, wraparound=False, cdivision=True
#cython --compile-args=-fopenmp --link-args=-fopenmp --force -a
## for compilation run: python setup.py build_ext --inplace

cimport cython
from cython.parallel import prange, parallel
from libc.math cimport exp
from libc.stdlib cimport rand, RAND_MAX
cimport numpy as np
import numpy as np
data_type = np.int8
ctypedef np.int8_t data_type_t



def draw_noparents_onechild(data_type_t[:,:] x,  # N x D
                           data_type_t[:,:] sibling, # D x Lc
                           data_type_t[:,:] child, # N x Lc
                           float lbda,
                           float prior,
                           data_type_t[:,:] sampling_indicator): # N x D
                               
    cdef float p, acc_child
    cdef int n, d, N = x.shape[0], D = x.shape[1]
    
    for n in prange(N, schedule=dynamic, nogil=True):
        for d in range(D):
            if sampling_indicator[n,d] is True:

                # compute the posterior
                acc_child = lbda*score_no_parents_unified(child[n,:], x[n,:], sibling, d)
                p = sigmoid(acc_child + prior)

                x[n, d] = swap_metropolised_gibbs_unified(p, x[n,d])


def draw_oneparent_nochild(
    data_type_t[:,:] x,  # N x D
    data_type_t[:,:] z_pa, # N x Lp 
    data_type_t[:,:] u_pa, # D x Lp
    double lbda_pa,
    float prior,
    data_type_t[:,:] sampling_indicator): # N x D

    cdef float p, acc_par
    cdef int n, d, N=x.shape[0], D=x.shape[1]
    for n in prange(N, schedule=dynamic, nogil=True): # parallelise
    # for n in range(N):
        for d in range(D):
            if sampling_indicator[n,d] is True:
                acc_par = lbda_pa*compute_g_alt_tilde_unified(u_pa[d,:], z_pa[n,:]) 
                p = sigmoid(acc_par + prior)
                x[n, d] = swap_metropolised_gibbs_unified(p, x[n,d])                           


def draw_twoparents_nochild(
        data_type_t[:,:] x,  # N x D
        data_type_t[:,:] z_pa1, # N x Lp1
        data_type_t[:,:] u_pa1, # D x Lp1
        double lbda_pa1,
        data_type_t[:,:] z_pa2, # N x Lp2
        data_type_t[:,:] u_pa2, # D x Lp2
        double lbda_pa2,
        float prior,
        data_type_t[:,:] sampling_indicator): # N x D

    cdef float p, acc_par
    cdef int n, d, N=x.shape[0], D=x.shape[1]
    for n in prange(N, schedule=dynamic, nogil=True): # parallelise
        for d in range(D):
            if sampling_indicator[n,d] is True:
               
                # accumulate over all parents
                acc_par = lbda_pa1*compute_g_alt_tilde_unified(u_pa1[d,:], z_pa1[n,:]) +\
                  lbda_pa2*compute_g_alt_tilde_unified(u_pa2[d,:], z_pa2[n,:])

                p = sigmoid(acc_par + prior)
                x[n, d] = swap_metropolised_gibbs_unified(p, x[n,d])


def draw_oneparent_onechild(
        data_type_t[:,:] x,  # N x D
        data_type_t[:,:] z_pa, # N x Lp 
        data_type_t[:,:] u_pa, # D x Lp
        double lbda_pa, 
        data_type_t[:,:] sibling, # D x Lc
        data_type_t[:,:] child, # N x Lc
        float lbda,
        float prior,
        data_type_t[:,:] sampling_indicator): # N x D
    
    cdef float p, acc_par, acc_child
    cdef int n, d, N = x.shape[0], D=x.shape[1]
    for n in prange(N, schedule=dynamic, nogil=True):
    # for n in range(N):
        for d in range(D):
            if sampling_indicator[n,d] is True:

                acc_par = lbda_pa*compute_g_alt_tilde_unified(u_pa[d,:], z_pa[n,:])
                    
                acc_child = lbda*score_no_parents_unified(child[n,:], x[n,:], sibling, d)  
                
                p = sigmoid(acc_par + acc_child + prior)
                
                x[n, d] = swap_metropolised_gibbs_unified(p, x[n,d])           


def draw_oneparent_nochild_maxdens(
    data_type_t[:,:] x,  # N x D
    data_type_t[:,:] z_pa, # N x Lp 
    data_type_t[:,:] u_pa, # D x Lp
    double lbda_pa,
    float prior,
    data_type_t[:] max_density,
    data_type_t[:,:] sampling_indicator): # N x D

    cdef float p, acc_par
    cdef int n, d, N=x.shape[0], D=x.shape[1]
    for n in range(N):
        for d in range(D):
            if sampling_indicator[n,d] is True:

                if density_magic(x, max_density, n, d):
                    continue

                # accumulate over all parents
                acc_par = lbda_pa*compute_g_alt_tilde_unified(u_pa[d,:], z_pa[n,:]) 
                p = sigmoid(acc_par + prior)
                x[n, d] = swap_metropolised_gibbs_unified(p, x[n,d])


def draw_twoparents_nochild_maxdens(
    data_type_t[:,:] x,  # N x D
    data_type_t[:,:] z_pa1, # N x Lp1
    data_type_t[:,:] u_pa1, # D x Lp1
    double lbda_pa1,
    data_type_t[:,:] z_pa2, # N x Lp2
    data_type_t[:,:] u_pa2, # D x Lp2
    double lbda_pa2,
    float prior,
    data_type_t[:] max_density,
    data_type_t[:,:] sampling_indicator): # N x D

    cdef float p, acc_par
    cdef int n, d, N=x.shape[0], D=x.shape[1]
    
    for n in range(N):
        for d in range(D):
            if sampling_indicator[n,d] is True:
                
                if density_magic(x, max_density, n, d):
                    continue
               
                # accumulate over all parents
                acc_par = lbda_pa1*compute_g_alt_tilde_unified(u_pa1[d,:], z_pa1[n,:]) +\
                  lbda_pa2*compute_g_alt_tilde_unified(u_pa2[d,:], z_pa2[n,:])

                p = sigmoid(acc_par + prior)
                x[n, d] = swap_metropolised_gibbs_unified(p, x[n,d])


def draw_noparents_onechild_maxdens(data_type_t[:,:] x,  # N x D
                           data_type_t[:,:] sibling, # D x Lc
                           data_type_t[:,:] child, # N x Lc
                           float lbda,
                           float prior,
                           data_type_t[:] max_density,
                           data_type_t[:,:] sampling_indicator): # N x D
                               
    cdef float p, acc_child
    cdef int n, d, N = x.shape[0], D = x.shape[1]
    cdef bint dens_switch
    
    for n in range(N):
        for d in range(D):
            if sampling_indicator[n,d] is True:

                # check codes are not too dense/sparse if applic.
                # the function may alter the state of the variable!
                if density_magic(x, max_density, n, d):
                    continue

                # compute the posterior
                acc_child = lbda*score_no_parents_unified(child[n,:], x[n,:], sibling, d)
                p = sigmoid(acc_child + prior)

                x[n, d] = swap_metropolised_gibbs_unified(p, x[n,d])               
                

def draw_oneparent_onechild_maxdens(
    data_type_t[:,:] x,  # N x D
    data_type_t[:,:] z_pa, # N x Lp 
    data_type_t[:,:] u_pa, # D x Lp
    double lbda_pa, 
    data_type_t[:,:] sibling, # D x Lc
    data_type_t[:,:] child, # N x Lc
    float lbda,
    float prior,
    data_type_t[:] max_density,
    data_type_t[:,:] sampling_indicator): # N x D
    
    cdef float p, acc_par, acc_child
    cdef int n, d, N = x.shape[0], D=x.shape[1]

    for n in range(N):
        for d in range(D):
            if sampling_indicator[n,d] is True:

                if density_magic(x, max_density, n, d):
                    continue

                acc_par = lbda_pa*compute_g_alt_tilde_unified(u_pa[d,:], z_pa[n,:])
                    
                acc_child = lbda*score_no_parents_unified(child[n,:],
                                                          x[n,:], sibling, d)  
                
                p = sigmoid(acc_par + acc_child + prior)
                
                x[n, d] = swap_metropolised_gibbs_unified(p, x[n,d])            

                

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
    # for n in range(x.shape[0]):
        for d in range(x.shape[1]):
            if compute_g_alt_tilde_unified(u[d,:], z[n,:]) == x[n, d]:
                P += 1
    return P


cpdef int no_of_ones(data_type_t[:] z) nogil:
    cdef int L = z.shape[0]
    cdef int acc = 0
    for i in range(L):
        if z[i] == 1:
            acc += 1
    return acc


cpdef bint max_density_checker(data_type_t[:] x, int max_density, int d) nogil:
    cdef int one_count
    
    if max_density != 0:
        one_count = no_of_ones(x)

        # too many ones in code
        if one_count > max_density:
            # if current value is one, set to zero
            # otherwise, do not update (return True -> no update)
            if x[d] == 1:
                x[d] = -1
            return True

        # maximum no of ones in code
        elif one_count == max_density:
            # if current value is one, allow update
            if x[d] == 1:
                return False
            # if current value is -1, do not update
            elif x[d] == -1:
                return True
            
    return False


cpdef bint min_density_checker(data_type_t[:] x, int min_density, int d) nogil:
    cdef int one_count
    
    if min_density != 0:
        one_count = no_of_ones(x)

        # too many ones in code
        if one_count < min_density:
            # if current value is one, set to zero
            # otherwise, do not update (return True -> no update)
            if x[d] == -1:
                x[d] = 1
            return True

        # maximum no of ones in code
        elif one_count == min_density:
            # if current value is one, allow update
            if x[d] == 1:
                return False
            # if current value is -1, do not update
            elif x[d] == -1:
                return True
            
    return False


cpdef bint density_magic(data_type_t[:,:] x,
                             data_type_t[:] density_conditions,
                             int n, int d):
    """
    return true if max density is reached in either rows or columns.
    if max density is violated, set x[n,d] from 1 to -1
    """

    cdef bint update1, update2, update3, update4

    # check whether minimum conditions are met in any dimension
    update0 = min_density_checker(x[n,:], density_conditions[0], d)
    update1 = min_density_checker(x[:,d], density_conditions[1], n)

    # if so terminate function
    if (update0 or update1):
        return True
    
    # if not, check whether maximum conditions are met
    else:
        update2 = max_density_checker(x[n,:], density_conditions[2], d)
        update3 = max_density_checker(x[:,d], density_conditions[3], n)

        return (update2 or update3)
    
    return False
    # if at least one constrained returns True (do NOT update), return True



cpdef void probabilistc_output(double[:,:] x,
                               double[:,:] u,
                               double[:,:] z,
                               double lbda,
                               int D, int N, int L):
    cdef float p_dn, sgmd_lbda
    """
    p_dn is the probability that every input is zero
    """

    sgmd_lbda = sigmoid(lbda)
    for d in range(D):
        for n in range(N):
            p_dn = 1
            for l in range(L):
                p_dn = p_dn * ( 1 - u[d,l]*z[n,l] )
            x[n, d] = (sgmd_lbda * (1-p_dn) + (p_dn*(1-sgmd_lbda) ) )