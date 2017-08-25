#!python
#cython: profile=False, language_level=3, boundscheck=True, wraparound=False, cdivision=True
#cython --compile-args=-fopenmp --link-args=-fopenmp --force -a
## for compilation run: python setup.py build_ext --inplace

cimport cython
from cython.parallel import prange, parallel
from libc.math cimport exp
from libc.math cimport log
from libc.stdlib cimport rand, RAND_MAX
from libc.stdlib cimport malloc
from IPython.core.debugger import Tracer
from libcpp cimport bool as bool_t

cimport numpy as np
import numpy as np

data_type = np.int8
ctypedef np.int8_t data_type_t

data_type2 = np.int16
ctypedef np.int16_t data_type_t2

# cdef float child_node_contribution_to_maxmachine(data_type_t[:,:] sibling,
#                                                  data_type_t[:] x_n,
#                                                  data_type_t[:] child_n,
#                                                  double[:] lbda,
#                                                  int l_idx,
#                                                  int[:] idx_sorted,
#                                                  int D, int L):
#     cdef int l_prime_idx
#     cdef bint break_accumulator
#     cdef float accumulator = 0
    
#     for d in range(D):
#         # connection to child is cut
#         if sibling[d,idx_sorted[l_idx]] == -1:
#             continue

#        # connection to child is intact
#         else:
#             break_accumulator = False # continue with next iteration over d, once acc is updated

#             # is any older sibling explaining away the child?
#             for l_prime_idx in range(l_idx):
#                 if (x_n[idx_sorted[l_prime_idx]] == 1) and (sibling[d,idx_sorted[l_prime_idx]] == 1):
#                     # break -> continue with next child
#                     break_accumulator = True
#                     break
#             if break_accumulator == True:
#                 continue

#             # is any younger sibling trying to explain away the child?
#             for l_prime_idx in range(l_idx+1,L):
#                 if (x_n[idx_sorted[l_prime_idx]] == 1) and (sibling[d,idx_sorted[l_prime_idx]] == 1):
#                     accumulator += child[d] * log(lbda[idx_sorted[l_idx]]/lbda[idx_sorted[l_prime_idx]])
#                     break_accumulator = True
#                     break
#             if break_accumulator == True:
#                 continue

#             # no one is explaining away (compare to clamped alpha=0)
#             accumulator += child_n[d] * log(lbda[idx_sorted[l_idx]]/lbda[-1])

#     return accumulator
    


def draw_noparents_onechild_maxmachine(data_type_t[:,:] x,  # N x D; z_nl
                                       data_type_t[:,:] sibling, # D x Lc; u_dl
                                       data_type_t[:,:] child, # N x Lc; x_nd
                                       double[:] lbda,
                                       int[:] idx_sorted,
                                       float prior,
                                       double[:] bbp_k,
                                       double[:] bbp_j,
                                       int[:] k, # vector of counts of length D
                                       int[:] j): # vector of counts of length N):
    """
    TODO:  sampling indicator, versions with parents
    """
                               
    # cdef float p
    cdef int n, l_idx, d, N = x.shape[0], L = x.shape[1], D = sibling.shape[0]
    cdef bint break_accumulator
    cdef float accumulator

    # iterate over codes in order of decreasing lbda
    # binomial prior over N breaks paralellism
    for n in range(N): #, schedule=dynamic, nogil=True):
        for l_idx in range(L):
            # iterate over children
            
            # accumulator = child_node_contribution_to_maxmachine(
            #     sibling, x[n,:], child[n,:], lbda, l_idx, idx_sorted, D, L)
                                       
            accumulator = 0
            for d in range(D):
                
                # connection to child is cut
                if sibling[d,idx_sorted[l_idx]] == -1:
                    continue
                
                # connection to child is intact
                else:
                    break_accumulator = False # continue with next iteration over d, once acc is updated

                    # is any older sibling explaining away the child?
                    for l_prime_idx in range(l_idx):
                        if (x[n,idx_sorted[l_prime_idx]] == 1) and (sibling[d,idx_sorted[l_prime_idx]] == 1):
                            # break -> continue with next child
                            break_accumulator = True
                            break
                    if break_accumulator == True:
                        continue
                    
                    # is any younger sibling trying to explain away the child?
                    for l_prime_idx in range(l_idx+1,L):
                        if (x[n,idx_sorted[l_prime_idx]] == 1) and (sibling[d,idx_sorted[l_prime_idx]] == 1):
                            accumulator += child[n,d] * log(lbda[idx_sorted[l_idx]]/lbda[idx_sorted[l_prime_idx]])

                            break_accumulator = True
                            break
                    if break_accumulator == True:
                        continue

                    # no one is explaining away (compare to clamped alpha=0)
                    accumulator += child[n,d] * log(lbda[idx_sorted[l_idx]]/lbda[L])
                    
            x_old = x[n,idx_sorted[l_idx]]

            prior = bbp_k[k[idx_sorted[l_idx]]] + bbp_j[j[n]]         

            # x[n,l] = swap_metropolised_gibbs_unified(sigmoid(accumulator), x[n,l])  ### + prio
            x[n, idx_sorted[l_idx]] = swap_gibbs(sigmoid(accumulator + prior)) # + prior

            # update row/column count of ones for prior
            k[idx_sorted[l_idx]] += x[n,idx_sorted[l_idx]] * (x[n,idx_sorted[l_idx]] != x_old)
            j[n] += x[n,idx_sorted[l_idx]] * (x[n,idx_sorted[l_idx]] != x_old)
                                       

# def draw_oneparent_onechild_maxmachine(
#         data_type_t[:,:] x,  # N x D
#         data_type_t[:,:] z_pa, # N x Lp 
#         data_type_t[:,:] u_pa, # D x Lp
#         double[:] lbda_pa, 
#         data_type_t[:,:] sibling, # D x Lc
#         data_type_t[:,:] child, # N x Lc
#         double[:] lbda,
#         float prior,
#         double[:] bbp_k,
#         double[:] bbp_j,
#         int[:] k, # vector of counts of length D
#         int[:] j):
    
#     cdef int n, l_idx, d, N = x.shape[0], L = x.shape[1], D = sibling.shape[0]
#     cdef bint break_accumulator
#     cdef float accumulator_child, accumulator_parent

#     for n in range(N):
#         for l_idx in range(L):
            
            

def draw_noparents_onechild_bbprior(data_type_t[:,:] x,  # N x D
                                    data_type_t[:,:] sibling, # D x Lc
                                    data_type_t[:,:] child, # N x Lc
                                    float lbda,
                                    double[:] bbp_k,
                                    double[:] bbp_j,
                                    int[:] k, # vector of counts of length D
                                    int[:] j, # vector of counts of length N
                                    data_type_t[:,:] sampling_indicator): # N x D
    """
    bbp_k is the binomial prior for across D (L) and bbp_j the binom. prior across N (D).
    """

    cdef float p, acc_child
    cdef int n, d, N = x.shape[0], D = x.shape[1]
    
    for n in range(N): # no parallel updates for bbp sampler :( u: N->D, D->L
        for d in range(D):
            if sampling_indicator[n,d] is True:

                # compute the posterior
                acc_child = lbda*score_no_parents_unified(child[n,:], x[n,:], sibling, d)
                p = sigmoid(acc_child + bbp_k[k[d]] + bbp_j[j[n]])
                
                x_old = x[n, d]
                
                x[n, d] = swap_metropolised_gibbs_unified(p, x[n,d])
                
                k[d] += x[n,d] * (x[n,d] != x_old)
                j[n] += x[n,d] * (x[n,d] != x_old)
    


def draw_noparents_onechild_indpn(data_type_t[:,:] x,  # N x D
                                  data_type_t[:,:] sibling, # D x Lc
                                  data_type_t[:,:] child, # N x Lc
                                  float mu, # TODO, pass mu and lbda as array (prepare for MaxMachine)
                                  float lbda,
                                  float prior,
                                  data_type_t[:,:] sampling_indicator): # N x D
    """
    prototype for independent noise model
    """
                               
    cdef float p, acc_child
    cdef int n, d, M, N = x.shape[0], D = x.shape[1]
    
    for n in prange(N, schedule=dynamic, nogil=True):
        for d in range(D):
            if sampling_indicator[n,d] is True:

                # compute the posterior
                #acc_child = lbda*score_no_parents_unified(child[n,:], x[n,:], sibling, d)
                M = score_no_parents_unified(child[n,:], x[n,:], sibling, d)
                p = 1 / ( 1 + ( ( 1 + exp( - M * lbda ) )/ ( 1 + exp( M * mu ) ) ) )
                #p = sigmoid(acc_child + prior)

                x[n, d] = swap_metropolised_gibbs_unified(p, x[n,d])


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

                
cpdef void predict_single_latent(data_type_t[:,:] x,
                                 data_type_t[:] u,
                                 data_type_t[:] z):
    """
    compute output matrix for a single latent dimension (deterministic).
    is equivalent to the product between to binary vectors
    """
    cdef int N = z.shape[0]
    cdef int D = u.shape[0]
    cdef int n, d

    for n in range(N):
        for d in range(D):
            if (u[d] == 1) and (z[n] == 1):
                x[n,d] = 1
                

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
cpdef inline int score_no_parents_unified(
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


cdef inline int swap_gibbs(float p) nogil:
    """
    Flip according to standard gibbs sampler
    """
    if rand()/float(RAND_MAX) > p:
        return -1
    else:
        return 1


# @cython.boundscheck(False)
# @cython.wraparound(False)
@cython.cdivision(True)
cdef inline int swap_metropolised_gibbs_unified(float p, data_type_t x) nogil:
    """
    Given the p(x=1) and the current state of x \in {-1,1}.
    Draw new x according to metropolised Gibbs sampler
    """
    cdef float alpha
    if x == 1:
        if p <= .5:
            alpha = 1 # TODO, can return -x here
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
    
            
cpdef inline float sigmoid(float x) nogil:
    cdef float p
    p = 1/(1+exp(-x))
    return p


cpdef inline void compute_pred_accuracy(data_type_t[:,:] x,
                                        data_type_t[:,:] u,
                                        data_type_t[:,:] z,
                                        long[:] rates) nogil:
    """ Compute false/true positive/negative rates. """
    cdef int TP=0, FP=0, TN=0, FN=0
    cdef int d, n, g_temp

    for n in prange(x.shape[0], schedule=dynamic, nogil=True):
        for d in range(x.shape[1]):
            g_temp = compute_g_alt_tilde_unified(u[d,:], z[n,:])
            if g_temp == -1:
                if x[n,d] == -1:
                    TN += 1
                else:
                    FN += 1
            else:
                if x[n,d] == 1:
                    TP += 1
                else:
                    FP += 1

    rates[0] = TP
    rates[1] = FP
    rates[2] = TN
    rates[3] = FN
    
    return


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


cpdef void probabilistic_output_maxmachine(double[:,:] x,
                                           double[:,:] u,
                                           double[:,:] z,
                                           double[:] alpha,
                                           int[:] l_dcr):

    cdef float s1
    cdef int D = u.shape[0]
    cdef int L = u.shape[1]
    cdef int N = z.shape[0]
    cdef int d, n, l
    pvec = np.zeros(L)
    # pvec_init = np.zeros(L, dtype=np.float32)
    # cdef double[:] pvec = pvec_init
    """
    p_dn is the probability that every input is zero
    """

    for d in range(D):
        for n in range(N):
            for l in range(L):
                pvec[l] = 1 - z[n,l_dcr[l]] * u[d,l_dcr[l]]
                for l_prime in range(l):
                    pvec[l] = pvec[l] * pvec[l_prime]
            
            s1 = z[n,l_dcr[0]]*u[d,l_dcr[0]]*alpha[l_dcr[0]]
            for l in range(1,L):
                s1 += z[n,l_dcr[l]]*u[d,l_dcr[l]]*alpha[l_dcr[l]]*pvec[l-1]
            # noise dimension
            s1 += pvec[L-1]*alpha[l_dcr[L]]
            
            x[n,d] = s1
                

            
cpdef void probabilistc_output_indpndt(double[:,:] x,
                                       double[:,:] u,
                                       double[:,:] z,
                                       double lbda,
                                       double mu,
                                       int D, int N, int L):
    cdef float p_dn, sgmd_lbda, sgmd_mu
    """
    p_dn is the probability that every input is zero
    """

    sgmd_lbda = sigmoid(lbda)
    sgmd_mu = sigmoid(mu)
    for d in range(D):
        for n in range(N):
            p_dn = 1
            for l in range(L):
                p_dn = p_dn * ( 1 - u[d,l]*z[n,l] )
            x[n, d] = (sgmd_lbda * (1-p_dn) + (p_dn*(1-sgmd_mu) ) )
            
