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


def draw_noparents_onechild_maxmachine(data_type_t[:,:] x,  # N x D; z_nl
                                       data_type_t[:,:] sibling, # D x Lc; u_dl
                                       data_type_t[:,:] child, # N x Lc; x_nd
                                       double[:] lbda,
                                       int[:] idx_sorted,
                                       list prior_config,
                                       float[:,:,:] lbda_ratios):
    """
    TODO:  sampling indicator
    """

    # 
    cdef int n, l_idx, d, N = x.shape[0], L = x.shape[1], D = sibling.shape[0]
    cdef bint break_accumulator
    cdef float accumulator
    cdef float prior = 0
    cdef data_type_t x_old
    
    # unpack prior config
    cdef np.ndarray[ndim=1, dtype=np.float64_t] row_binom = prior_config[2]
    cdef np.ndarray[ndim=1, dtype=np.float64_t] col_binom = prior_config[3]
    cdef np.ndarray[ndim=1, dtype=np.int32_t] row_densities = prior_config[4]
    cdef np.ndarray[ndim=1, dtype=np.int32_t] col_densities = prior_config[5]
    cdef int prior_code = prior_config[0]
    
    # flat prior
    if prior_code == 0:
        prior = 0
    # independent bernoulli prior
    elif prior_code == 1:
        prior = prior_config[1]

    # iterate over codes in order of decreasing lbda
    # binomial prior over N breaks paralellism
    # (could implement parallel sampling, conditional on a simpler prior)
    for n in range(N): #, schedule=dynamic, nogil=True):
        for l_idx in range(L):
            # iterate over children
            
            accumulator = child_node_contribution_to_maxmachine(
                x, sibling, child, lbda, l_idx, idx_sorted, n, D, L, lbda_ratios)

            # compute prior contributions to the conditional for binomial priors
            # everything inside this loops is executed
            # (N*L + N*D) * MCMC_iters times.
            # Therefore it's better to avoid the modularisation overhead.
            if prior_code > 1:
                if prior_code == 2:
                    prior = row_binom[row_densities[n]]
                elif prior_code == 3:
                    prior = col_binom[col_densities[idx_sorted[l_idx]]]
                elif prior_code == 4:
                    prior = row_binom[row_densities[n]] + col_binom[col_densities[idx_sorted[l_idx]]]

            x_old = x[n,idx_sorted[l_idx]]
            
            x[n, idx_sorted[l_idx]]= swap_metropolised_gibbs_unified(
                sigmoid(accumulator + prior), x[n, idx_sorted[l_idx]])

            if (prior_code > 1) and (x[n, idx_sorted[l_idx]] != x_old):
                if (prior_code == 2) or (prior_code == 4):
                    row_densities[n] += x[n, idx_sorted[l_idx]]
                elif (prior_code == 3) or (prior_code == 4):
                    col_densities[idx_sorted[l_idx]] += x[n, idx_sorted[l_idx]]
                # update_prior_counts(row_binom, col_binom,
                #                     row_densities, col_densities,
                #                     x[n, idx_sorted[l_idx]], n, idx_sorted[l_idx],
                #                     prior_code)


                 
def draw_oneparent_onechild_maxmachine(data_type_t[:,:] x,  # N x D; z_nl
                                       data_type_t[:,:] sibling, # D x Lc; u_dl
                                       data_type_t[:,:] child, # N x Lc; x_nd
                                       double[:] lbda,
                                       int[:] idx_sorted,
                                       list prior_config,
                                       data_type_t[:,:] u_pa,
                                       data_type_t[:,:] z_pa,
                                       double[:] logit_lbda_pa,
                                       int[:] idx_sorted_pa,
                                       float[:,:,:] lbda_ratios):
    """
    TODO:  sampling indicator, versions with parents
    """

    # unpack prior config
    cdef np.ndarray[ndim=1, dtype=np.float64_t] row_binom = prior_config[2]
    cdef np.ndarray[ndim=1, dtype=np.float64_t] col_binom = prior_config[3]
    cdef np.ndarray[ndim=1, dtype=np.int32_t] row_densities = prior_config[4]
    cdef np.ndarray[ndim=1, dtype=np.int32_t] col_densities = prior_config[5]
    cdef int prior_code = prior_config[0]

    cdef int n, l_idx, d, N = x.shape[0], L = x.shape[1], D = sibling.shape[0]
    cdef bint break_accumulator
    cdef float accumulator_par, accumulator_child
    cdef float prior = 0
    cdef data_type_t x_old

    # for l in range(len(logit_lbda_pa)):
    #     logit_lbda_pa[l] = scipy.special.cython_special.logit(logit_lbda_pa[l])

    if prior_code == 1:
        prior = prior_config[1] # already on logit scale
            
    # iterate over codes in order of decreasing lbda
    # binomial prior over N breaks paralellism
    for n in range(N): #, schedule=dynamic, nogil=True):
        for l_idx in range(L):
            # iterate over children
            
            accumulator_child = child_node_contribution_to_maxmachine(
                x, sibling, child, lbda, l_idx, idx_sorted, n, D, L, lbda_ratios)

            accumulator_par = parent_contribution_to_maxmachine(
                u_pa, z_pa, logit_lbda_pa, idx_sorted_pa, n, idx_sorted[l_idx])
            
            if prior_code > 1:
                if prior_code == 2:
                    prior = row_binom[row_densities[n]]
                elif prior_code == 3:
                    prior = col_binom[col_densities[idx_sorted[l_idx]]]
                elif prior_code == 4:
                    prior = row_binom[row_densities[n]] + col_binom[col_densities[idx_sorted[l_idx]]]
                
                # get_prior(row_binom, col_binom,
                #           row_densities, col_densities,
                #           n, idx_sorted[l_idx], prior_code, prior)
                
            x_old = x[n,idx_sorted[l_idx]]

            x[n, idx_sorted[l_idx]] = swap_metropolised_gibbs_unified(
                sigmoid(accumulator_child + accumulator_par + prior),
                x[n, idx_sorted[l_idx]])
            # x[n, idx_sorted[l_idx]] = swap_gibbs(sigmoid(accumulator_child + prior)) # + prior
  
            if (prior_code > 1) and (x[n, idx_sorted[l_idx]] != x_old):
                if (prior_code == 2) or (prior_code == 4):
                    row_densities[n] += x[n, idx_sorted[l_idx]]
                elif (prior_code == 3) or (prior_code == 4):
                    col_densities[idx_sorted[l_idx]] += x[n, idx_sorted[l_idx]]

                    
def draw_noparents_onechild(data_type_t[:,:] x,  # N x D
                           data_type_t[:,:] sibling, # D x Lc
                           data_type_t[:,:] child, # N x Lc
                           float lbda,
                           list prior_config,                            
                           data_type_t[:,:] sampling_indicator): # N x D

    # unpack prior config
    cdef np.ndarray[ndim=1, dtype=np.float64_t] row_binom = prior_config[2]
    cdef np.ndarray[ndim=1, dtype=np.float64_t] col_binom = prior_config[3]
    cdef np.ndarray[ndim=1, dtype=np.int32_t] row_densities = prior_config[4]
    cdef np.ndarray[ndim=1, dtype=np.int32_t] col_densities = prior_config[5]
    cdef int prior_code = prior_config[0]
    cdef float p, acc_child
    cdef float prior = 0
    cdef int n, d, N = x.shape[0], D = x.shape[1]
    cdef data_type_t x_old
    
    if prior_code == 1:
        prior = prior_config[1]
    
    # for n in prange(N, schedule=dynamic, nogil=True):
    for n in range(N):
        for d in range(D):
            if sampling_indicator[n,d] == 1:

                # compute the posterior
                acc_child = lbda*score_no_parents_unified(child[n,:], x[n,:], sibling, d)
                
                if prior_code > 1:
                    if prior_code == 2:
                        prior = row_binom[row_densities[n]]
                    elif prior_code == 3:
                        prior = col_binom[col_densities[d]]
                    elif prior_code == 4:
                        prior = row_binom[row_densities[n]] + col_binom[col_densities[d]]
                        
                p = sigmoid(acc_child + prior)

                x_old = x[n,d]
                x[n, d] = swap_metropolised_gibbs_unified(p, x[n,d])
  
                if (prior_code > 1) and (x[n, d] != x_old):
                    if (prior_code == 2) or (prior_code == 4):
                        row_densities[n] += x[n, d]
                    elif (prior_code == 3) or (prior_code == 4):
                        col_densities[d] += x[n, d]

                #if x[n, d] != x_old:
                #   update_prior_counts(prior_config, x[n,d], n, d)


def draw_oneparent_nochild(
    data_type_t[:,:] x,  # N x D
    data_type_t[:,:] z_pa, # N x Lp 
    data_type_t[:,:] u_pa, # D x Lp
    double lbda_pa,
    list prior_config,
    data_type_t[:,:] sampling_indicator): # N x D

    cdef int n, d, N = x.shape[0], D=x.shape[1]
    cdef float acc_par, p
    cdef float prior = 0
    cdef data_type_t x_old
    # unpack prior config
    cdef np.ndarray[ndim=1, dtype=np.float64_t] row_binom = prior_config[2]
    cdef np.ndarray[ndim=1, dtype=np.float64_t] col_binom = prior_config[3]
    cdef np.ndarray[ndim=1, dtype=np.int32_t] row_densities = prior_config[4]
    cdef np.ndarray[ndim=1, dtype=np.int32_t] col_densities = prior_config[5]
    cdef int prior_code = prior_config[0]
    
    if prior_code == 1:
        prior = prior_config[1]    

    for n in prange(N, schedule=dynamic, nogil=True): # parallelise
    # for n in range(N):
        for d in range(D):
            if sampling_indicator[n,d] == 1:

                if prior_code > 1:
                    if prior_code == 2:
                        prior = row_binom[row_densities[n]]
                    elif prior_code == 3:
                        prior = col_binom[col_densities[d]]
                    elif prior_code == 4:
                        prior = row_binom[row_densities[n]] + col_binom[col_densities[d]]
                        
                acc_par = lbda_pa*compute_g_alt_tilde_unified(u_pa[d,:], z_pa[n,:]) 
                p = sigmoid(acc_par + prior)
                x_old = x[n,d]
                x[n, d] = swap_metropolised_gibbs_unified(p, x[n,d])
                
                if (prior_code > 1) and (x[n, d] != x_old):
                    if (prior_code == 2) or (prior_code == 4):
                        row_densities[n] += x[n, d]
                    elif (prior_code == 3) or (prior_code == 4):
                        col_densities[d] += x[n, d]
                


def draw_twoparents_nochild(
        data_type_t[:,:] x,  # N x D
        data_type_t[:,:] z_pa1, # N x Lp1
        data_type_t[:,:] u_pa1, # D x Lp1
        double lbda_pa1,
        data_type_t[:,:] z_pa2, # N x Lp2
        data_type_t[:,:] u_pa2, # D x Lp2
        double lbda_pa2,
        list prior_config,
        data_type_t[:,:] sampling_indicator): # N x D

    cdef int n, d, N = x.shape[0], D=x.shape[1]
    cdef float acc_par, p
    cdef float prior = 0
    cdef data_type_t x_old
    
    # unpack prior config
    cdef np.ndarray[ndim=1, dtype=np.float64_t] row_binom = prior_config[2]
    cdef np.ndarray[ndim=1, dtype=np.float64_t] col_binom = prior_config[3]
    cdef np.ndarray[ndim=1, dtype=np.int32_t] row_densities = prior_config[4]
    cdef np.ndarray[ndim=1, dtype=np.int32_t] col_densities = prior_config[5]
    cdef int prior_code = prior_config[0]


    # independent bernoulli prior
    if prior_code == 1:
        prior = prior_config[1]    

    for n in prange(N, schedule=dynamic, nogil=True): # parallelise
        for d in range(D):
            if sampling_indicator[n,d] == 1:

                if prior_code > 1:
                    if prior_code == 2:
                        prior = row_binom[row_densities[n]]
                    elif prior_code == 3:
                        prior = col_binom[col_densities[d]]
                    elif prior_code == 4:
                        prior = row_binom[row_densities[n]] + col_binom[col_densities[d]]
                        
                # accumulate over all parents
                acc_par = lbda_pa1*compute_g_alt_tilde_unified(u_pa1[d,:], z_pa1[n,:]) +\
                  lbda_pa2*compute_g_alt_tilde_unified(u_pa2[d,:], z_pa2[n,:])

                p = sigmoid(acc_par + prior)
                x_old = x[n,d]
                x[n, d] = swap_metropolised_gibbs_unified(p, x[n,d])

                if (prior_code > 1) and (x[n, d] != x_old):
                    if (prior_code == 2) or (prior_code == 4):
                        row_densities[n] += x[n, d]
                    elif (prior_code == 3) or (prior_code == 4):
                        col_densities[d] += x[n, d]                
                


def draw_oneparent_onechild(
        data_type_t[:,:] x,  # N x D
        data_type_t[:,:] z_pa, # N x Lp 
        data_type_t[:,:] u_pa, # D x Lp
        double lbda_pa, 
        data_type_t[:,:] sibling, # D x Lc
        data_type_t[:,:] child, # N x Lc
        float lbda,
        list prior_config,
        data_type_t[:,:] sampling_indicator): # N x D

    cdef int n, d, N = x.shape[0], D=x.shape[1]
    cdef float acc_par, acc_child, p
    cdef float prior = 0
    cdef data_type_t x_old
    
    # unpack prior config
    cdef int prior_code = prior_config[0]
    cdef np.ndarray[ndim=1, dtype=np.float64_t] row_binom = prior_config[2]
    cdef np.ndarray[ndim=1, dtype=np.float64_t] col_binom = prior_config[3]
    cdef np.ndarray[ndim=1, dtype=np.int32_t] row_densities = prior_config[4]
    cdef np.ndarray[ndim=1, dtype=np.int32_t] col_densities = prior_config[5]
    
    # independent bernoulli prior
    if prior_code == 1:
        prior = prior_config[1]     
    
    for n in prange(N, schedule=dynamic, nogil=True):
    # for n in range(N):
        for d in range(D):
            if sampling_indicator[n,d] is True:

                if prior_code > 1:
                    if prior_code == 2:
                        prior = row_binom[row_densities[n]]
                    elif prior_code == 3:
                        prior = col_binom[col_densities[d]]
                    elif prior_code == 4:
                        prior = row_binom[row_densities[n]] + col_binom[col_densities[d]]
                        
                acc_par = lbda_pa*compute_g_alt_tilde_unified(u_pa[d,:], z_pa[n,:])
                    
                acc_child = score_no_parents_unified(child[n,:], x[n,:], sibling, d)  
                acc_child = acc_child * lbda
                
                p = sigmoid(acc_par + acc_child + prior)

                x_old = x[n,d]
                x[n, d] = swap_metropolised_gibbs_unified(p, x[n,d])           
                               
                if (prior_code > 1) and (x[n, d] != x_old):
                    if (prior_code == 2) or (prior_code == 4):
                        row_densities[n] += x[n, d]
                    elif (prior_code == 3) or (prior_code == 4):
                        col_densities[d] += x[n, d]

                
cpdef np.ndarray[data_type_t, ndim = 2] predict_single_latent(data_type_t[:] u,
                                                               data_type_t[:] z):
    """
    compute output matrix for a single latent dimension (deterministic).
    is equivalent to the product between to binary vectors
    """
    cdef int N = z.shape[0]
    cdef int D = u.shape[0]
    cdef int n, d

    cdef np.ndarray[data_type_t, ndim=2] x = np.zeros([z.shape[0], u.shape[0]], dtype=np.int8)

    for n in range(N):
        for d in range(D):
            if (u[d] == 1) and (z[n] == 1):
                x[n,d] = 1
    return x
                

cdef int compute_g_alt_tilde_unified(data_type_t[:] u,
                                     data_type_t[:] z) nogil:
    """
    for two vectors of same length, u and z, comput
    2*np.min(1, u^T z)-1
    """
    cdef int i
    for i in range(u.shape[0]):
        if u[i] == 1 and z[i] == 1:
            return 1
    return -1
    
# @cython.boundscheck(False)
# @cython.wraparound(False)
cdef int score_no_parents_unified(
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

    #for n in prange(x.shape[0], schedule=dynamic, nogil=True):
    for n in range(x.shape[0]):
        for d in range(x.shape[1]):
            if compute_g_alt_tilde_unified(u[d,:], z[n,:]) == x[n, d]:
                P += 1
    return P

    
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
                                           double[:] pvec,
                                           int[:] l_dcr):

    cdef float s1
    cdef int D = u.shape[0]
    cdef int L = u.shape[1]
    cdef int N = z.shape[0]
    cdef int d, n, l
    #cdef int[:] l_dcr
    #cdef double[:] pvec
    #pvec[:] = 0
    #cdef np.ndarray[np.float_t, ndim=1] pvec = np.zeros(L, dtype=np.float16)
    # pvec_init = np.zeros(L, dtype=np.float32)
    # cdef double[:] pvec = pvec_init
    """
    p_dn is the probability that every input is zero
    """

    for d in range(D): #, schedule=dynamic, nogil=False):
        for n in range(N):
            for l in range(L):
                l_dcr = np.array(np.argsort(np.multiply(
                    np.multiply(alpha[:-1], z[n,:]),u[d,:]))[::-1],
                                 dtype=np.int32)
                #print(l_dcr)
                #print('lala')
                pvec[l] = 1 - (z[n,l_dcr[l]] * u[d,l_dcr[l]])
                for l_prime in range(l):
                    pvec[l] = pvec[l] * pvec[l_prime]

            x[n,d] = z[n,l_dcr[0]]*u[d,l_dcr[0]]*alpha[l_dcr[0]]
            for l in range(1,L):
                x[n,d] += z[n,l_dcr[l]]*u[d,l_dcr[l]]*alpha[l_dcr[l]]*pvec[l-1]
            # noise dimension
            x[n,d] += pvec[L-1]*alpha[L]
            
            
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
            

            


cdef float child_node_contribution_to_maxmachine(data_type_t[:,:] x,
                                                 data_type_t[:,:] sibling,
                                                 data_type_t[:,:] child,
                                                 double[:] lbda,
                                                 int l_idx,
                                                 int[:] idx_sorted,
                                                 int n, int D, int L,
                                                 float[:,:,:] lbda_ratios):
    """
    todo: precompute 1-lbda ?
    """
    cdef int l_prime_idx
    cdef bint my_break_accumulator
    cdef float my_accumulator = 0
    
    for d in range(D):
        # connection to child is cut
        if sibling[d,idx_sorted[l_idx]] == -1:
            continue

       # connection to child is intact
        else:
            my_break_accumulator = False # continue with next iteration over d, once acc is updated

            # is any older sibling explaining away the child?
            for l_prime_idx in range(l_idx):
                if (x[n,idx_sorted[l_prime_idx]] == 1) and (sibling[d,idx_sorted[l_prime_idx]] == 1):
                    # break -> continue with next child
                    my_break_accumulator = True
                    break
            if my_break_accumulator == True:
                continue

            # is any younger sibling trying to explain away the child?
            for l_prime_idx in range(l_idx+1,L):
                if (x[n,idx_sorted[l_prime_idx]] == 1) and (sibling[d,idx_sorted[l_prime_idx]] == 1):
                    if child[n,d] == 1:
                        my_accumulator += lbda_ratios[0, idx_sorted[l_idx], idx_sorted[l_prime_idx]]

                        ## for debugging
                        # print('\n')
                        # print( lbda_ratios[0, idx_sorted[l_idx], idx_sorted[l_prime_idx]] )
                        # print( log((lbda[idx_sorted[l_idx]])/(lbda[idx_sorted[l_prime_idx]])))
                        # assert ( np.abs( lbda_ratios[0, idx_sorted[l_idx], idx_sorted[l_prime_idx]] -
                        #         log(lbda[idx_sorted[l_idx]]/lbda[idx_sorted[l_prime_idx]])) < 1e-5 )
                        #my_accumulator += log(lbda[idx_sorted[l_idx]]/lbda[idx_sorted[l_prime_idx]])
                        
                    elif child[n,d] == -1:
                        my_accumulator += lbda_ratios[1, idx_sorted[l_idx], idx_sorted[l_prime_idx]]
                        
                        ## for debugging
                        # print('\n')
                        # print( lbda_ratios[1, idx_sorted[l_idx], idx_sorted[l_prime_idx]] )
                        # print( log((1-lbda[idx_sorted[l_idx]])/(1-lbda[idx_sorted[l_prime_idx]])) )
                        # assert ( np.abs( lbda_ratios[1, idx_sorted[l_idx], idx_sorted[l_prime_idx]] -
                        #        log((1-lbda[idx_sorted[l_idx]])/(1-lbda[idx_sorted[l_prime_idx]]))) < 1e-5 )
                        #my_accumulator += log((1-lbda[idx_sorted[l_idx]])/(1-lbda[idx_sorted[l_prime_idx]]))
                        
                    my_break_accumulator = True
                    break
            if my_break_accumulator == True:
                continue

            # no one is explaining away (compare to clamped alpha=0)
            # print(myaccumulator, child[n,d])
            # TODO: use precomputed values here.
            if child[n,d] == 1:
                my_accumulator += lbda_ratios[0, idx_sorted[l_idx], L]
                #my_accumulator += log(lbda[idx_sorted[l_idx]]/lbda[L])
            elif child[n,d] == -1:
                my_accumulator += lbda_ratios[1, idx_sorted[l_idx], L]
                #my_accumulator += log((1-lbda[idx_sorted[l_idx]])/(1-lbda[L]))

    return my_accumulator


cdef float parent_contribution_to_maxmachine(data_type_t[:,:] u_pa,
                                             data_type_t[:,:] z_pa,
                                             double[:] logit_lbda_pa,
                                             int[:] idx_sorted,
                                             int n, int l):
    """
    Compute parent contribution for a given particular z_nl
    """
    cdef int m
    cdef int M = u_pa.shape[1]
    
    for m in range(M):
        if (z_pa[n,idx_sorted[m]] == 1) and (u_pa[l,idx_sorted[m]] == 1):
            return logit_lbda_pa[idx_sorted[m]]

    # if no inpute was on, return clamped
    return logit_lbda_pa[M]


cdef void get_prior(np.float64_t[:] row_binom, np.float64_t[:] col_binom,
                     np.int32_t[:] row_densities, np.int32_t[:] col_densities,
                     int n, int l, int prior_code, float prior):
    """
    Compute logit prior contribution, for priors as specified in prior_config.
    This function is applicable for all sampling functions.
    """

    if prior_code == 2:
        prior = row_binom[row_densities[n]]
    elif prior_code == 3:
        prior = col_binom[col_densities[l]]
    elif prior_code == 4:
        prior = row_binom[row_densities[n]] + col_binom[col_densities[l]]


cdef void update_prior_counts(np.float64_t[:] row_binom, np.float64_t[:] col_binom,
                         np.int32_t[:] row_densities, np.int32_t[:] col_densities,
                         data_type_t x_new, int n, int l, int prior_code) nogil:
    """
    after an entry in the factor matrix has change, 
    update the prior count vectors accordingly
    """

    if prior_code == 2:
        row_densities[n] += x_new
    elif prior_code == 3:
        col_densities[l] += x_new
    elif prior_code == 4:
        row_densities[n] += x_new
        col_densities[l] += x_new
            
