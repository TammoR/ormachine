import cython_functions as cf
import numpy as np
from IPython.core.debugger import Tracer
    
def draw_z_oneparent_nochild_maxdens_wrapper(mat):
    
    cf.draw_oneparent_nochild_maxdens(
        mat(), # NxD
        mat.parents[0](), # parents obs: N x Lp
        mat.parents[1](), # parents feat: D x Lp
        mat.parents[0].lbda(), # parent lbdas: 
        mat.logit_bernoulli_prior,
        mat.density_conditions,
        mat.sampling_indicator)
    
def draw_u_oneparent_nochild_maxdens_wrapper(mat):
    
    cf.draw_oneparent_nochild_maxdens(
        mat(), # NxD
        mat.parents[1](), # parents obs: N x Lp
        mat.parents[0](), # parents feat: D x Lp
        mat.parents[0].lbda(), # parent lbdas: K (KxL for MaxM)
        mat.logit_bernoulli_prior,
        mat.density_conditions,
        mat.sampling_indicator)    
        
def draw_z_twoparents_nochild_maxdens_wrapper(mat):

    cf.draw_twoparents_nochild_maxdens(
        mat(), # NxD
        mat.parent_layers[0].z(), # parents obs: N x Lp
        mat.parent_layers[0].u(), # parents feat: D x Lp
        mat.parent_layers[0].u.lbda(), # parent lbda
        mat.parent_layers[1].z(), # parents obs: N x Lp
        mat.parent_layers[1].u(), # parents feat: D x Lp
        mat.parent_layers[1].u.lbda(), # parent lbda
        mat.logit_bernoulli_prior,
        mat.density_conditions,
        mat.sampling_indicator)
    
def draw_u_twoparents_nochild_maxdens_wrapper(mat):

    cf.draw_twoparents_nochild_maxdens(
        mat(), # NxD
        mat.parent_layers[0].u(), # parents obs: N x Lp
        mat.parent_layers[0].z(), # parents feat: D x Lp
        mat.parent_layers[0].u.lbda(), # parent lbda
        mat.parent_layers[1].u(), # parents obs: N x Lp
        mat.parent_layers[1].z(), # parents feat: D x Lp
        mat.parent_layers[1].u.lbda(), # parent lbda
        mat.logit_bernoulli_prior,
        mat.density_conditions,
        mat.sampling_indicator)        

def draw_z_noparents_onechild_maxdens_wrapper(mat):

    cf.draw_noparents_onechild_maxdens(
        mat(),  # NxD
        mat.sibling(),  # sibling u: D x Lc
        mat.child(),  # child observation: N x Lc
        mat.lbda(),  # own parameter: double
        mat.logit_bernoulli_prior,
        mat.density_conditions,
        mat.sampling_indicator)
    
def draw_u_noparents_onechild_maxdens_wrapper(mat):
    
    cf.draw_noparents_onechild_maxdens(
        mat(), # NxD
        mat.sibling(), # sibling u: D x Lc
        mat.child().transpose(), # child observation: N x Lc
        mat.lbda(), # own parameter: double
        mat.logit_bernoulli_prior,
        mat.density_conditions,
        mat.sampling_indicator)

def draw_z_oneparent_onechild_maxdens_wrapper(mat):

    cf.draw_oneparent_onechild_maxdens(
        mat(), # N x D
        mat.parents[0](), # parent obs: N x Lp
        mat.parents[1](), # parent features, D x Lp
        mat.parents[1].lbda(), # parent lbda
        mat.sibling(), # sibling u: D x Lc
        mat.child(), # child observation: N x Lc
        mat.lbdas(), # own parameter: double
        mat.logit_bernoulli_prior,
        mat.density_conditions,
        mat.sampling_indicator)

def draw_u_oneparent_onechild_maxdens_wrapper(mat):

    cf.draw_oneparent_onechild_maxdens(
        mat(), # NxD
        mat.parents[1](), # parent obs: N x Lp
        mat.parents[0](), # parent features, D x Lp
        mat.parents[1].lbda(), # parent lbda
        mat.sibling(), # sibling u: D x Lc
        mat.child().transpose(), # child observation: N x Lc
        mat.lbdas(), # own parameter: double
        mat.logit_bernoulli_prior,
        mat.density_conditions,
        mat.sampling_indicator)


def draw_z_oneparent_nochild_wrapper(mat):
    
    cf.draw_oneparent_nochild(
        mat(), # NxD
        mat.parents[0](), # parents obs: N x Lp
        mat.parents[1](), # parents feat: D x Lp
        mat.parents[0].lbda(), # parent lbdas: 
        mat.logit_bernoulli_prior,
        mat.sampling_indicator)
    
def draw_u_oneparent_nochild_wrapper(mat):
    
    cf.draw_oneparent_nochild(
        mat(), # NxD
        mat.parents[1](), # parents obs: N x Lp
        mat.parents[0](), # parents feat: D x Lp
        mat.parents[0].lbda(), # parent lbdas: K (KxL for MaxM)
        mat.logit_bernoulli_prior,
        mat.sampling_indicator)    
        
def draw_z_twoparents_nochild_wrapper(mat):

    cf.draw_twoparents_nochild(
        mat(), # NxD
        mat.parent_layers[0].z(), # parents obs: N x Lp
        mat.parent_layers[0].u(), # parents feat: D x Lp
        mat.parent_layers[0].u.lbda(), # parent lbda
        mat.parent_layers[1].z(), # parents obs: N x Lp
        mat.parent_layers[1].u(), # parents feat: D x Lp
        mat.parent_layers[1].u.lbda(), # parent lbda
        mat.logit_bernoulli_prior,
        mat.sampling_indicator)
    
def draw_u_twoparents_nochild_wrapper(mat):

    cf.draw_twoparents_nochild(
        mat(), # NxD
        mat.parent_layers[0].u(), # parents obs: N x Lp
        mat.parent_layers[0].z(), # parents feat: D x Lp
        mat.parent_layers[0].u.lbda(), # parent lbda
        mat.parent_layers[1].u(), # parents obs: N x Lp
        mat.parent_layers[1].z(), # parents feat: D x Lp
        mat.parent_layers[1].u.lbda(), # parent lbda
        mat.logit_bernoulli_prior,
        mat.sampling_indicator)        

def draw_z_noparents_onechild_wrapper(mat):

    cf.draw_noparents_onechild(
        mat(),  # NxD
        mat.sibling(),  # sibling u: D x Lc
        mat.child(),  # child observation: N x Lc
        mat.lbda(),  # own parameter: double
        mat.logit_bernoulli_prior,
        mat.sampling_indicator)
    
def draw_u_noparents_onechild_wrapper(mat):
    from scipy.special import logit

    if False:
        # standard prior
        logit_bernoulli_prior = mat.logit_bernoulli_prior

        #binomial prior in EM fashion
        logitq = logit(1./3) # expected code fractional length
        D, L = mat().shape # naming of varibles different than in cython fcts.
        logit_bernoulli_priors = np.zeros(L) # initialise
        k = np.sum(mat()==1,0) # this also takes the current valu z_nl into account.
        for l in range(L):
            logit_bernoulli_priors[l]
    
    cf.draw_noparents_onechild(
        mat(), # NxD
        mat.sibling(), # sibling u: D x Lc
        mat.child().transpose(), # child observation: N x Lc
        mat.lbda(), # own parameter: double
        mat.logit_bernoulli_prior,
        mat.sampling_indicator)

def draw_z_oneparent_onechild_wrapper(mat):
    cf.draw_oneparent_onechild(
        mat(), # N x D
        mat.parents[0](), # parent obs: N x Lp
        mat.parents[1](), # parent features, D x Lp
        mat.parents[1].lbdas(), # parent lbda
        mat.sibling(), # sibling u: D x Lc
        mat.child(), # child observation: N x Lc
        mat.lbda(), # own parameter: double
        mat.logit_bernoulli_prior,
        mat.sampling_indicator)

def draw_u_oneparent_onechild_wrapper(mat):

    cf.draw_oneparent_onechild(
        mat(), # NxD
        mat.parents[1](), # parent obs: N x Lp
        mat.parents[0](), # parent features, D x Lp
        mat.parents[1].lbda(), # parent lbda
        mat.sibling(), # sibling u: D x Lc
        mat.child().transpose(), # child observation: N x Lc
        mat.lbda(), # own parameter: double
        mat.logit_bernoulli_prior,
        mat.sampling_indicator)


def draw_lbda_maxmachine_wrapper(parm):
    """
    Set each lambda to its maxmachine mle. TODO: needs optimisation/cythonisation
    """
    assert parm.layer.size == len(parm())-1

    z=parm.attached_matrices[0]
    u=parm.attached_matrices[1]
    x=parm.attached_matrices[0].child
    N = z().shape[0]
    D = u().shape[0]
    L = z().shape[1]

    mask = np.zeros([N,D], dtype=bool)
    l_list = range(L)

    for iter_index in range(L):

        # get positive predictive rate for each code
        # todo: should only compute predictions once! -> but then need to store -> acutally takes longer
        # = TP/ND
        predictions = [cf.predict_single_latent(u()[:,l], z()[:,l])==1 for l in l_list]

        # l_pp_rate = [np.mean(x()[predictions[l] & ~mask] == 1) for l in range(len(l_list))]

        TP = [np.sum(x()[predictions[l] & ~mask] == 1) for l in range(len(l_list))]
        FP = [np.sum(x()[predictions[l] & ~mask] == -1) for l in range(len(l_list))]
        l_pp_rate = [tp/float(tp+fp) for tp, fp in zip(TP, FP)]

        #if np.any(l_pp_rate == < .01):
        #l_pp_rate = [y if not np.isnan(y) else 0.0 for y in l_pp_rate]

        # find l with max predictive power
        l_max_idx = np.argmax(l_pp_rate)
        l_max = l_list[l_max_idx]


        # assign corresponding alpha
        # mind: parm is of (fixed) size L, l_pp_rate gets smaller every iteration
        ## flat prior on alpha
        # parm()[l_max] = np.max([0.0,l_pp_rate[l_max_idx]])
        # parm()[l_max] = TP[l_max_idx]/float(TP[l_max_idx]+FP[l_max_idx])
        ## beta_prior on alpha
        
        parm()[l_max] = ( ( TP[l_max_idx] + parm.beta_prior[0] - 1) /
                  float(TP[l_max_idx] + FP[l_max_idx] +
                        parm.beta_prior[0] + parm.beta_prior[1] - 2) )

        # compute prediction for the current latent dimension
        # prediction = cf.predict_single_latent(u()[:,l_max], z()[:,l_max])# l_predict(l_max,u,z)
        
        
        if np.isnan(parm()[l_max]):
            parm()[l_max] = 0
       
        # mask the predicted values
        # mask_old = mask
        mask +=  predictions[l_max_idx]==1
        # assert np.mean(mask_old) <= np.mean(mask)
            
        # remove the dimenson from l_list
        # l_list = [i for i in l_list if i != l_max]
        l_list = [l_list[i] for i in range(len(l_list)) if i != l_max_idx]

    assert len(l_list) == 0
    
    if np.isnan(parm()[-1]):
        parm()[-1] = 0.0
        Tracer()()
    else:
        P_remain = np.sum(x()[~mask]==1)
        N_remain = np.sum(x()[~mask]==-1)

        # old version does not work with missingd data (x=0)
#        p_old = np.max([0, np.min([.5, np.mean(x()[~mask]==1)] )])
        p_new = ((P_remain + parm.beta_prior_clamped[0] - 1)/
                 float(P_remain + N_remain + parm.beta_prior_clamped[0] +
                       parm.beta_prior_clamped[1] - 2))

        if np.isnan(p_new):
            Tracer()()
        parm()[-1] = p_new
        
def clean_up_codes(layer, noise_model):
    
    def remove_dimension(l_prime, layer):
        u = layer.u; z = layer.z; lbda = layer.lbda
        u.val = np.delete(u.val, l_prime, axis=1)
        z.val = np.delete(z.val, l_prime, axis=1)
        lbda.val = np.delete(u.layer.lbda(), l_prime)
        layer.size -= 1
        for iter_mat in [u,z]:
            if len(iter_mat.parents) != 0:
                for parent in iter_mat.parents:
                    if parent.role == 'features':
                        parent.val = np.delete(parent.val, l_prime, axis=0)
                        if noise_model is 'maxmachine':
                            # binomial prior per code
                            parent.bbp_k = compute_bp(u().shape[0], .1, 10)
                            parent.k = np.array(np.sum(u()==1, 0), dtype=np.int32)    

                            # binomial prior per attribute
                            parent.bbp_j = compute_bp(u().shape[1], .1, 2)
                            parent.j = np.array(np.sum(u()==1, 1), dtype=np.int32)
            
        if noise_model is 'maxmachine': 
            #z.k[l_prime] = 0
            #u.k[l_prime] = 0

            # binomial prior per code
            u.bbp_k = compute_bp(u().shape[0], .1, 10)
            u.k = np.array(np.sum(u()==1, 0), dtype=np.int32)    

            # binomial prior per attribute
            u.bbp_j = compute_bp(u().shape[1], .1, 2)
            u.j = np.array(np.sum(u()==1, 1), dtype=np.int32)

            # dummy prios on z
            z.bbp_k = np.zeros(z().shape[0]+1)
            z.k = np.array(np.sum(z()==1, 0), dtype=np.int32)    

            # dummy prios on z per attribute
            z.bbp_j = np.zeros(z().shape[1]+1)
            z.j = np.array(np.sum(z()==1, 1), dtype=np.int32)


    # clean duplicates
    l = 0
    reduction_applied = False
    while l < layer.size:
        l_prime = l+1
        while l_prime < layer.size:
            if (np.all(layer.u()[:,l] == layer.u()[:,l_prime]) or
            np.all(layer.z()[:,l] == layer.z()[:,l_prime])):
                # print('remove duplicate dimension')
                reduction_applied = True
                remove_dimension(l_prime, layer)
            l_prime += 1
        l += 1

    # clean by alpha threshold
    l = 0
    while l < layer.size:
        if layer.lbda()[l] < 1e-3:
            # print('remove useless dimension')
            reduction_applied = True
            remove_dimension(l, layer)
        l += 1

    return reduction_applied
        
def reset_codes_wrapper(z, u, noise_model):
    if True:
        reset_codes(z, u, noise_model)
        reset_codes(u, z, noise_model)
  
def reset_codes(z, u, noise_model):

    # reset duplicates
    if False:
        for l in range(u().shape[1]):
            for l_prime in range(l+1, u().shape[1]):
                if np.all(u()[:,l] == u()[:,l_prime]):
                    print('reset duplicates')
                    z()[:,l_prime] = -1
                    u()[:,l_prime] = -1
                    if noise_model is 'maxmachine': 
                        #z.k[l_prime] = 0
                        #u.k[l_prime] = 0
                        
                        u.j = np.array(np.sum(u()==1, 1), dtype=np.int32)
                        z.j = np.array(np.sum(z()==1, 1), dtype=np.int32)
                        
                        u.k[l_prime] = 0 # = np.array(np.sum(u()==1, 0), dtype=np.int32)
                        z.k[l_prime] = 0 # np.array(np.sum(z()==1, 0), dtype=np.int32)

    # reset zeroed codes
    if False:
        for l in range(u().shape[1]):
            if np.all(u()[:,l] == -1):
                print('reset zeroed')
                z()[:,l] = -1
                # only needed for binomial / beta-binomial prior
                # TODO implement these prior for ormachine
                if noise_model is 'maxmachine': 
                    z.k[l] = 0
                    z.j = np.array(np.sum(z()==1, 1), dtype=np.int32)

    # reset nested codes
    if False:
        for l in range(u().shape[1]):
            for l_prime in range(u().shape[1]):
                if l == l_prime:
                    continue
                # smaller code needs to have at least one 1.
                elif np.sum(u()[:,l_prime]==1) > 1 and np.all(u()[u()[:,l_prime]==1, l]==1):
                    print('reset nesting '+str(l)+' '+str(l_prime) +
                          ' ' + str(np.sum(u()[:,l]==1)) +
                          ' ' + str(np.sum(u()[:,l_prime]==1)))
                    u()[u()[:,l_prime]==1,l] = -1
                    # z()[z()[:,l]==1,l_prime] = 1

                    if noise_model is 'maxmachine':
                        u.k = np.array(np.sum(u()==1, 0), dtype=np.int32)
                        u.j = np.array(np.sum(u()==1, 1), dtype=np.int32)
                        z.k = np.array(np.sum(z()==1, 0), dtype=np.int32)
                        z.j = np.array(np.sum(z()==1, 1), dtype=np.int32)


def draw_lbda_wrapper(parm):

    P = cf.compute_P_parallel(parm.attached_matrices[0].child(),
                                parm.attached_matrices[1](),
                                parm.attached_matrices[0]())

    # effectie number of observations (precompute for speedup TODO (not crucial))
    ND = (np.prod(parm.attached_matrices[0].child().shape) -\
                    np.sum(parm.attached_matrices[0].child() == 0))

    # set to MLE ( exp(-lbda_mle) = ND/P - 1 )
    if ND==P:
        parm.val = 10e10
    else:
        parm.val = np.max([0, np.min([1000,-np.log(ND/float(P)-1)])])

        
def draw_lbda_indpndt_wrapper(parm):
    
    if parm.layer.predictive_accuracy_updated is False:
        parm.layer.update_predictive_accuracy()

    TP, FP = parm.layer.pred_rates[:2]
    if not FP==0:
        parm()[1] = np.max([0, np.log(TP) - np.log(FP)]) - parm.layer.child.log_bias
    else:
        parm()[1] = 1e3
        
    TN, FN = parm.layer.pred_rates[2:4]
    if not FN==0:
        parm()[0] = np.max([0, np.log(TN) - np.log(FN)]) + parm.layer.child.log_bias
    else:
        parm()[0] = 1e3
        
    return 0


def draw_z_noparents_onechild_indpn_wrapper(mat):

    cf.draw_noparents_onechild_indpn(
        mat(),  # NxD
        mat.sibling(),  # sibling u: D x Lc
        mat.child(),  # child observation: N x Lc
        mat.layer.mu, # mu own parameter: double
        mat.layer.nu, # lbda
        mat.logit_bernoulli_prior,
        mat.sampling_indicator)
    mat.layer.predictive_accuracy_updated = False

    
def draw_u_noparents_onechild_indpn_wrapper(mat):
    
    cf.draw_noparents_onechild_indpn(
        mat(), # NxD
        mat.sibling(), # sibling u: D x Lc
        mat.child().transpose(), # child observation: N x Lc
        mat.layer.mu, # mu own parameter: double
        mat.layer.nu, # lbda
        mat.logit_bernoulli_prior,
        mat.sampling_indicator)
    mat.layer.predictive_accuracy_updated = False

    
# missing: twoparents_onechild, (arbitraryparents_onechild, arbitaryparents_nochild)
# unified wrapper aren't working
def draw_unified_wrapper_z(mat):
    cf.draw_unified(
        mat(), # NxD
        np.array([l.z() for l in mat.parent_layers]), # parents obs: K x N x Lp
        np.array([l.u() for l in mat.parent_layers]), # parents feat: K x D x Lp
        np.array([x.lbda() for x in mat.parent_layers],dtype=float), # parent lbdas: K (KxL for MaxM)
        mat.sibling(), # sibling u: D x Lc
        mat.child(), # child observation: N x Lc
        mat.lbda(), # own parameter: double
        mat.logit_bernoulli_prior,
        mat.sampling_indicator)
    
def draw_unified_wrapper_u(mat):
    cf.draw_unified(
        mat(), # NxD
        np.array([l.u() for l in mat.parent_layers]), # parents obs: K x N x Lp
        np.array([l.z() for l in mat.parent_layers]), # parents feat: K x D x Lp
        np.array([x.lbda() for x in mat.parent_layers],dtype=float), # parent lbdas: K (KxL for MaxM)
        mat.sibling(), # sibling u: D x Lc
        mat.child().transpose(), # child observation: N x Lc
        mat.lbda(), # own parameter: double
        mat.logit_bernoulli_prior,
        mat.sampling_indicator)


def compute_bp(n, q, tau=1):
    exp_bp = [(q*(n-k*tau)) / ((1-q)*(k*tau+1)) for k in range(n+1)]
    bp = [np.log(x) if (x > 0) else -np.infty for x in exp_bp]
    return np.array(bp, dtype=float)    
