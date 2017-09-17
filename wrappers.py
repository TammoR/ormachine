import cython_functions as cf
import numpy as np
from IPython.core.debugger import Tracer
import warnings


def draw_z_oneparent_nochild_wrapper(mat):
    
    cf.draw_oneparent_nochild(
        mat(), # NxD
        mat.parents[0](), # parents obs: N x Lp
        mat.parents[1](), # parents feat: D x Lp
        mat.parents[0].lbda(), # parent lbdas:
        mat.prior_config,
        mat.sampling_indicator)
    
def draw_u_oneparent_nochild_wrapper(mat):
    
    cf.draw_oneparent_nochild(
        mat(), # NxD
        mat.parents[1](), # parents obs: N x Lp
        mat.parents[0](), # parents feat: D x Lp
        mat.parents[0].lbda(), # parent lbdas: K (KxL for MaxM)
        mat.prior_config,        
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
        mat.prior_config,        
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
        mat.prior_config,        
        mat.sampling_indicator)        

def draw_z_noparents_onechild_wrapper(mat):

    cf.draw_noparents_onechild(
        mat(),  # NxD
        mat.sibling(),  # sibling u: D x Lc
        mat.child(),  # child observation: N x Lc
        mat.lbda(),  # own parameter: double
        mat.prior_config,
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
        mat.prior_config,
        mat.sampling_indicator)

def draw_z_oneparent_onechild_wrapper(mat):
    cf.draw_oneparent_onechild(
        mat(), # N x D
        mat.parents[0](), # parent obs: N x Lp
        mat.parents[1](), # parent features, D x Lp
        mat.parents[1].lbda(), # parent lbda
        mat.sibling(), # sibling u: D x Lc
        mat.child(), # child observation: N x Lc
        mat.lbda(), # own parameter: double
        mat.prior_config,
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
        mat.prior_config,
        mat.sampling_indicator)


def draw_lbda_maxmachine_wrapper(parm):
    """
    Set each lambda to its maxmachine mle. 
    This should be cythonised, but contains nasty things like argsort.
    """

    z=parm.attached_matrices[0]
    u=parm.attached_matrices[1]
    x=parm.attached_matrices[0].child
    N = z().shape[0]
    D = u().shape[0]
    L = z().shape[1]

    mask = np.zeros([N,D], dtype=bool)
    l_list = range(L)
    
    predictions = [cf.predict_single_latent(u()[:,l], z()[:,l])==1 for l in l_list]

    TP2 = [np.count_nonzero(x()[predictions[l]] == 1) for l in range(L)]
    FP2 = [np.count_nonzero(x()[predictions[l]] == -1) for l in range(L)]

    for iter_index in range(L):

        # use Laplace rule of succession here, to avoid numerical issues
        l_pp_rate = [(tp+1)/float(tp+fp+2) for tp, fp in zip(TP2, FP2)]        

        # find l with max predictive power
        l_max_idx = np.argmax(l_pp_rate)
        l_max = l_list[l_max_idx]

        # assign corresponding alpha
        # mind: parm is of (fixed) size L, l_pp_rate gets smaller every iteration
        if parm.prior_config[0] == 0:
            # again, using Laplace rule of succession
            parm()[l_max] = ( ( TP2[l_max_idx] + 1 ) / float(TP2[l_max_idx] + FP2[l_max_idx] + 2 ) )

        elif parm.prior_config[0] == 1:
            alpha = parm.prior_config[1][0]
            beta  = parm.prior_config[1][1]
            parm()[l_max] = ( ( TP2[l_max_idx] + alpha - 1) /
                              float(TP2[l_max_idx] + FP2[l_max_idx] + alpha + beta - 2) )

        # remove the dimenson from l_list
        l_list = [l_list[i] for i in range(len(l_list)) if i != l_max_idx]

        # the following large binary arrays need to be computed L times -> precompute here
        temp_array = predictions[l_max] & ~mask
        temp_array1 = temp_array & (x()==1)
        temp_array2 = temp_array & (x()==-1)
        
        TP2 = [TP2[l + (l >= l_max_idx)] - np.count_nonzero(temp_array1 & predictions[l_list[l]])
               for l in range(len(l_list))]
        FP2 = [FP2[l + (l >= l_max_idx)] - np.count_nonzero(temp_array2 & predictions[l_list[l]])
               for l in range(len(l_list))]

        mask += predictions[l_max]==1
        
    assert len(l_list) == 0
    
    # if np.isnan(parm()[-1]):
    #     Tracer()()
    #     warnings.warn('Noise parameter was nan, setting to 0.')         
    #     parm()[-1] = 0.0

    P_remain = np.count_nonzero(x()[~mask]==1)
    N_remain = np.count_nonzero(x()[~mask]==-1)

    if parm.prior_config[0] == 1:
        alpha = parm.prior_config[2][0]
        beta  = parm.prior_config[2][1]        
        p_new = (P_remain + alpha - 1)/float(P_remain + N_remain + alpha + beta - 2)
    elif parm.prior_config[0] == 0:
        p_new = (P_remain + 1)/float(P_remain + N_remain + 2) # get into trouble here if P=N=0

    if np.isnan(p_new):
        Tracer()()
        warnings.warn('Noise parameter was nan, setting to 0.')             
        parm()[-1] = p_new            

    parm()[-1] = p_new

    parm.layer.precompute_lbda_ratios()
    
        
def clean_up_codes(layer, noise_model):
    
    def remove_dimension(l_prime, layer):
        u = layer.u; z = layer.z; lbda = layer.lbda
        u.val = np.delete(u.val, l_prime, axis=1)
        z.val = np.delete(z.val, l_prime, axis=1)
        layer.size -= 1
        if layer.noise_model == 'maxmachine':
            lbda.val = np.delete(u.layer.lbda(), l_prime)
            layer.precompute_lbda_ratios()
        z.update_prior_config()
        u.update_prior_config()
        for iter_mat in [u,z]:
            if len(iter_mat.parents) != 0:
                for parent in iter_mat.parents:
                    if parent.role == 'features':
                        parent.val = np.delete(parent.val, l_prime, axis=0)
                        parent.update_prior_config()

    reduction_applied = False                        
    # remove inactive codes
    l = 0
    while l < layer.size:
        if np.all(layer.z()[:,l] == -1) or np.all(layer.u()[:,l] == -1):
            # print('remove inactive dimension')
            reduction_applied = True
            remove_dimension(l, layer)
        l += 1
                        
    # clean duplicates
    l = 0
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
    if layer.noise_model == 'maxmachine':
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
                        
                        u.j = np.array(np.count_nonzero(u()==1, 1), dtype=np.int32)
                        z.j = np.array(np.count_nonzero(z()==1, 1), dtype=np.int32)
                        
                        u.k[l_prime] = 0 # = np.array(np.count_nonzero(u()==1, 0), dtype=np.int32)
                        z.k[l_prime] = 0 # np.array(np.count_nonzero(z()==1, 0), dtype=np.int32)

    # reset zeroed codes
    if False:
        for l in range(u().shape[1]):
            if np.all(u()[:,l] == -1):
                print('reset zeroed')
                z()[:,l] = -1
                # only needed for binomial / beta-binomial prior
                # TODO implement these prior for ormgachine
                if noise_model is 'maxmachine': 
                    z.k[l] = 0
                    z.j = np.array(np.count_nonzero(z()==1, 1), dtype=np.int32)

    # reset nested codes
    if False:
        for l in range(u().shape[1]):
            for l_prime in range(u().shape[1]):
                if l == l_prime:
                    continue
                # smaller code needs to have at least one 1.
                elif np.count_nonzero(u()[:,l_prime]==1) > 1 and np.all(u()[u()[:,l_prime]==1, l]==1):
                    print('reset nesting '+str(l)+' '+str(l_prime) +
                          ' ' + str(np.count_nonzero(u()[:,l]==1)) +
                          ' ' + str(np.count_nonzero(u()[:,l_prime]==1)))
                    u()[u()[:,l_prime]==1,l] = -1
                    # z()[z()[:,l]==1,l_prime] = 1

                    if noise_model is 'maxmachine':
                        u.k = np.array(np.count_nonzero(u()==1, 0), dtype=np.int32)
                        u.j = np.array(np.count_nonzero(u()==1, 1), dtype=np.int32)
                        z.k = np.array(np.count_nonzero(z()==1, 0), dtype=np.int32)
                        z.j = np.array(np.count_nonzero(z()==1, 1), dtype=np.int32)


def draw_lbda_wrapper(parm):
    from scipy.special import logit
    P = cf.compute_P_parallel(parm.attached_matrices[1].child(),
                              parm.attached_matrices[1](),
                              parm.attached_matrices[0]())

    # effectie number of observations (precompute for speedup TODO (not crucial))
    ND = (np.prod(parm.attached_matrices[0].child().shape) -\
                    np.count_nonzero(parm.attached_matrices[0].child() == 0))

    # Flat prior
    if parm.prior_config[0] == 0:
        # use Laplace rule of succession
        parm.val = -np.log( ( (ND) / (float(P) ) ) - 1 )
        #parm.val = np.max([0, np.min([1000, -np.log( (ND) / (float(P)-1) )])])
        
    # Beta prior
    elif parm.prior_config[0] == 1:
        alpha = parm.prior_config[1][0]
        beta  = parm.prior_config[1][1]
        parm.val = -np.log( (ND + alpha - 1) / (float(P) + alpha + beta - 2) - 1 )        


def draw_u_noparents_onechild_maxmachine(mat):
    ## order by accuracy of code
    l_statistic = mat.layer.lbda()[:-1]    
    
    ## order by number of predicted data-points
    # l_statistic = np.sum(mat.layer.z()==1,0)*np.sum(mat.layer.u()==1,0)

    ## order by usage
    # l_statistic = np.sum(mat.layer.z()==1,0)
    
    ## order by no of accurately predicted data points
    # l_statistic = mat.layer.lbda()[:-1] * np.sum(mat.layer.z()==1,0)*np.sum(mat.layer.u()==1,0)
    
    l_order = np.array(np.argsort(-l_statistic), dtype=np.int32)
    # l_order = np.array(range(len(l_statistic)), dtype=np.int32)
    # l_order = np.array([1,0,2], dtype=np.int32)
    # print(l_order)
    
    cf.draw_noparents_onechild_maxmachine(
        mat(), 
        mat.sibling(),
        mat.child().transpose(),
        mat.lbda(),
        l_order,
        mat.prior_config,
        mat.layer.lbda_ratios)


def draw_z_noparents_onechild_maxmachine(mat):           
    ## order by accuracy of code
    l_order = np.array(np.argsort(-mat.layer.lbda()[:-1]), dtype=np.int32)
    
    cf.draw_noparents_onechild_maxmachine(
        mat(),
        mat.sibling(),
        mat.child(),
        mat.lbda(),
        l_order,
        mat.prior_config,
        mat.layer.lbda_ratios)

            
def draw_u_oneparent_onechild_maxmachine(mat):
    l_order = np.array(np.argsort(-mat.layer.lbda()[:-1]), dtype=np.int32)
    l_order_pa = np.array(np.argsort(-mat.parent_layers[0].lbda()[:-1]), dtype=np.int32)
    from scipy.special import logit
    
    cf.draw_oneparent_onechild_maxmachine(
        mat(),
        mat.sibling(),
        mat.child().transpose,
        mat.lbda(),
        l_order,
        mat.prior_config,
        mat.parents[0](),
        mat.parents[1](),
        logit(mat.parent_layers[0].lbda()), # TODO compute logit inside function
        l_order_pa,
        mat.layer.lbda_ratios)    
            
        
def draw_z_oneparent_onechild_maxmachine(mat):
    l_order = np.array(np.argsort(-mat.layer.lbda()[:-1]), dtype=np.int32)
    l_order_pa = np.array(np.argsort(-mat.parent_layers[0].lbda()[:-1]), dtype=np.int32)
    from scipy.special import logit
    
    cf.draw_oneparent_onechild_maxmachine(
        mat(),
        mat.sibling(),
        mat.child(),
        mat.lbda(),
        l_order,
        mat.prior_config,
        mat.parents[1](),
        mat.parents[0](),
        logit(mat.parent_layers[0].lbda()), # TODO compute logit inside function
        l_order_pa,
        mat.layer.lbda_ratios)    
            
    
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
    """
    Compute binomial logit
    """
    exp_bp = [(q*(n-k*tau)) / ((1-q)*(k*tau+1)) for k in range(n+1)]
    bp = [np.log(x) if (x > 0) else -np.infty for x in exp_bp]
    return np.array(bp, dtype=float)    


def compute_bbp(n, a, b, tau=1):
    """
    Compute beta-binomial logit
    """
    exp_bbp = [(float((n-k*tau)*(k*tau+a))/float((k*tau+1)* (n-k*tau+b-1))) for k in range(n+1)]
    bbp = [np.log(x) if (x > 0) else -np.infty for x in exp_bbp]
    return np.array(bbp, dtype=float)

