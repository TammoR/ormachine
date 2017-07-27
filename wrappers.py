import cython_functions as cf
import numpy as np

    
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
        mat.lbdas[0](),  # own parameter: double
        mat.logit_bernoulli_prior,
        mat.density_conditions,
        mat.sampling_indicator)
    
def draw_u_noparents_onechild_maxdens_wrapper(mat):
    
    cf.draw_noparents_onechild_maxdens(
        mat(), # NxD
        mat.sibling(), # sibling u: D x Lc
        mat.child().transpose(), # child observation: N x Lc
        mat.lbdas[0](), # own parameter: double
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
        mat.lbdas[0](), # own parameter: double
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
        mat.lbdas[0](), # own parameter: double
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
        mat.parent_layers[0].u.lbdas[0](), # parent lbda
        mat.parent_layers[1].z(), # parents obs: N x Lp
        mat.parent_layers[1].u(), # parents feat: D x Lp
        mat.parent_layers[1].u.lbdas[0](), # parent lbda
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
        mat.lbdas[0](),  # own parameter: double
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
        mat.lbdas[0](), # own parameter: double
        mat.logit_bernoulli_prior,
        mat.sampling_indicator)

def draw_z_oneparent_onechild_wrapper(mat):
    cf.draw_oneparent_onechild(
        mat(), # N x D
        mat.parents[0](), # parent obs: N x Lp
        mat.parents[1](), # parent features, D x Lp
        mat.parents[1].lbdas[0](), # parent lbda
        mat.sibling(), # sibling u: D x Lc
        mat.child(), # child observation: N x Lc
        mat.lbdas[0](), # own parameter: double
        mat.logit_bernoulli_prior,
        mat.sampling_indicator)

def draw_u_oneparent_onechild_wrapper(mat):

    cf.draw_oneparent_onechild(
        mat(), # NxD
        mat.parents[1](), # parent obs: N x Lp
        mat.parents[0](), # parent features, D x Lp
        mat.parents[1].lbdas[0](), # parent lbda
        mat.sibling(), # sibling u: D x Lc
        mat.child().transpose(), # child observation: N x Lc
        mat.lbdas[0](), # own parameter: double
        mat.logit_bernoulli_prior,
        mat.sampling_indicator)



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
        
    if parm is parm.layer.lbda:
         # TODO can save from overflow? more efficient?
        TP, FP = parm.layer.pred_rates[:2]
        if not FP==0:
            parm.val = np.max([0, np.log(TP) - np.log(FP)]) - parm.layer.child.log_bias
        else:
            parm.val = 1e3
    elif parm is parm.layer.mu:
        TN, FN = parm.layer.pred_rates[2:4]
        if not FN==0:
            parm.val = np.max([0, np.log(TN) - np.log(FN)]) + parm.layer.child.log_bias
        else:
            parm.val = 1e3
        
    return 0


def draw_z_noparents_onechild_indpn_wrapper(mat):

    cf.draw_noparents_onechild_indpn(
        mat(),  # NxD
        mat.sibling(),  # sibling u: D x Lc
        mat.child(),  # child observation: N x Lc
        mat.lbdas[0](), # mu own parameter: double
        mat.lbdas[1](), # lbda
        mat.logit_bernoulli_prior,
        mat.sampling_indicator)
    mat.layer.predictive_accuracy_updated = False

    
def draw_u_noparents_onechild_indpn_wrapper(mat):
    
    cf.draw_noparents_onechild_indpn(
        mat(), # NxD
        mat.sibling(), # sibling u: D x Lc
        mat.child().transpose(), # child observation: N x Lc
        mat.lbdas[0](), # mu own parameter: double
        mat.lbdas[1](), # lbda
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
        mat.lbdas[0](), # own parameter: double
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
        mat.lbdas[0](), # own parameter: double
        mat.logit_bernoulli_prior,
        mat.sampling_indicator)

