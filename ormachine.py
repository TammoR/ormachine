import numpy as np
from numpy.random import binomial
#import cython
from random import shuffle
# from scipy.special import expit
import cython_functions as cf
import unittest

def expit(x):
    """
    better implementation in scipy.special, 
    but can avoid dependency
    """
    return 1/(1+np.exp(-x))


class trace():
    """
    abstract base class implementing methods for
    arrays of posterior traces,
    inherits to machine matrix and lbda instances.
    """
    
    def __call__(self):
        return self.val
        
    def allocate_trace_arrays(self, no_of_samples):
        if type(self.val) == np.ndarray:
            self.trace = np.empty([no_of_samples, *self.val.shape], dtype=np.int8)
        else:
            self.trace = np.empty([no_of_samples], dtype=np.float32)
        
    def update_trace(self):
        self.trace[self.trace_index] = self.val
        self.trace_index += 1
        
    def mean(self):
        return np.mean(self.trace, axis=0)
    
    def check_convergence(self, eps):
        """
        split trace in half and check difference between means < eps
        """
        l = int(len(self.trace)/2)
        r1 = expit(np.mean(self.trace[:l]))
        r2 = expit(np.mean(self.trace[l:]))
        r = expit(np.mean(self.trace))
        
        if np.abs(r1-r2) < eps:
            return True
        else:
            # print('reconstr. accuracy: '+str(r))
            return False

        
class machine_parameter(trace):
    """
    Parameters are attached to corresponding matrices
    """
    
    def __init__(self, val, attached_matrices=None, sampling_indicator=True):
        self.val = val
        self.attached_matrices = attached_matrices
        self.correct_role_order()
        # make sure they are a tuple in order (observations/features)
        self.sampling_indicator = sampling_indicator
        self.trace_index = 0
        
        # assign matrices the parameter
        for mat in attached_matrices:
            mat.lbda = self
            
    def correct_role_order(self):
        """
        the assigne matrices have to be ordere (observations,features),
        i.e. (z,u)
        """
        
        if self.attached_matrices[0].role == 'features'\
        and self.attached_matrices[1].role == 'observations':
            self.attached_matrices = (self.attached_matrices[1],self.attached_matrices[0])
            print('swapped role order')
        elif self.attached_matrices[0].role == 'observations'\
        and self.attached_matrices[1].role == 'features':
            pass
        else:
            raise ValueError('something is wrong with the role assignments')
            
    def update(self):
        """
        set lbda to its MLE
        """
        
        # compute the number of correctly reconstructed data points
        P = cf.compute_P_parallel(self.attached_matrices[0].child(),
                               self.attached_matrices[1](),
                               self.attached_matrices[0]())
        
        # effectie number of observations (precompute for speedup TODO (not crucial))
        ND = (np.prod(self.attached_matrices[0].child().shape) -\
                    np.sum(self.attached_matrices[0].child() == 0))
        
        # set to MLE ( exp(-lbda_mle) = ND/P - 1 )
#         print('\n'+str(arg)+'\n')
        if ND==P:
            self.val = 10e10
        else:
            self.val = np.max([0, np.min([1000, -np.log( ND / float(P) - 1)])])
            
        

class machine_matrix(trace):
    """
    all matrices data/factor inherit from this class
    access states, sampling traces
    track parents, sibling relationship
    class variable tracking no of matrices
    @classmethods needed?
    """
    
    def __init__(self, val=None, shape=None, sibling=None, 
                 parents=None, child=None, lbda=None, bernoulli_prior=.5,
                 p_init=.5, role=None, sampling_indicator = True,
                 parent_layers = None):
        """
        role (str): 'features' or 'observations' or 'data'. Try to infer if not provided
        """
        
        self.trace_index = 0
        
        # never use empty lists as default arguments. bad things will happen
        if parents is None:
            parents = []
        if parent_layers is None:
            parent_layers = []
        self.parent_layers = parent_layers

        # assign family
        self.parents = parents
        self.child = child
        self.sibling = sibling
        self.lbda = lbda
        
        
        # assign prior
        self.set_prior(bernoulli_prior)

        # ascertain that we have enough information to initiliase the matrix
        assert (val is not None) or (shape is not None and p_init is not None)
        
        # initiliase matrix. TODO sanity checks for matrices (-1,1?)
        if val is not None:
            self.val = np.array(val, dtype=np.int8)
        else:
            # assign random matrix
            # print('initialise matrix randomly')
            self.val = 2*np.array(binomial(n=1, p=p_init, size=shape), dtype=np.int8)-1

        self.family_setup()
        
        if role is None:
            self.infer_role()
        else:
            self.role = role
            
        # sampling indicator is boolean matrix of the same size, type=np.int8
        self.set_sampling_indicator(sampling_indicator)
        
    @staticmethod
    def layer():
        print('no corresponding layer defined')
        
    def add_parent_layer(self, layer):
        if self.parent_layers is None:
            self.parent_layers = [layer]
        else:
            self.parent_layers.append(layer)
        
    def set_sampling_indicator(self, sampling_indicator):
        """
        matrix of same size, indiactor for every value whether it stays fixed
        or is updated. Could hardcode this, if all or no values are samples
        but speed advantage is negligible.
        """
        if sampling_indicator is True:
            self.sampling_indicator = np.ones(self().shape, dtype=np.int8)
        elif sampling_indicator is False:
            self.sampling_indicator = np.zeros(self().shape, dtype=np.int8)
        elif type(sampling_indicator) is np.ndarray:
            assert sampling_indicator.shape == self().shape
            self.sampling_indicator = np.array(sampling_indicator, dtype=np.int8)

    def set_prior(self, bernoulli_prior=None):
        self.bernoulli_prior = bernoulli_prior
        
        if bernoulli_prior is None:
            self.logit_bernoulli_prior = 0
        else:
            self.logit_bernoulli_prior = np.log(bernoulli_prior/(1-bernoulli_prior))
    
    def family_setup(self):
        """
        Set family relationship of relatives
        Mind: every parent has only one child (no multiview), a child can have many parents.
        Every sibling has only one sibling
        """
        # register as child of all my parents
        if self.parents is not None:
            for parent in self.parents:
                # check whether tere is another child registered
                assert (parent.child is not None and parent.child != self)
                parent.child = self

        # register as parent of all my children
        if (self.child is not None) and (self not in self.child.parents):

            self.child.parents.append(self)

        # register as sibling of all my siblings
        if (self.sibling is not None) and (self != self.sibling.sibling):
            self.sibling.sibling = self
            
        # register as attached to my parameter
        if self.lbda is not None:
            if self.lbda.attached_matrices is None:
                self.lbda.attached_matrices = [self]
            elif type(self.lbda.attached_matrices == list) and (self not in self.lbda.attached_matrices):
                self.lbda.attached_matrices.append(self)
                
    def infer_role(self):
        """
        based on the dimensionality, infer whether self is a
        observation (z) or a feature matrix (u) or a data matrix (x)
        """
        
        # print('infer roles of matrix (data/observations/features)')
        
        if self.child is None:
            self.role = 'observations'
            return 

        # if dimensions N=D, role can't be inferred
        if self.child().shape[0] == self.child().shape[1]:
            raise ValueError('Data dimensions are the same. Can not infer parents role.')
        
        if self().shape[0] == self.child().shape[0]:
            self.role = 'observations'
        elif self().shape[0] == self.child().shape[1]:
            self.role = 'features'
        else:
            raise ValueError('dimension mismatch')
            
    def set_to_map(self):
        self.val = np.array(self.mean()>0, dtype=np.int8)
        self.val[self.val==0] = -1
                        
    def get_sampling_fct(self):
        """
        Return appropriate sampling function wrapper, depending
        on family status, sampling status etc.
        """
        # first do some sanity checks, no of children etc. Todo

        # matrix without parents
        if not self.parents:
            if self.role == 'observations':
                return draw_unified_z_noparents_wrapper
            elif self.role == 'features':
                return draw_unified_u_noparents_wrapper

        # matrix without child
        if not self.child:
            if self.role == 'observations':
                return draw_unified_z_nochild_wrapper
            elif self.role == 'features':
                return draw_unified_u_nochild_wrapper

        # matrix with child and parent
        else:
            if self.role == 'observations':
                return draw_unified_wrapper_z
            elif self.role == 'features':
                return draw_unified_wrapper_u


class machine_layer():
    """
    Essentially a container for (z,u,lbda) to allow
    convenient access to different layers for the sampling routine
    and the user.
    """
    
    def __init__(self, z, u, lbda, size, child):
        self.z = z
        self.u = u
        self.lbda = lbda
        self.size = size
        # register as layer of members
        self.z.layer = self
        self.u.layer = self
        self.lbda.layer = self
        self.child = child
        self.child.add_parent_layer(self)
        
    def __call__(self):
        return(self.z(), self.u(), self.lbda())
        
    def members(self):
        return [self.z, self.u]
    
    def lbda(self):
        return self.lbda
    
    def child(self):
        assert self.z.child is self.u.child
        return self.z.child


class machine():
    """
    main class
    attach factor_matrices to this class
    methods:
    
    add_data(data, infer=False, parents=None)
    add_layer(L, infer=True, children=None, parents=None)    
    """

    def __init__(self):
        self.layers = [] # list of dictionaries with factor matrices and parameters, this is for convenivence only
        self.members = [] #
        self.lbdas = []

        
    def add_matrix(self, val=None, shape=None, sibling=None,
                   parents=None, child=None, lbda=None, p_init=.5,
                   role=None, sampling_indicator=True, bernoulli_prior=None):

        mat = machine_matrix(val, shape, sibling, parents, child, 
                             lbda, bernoulli_prior, p_init,
                             role, sampling_indicator)

        self.members.append(mat)
        return mat
        
        
    def add_parameter(self, val=2, attached_matrices=None):
        
        lbda = machine_parameter(val=val, attached_matrices=attached_matrices)
        self.lbdas.append(lbda)
        
        return lbda
    
        
    def add_layer(self, size, child=None, 
                  lbda_init=2, z_init=.5, u_init=.5, 
                  z_prior=None, u_prior=None):
        """
        This essentially wraps the necessary calls to
        add_parameter, add_matrix
        """
        z = self.add_matrix(shape=(child().shape[0], size), 
                            child=child, p_init=z_init, bernoulli_prior=z_prior,
                            role='observations')
        
        u = self.add_matrix(shape=(child().shape[1], size), sibling=z, 
                            child=child, p_init=u_init, bernoulli_prior=u_prior,
                            role='features')
        
        lbda = self.add_parameter(attached_matrices=(z,u), val=lbda_init)
        
        layer = machine_layer(z, u, lbda, size, child)
        
        self.layers.append(layer)
        
        return layer
    
        
    def burn_in(self, sampling_tuples, sampling_lbdas, eps=1e-2, 
                convergence_window=20, burn_in_min=100, burn_in_max=20000):
        
        
        # allocate array for lamabda traces for burn in detection
        for lbda in sampling_lbdas:
            lbda.allocate_trace_arrays(2*convergence_window)
        
        # first sample without checking for convergence, if burn_in_min > 0
        pre_burn_in_iter = 0
        while True:
            pre_burn_in_iter += 1
            if pre_burn_in_iter%10 == 0:
                print('\r\titeration: '+str(pre_burn_in_iter),end='')
            
            if pre_burn_in_iter == burn_in_min:
                break
            else:
                [s[0](s[1]) for s in sampling_tuples]
                [x.update() for x in sampling_lbdas]
                shuffle(sampling_tuples)
                
        # now check for convergence
        burn_in_iter = 0
        while True:
            burn_in_iter += 1
            if (burn_in_iter)%10 == 0:
                print('\r\titeration: '+str(pre_burn_in_iter+burn_in_iter),end='')
            # sample matrices
            [s[0](s[1]) for s in sampling_tuples] # TODO shuffle order or by size? test with problematic cases
            # update lbda
            [x.update() for x in sampling_lbdas]
            # update lbda trace
            [x.update_trace() for x in sampling_lbdas]
            shuffle(sampling_tuples)
                
            # check convergence every convergence_window iterations
            if (burn_in_iter%convergence_window) == 0 and (burn_in_iter > convergence_window):
                
                # reset trace index
                for lbda in sampling_lbdas:
                    lbda.trace_index = 0
                
                # check convergence for all lbdas
                if np.all([x.check_convergence(eps=eps) for x in sampling_lbdas]):
                    print('\n\tconverged at reconstr. accuracy: ' +
                          str([expit(np.mean(x.trace)) for x in sampling_lbdas]))
                    break;
            
            if burn_in_iter > burn_in_max:
                print('\n\tmax burn-in iterations reached without convergence')
                # reset trace index
                for lbda in sampling_lbdas:
                    lbda.trace_index = 0
                break;
        
    
    def infer(self, mats='all', no_samples=100, 
               convergence_window=20, convergence_eps=1e-2,
               burn_in_min=100, burn_in_max=20000):
        """
        members can be a list of machine_layers or machine_matrices
        or a mix of both types.
        eps is on expit scale, i.e is the fractional reconstr. accuracy
        """

        # create list of tuples of the form (sampling_function, mat)
        if mats == 'all':
            sampling_tuples = [(x.get_sampling_fct(),x) 
                                for x in self.members 
                                if not np.all(x.sampling_indicator==0)]
        elif type(mats) is list:
            sampling_tuples = [(x.get_sampling_fct(),x) 
                                for x in mats if not np.all(x.sampling_indicator==0)]
        else:
            raise TypeError('Sampling matrices not properly specified. Should be list of matrix objects.')
        
        # a list of all parameters (lbdas) that need to be updated
        sampling_lbdas = [x[1].lbda for x in sampling_tuples] # own lbdas
        sampling_lbdas_pa = []
        for x in sampling_tuples:
            if len(x[1].parents) > 0:
                sampling_lbdas_pa.append(x[1].parents[0].lbda)
        sampling_lbdas = list(set(sampling_lbdas).union(set(sampling_lbdas_pa)))
        sampling_lbdas = [x for x in sampling_lbdas if x is not None]

        # make sure all trace indicies are zero
        for s in sampling_tuples:
            s[1].trace_index = 0
        for s in sampling_lbdas:
            if s is not None:
                s.trace_index = 0
        
        # burn in markov chain # TODO build in stopping mechanism if it does not converge
        print('burning in markov chain...')
        self.burn_in(sampling_tuples, sampling_lbdas, 
                     eps=convergence_eps, convergence_window=convergence_window,
                     burn_in_min=burn_in_min, burn_in_max=burn_in_max)        
        
        # allocate memory to save samples
        print('allocating memory to save samples...')
        for _, mat in sampling_tuples:
            mat.allocate_trace_arrays(no_samples)
        for lbda in sampling_lbdas:
            lbda.allocate_trace_arrays(no_samples)
        
        print('drawing samples...')
        for sampling_iter in range(1, no_samples+1):
            shuffle(sampling_tuples) # TODO
            
            [s[0](s[1]) for s in sampling_tuples]
            [s[1].update_trace() for s in sampling_tuples]
            
            [x.update() for x in sampling_lbdas]
            [x.update_trace() for x in sampling_lbdas]
            
            if (sampling_iter)%10 == 0:
                print('\r\t' + 'iteration ' + str(sampling_iter) + '; recon acc.: ' +
                  str([round(expit(x()),2) for x in sampling_lbdas]), end='')
            
        # set all parameters to MAP estimate
        [x.set_to_map() for _,x in sampling_tuples]
        [x.update() for x in sampling_lbdas]
        print('\nfinished.')


def draw_unified_z_noparents_wrapper(mat):
    
    cf.draw_unified_noparents(mat(), # NxD
                 mat.sibling(), # sibling u: D x Lc
                 mat.child(), # child observation: N x Lc
                 mat.lbda(), # own parameter: double
                 mat.logit_bernoulli_prior,
                 mat.sampling_indicator)
    
def draw_unified_u_noparents_wrapper(mat):
    
    cf.draw_unified_noparents(mat(), # NxD
                 mat.sibling(), # sibling u: D x Lc
                 mat.child().transpose(), # child observation: N x Lc
                 mat.lbda(), # own parameter: double
                 mat.logit_bernoulli_prior,
                 mat.sampling_indicator)
    
def draw_unified_z_nochild_wrapper(mat):
    
    cf.draw_unified_nochild(mat(), # NxD
                 np.array([l.z() for l in mat.parent_layers]), # parents obs: K x N x Lp
                 np.array([l.u() for l in mat.parent_layers]), # parents feat: K x D x Lp
                 np.array([x.lbda() for x in mat.parent_layers],dtype=float), # parent lbdas: K (KxL for MaxM)
                 mat.logit_bernoulli_prior,
                 mat.sampling_indicator)
    
def draw_unified_u_nochild_wrapper(mat):
    
    cf.draw_unified_nochild(mat(), # NxD
                 np.array([l.u() for l in mat.parent_layers]), # parents obs: K x N x Lp
                 np.array([l.z() for l in mat.parent_layers]), # parents feat: K x D x Lp
                 np.array([x.lbda() for x in mat.parent_layers],dtype=float), # parent lbdas: K (KxL for MaxM)
                 mat.logit_bernoulli_prior,
                 mat.sampling_indicator)    
    
def draw_unified_wrapper_z(mat):
    
    cf.draw_unified(mat(), # NxD
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
