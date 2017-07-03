from __future__ import absolute_import, division, print_function # for python2 support
import numpy as np
from numpy.random import binomial
#import cython
from random import shuffle
# from scipy.special import expit
import cython_functions as cf
import unittest
from IPython.core.debugger import Tracer

def expit(x):
    """
    better implementation in scipy.special, 
    but can avoid dependency
    """
    return 1/(1+np.exp(-x))

def unique_ordered(seq):
    """
    return unique list entries preserving order.
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

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
            # nicer but no python2 compatible
            # self.trace = np.empty([no_of_samples, *self.val.shape], dtype=np.int8)        
            self.trace = np.empty([no_of_samples]+[x for x in self.val.shape], dtype=np.int8)
        else:
            self.trace = np.empty([no_of_samples], dtype=np.float32)
        
    def update_trace(self):
        self.trace[self.trace_index] = self.val
        self.trace_index += 1
        
    def mean(self):
        if 'trace' in dir(self):
            return np.mean(self.trace, axis=0)
        # if no trace is defined, return current state
        else:
            return self()
    
    def check_convergence(self, eps):
        """
        split trace in half and check difference between means < eps
        """
        l = int(len(self.trace)/2)
        r1 = expit(np.mean(self.trace[:l]))
        r2 = expit(np.mean(self.trace[l:]))
        r = expit(np.mean(self.trace))

        # print('\n\n',r1\,'\n',r2,'\n',np.abs(r1-r2))
        
        if np.abs(r1-r2) < eps:
            return True
        else:
            # print('reconstr. accuracy: '+str(r))
            return False

    def set_sampling_fct(self, sampling_fct=None):

        if self.sampling_fct is not None:
            return
        
        # if user provides function, us it
        if sampling_fct is not None:
            self.sampling_fct = sampling_fct

        # otherwise infer it
        else:
            self.infer_sampling_fct()

class machine_parameter(trace):
    """
    Parameters are attached to corresponding matrices
    """
    
    def __init__(self, val, attached_matrices=None, sampling_indicator=True):
        self.trace_index = 0
        self.sampling_fct = None
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
        the assigne matrices have to be ordered (observations,features),
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

    def infer_sampling_fct(self):
        """
        Can interface other functions here easily.
        """
        self.sampling_fct = draw_lbda_wrapper

    def update(self):
        """
        set lbda to its MLE
        todo: make this completely modular like the matrix sampling functions.
        then we can update using priors and implement the maxmachine.
        """
        
        # compute the number of correctly reconstructed data points

        if False:
            x=.5*(self.attached_matrices[0].child()+1)
            u=.5*(self.attached_matrices[1]()+1)
            z=.5*(self.attached_matrices[0]()+1)
            prod = np.dot(z, u.transpose()) > 0

            assert prod.shape == x.shape
            print('\n\n', np.mean(prod == x))
            print(x[x!=prod])

            Tracer()()
              
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
                 density_conditions=None, p_init=.5, role=None,
                 sampling_indicator = True, parent_layers = None):
        """
        role (str): 'features' or 'observations' or 'data'. Try to infer if not provided
        """

        self.trace_index = 0
        self.sampling_fct = None
        
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
        self.set_prior(bernoulli_prior, density_conditions)

        # ascertain that we have enough information to initiliase the matrix
        assert (val is not None) or (shape is not None and p_init is not None)
        
        # initiliase matrix. TODO sanity checks for matrices (-1,1?)

        # if value is given, assign
        if val is not None:
            self.val = np.array(val, dtype=np.int8)

        # otherwise, if p_init is a matrix, assign it as value
        elif type(p_init) is np.ndarray:
            self.val = np.array(p_init, dtype=np.int8)            

        # otherwise, initialise iid random with p_init
        else:
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

    def set_prior(self, bernoulli_prior=None, density_conditions=None):
        """
        density_conditions is the max no of ones in each dimension
        [min_row, min_col, max_row, max_col].
        zero means unrestricted
        """
        if density_conditions is None:
            self.density_conditions = np.array([0,0,0,0], dtype=np.int8)
        else:
            assert len(density_conditions) == 4
            self.density_conditions = np.array(density_conditions, dtype=np.int8)
       
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

    def infer_sampling_fct(self):
        """
        Assing appropriate sampling function as attribute, depending
        on family status, sampling status etc. or assign sampling_fct
        if provided as argument.
        Functions take mat object as only argument.
        """
        # first do some sanity checks, no of children etc. Todo

        # assign different sampling fcts if matrix row/col density is constrained
        if np.any(self.density_conditions):
            # matrix without child...
            if not self.child:
                # ...and one parent
                if len(self.parent_layers) == 1:                   
                    if self.role == 'observations':
                        self.sampling_fct = draw_z_oneparent_nochild_maxdens_wrapper
                    elif self.role == 'features':
                        self.sampling_fct = draw_u_oneparent_nochild_maxdens_wrapper
                # ...and two parents
                if len(self.parent_layers) == 2:
                    if self.role == 'observations':
                        self.sampling_fct = draw_z_twoparents_nochild_maxdens_wrapper
                    elif self.role == 'features':
                        self.sampling_fct = draw_u_twoparents_nochild_maxdens_wrapper

            # matrix with one child...
            elif self.child:
                # ... and no parent
                if not self.parents:
                    if self.role == 'observations':
                        self.sampling_fct = draw_z_noparents_onechild_maxdens_wrapper
                    elif self.role == 'features':
                        self.sampling_fct = draw_u_noparents_onechild_maxdens_wrapper                    
                # ... and one parent 
                elif len(self.parent_layers) == 1:
                    if self.role == 'observations':
                        self.sampling_fct = draw_z_oneparent_onechild_maxdens_wrapper
                    elif self.role == 'features':
                        self.sampling_fct = draw_u_oneparent_onechild_maxdens_wrapper
                # ... and two parents (not implemented, throwing error)
                elif len(self.parent_layers) == 2:
                    if self.role == 'observations':
                        self.sampling_fct = draw_z_twoparents_onechild_maxdens_wrapper
                    elif self.role == 'features':
                        self.sampling_fct = draw_u_twoparents_onechild_maxdens_wrapper
            else:
                raise Warning('Sth is wrong with allocting sampling functions')

        else:
            # matrix without child...
            if not self.child:
                # ...and one parent
                if len(self.parent_layers) == 1:                   
                    if self.role == 'observations':
                        self.sampling_fct = draw_z_oneparent_nochild_wrapper
                    elif self.role == 'features':
                        self.sampling_fct = draw_u_oneparent_nochild_wrapper
                # ...and two parents
                if len(self.parent_layers) == 2:
                    if self.role == 'observations':
                        self.sampling_fct = draw_z_twoparents_nochild_wrapper
                    elif self.role == 'features':
                        self.sampling_fct = draw_u_twoparents_nochild_wrapper

            # matrix with one child...
            elif self.child:
                # ... and no parent
                if not self.parents:
                    if self.role == 'observations':
                        self.sampling_fct = draw_z_noparents_onechild_wrapper
                    elif self.role == 'features':
                        self.sampling_fct = draw_u_noparents_onechild_wrapper                    
                # ... and one parent 
                elif len(self.parent_layers) == 1:
                    if self.role == 'observations':
                        self.sampling_fct = draw_z_oneparent_onechild_wrapper
                    elif self.role == 'features':
                        self.sampling_fct = draw_u_oneparent_onechild_wrapper       
                # ... and two parents (not implemented, throwing error)
                elif len(self.parent_layers) == 2:
                    if self.role == 'observations':
                        self.sampling_fct = draw_z_twoparents_onechild_wrapper
                    elif self.role == 'features':
                        self.sampling_fct = draw_u_twoparents_onechild_wrapper
            else:
                raise Warning('Sth is wrong with allocting sampling functions')

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

    def output(self, u=None, z=None):
        """
        propagate probabilities to child layer
        u and z are optional and intended for use
        when propagating through mutliple layers.
        outputs a probability of x being 1.
        """
        if u is None:
            u = self.u.mean()
        if z is None:
            z = self.z.mean()

        L = z.shape[1]
        N = z.shape[0]
        D = u.shape[0]
    
        x = np.empty((N, D))
        
        cf.probabilistc_output(
            x, .5*(u+1), .5*(z+1), self.lbda.mean(), D, N, L)
    
        return x
    def output(self, u=None, z=None):
        """
        propagate probabilities to child layer
        u and z are optional and intended for use
        when propagating through mutliple layers.
        outputs a probability of x being 1.
        """
        if u is None:
            u = self.u.mean()
        if z is None:
            z = self.z.mean()

        L = z.shape[1]
        N = z.shape[0]
        D = u.shape[0]
    
        x = np.empty((N, D))
        
        cf.probabilistc_output(
            x, .5*(u+1), .5*(z+1), self.lbda.mean(), D, N, L)
    
        return x

    
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
                   role=None, sampling_indicator=True,
                   density_conditions=None, bernoulli_prior=None):

        mat = machine_matrix(val, shape, sibling, parents, child,
                             lbda, bernoulli_prior, density_conditions,
                             p_init, role, sampling_indicator)

        self.members.append(mat)
        return mat
        
        
    def add_parameter(self, val=2, attached_matrices=None):
        
        lbda = machine_parameter(val=val, attached_matrices=attached_matrices)
        self.lbdas.append(lbda)
        
        return lbda
    
        
    def add_layer(self, size=None, child=None, 
                  lbda_init=1.5, z_init=.5, u_init=0.0, 
                  z_prior=None, u_prior=None,
                  z_density_conditions=None, u_density_conditions=None):
        """
        This essentially wraps the necessary calls to
        add_parameter, add_matrix
        z/u_density_conditions: [min_row, min_col, max_row, max_col]
        """

        # infer layer shape
        if (size is None) and (child is not None):
            if type(z_init) is np.ndarray:
              shape_z = (child().shape[0], z_init.shape[1])
              shape_u = (child().shape[1], z_init.shape[1])              
            elif type(u_init) is np.ndarray:
              shape_z = (child().shape[0], u_init.shape[1])
              shape_u = (child().shape[1], u_init.shape[1])              
        elif (size is not None) and (child is not None):
            shape_z = (child().shape[0], size)
            shape_u = (child().shape[1], size)    
        else:
            raise Warning('Can not infer layer size')
            
        z = self.add_matrix(shape=shape_z, 
                            child=child, p_init=z_init, bernoulli_prior=z_prior,
                            density_conditions=z_density_conditions,
                            role='observations')
        
        u = self.add_matrix(shape=shape_u, sibling=z, 
                            child=child, p_init=u_init, bernoulli_prior=u_prior,
                            density_conditions=u_density_conditions,
                            role='features')
        
        lbda = self.add_parameter(attached_matrices=(z,u), val=lbda_init)
        
        layer = machine_layer(z, u, lbda, size, child)
        
        self.layers.append(layer)
        
        return layer

    def burn_in(self,
                mats,
                lbdas,
                eps=1e-2,
                convergence_window=15,
                burn_in_min=0,
                burn_in_max=2000,
                print_step=10,
                fix_lbda_iters=0):

        """
        fix_lbda_iters (int): no of iterations that lambda is not updated for.
            this can help convergence.
        """  

        # first sample without checking for convergence or saving lbda traces
        # this is a 'pre-burn-in-phase'
        pre_burn_in_iter = 0
        while True:
            # stop pre-burn-in if minimum numbers
            # if burn-in iterations is reached
            if pre_burn_in_iter == burn_in_min:
                break

            pre_burn_in_iter += 1

            if pre_burn_in_iter % print_step == 0:
                print('\r\titeration: ' +
                      str(pre_burn_in_iter) +
                      ' recon acc.: ' +
                      ', '.join([str(round(expit(np.mean(x())),3)) for x in lbdas]),
                      end='')

            # draw samples
            [mat.sampling_fct(mat) for mat in mats]
            if pre_burn_in_iter > fix_lbda_iters:
                [lbda.sampling_fct(lbda) for lbda in lbdas]
                shuffle(mats)

                    
        # allocate array for lamabda traces for burn in detection
        for lbda in lbdas:
            lbda.allocate_trace_arrays(convergence_window)
            lbda.trace_index = 0          # reset trace index

        # now cont. burn in and check for convergence
        burn_in_iter = 0
        while True:
            burn_in_iter += 1

            # write output to terminal
            if burn_in_iter % print_step == 0:
                print('\r\titeration: ' +
                      str(pre_burn_in_iter+burn_in_iter) +
                      ' recon acc.: ' +
                       ', '.join([str(round(expit(np.mean(x())),3)) for x in lbdas]),
                      end='')
                  
            # check convergence every convergence_window iterations
            if burn_in_iter % convergence_window == 0:
                # reset trace index
                for lbda in lbdas:
                    lbda.trace_index = 0
                # check convergence for all lbdas
                if np.all([x.check_convergence(eps=eps) for x in lbdas]):
                    print('\n\tconverged at reconstr. accuracy: ' +
                           ', '.join([str(round(expit(np.mean(x())),3)) for x in lbdas]))
                    break

            # stop if max number of burn in inters is reached
            if (burn_in_iter+pre_burn_in_iter) > burn_in_max:
                print('\n\tmax burn-in iterations reached without convergence')
                # reset trace index
                for lbda in lbdas:
                    lbda.trace_index = 0
                break
            
            # draw sampels
            shuffle(mats)
            [mat.sampling_fct(mat) for mat in mats]
            [lbda.sampling_fct(lbda) for lbda in lbdas]
            [x.update_trace() for x in lbdas]

            # 
            # print([x.trace_index for x in lbdas], burn_in_iter)


    def infer(self,
              mats='all',
              no_samples=100,
              convergence_window=15,
              convergence_eps=1e-3,
              burn_in_min=100,
              burn_in_max=20000,
              print_step=5,
              fix_lbda_iters=10):
        """
        members can be a list of machine_layers or machine_matrices
        or a mix of both types.
        eps is on expit scale, i.e is the fractional reconstr. accuracy
        """

        # create list of matrices to draw samples from
        if mats == 'all':
            mats = self.members
        mats = [mat for mat in mats if not np.all(mat.sampling_indicator == 0)]

        # sort from large to small. this is crucial for convergence.
        # mats = sorted(mats, key=lambda x: x.val.shape[0], reverse=True)

        # list of parameters (lbdas) of matrix and parents
        lbdas = []
        for mat in mats:
            lbdas.append(mat.lbda)
            if len(mat.parents) > 0:
                lbdas.append(mat.parents[0].lbda)
        # remove dubplicates preserving order
        lbdas = [x for x in unique_ordered(lbdas) if x is not None]

        # assign sampling function to each mat
        for thing_to_update in mats+lbdas:
            if not thing_to_update.sampling_fct:
                thing_to_update.set_sampling_fct()

        # make sure all trace indicies are zero
        for mat in mats:
            mat.trace_index = 0
        for lbda in lbdas:
            if lbda is not None:
                lbda.trace_index = 0

        # burn in markov chain
        print('burning in markov chain...')
        self.burn_in(mats,
                     lbdas,
                     eps=convergence_eps,
                     convergence_window=convergence_window,
                     burn_in_min=burn_in_min,
                     burn_in_max=burn_in_max,
                     print_step=print_step,
                     fix_lbda_iters=fix_lbda_iters)

        # allocate memory to save samples
        print('allocating memory to save samples...')
        for mat in mats:
            mat.allocate_trace_arrays(no_samples)
        for lbda in lbdas:
            lbda.allocate_trace_arrays(no_samples)

        print('drawing samples...')
        for sampling_iter in range(1, no_samples+1):

            shuffle(mats)  # TODO
            # sample mats and write to trace
            [mat.sampling_fct(mat) for mat in mats]
            [mat.update_trace() for mat in mats]

            # sample lbdas and write to trace
            [lbda.sampling_fct(lbda) for lbda in lbdas]
            [lbda.update_trace() for lbda in lbdas]

            if sampling_iter % print_step == 0:
                print('\r\t' + 'iteration ' +
                      str(sampling_iter) +
                      '; recon acc.: ' +
                      ', '.join([str(round(expit(np.mean(x())),3)) for x in lbdas]),
                      end='')

        # set all parameters to MAP estimate
        # [mat.set_to_map() for mat in mats]
        # [lbda.update() for lbda in lbdas]
        print('\nfinished.')
    
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
        mat.lbda(), # own parameter: double
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
        mat.lbda(), # own parameter: double
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
        mat.parents[1].lbda(), # parent lbda
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
