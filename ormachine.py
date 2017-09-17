#!/usr/bin/env python
"""
MaxMachine

This module implements classes for sampling form hierarchical binary matrix
factorisation models, in particular for the MaxMachine.

Flexible hierarchies of binary matrices with MaxMachine or OrMachine links
can be specified. It features iid Bernoulli or row/column wise Binomial priors
on the matrices, as well as beta priors on dispersion parameters.

Example usage: see tests.

Implemented classes are:
- machine
- machine_matrix
- machine_parameter
- trace

"""

from __future__ import absolute_import, division, print_function # for python2 support
from IPython.core.debugger import Tracer
import pyximport; pyximport.install()
import wrappers
import cython_functions as cf
import numpy as np
from numpy.random import binomial
from random import shuffle
import unittest
# from IPython.core.debugger import Tracer

try:
    from scipy.special import expit
except:
    # makes scipy dependency optional
    def expit(x):
        """
        better implementation in scipy.special, 
        but can avoid dependency
        """
        return 1/(1+np.exp(-x))

__author__ = "Tammo Rukat"
__status__ = "Development"

class trace():
    """
    abstract base class implementing methods for
    arrays of posterior traces,
    inherits to machine matrix and lbda instances.
    """
    def __call__(self):
        return self.val
        
    def allocate_trace_arrays(self, no_of_samples):
        no_of_samples = int(no_of_samples)
        if type(self.val) == np.ndarray:
            # nicer but no python2 compatible
            # self.trace = np.empty([no_of_samples, *self.val.shape], dtype=np.int8)

            self.trace = np.empty([no_of_samples]+[x for x in self.val.shape], dtype=self().dtype)
        else: # not needed (?)
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

        def check_convergence_single_trace(trace, eps):
            l = int(len(trace)/2)
            r1 = expit(np.mean(trace[:l]))
            r2 = expit(np.mean(trace[l:]))
            r = expit(np.mean(trace))
            
            if np.abs(r1-r2) < eps:
                return True
            else:
                return False            
        
        if self.trace.ndim == 1:
            return check_convergence_single_trace(self.trace, eps)

        # if we have multiple dispersion parameter every one of them needs to have converged
        elif self.trace.ndim == 2:
            return np.all([check_convergence_single_trace(self.trace[:,l], eps)
                           for l in range(self.trace.shape[1])])

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
    
    def __init__(self, val, attached_matrices=None, sampling_indicator=True, noise_model='coupled'):
        self.trace_index = 0
        self.sampling_fct = None
        self.val = val
        self.attached_matrices = attached_matrices
        self.correct_role_order()
        # make sure they are a tuple in order (observations/features)
        self.sampling_indicator = sampling_indicator
        self.trace_index = 0
        self.noise_model = noise_model
        
        # assign matrices the parameter
        for mat in attached_matrices:
            mat.lbda = self
            # if not hasattr(mat, 'lbdas'):
            #     mat.lbdas = [self]
            # else:
            #     mat.lbdas.append(self)
        self.noise_mode = noise_model
        self.set_prior()

    def print_value(self):
        if self.noise_model == 'coupled':
            return ', '.join([str(round(expit(np.mean(x)),3)) for x in [self.val]])
        elif self.noise_model == 'independent':
            return ', '.join([str(round(expit(x),3)) for x in self.val])
        elif self.noise_model == 'maxmachine':
            return '\t'.join([str("%.1f"%round(100*x,2)) for x in self.val])
            #return ', '.join([str(str.format('{0:3f}',x)) for x in self.val])

            
    def set_prior(self, prior_parms=None, clamped_unit_prior_parms=None):
        """
        In analaogy to the set_prior method of machine_matrix,
        initialise all the necessary prior properties for posterior sampling.
        Only option here is a beta prior.
        Creates a property for the machine_parameter called .prior_config:
        list of [prior_type, [alpha, beta], [alpha_c, beta_c]] = [int, [float, float], [float, float]] 
        with
        prior_type = 0 -> Flat prior
        prior_type = 1 -> beta prior with parameters [alpha, beta] and [alpha_c, beta_c]
            on the clamped unit.
        """

        # beta priors on the noise parameters are only available in maxmachine (todo)


        if (((prior_parms == [1,1]) or (prior_parms is None)) and
           ((clamped_unit_prior_parms == [1,1]) or (clamped_unit_prior_parms is None))):
            prior_type = 0
            self.prior_config = [0, None, None]
        else:
            prior_type = 1
            if prior_parms is None:
                prior_parms = [1,1]
            if clamped_unit_prior_parms is None:
                clamped_unit_prior_parms = prior_parms
            
            self.prior_config = [1, prior_parms, clamped_unit_prior_parms]

            
    def correct_role_order(self):
        """
        the assigned matrices have to be ordered (observations,features),
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
        if 'coupled' in self.layer.noise_model:
            self.sampling_fct = wrappers.draw_lbda_wrapper    
        elif 'independent' in self.layer.noise_model:
            self.sampling_fct = wrappers.draw_lbda_indpndt_wrapper
        elif 'maxmachine' in self.noise_model:
            self.sampling_fct = wrappers.draw_lbda_maxmachine_wrapper
        else:
            raise StandardError('Can not infer appropriate samping function for lbda/mu') 

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

        # initiliase some parameters to None
        # for type checking once they're used.
        self.logit_bernoulli_prior = None
        self.row_binomial_prior = None
        self.col_binomial_prior = None
        self.row_densities = None
        self.col_densities = None
        self.set_prior()
        if density_conditions == None:
            self.density_conditions = None
        
    
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
            self.sampling_indicator = False
        elif type(sampling_indicator) is np.ndarray:
            assert sampling_indicator.shape == self().shape
            self.sampling_indicator = np.array(sampling_indicator, dtype=np.int8)

    def set_prior(self, prior_type=None, prior_parms=None, axis=None, prior_code=None):
        """
        prior_type (str): 'bernoulli', 'binomial', 'beta-binomial'
        prior_parms (list of mixed types):
            bernoulli: [p (float)]
            binomial: [q (float), N (int)]
            beta-bernoulli: [a (float) ,b (float), N (int)]
        Setting N different from the acutal size of the corresponding
        dimension enforces sparsit.

        the property .prior_code is
        0: no prior
        1: Bernoulli prior
        2: Binomial prior across columns
        3: Binomial prior across rows
        4: Binomial prior across rows and columns (not very reasonable)

        Every machine_matrix exposes: .prior_type, .prior_parms, 
        and then: .row_densities, .col_densities, .binomial_prior_col, .binomial_prior_row, .bernoulli_prior
        """
        if prior_type is None:
            self.prior_code = 0
        
        if (prior_type is not None) and (prior_type not in ['bernoulli', 'binomial', 'beta-binomial']):
            print('No proper prior specified. Choices are bernoulli, binomial, beta binomial.')
            print('Defaulting to flat priors')
            self.set_prior()

        if prior_type is 'bernoulli':
            self.prior_code = 1
            if type(prior_parms) is list:
                assert len(prior_parms) == 1
                prior_parms = prior_parms[0]
            else:
                assert type(prior_parms) is float
            self.logit_bernoulli_prior = logit(prior_parms)

        if prior_type is 'binomial':
            # set number of draws to axis length, if not provided.
            if type(prior_parms) is list:
                if len(prior_parms) == 1:
                    prior_parms.append(self().shape[1-axis])
            elif type(prior_parms) is float:
                prior_parms = [prior_parms, self().shape[1-axis]]
            if axis is None:
                print('No axis for binomial prior provided. Defaulting to 0')
                axis = 0

            # set counts, summing the ones across rows/columns
            if axis == 0: # prior on each row, sum across columns
                if self.prior_code == 3:
                    self.prior_code = 4
                else:
                    self.prior_code = 2
                    
                self.row_densities = np.array(np.sum(self()==1, 1), dtype=np.int32)
                assert len(self.row_densities) == self().shape[0]
                # need to add +1 to also count all zero state.
                self.row_binomial_prior = compute_bp(prior_parms[0], prior_parms[1]+1, self().shape[1-axis]+1)
                    
            elif axis == 1:
                if self.prior_code == 2:
                    self.prior_code = 4
                else:
                    self.prior_code = 3
                self.col_densities = np.array(np.sum(self()==1, 0), dtype=np.int32)
                assert len(self.col_densities) == self().shape[1]
                self.col_binomial_prior = compute_bp(prior_parms[0], prior_parms[1]+1, self().shape[1-axis]+1)

            elif prior_type is 'beta-binomial':
                print('beta-binomial not yet implemented.') # works just like above

        self.prior_config = [self.prior_code, self.logit_bernoulli_prior, self.row_binomial_prior,
                             self.col_binomial_prior, self.row_densities, self.col_densities]

        
    def update_prior_config(self):
        """
        If the model dimensions change, binomial priors need to be updated.
        """
        # No updating for bernoulli priors
        if self.prior_config[0] < 2:
            return
        if (self.prior_config[0] == 2) or (self.prior_config == 4):
            self.row_densities = np.array(np.sum(self()==1, 1), dtype=np.int32)
            assert len(self.row_densities) == self().shape[0]
            self.row_binomial_prior = compute_bp(prior_parms[0], prior_parms[1]+1, self().shape[1-axis]+1)
        if (self.prior_config == 3)  or (self.prior_config == 4):
            self.col_densities = np.array(np.sum(self()==1, 0), dtype=np.int32)
            assert len(self.col_densities) == self().shape[1]
            self.col_binomial_prior = compute_bp(prior_parms[0], prior_parms[1]+1, self().shape[1-axis]+1)
        self.prior_config = [self.prior_code, self.logit_bernoulli_prior, self.row_binomial_prior,
                             self.col_binomial_prior, self.row_densities, self.col_densities]            
            
        
    
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
        Assign appropriate sampling function as attribute, depending
        on family status, sampling status etc. or assign sampling_fct
        if provided as argument.
        Functions take mat object as only argument.
        """
        # first do some sanity checks, no of children etc. Todo
        
        if 'independent' in self.layer.noise_model:
            if not np.any(self.density_conditions) and not self.parents:
                if self.role == 'observations':
                    self.sampling_fct = wrappers.draw_z_noparents_onechild_indpn_wrapper
                elif self.role == 'features':
                    self.sampling_fct = wrappers.draw_u_noparents_onechild_indpn_wrapper
            else:
                raise StandardError('Appropriate sampling function for independent '+
                                    'noise model is not defined')
            return
            
        # elif self.layer.noise_model == 'coupled':
        # assign different sampling fcts if matrix row/col density is constrained
        if np.any(self.density_conditions):
            # matrix without child...
            if not self.child:
                # ...and one parent
                if len(self.parent_layers) == 1:                   
                    if self.role == 'observations':
                        self.sampling_fct = wrappers.draw_z_oneparent_nochild_maxdens_wrapper
                    elif self.role == 'features':
                        self.sampling_fct = wrappers.draw_u_oneparent_nochild_maxdens_wrapper
                # ...and two parents
                if len(self.parent_layers) == 2:
                    if self.role == 'observations':
                        self.sampling_fct = wrappers.draw_z_twoparents_nochild_maxdens_wrapper
                    elif self.role == 'features':
                        self.sampling_fct = wrappers.draw_u_twoparents_nochild_maxdens_wrapper

            # matrix with one child...
            elif self.child:
                # ... and no parent
                if not self.parents:
                    if self.role == 'observations':
                        self.sampling_fct = wrappers.draw_z_noparents_onechild_maxdens_wrapper
                    elif self.role == 'features':
                        self.sampling_fct = wrappers.draw_u_noparents_onechild_maxdens_wrapper                    
                # ... and one parent 
                elif len(self.parent_layers) == 1:
                    if self.role == 'observations':
                        self.sampling_fct = wrappers.draw_z_oneparent_onechild_maxdens_wrapper
                    elif self.role == 'features':
                        self.sampling_fct = wrappers.draw_u_oneparent_onechild_maxdens_wrapper
                # ... and two parents (not implemented, throwing error)
                elif len(self.parent_layers) == 2:
                    if self.role == 'observations':
                        self.sampling_fct = wrappers.draw_z_twoparents_onechild_maxdens_wrapper
                    elif self.role == 'features':
                        self.sampling_fct = wrappers.draw_u_twoparents_onechild_maxdens_wrapper
            else:
                raise Warning('Sth is wrong with allocating sampling functions')

        else:
            # matrix without child...
            if not self.child:
                # ...and one parent
                if len(self.parent_layers) == 1:                   
                    if self.role == 'observations':
                        self.sampling_fct = wrappers.draw_z_oneparent_nochild_wrapper
                    elif self.role == 'features':
                        self.sampling_fct = wrappers.draw_u_oneparent_nochild_wrapper
                # ...and two parents
                if len(self.parent_layers) == 2:
                    if self.role == 'observations':
                        self.sampling_fct = wrappers.draw_z_twoparents_nochild_wrapper
                    elif self.role == 'features':
                        self.sampling_fct = wrappers.draw_u_twoparents_nochild_wrapper

            # matrix with one child...
            elif self.child:
                # ... and no parent
                if not self.parents:
                    if self.role == 'observations':
                        if self.layer.noise_model == 'maxmachine':
                            self.sampling_fct = wrappers.draw_z_noparents_onechild_maxmachine
                        elif self.layer.noise_model == 'coupled':
                            self.sampling_fct = wrappers.draw_z_noparents_onechild_wrapper
                    elif self.role == 'features':
                        if self.layer.noise_model == 'maxmachine':
                            self.sampling_fct = wrappers.draw_u_noparents_onechild_maxmachine
                        elif self.layer.noise_model == 'coupled':
                            self.sampling_fct = wrappers.draw_u_noparents_onechild_wrapper       
                            

                # ... and one parent 
                elif len(self.parent_layers) == 1:
                    if self.role == 'observations':
                        if self.layer.noise_model == 'maxmachine':
                            self.sampling_fct = wrappers.draw_z_oneparent_onechild_maxmachine
                        elif self.layer.noise_model == 'coupled':
                            self.sampling_fct = wrappers.draw_z_oneparent_onechild_wrapper
                    elif self.role == 'features':
                        self.sampling_fct = wrappers.draw_u_oneparent_onechild_wrapper       
                # ... and two parents (not implemented, throwing error)
                elif len(self.parent_layers) == 2:
                    if self.role == 'observations':
                        self.sampling_fct = wrappers.draw_z_twoparents_onechild_wrapper
                    elif self.role == 'features':
                        self.sampling_fct = wrappers.draw_u_twoparents_onechild_wrapper
            else:
                raise Warning('Sth is wrong with allocting sampling functions')

class machine_layer():
    """
    Essentially a container for (z,u,lbda) to allow
    convenient access to different layers for the sampling routine
    and the user.
    """
    
    def __init__(self, z, u, lbda, size, child, noise_model):
        self.z = z
        self.u = u
        self.lbda = lbda
        self.size = size
        
        # register as layer of members
        self.z.layer = self
        self.u.layer = self
        self.lbda.layer = self
        # self.lbdas.layer = self # not working in independent noise implementation
        self.child = child
        self.child.add_parent_layer(self)
        self.noise_model = noise_model
        # Keep track of whehter P (OrM) or TP, FN, etc need updating

        if noise_model == 'maxmachine':
            self.precompute_lbda_ratios()
        
        if 'independent' in noise_model or True: # we can compute predict accuracies for the original model, too.
            self.predictive_accuracy_updated = False # switch to update only if needed
            self.pred_rates = np.array([0,0,0,0], dtype=np.int) # TP, FP, TN, FN
            # fraction of ones in the data for unbiased inference of 0/1
            # as implement in draw_lbda_indpndt_wrapper()
            if 'unbias' in noise_model:
                self.child.log_bias = np.log(float(np.sum(self.child()==1))/np.sum(self.child()==-1))
            else:
                self.child.log_bias = 0
        
    def __call__(self):
        return(self.z(), self.u(), self.lbda())
        
    def members(self):
        return [self.z, self.u]

    @property
    def lbda_property(self):
        """
        deprecated
        """
        if 'coupled' in self.noise_model:
            return self.lbdas[0]
        elif 'independent' in self.noise_model:
            assert len(self.lbdas) == 2
            return self.lbdas[1]
        elif 'maxmachine' in self.noise_model:
            return [x() for x in self.lbdas]

    @property
    def mu(self):
        assert 'independent' in self.noise_model
        return self.lbda()[0]

    @property
    def nu(self):
        assert 'independent' in self.noise_model
        return self.lbda()[1]

    def child(self):
        assert self.z.child is self.u.child
        return self.z.child

    def output(self, u=None, z=None, recon_model=None):
        """
        propagate probabilities to child layer
        u and z are optional and intended for use
        when propagating through mutliple layers.
        outputs a probability of x being 1.
        """
        if recon_model is None:
            noise_model = self.noise_model
        else:
            noise_model = recon_model
            
        if u is None:
            u = self.u.mean()
        if z is None:
            z = self.z.mean()

        L = z.shape[1]
        N = z.shape[0]
        D = u.shape[0]
    
        x = np.empty((N, D))

        if noise_model == 'coupled':
            cf.probabilistc_output(
                x, .5*(u+1), .5*(z+1), self.lbda.mean(), D, N, L)
        elif 'independent' in noise_model:
            cf.probabilistc_output_indpndt(
                x, .5*(u+1), .5*(z+1), self.lbda.mean()[1], self.lbda.mean()[0], D, N, L)
        elif noise_model is 'maxmachine_plugin':
            # check that the background noise is smaller than any latent dimension's noise
            if self.lbda.mean()[-1] != np.min(self.lbda.mean()):
                print('we have alphas < alpha[-1]')
            cf.probabilistic_output_maxmachine(
                x, .5*(u+1), .5*(z+1), self.lbda.mean(),
                np.zeros(len(self.lbda()), dtype=np.float64),
                np.zeros(len(self.lbda()), dtype=np.int32))
                #np.array(np.argsort(-self.lbda.mean()[:-1]), dtype=np.int32))
        elif noise_model is 'maxmachine':
            print('Computing MC estimate of data reconstruction')
            u_tr = self.u.trace
            z_tr = self.z.trace
            alpha_tr = self.lbda.trace
            x = np.zeros([N,D])
            trace_len = u_tr.shape[0]
            for tr_idx in range(trace_len):
                x += maxmachine_forward_pass(u_tr[tr_idx,:,:]==1,
                                             z_tr[tr_idx,:,:]==1,
                                             alpha_tr[tr_idx,:])
            x /= trace_len
            
        else:
            raise StandardError('Output function not defined for given noise model.')
    
        return x
    
    
    def log_likelihood(self):
        """
        Return log likelihood of the assoicated child, given the layer.
        """
        
        N = self.z().shape[0]
        D = self.u().shape[0]

        if 'coupled' in self.noise_model:

            P = cf.compute_P_parallel(self.lbda.attached_matrices[0].child(),
                                      self.lbda.attached_matrices[1](),
                                      self.lbda.attached_matrices[0]())

            return (-P*logsumexp([0,-self.lbda.mean()]) -
                    (N*D-P)*logsumexp([0,self.lbda.mean()]) )

        elif 'independent' in self.noise_model:
            self.update_predictive_accuracy()
            TP, FP, TN, FN = self.pred_rates

            return (-TP*logsumexp([0,-self.lbda()])
                    -FP*logsumexp([0,self.lbda()])
                    -TN*logsumexp([0,-self.mu()])
                    -FN*logsumexp([0,self.lbda()]))


    def update_predictive_accuracy(self):
        """
        update values for TP/FP and TN/FN
        """
        cf.compute_pred_accuracy(self.child(), self.u(), self.z(), self.pred_rates)
        self.predictive_accuracy_updated = True


    def precompute_lbda_ratios(self):
        """
        TODO: speedup (cythonise and parallelis)e
        precompute matrix of size [2,L+1,L+1],
        with log(lbda/lbda') / log( (1-lbda) / (1-lbda') )
        as needed for maxmachine gibbs updates.
        """

        if self.noise_model != 'maxmachine':
            return
        
        L = self.size + 1

        lbda_ratios = np.zeros([2,L,L], dtype=np.float32)

        for l1 in range(L):
            for l2 in range(l1+1):
                lratio_p = np.log(self.lbda()[l1]/self.lbda()[l2])
                lratio_m = np.log( (1-self.lbda()[l1])/ (1-self.lbda()[l2]) )
                lbda_ratios[0,l1,l2] = lratio_p
                lbda_ratios[0,l2,l1] = -lratio_p
                lbda_ratios[1,l1,l2] = lratio_m
                lbda_ratios[1,l2,l1] = -lratio_m

        self.lbda_ratios = lbda_ratios
            
    
class machine():
    """
    main class
    attach factor_matrices to this class
    methods:
    
    add_data(data, infer=False, parents=None)
    add_layer(L, infer=True, children=None, parents=None)    
    TODO maxmachine lbda initialisation MUST be within bounds!
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
        
        
    def add_parameter(self, val=2, attached_matrices=None, noise_model='coupled'):
        lbda = machine_parameter(val=val, attached_matrices=attached_matrices, noise_model=noise_model)
        self.lbdas.append(lbda)
        return lbda
    
        
    def add_layer(self, size=None, child=None, 
                  lbda_init=1.5, z_init=0.5, u_init=0.5, 
                  z_prior=None, u_prior=None,
                  noise_model='coupled'):
        """
        This essentially wraps the necessary calls to
        add_parameter, add_matrix
        z/u_density_conditions: [min_row, min_col, max_row, max_col]
        noise-mode (str): 'coupled', 'indepdent', 'max-machine' (not implemented)
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
                            role='observations')
        
        u = self.add_matrix(shape=shape_u, sibling=z, 
                            child=child, p_init=u_init, bernoulli_prior=u_prior,
                            role='features')

        if 'coupled' in noise_model:
            lbda = self.add_parameter(attached_matrices=(z,u), val=lbda_init, noise_model=noise_model)
            # lbdas = [lbda]
            
        elif 'independent' in noise_model:
            print('All noise parameters are initialised with lbda_init')
            lbda = self.add_parameter(attached_matrices=(z,u), val=np.array(2*[lbda_init]), noise_model=noise_model)
            #mu = self.add_parameter(attached_matrices=(z,u), val=lbda_init, noise_model=noise_model)
            #lbda = self.add_parameter(attached_matrices=(z,u), val=lbda_init, noise_model=noise_model)
            # lbdas = [mu, lbda]
            
        elif noise_model == 'maxmachine': # experimental
            if type(lbda_init) is not list:
                lbda = self.add_parameter(attached_matrices=(z,u),
                                          val=np.array([lbda_init for i in range(size+1)]),
                                          noise_model=noise_model)
                lbda()[-1] = .01
            elif len(lbda_init) == size + 1:
                lbda = self.add_parameter(attached_matrices=(z,u), val=np.array(lbda_init), noise_model=noise_model)
            else:
                raise StandardError('lambda not properly initiliased for maxmachine')
        else:
            raise StandardError('No proper generative model specified.')

        layer = machine_layer(z, u, lbda, size, child, noise_model)

        layer.auto_clean_up = False

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
                # clean up codes                         
                break

            pre_burn_in_iter += 1

            if pre_burn_in_iter % print_step == 0:
                print('\r\titeration: ' +
                      str(pre_burn_in_iter) +
                      ' disperion.: ' +
                      '\t--\t '.join([x.print_value() for x in lbdas]),
                      end='')

            # draw samples
            [mat.sampling_fct(mat) for mat in mats]
            #if pre_burn_in_iter % 25 == 0:
                # [wrappers.reset_codes_wrapper(lr.u, lr.z, lr.noise_model) for lr in self.layers]
            if pre_burn_in_iter > fix_lbda_iters:
                [lbda.sampling_fct(lbda) for lbda in lbdas]
                # shuffle(mats)

                    
        # allocate array for lambda traces for burn in detection
        for lbda in lbdas:
            lbda.allocate_trace_arrays(convergence_window)
            lbda.trace_index = 0          # reset trace index

        # now cont. burn in and check for convergence
        burn_in_iter = 0
        while True:
            burn_in_iter += 1

            # write output to terminal # todo modularise
            if burn_in_iter % print_step == 0:
                print('\r\titeration: ' +
                      str(pre_burn_in_iter+burn_in_iter) +
                      ' recon acc.: ' +
                       '\t--\t '.join([x.print_value() for x in lbdas]),
                      end='')
                  
            # check convergence every convergence_window iterations
            if burn_in_iter % convergence_window == 0:
                # reset trace index
                for lbda in lbdas:
                    lbda.trace_index = 0
                # check convergence for all lbdas
                if np.all([x.check_convergence(eps=eps) for x in lbdas]):
                    print('\n\tconverged at reconstr. accuracy: ' +
                          '\t--\t'.join([x.print_value() for x in lbdas]))
                    
                    if True:
                        # check for dimensios to be removed and restart burn-in i.a.
                        if np.any([wrappers.clean_up_codes(lr, lr.noise_model)
                                   for lr in self.layers if lr.auto_clean_up is True]):
                            print('\tremove duplicate or useless latent dimensions and restart burn-in. New L='
                                  +str([lr.size for lr in self.layers]))
                            for lbda in lbdas:
                                # reallocate arrays for lbda trace
                                lbda.allocate_trace_arrays(convergence_window)
                                lbda.trace_index = 0
                        else:
                            self.burn_in_iters = burn_in_iter+pre_burn_in_iter # save no of burn in iters
                            break
                    else:
                        self.burn_in_iters = burn_in_iter+pre_burn_in_iter # save no of burn in iters
                        break

            # stop if max number of burn in inters is reached
            if (burn_in_iter+pre_burn_in_iter) > burn_in_max:
                print('\n\tmax burn-in iterations reached without convergence')
                # reset trace index
                for lbda in lbdas:
                    lbda.trace_index = 0
                self.burn_in_iters = burn_in_iter
                break
            
            # draw sampels
            # shuffle(mats)
            [mat.sampling_fct(mat) for mat in mats]
            [lbda.sampling_fct(lbda) for lbda in lbdas]
            if False: # pre_burn_in_iter % 10 == 0:
                [wrappers.reset_codes_wrapper(lr.z, lr.u, lr.noise_model) for lr in self.layers]
            [x.update_trace() for x in lbdas]
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
        mats = [mat for mat in mats if not mat.sampling_indicator is False]

        # sort from large to small. this is crucial for convergence.
        # mats = sorted(mats, key=lambda x: x.val.shape[0], reverse=True)

        # list of parameters (lbdas) of matrix and parents
        lbdas = []
        for mat in mats:
            lbdas += [mat.lbda]
            if len(mat.parents) > 0:
                lbdas += [mat.parents[0].lbda]
        # remove dubplicates preserving order
        lbdas = [x for x in unique_ordered(lbdas) if x is not None]


        # assign sampling function to each mat
        for thing_to_update in mats+lbdas:
            try:
                if not thing_to_update.sampling_fct:
                    thing_to_update.set_sampling_fct()
            except:
                Tracer()()

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

            # shuffle(mats)  # TODO
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
                      '\t--\t'.join([x.print_value() for x in lbdas]),
                      end='')

        # some sanity checks
        for layer in self.layers:
            if False: # and layer.noise_model == 'maxmachine':
                if layer.u.row_densities is not None:
                    assert np.all(layer.u.row_densities == np.sum(layer.u()==1,1))
                if layer.u.col_densities is not None:
                    assert np.all(layer.u.row_densities == np.sum(layer.u()==1,0))    
                if layer.z.row_densities is not None:
                    assert np.all(layer.z.row_densities == np.sum(layer.z()==1,1))
                if layer.z.col_densities is not None:
                    assert np.all(layer.z.row_densities == np.sum(layer.z()==1,0))
                    
        # set all parameters to MAP estimate
        # [mat.set_to_map() for mat in mats]
        # [lbda.update() for lbda in lbdas]
        print('\nfinished.')

def logsumexp(a):
    a_max = np.max(a)
    out = np.log(np.sum(np.exp(a - a_max)))
    out += a_max
    return out


def maxmachine_relevance(layer, model_type='mcmc'):
    """
    Compute the fraction of 1s that are modelled by a given latent dimension,
    somewhat akin to eigenvalues in PCA
    """

    if model_type is 'plugin': 
        alphas = layer.lbda()
        x = layer.child()
        # add clamped units
        z = np.concatenate([(layer.z.mean()+1)*.5,np.ones([layer.z().shape[0],1])], axis=1)
        u = np.concatenate([(layer.u.mean()+1)*.5,np.ones([layer.u().shape[0],1])], axis=1)
        N = z.shape[0]
        D = u.shape[0]
        L = z.shape[1]

        # # we need to argsort all z*u*alpha. This is of size NxDxL!
        # l_sorted = np.zeros([N,D,L], dtype=np.int8)
        # for n in range(N): # could be parallelised -> but no argsort in cython without extra pain
        #     for d in range(D):
        #         l_sorted[n,d,:] = np.argsort(alphas*u[d,:]*z[n,:])

        eigenvals = np.zeros(len(alphas)) # array for results
        idxs = np.where(layer.child()==1)
        idxs = zip(idxs[0],idxs[1]) # indices of n,d where x[n,d]=1
        no_ones = len(idxs)

        # iterate from largest to smallest alpha doesn't work
        for l in range(L):
            for n,d in idxs:
                eigenvals[l] += z[n,l]*u[d,l]*alphas[l] * np.prod(
                    [1-z[n,l_prime]*u[d,l_prime] for l_prime in range(L) 
                     if z[n,l_prime]*u[d,l_prime]*alphas[l_prime] > z[n,l]*u[d,l]*alphas[l]])
        eigenvals /= len(idxs)

    elif model_type is 'mcmc':
        alpha_tr = layer.lbda.trace
        z_tr = layer.z.trace
        u_tr = layer.u.trace
        x = layer.child()
        tr_len = u_tr.shape[0]
        eigenvals = np.zeros(alpha_tr.shape[1])

        for tr_idx in range(tr_len):
            for l in range(u_tr.shape[2]):
                eigenvals[l] += alpha_tr[tr_idx,l] * np.sum(
                    x[np.dot(z_tr[tr_idx,:,l:l+1]==1,
                           u_tr[tr_idx,:,l:l+1].transpose()==1)]==1)
            eigenvals[-1] += alpha_tr[tr_idx, -1]

        eigenvals /= float(tr_len)*np.sum(x==1)

    return eigenvals

        
def maxmachine_forward_pass(u, z, alpha):
    x = np.zeros([z.shape[0], u.shape[0]])

    for l in np.argsort(-alpha[:-1]):
        x[x==0] += alpha[l]*np.dot(z[:,l:l+1],u[:,l:l+1].transpose())[x==0]

    x[x==0] = alpha[-1]

    return x


def compute_bp(q, n, N, tau=1):
    """
    compute coefficients for binomial prior
    n = number of draws
    q = success probability per draw
    N = size of output (output gets 0-padded)
    """
    
    exp_bp = [(q*(n-k*tau)) / ((1-q)*(k*tau+1)) for k in range(n)]
    
    bp = [np.log(x) if (x > 0) else -np.infty for x in exp_bp]

    if N != n:
        bp_new = [-np.infty for i in range(N)]
        bp_new[:n-1] = bp
        bp = bp_new
        
    return np.array(bp, dtype=float)


def unique_ordered(seq):
    """
    return unique list entries preserving order.
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

