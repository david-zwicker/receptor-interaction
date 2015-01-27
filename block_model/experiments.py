'''
Created on Jan 23, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import itertools
import random

import numpy as np
import scipy

from .utils import calc_entropy, copy_func



class DetectSingleSubstrate(object):
    """ experiment in which a receptor array is used to identify a single
    substrate out of a given list """ 
    
    def __init__(self, temperature=1, threshold='auto'):
        """ initialize the experiment with parameters:
        `temperature` influences the binding probabilities
        `threshold` determines when receptors signal
        """
        self.temperature = temperature
        self.threshold = threshold
        
        self._cache = {}


    def __repr__(self):
        return ('%s(temperature=%r, threshold=%r)' %
                (self.__class__.__name__, self.temperature, self.threshold))
        
        
    @classmethod
    def create_test_instance(cls):
        """ creates a instance of the class with random parameters """
        obj = cls()
        # choose random parameters
        if random.randrange(2):
            obj.temperature = 0
        else:
            obj.temperature = random.randrange(0, 3)
        if random.randrange(2):
            obj.threshold = 'auto'
        else:
            obj.threshold = random.random() + 0.5
        return obj

    
    def binary_base(self, cnt_r):
        """ return repeated sub_ids to implement periodic boundary
        conditions """
        try:
            return self._cache['binary_base'][cnt_r]
        except KeyError:
            # one of the keys was not set
            binary_base = np.exp2(np.arange(cnt_r)).astype(np.int)
            try:
                self._cache['binary_base'][cnt_r] = binary_base
            except KeyError:
                # the cache was not initialized at all
                self._cache['binary_base'] = {cnt_r: binary_base}
                
            return binary_base


    def get_binding_probabilities(self, state):
        """
        Calculates the probability of sub_ids binding to receptors given
        interaction energies described by a matrix energies[substrate, receptor].
        Here, we consider the case that a single substrate must bind to any
        receptor and calculate the coverage ratio of the receptors given many
        realizations.
        """
        if self.temperature == 0:
            # determine maximal energies for each substrate
            Emax = state.energies.max(axis=1)
            # determine the receptors that are activated
            probs = (state.energies == Emax[:, np.newaxis]).astype(np.double)
            
        else:
            # calculate interaction probabilities
            probs = np.exp(state.energies/self.temperature)
            
        # normalize for each substrate across all receptors
        # => scenario in which each substrate binds to exactly one receptor
        probs /= np.sum(probs, axis=1)[:, None]
        
        return probs
    
    
    def get_output_vector(self, state):
        """ calculate output vector for given input """
        # calculate the resulting binding characteristics
        probs = self.get_binding_probabilities(state)

        # threshold to get the response
        cnt_r = state.num_receptors
        if self.threshold == 'auto':
            output = (probs >= 1/cnt_r)
        else:
            output = (probs >= self.threshold)

        # encode output in single integer
        binary_base = self.binary_base(cnt_r)
        return np.dot(output, binary_base)
        
            
    def mutual_information_from_output(self, output_vector):
        """ determine the mutual information from the output distribution """        
        entropy_o = calc_entropy(output_vector)
        
        cnt_i = len(output_vector)
        return np.log2(cnt_i) - entropy_o/cnt_i    
        
    
    def get_mutual_information(self, state):
        """ calculate the mutual information for a state """
        output = self.get_output_vector(state)
        return self.mutual_information_from_output(output)


    def get_input_dim(self, model_or_state):
        """ number of different inputs """
        return model_or_state.num_substrates
    
        
    def get_output_dim(self, model_or_state):
        """ number of different outputs """
        # the 2 is due to the binary output of the receptors
        return 2 ** model_or_state.num_receptors
    
    
    def get_max_mutual_information(self, model):
        """ return upper bound for mutual information of a model """
        # maximal mutual information restricted by sub_ids
        MI_input = np.log2(self.get_input_dim(model))
        # maximal mutual information restricted by the output
        MI_output = np.log2(self.get_output_dim(model))
        return min(MI_input, MI_output)
    
    
    
class DetectMultipleSubstrates(DetectSingleSubstrate):
    """ experiment in which a receptor array is used to identify a mixture of
    substrate out of a given list """ 
    
    def __init__(self, num_substrates=2, temperature=1, threshold='auto'):
        """ initialize the experiment with parameters:
        `num_substrates` is the number of sub_ids perceived concurrently
        `temperature` influences the binding probabilities
        `threshold` determines when receptors signal
        """
        self.num = num_substrates
        self.temperature = temperature
        self.threshold = threshold
        
        self._cache = {}


    def __repr__(self):
        return ('%s(num=%r, temperature=%r, threshold=%r)' %
                (self.__class__.__name__, self.num, self.temperature,
                 self.threshold))
        
        
    @classmethod
    def create_test_instance(cls):
        """ creates a instance of the class with random parameters """
        obj = super(DetectMultipleSubstrates, cls).create_test_instance()
        obj.num = random.randrange(1, 4)
        return obj
    
    
    def get_binding_probabilities(self, state):
        """
        Calculates the probability of sub_ids binding to receptors given
        interaction energies described by a matrix energies[substrate, receptor].
        Here, we consider the case that a single substrate must bind to any
        receptor and calculate the coverage ratio of the receptors given many
        realizations.
        """
        cnt_s, cnt_r = state.energies.shape
        subs = range(cnt_s) #< all possible substrate indices

        if self.temperature == 0:
            # determine maximal energies for each substrate
            Emax = state.energies.max(axis=1)
            weights = (state.energies == Emax[:, np.newaxis]).astype(np.double)
        
        else:
            # calculate Boltzmann factors
            weights = np.exp(state.energies/self.temperature)
            
        # iterate over all substrate combinations
        probs = np.empty((self.get_input_dim(state), cnt_r))
        for k, sub in enumerate(itertools.combinations(subs, self.num)):
            # calculate interaction probabilities
            probs[k, :] = weights[sub, :].sum(axis=0)
                
        # normalize for each substrate across all receptors
        # => scenario in which sub_ids binds to exactly one receptor
        probs /= np.sum(probs, axis=1)[:, None]
        # TODO: check whether this is the natural normalization
        
        return probs
    
    
    # Copy the `get_output_vector` method from the base class. This copy
    # is necessary, because we want to optimize the base method and its sub
    # functions using monkey patching
    get_output_vector = copy_func(DetectSingleSubstrate.get_output_vector)


    def get_input_dim(self, model_or_state):
        """ number of different inputs """
        # number of possibilities of choosing `num` items out of `cnt_s`
        return scipy.misc.comb(model_or_state.num_substrates, self.num,
                               exact=True)
    
    
        
class MeasureMultipleSubstrates(DetectSingleSubstrate):
    """ experiment in which a receptor array is used to measure the
    concentrations of a mixture of substrate out of a given list """ 
    input_dim = 100 #< how many concentrations combinations should we try
    
    def __init__(self, num_substrates, concentration_range, temperature=1,
                 threshold=1):
        """ initialize the experiment with parameters:
        `num_substrates` is the number of sub_ids perceived concurrently
        `concentration_range` is a tuple defining the minimal and maximal conc. 
        `temperature` influences the binding probabilities
        `threshold` determines when receptors signal
        """
        self.num = num_substrates
        self.cmin, self.cmax = concentration_range
        self.temperature = temperature
        self.threshold = threshold
        
        # arrays that will contain the sub_ids concentrations
        self.sub_ids = None
        self.concs = None
        
        self._cache = {}


    def __repr__(self):
        return ('%s(num=%r, concentration_range=%r, temperature=%r, '
                'threshold=%r)' %
                (self.__class__.__name__, self.num, (self.cmin, self.cmax),
                 self.temperature, self.threshold))
        
        
    def choose_input(self, cnt_s):
        """ choose random input """
        # choose the substrates that will appear in the input
        sub_ids = [np.random.choice(cnt_s, self.num, replace=False)
                   for _ in xrange(self.input_dim)]
        self.sub_ids = np.array(sub_ids, np.int)
        # choose the associated concentrations
        lmin, lmax = np.log(self.cmin), np.log(self.cmax)
        lconcs = np.random.uniform(lmin, lmax, (self.input_dim, self.num))
        self.concs = np.exp(lconcs)
        
        assert self.sub_ids.shape == self.concs.shape
        assert np.all(self.cmin <= self.concs)
        assert np.all(self.concs <= self.cmax)
        
        
    @classmethod
    def create_test_instance(cls):
        """ creates a instance of the class with random parameters """
        obj = cls(num_substrates=random.randrange(1, 4),
                  concentration_range=(0.01 * random.random(),
                                       0.1 + random.random()),
                  temperature=0.1 + 3*random.random(),
                  threshold=random.random())
        return obj
    
    
    def get_binding_probabilities(self, state):
        """
        Calculates the probability of substrates binding to receptors given
        interaction energies described by a matrix energies[substrate, receptor].
        Here, we consider the case that a single substrate must bind to any
        receptor and calculate the coverage ratio of the receptors given many
        realizations.
        """
        cnt_s, cnt_r = state.energies.shape

        # choose sub_ids during the first call 
        if self.sub_ids is None:
            self.choose_input(cnt_s)
            
        if self.temperature == 0:
            # determine maximal energies for each substrate
            Emax = state.energies.max(axis=1)
            weights = (state.energies == Emax[:, np.newaxis]).astype(np.double)
        
        else:
            # calculate Boltzmann factors
            weights = np.exp(state.energies/self.temperature)
            
        # iterate over all chosen inputs
        probs = np.empty((self.input_dim, cnt_r))
        for k, (s, c) in enumerate(itertools.izip(self.sub_ids, self.concs)):
            # calculate interaction probabilities
            probs[k, :] = (weights[s, :] * c[:, np.newaxis]).sum(axis=0)
                
        return probs
    
    
    def get_output_vector(self, state):
        """ calculate output vector for given input """
        # calculate the resulting binding characteristics
        probs = self.get_binding_probabilities(state)
        # threshold to get the response
        output = (probs >= self.threshold)
        # encode output in single integer
        binary_base = self.binary_base(state.num_receptors)
        return np.dot(output, binary_base)


    def get_input_dim(self, model_or_state):
        """ number of different inputs """
        return self.input_dim
    
                
