'''
Created on Jan 23, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import random

import numpy as np

from .utils import calc_entropy



class DetectSingleSubstrate(object):
    """ experiment in which a receptor array is used to identify a substrate
    out of a given list """ 
    
    def __init__(self, temperature=1, threshold=1):
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
        if random.randrange(0, 2):
            obj.temperature = random.randrange(0, 3)
        else:
            obj.temperature = 0
        obj.threshold = random.random() + 0.5
        return obj

    
    def binary_base(self, cnt_r):
        """ return repeated substrates to implement periodic boundary
        conditions """
        try:
            return self._cache['binary_base'][cnt_r]
        except KeyError:
            # one of the keys was not set
            binary_base = np.exp2(np.arange(cnt_r))
            try:
                self._cache['binary_base'][cnt_r] = binary_base
            except KeyError:
                # the cache was not initialized at all
                self._cache['binary_base'] = {cnt_r: binary_base}
                
            return binary_base


    def get_binding_probabilities(self, state):
        """
        Calculates the probability of substrates binding to receptors given
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
        """ calculate output vector for given receptors """
        # calculate the resulting binding characteristics
        probs = self.get_binding_probabilities(state)
        # threshold to get the response
        cnt_r = state.num_receptors
        output = (probs > self.threshold/cnt_r)
        # encode output in single integer
        binary_base = self.binary_base(cnt_r)
        return np.dot(output, binary_base) 
        
    
    def get_mutual_information(self, state):
        """ calculate the mutual information for a state """
        output = self.get_output_vector(state)
        
        # determine the contribution from the output distribution        
        entropy_o = calc_entropy(output)
        
        cnt_s = len(output)
        return np.log2(cnt_s) - entropy_o/cnt_s    


    def get_max_mutual_information(self, model):
        """ return upper bound for mutual information of a model """
        # maximal mutual information restricted by the output
        MI_receptors = np.log2(model.output_dim)
        
        # maximal mutual information restricted by substrates
        MI_substrates = np.log2(model.num_substrates)
        
        return min(MI_receptors, MI_substrates)
    
    