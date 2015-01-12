'''
Created on Jan 12, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

from collections import Counter
import random

import numpy as np

from simanneal import Annealer



class SubstrateReceptorInteraction(object):
    """ class that evaluates the interaction energies between a set of
    substrates and a set of receptors.
    """
    
    clone_cache_keys = ('energies2',)
    
    def __init__(self, substrates, receptors):
        self.substrates = substrates
        self.receptors = receptors
        self._cache = {}
        
        
    def copy(self):
        """ copies the current interaction state to allow the receptors to
        be mutated. The substrates will be shared between this object and its
        copy """
        obj = self.__class__(self.substrates, self.receptors.copy())
        for key in self.clone_cache_keys:
            obj._cache[key] = self._cache[key]

        
    @property
    def substrates2(self):
        if 'substrates2' in self._cache:
            self._cache['substrates2'] = np.repeat(self.substrates, 2, axis=1)
        return self._cache['substrates2']

    
    @property
    def energies(self):
        if 'energies' not in self._cache:
            self._cache['energies'] = self.get_energies()
        return self._cache['energies']
        

    def get_energies(self):
        """ this assumes small receptors and large substrates
        TODO: lift this constraint
        """
        # get dimensions
        cnt_s, l_s = self.substrates.shape
        cnt_r, l_r = self.receptors.shape

        # repeat substrate to implement periodic boundary conditions
        substrates = np.repeat(self.substrates, 2, axis=1)

        # calculate the energies with the sliding window
        Es = np.array([
            np.sum(substrates[:, np.newaxis, i:i+l_r] ==
                       self.receptors[np.newaxis, :, :],
                   axis=2)
            for i in xrange(l_s)
        ])

        return Es.max(axis=0)
           
        
    def mutate(self):
        pass



class OptimalReceptors(Annealer):
    """ class for finding optimal receptor distribution """
    
    Tmax =  1e3     # Max (starting) temperature
    Tmin =  1e-4    # Min (ending) temperature
    steps = 1e5     # Number of iterations
    updates = 2     # Number of outputs
    copy_strategy = 'method'

    temperature = 1
    threshold = 0.2
    
    
    def __init__(self, substrates, l_r, cnt_r):
        """
        l_r length of a single receptor
        cnt_r number of receptors
        """
        self.substrates = substrates
        # only support case K=2 first
        receptors = np.random.randint(0, 2, size=(cnt_r, l_r))
        super(OptimalReceptors, self).__init__(receptors)
    
    
    def move(self):
        """ change a single bit in any of the receptor vectors """
        # TODO: only update the single row that changed in the energy matrix
        lx, ly = self.state.shape
        x = random.randint(0, lx - 1)
        y = random.randint(0, ly - 1)
        # restrict to case K=2 first => flip color
        self.state[x, y] = 1 - self.state[x, y]
       
        
    def get_energies(self, substrates, receptors):
        """ this assumes small receptors and large substrates
        TODO: lift this constraint
        """
        # get dimensions
        cnt_s, l_s = substrates.shape
        cnt_r, l_r = receptors.shape

        # repeat substrate to implement periodic boundary conditions
        substrates = np.repeat(substrates, 2, axis=1)

        # calculate the energies with the sliding window
        Es = np.array([
            np.sum(substrates[:, np.newaxis, i:i+l_r] == receptors[np.newaxis, :, :], axis=2)
            for i in xrange(l_s)
        ])

        return Es.max(axis=0)
    

    def get_binding_probabilities(self, energies):
        """
        Calculates the probability of substrates binding to receptors given
        interaction energies described by a matrix energies[substrate, receptor].
        Here, we consider the case that a single substrate must bind to any
        receptor and calculate the coverage ratio of the receptors given many
        realizations.
        """
        # calculate interaction probabilities
        probs = np.exp(energies/self.temperature)
        # normalize for each substrate across all receptors
        probs /= np.sum(probs, axis=1)[:, None]
        return probs        
    
        
    def get_output_vector_probs(self, probs):
        """ calculate output vector from binding probabilities """
        cnt_s, cnt_r = probs.shape
        # threshold
        output = (probs > self.threshold)
        # convert to integer
        vec_r = 2**np.arange(cnt_r)
        return np.dot(output, vec_r) 
    
       
    def mutual_information(self, output):
        """ calculate mutual information between output signal
        output is a matrix that maps the substrate onto an integer
            indicating which receptors are activated
        """
        cnt_s = len(output)
        ns_a = Counter(output)
        # assert cnt_s == sum(a for a in ns_a.itervalues())

        logsum_a = sum(a*np.log(a) for a in ns_a.itervalues())
        MI = np.log(cnt_s) - logsum_a/cnt_s
        return MI  
    
    
    def get_output_vector(self, receptors):
        """ calculate output vector for given receptors """
        # calculate their interaction energies
        energies = self.get_energies(self.substrates, receptors)
        # calculate the resulting binding characteristics
        probs = self.get_binding_probabilities(energies)
        # threshold to get the response
        return self.get_output_vector_probs(probs)
    
    
    @property
    def mutual_information_max(self):
        """ return upper bound for mutual information """
        cnt_r, l_r = self.state.shape
        # there is a vector space of possible receptors, spanned
        # by the dim=min(cnt_r, l_r) basis vectors
        # => the possible number of receptors is 2^dim
        return min(cnt_r, l_r) * np.log(2)
    
    
    def get_mutual_information(self, substrates, receptors):
        """ calculate the mutual information between substrate in receptors """
        # calculate their interaction energies
        #energies = get_energies(self.substrates, receptors)
        # calculate the resulting binding characteristics
        #probs = get_binding_probabilities(energies)
        # threshold to get the response
        output = self.get_output_vector(receptors)
        # calculate the mutual information
        return self.mutual_information(output)
        
        
    def energy(self):
        """ returns the energy of the current state """
        return -self.get_mutual_information(self.substrates, self.state)