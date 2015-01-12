'''
Created on Jan 12, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import random
import math

import numpy as np

np_version = [int(t) for t in np.__version__.split('.')]
if np_version >= [1, 9]:
    # implement a count function using numpy
    def count_items(arr):
        return np.unique(arr, return_counts=True)[1]
else:
    # implement a count function using python Counter
    from collections import Counter
    def count_items(arr):
        return list(Counter(arr).itervalues())

try:
    import numba
except ImportError:
    numba = None
    print('Numba was not found. Slow functions will be used')


if numba:
    # define fast numerical functions using numba
    
    @numba.jit('void(i8[:, :], i8[:], i8[:])')
    def _interaction_substrates_receptor(substrates2, receptor, out):
        """ update interaction energies of the `receptor` with the substrates
        and save them in `out` """ 
        cnt_s, l_s2 = substrates2.shape
        l_r = len(receptor)
        for s in xrange(cnt_s):
            overlap_max = -1
            for i in xrange(l_s2//2):
                overlap = 0
                for k in xrange(l_r):
                    if substrates2[s, i + k] == receptor[k]:
                        overlap += 1
                overlap_max = max(overlap_max, overlap)
            out[s] = overlap_max


else:
    # define the slow substitute functions working without numba
    
    def _interaction_substrates_receptor(substrates2, receptor, out):
        """ update interaction energies of the `receptor` with the substrates
        and save them in `out` """ 
        l_s, l_r = substrates2.shape[1]//2, len(receptor)
        out[:] = np.max([
            np.sum(substrates2[:, i:i+l_r] == receptor[np.newaxis, :],
                   axis=1)
            for i in xrange(l_s)
        ], axis=0)



class SubstrateReceptorInteraction1D(object):
    """ class that evaluates the interaction energies between a set of
    substrates and a set of receptors.
    """
    temperature = 1
    threshold = 0.2
        
    clone_cache_keys = ('energies2',)
    
    def __init__(self, substrates, receptors, colors=None, cache=None,
                 energies=None):
        self.substrates = substrates
        self.receptors = receptors
        if colors is None:
            self.colors = max(substrates.max(), receptors.max()) + 1
        else:
            self.colors = colors
        if cache is None:
            self._cache = {}
        else:
            self._cache = cache
        if energies is None:
            self.energies = self.get_energies()
        else:
            self.energies = energies
        
        
    def __repr__(self):
        cnt_s, l_s = self.substrates.shape
        cnt_r, l_r = self.receptors.shape
        return ('%s(%d substrates of l=%d, %d receptors of l=%d)' %
                (self.__class__.__name__, cnt_s, l_s, cnt_r, l_r))
        
        
    def copy(self):
        """ copies the current interaction state to allow the receptors to
        be mutated. The substrates will be shared between this object and its
        copy """
        return self.__class__(self.substrates, self.receptors.copy(),
                              self.colors, self._cache, self.energies.copy())
        
        
    @property
    def substrates2(self):
        """ return repeated substrates to implement periodic boundary
        conditions """
        if 'substrates2' not in self._cache:
            self._cache['substrates2'] = np.repeat(self.substrates, 2, axis=1)
        return self._cache['substrates2']
        

    def get_energies(self):
        """ this assumes small receptors and large substrates
        TODO: lift this constraint
        """
        # get dimensions
        l_s = self.substrates.shape[1]
        l_r = self.receptors.shape[1]

        # calculate the energies with a sliding window
        Es = np.array([
            np.sum(self.substrates2[:, np.newaxis, i:i+l_r] ==
                       self.receptors[np.newaxis, :, :],
                   axis=2)
            for i in xrange(l_s)
        ])

        return Es.max(axis=0)
           
        
    def randomize_receptors(self):
        """ choose a completely new set of receptors """
        self.receptors = np.random.randint(0, 2, size=self.receptors.shape)
        self.energies = self.get_energies()
    
    
    @property
    def color_alternatives(self):
        """ return repeated substrates to implement periodic boundary
        conditions """
        if 'color_alternatives' not in self._cache:
            colors = [np.r_[0:c, c+1:self.colors]
                      for c in xrange(self.colors)] 
            self._cache['color_alternatives'] = colors
        return self._cache['color_alternatives']
    
        
    def mutate_receptors(self):
        """ mutate a single, random receptor """
        cnt_r, l_r = self.receptors.shape

        # choose one point on one receptor that will be mutated        
        x = random.randint(0, cnt_r - 1)
        y = random.randint(0, l_r - 1)
        if self.colors == 2:
            # restricted to two colors => flip color
            self.receptors[x, y] = 1 - self.receptors[x, y]
        else:
            # more than two colors => use random choice
            clrs = self.color_alternatives[self.receptors[x, y]]
            self.receptors[x, y] = np.random.choice(clrs)

        # recalculate the interaction energies of the changed receptor        
        _interaction_substrates_receptor(self.substrates2,
                                         self.receptors[x],
                                         self.energies[:, x])
        
        #assert np.sum(np.abs(self.energies - self.get_energies())) == 0
        

    def get_binding_probabilities(self):
        """
        Calculates the probability of substrates binding to receptors given
        interaction energies described by a matrix energies[substrate, receptor].
        Here, we consider the case that a single substrate must bind to any
        receptor and calculate the coverage ratio of the receptors given many
        realizations.
        """
        # calculate interaction probabilities
        probs = np.exp(self.energies/self.temperature)
        # normalize for each substrate across all receptors
        probs /= np.sum(probs, axis=1)[:, None]
        return probs     
    
        
    def get_output_vector(self):
        """ calculate output vector for given receptors """
        # calculate the resulting binding characteristics
        probs = self.get_binding_probabilities()
        # threshold to get the response
        cnt_r = probs.shape[1]
        # threshold
        output = (probs > self.threshold)
        
        
#         res = 0
#         base = 1
#         for out in output:
#             res += out*base
#             base *= self.colors
#             
#         return res
        # encode output in single integer
        vec_r = self.colors**np.arange(cnt_r)
        return np.dot(output, vec_r) 
        
    
    def get_mutual_information(self):
        """ calculate the mutual information between substrates and receptors
        """
        output = self.get_output_vector()
        
        # determine the contribution from the frequency distribution        
        fs = count_items(output)
        # assert cnt_s == sum(fs)
        logsum_a = np.sum(fs*np.log(fs))
        
        cnt_s = len(output)
        return math.log(cnt_s) - logsum_a/cnt_s

    
    @property
    def mutual_information_max(self):
        """ return upper bound for mutual information """
        #TODO determine theoretical mutual information based on 
        # length and number of receptors/substrates
        
        cnt_r, l_r = self.receptors.shape
        # there is a vector space of possible receptors, spanned
        # by the dim=min(cnt_r, l_r) basis vectors
        # => the possible number of receptors is 2^dim
        return min(cnt_r, l_r) * np.log(2)

