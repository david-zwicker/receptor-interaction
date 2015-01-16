'''
Created on Jan 12, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>

Module for handling the interaction between a set of substrates and
corresponding receptors. Both substrates and receptors are described by chains
of chains, where each chain can have a color. We consider periodic boundary
conditions, such that these chains are equivalent to necklaces in combinatorics.

Each chain is represented by an integer numpy array of length l. Sets of 
chains are represented by a 2D-numpy array where the first dimension
corresponds to different chains.

Special objects are used to represent the set of all unique chains and the
interaction between sets of substrate and receptor chains.
'''

from __future__ import division

import fractions
import itertools
import random
import math
import numpy as np
import timeit

from collections import Counter
from scipy.misc import comb
from scipy.stats import itemfreq


def get_fastest_entropy_function():
    """ returns a function that calculates the entropy of a array of integers
    Here, several alternative definitions are tested an the fastest one is
    returned """ 
    def entropy_numpy(arr):
        """ entropy function based on numpy.unique """
        fs = np.unique(arr, return_counts=True)[1]
        return np.sum(fs*np.log(fs))
    def entropy_scipy(arr):
        """ entropy function based on scipy.stats.itemfreq """
        fs = itemfreq(arr)[:, 1]
        return np.sum(fs*np.log(fs))
    def entropy_counter(arr):
        """ entropy function based on collections.Counter """
        return sum(val*math.log(val)
                   for val in Counter(arr).itervalues())

    test_array = np.random.random_integers(0, 10, 100)
    func_fastest, dur_fastest = None, np.inf
    for test_func in (entropy_numpy, entropy_scipy, entropy_counter):
        try:
            dur = timeit.timeit(lambda: test_func(test_array), number=1000)
            print dur
        except TypeError:
            # older numpy versions don't support `return_counts`
            pass
        else:
            if dur < dur_fastest:
                func_fastest, dur_fastest = test_func, dur

    return func_fastest

calc_entropy = get_fastest_entropy_function()



# try importing numba for speeding up calculations
try:
    import numba
except ImportError:
    numba = None
    print('Numba was not found. Slow functions will be used')


#===============================================================================
# BASIC CHAIN/NECKLACE FUNCTIONS
#===============================================================================


def remove_redundant_chains(chains):
    """ removes chains that are the same (because of periodic boundary
    conditions) """
    l_s = len(chains[0])
    colors = chains.max() + 1
    base = colors**np.arange(l_s)
    
    # calculate an characteristic number for each substrate
    characters = [min(np.dot(s2[k:k + l_s], base)
                      for k in xrange(l_s))
                  for s2 in np.c_[chains, chains]]
    
    _, idx = np.unique(characters, return_index=True)
    return chains[idx]

    

class Chains(object):
    """ class that represents all chains of length l """
    
    def __init__(self, l, colors=2):
        self.l = l
        self.colors = colors
        
        
    def __repr__(self):
        return ('%s(l=%d, colors=%d)' %
                (self.__class__.__name__, self.l, self.colors))
        
        
    def __len__(self):
        """ returns the number of unique chains of length l
        This number is equivalent to len(list(self)), but uses a more efficient
        mathematical solution based on Eq. 1 in the paper
            F. Ruskey, C. Savage, T. Min Yih Wang, J. of Algorithms, 13 (1992).
        """
        Nb = 0
        for d in xrange(1, self.l + 1):
            Nb += self.colors ** fractions.gcd(self.l, d)
        Nb //= self.l
        
        return Nb
    
    
    def __iter__(self):
        """ returns a generator for all unique chains.

        Note that the return value is a view into an internal array. Use
            [v.copy() for v in self]
        instead of
            list(self)
        to create a list of all chains.  

        This method uses the FKM algorithm published in
        * H. Fredricksen and J. Maiorana, Necklaces of beads in k colors and
            k-ary de Bruijn sequences, Discrete Math. 23 (1978), 207-210.
        * H. Fredricksen and I. J. Kessler, An algorithm for generating 
            necklaces of beads in two colors, Discrete Math. 61 (1986), 181-188.
        """
        l, k = self.l, self.colors
        a = np.zeros(l, np.int)
        yield a
        
        i = l - 1
        while True:
            a[i] += 1
            for j in xrange(l - i - 1):
                a[j + i + 1] = a[j]
            if l % (i + 1) == 0:
                yield a
            i = l - 1
            while a[i] == k - 1:
                i -= 1
                if i < 0:
                    return


    def to_list(self):
        """ returns an array of all unique chains of length `l` with `colors`
        possible colors per chain """
        return [v.copy() for v in self]
    
    
    def to_array(self):
        """ returns an array of all unique chains of length `l` with `colors`
        possible colors per chain """
        res = np.zeros((len(self), self.l), np.int)
        for k, c in enumerate(self):
            res[k, :] = c
        return res
    
    
    def choose_unique(self, cnt):
        """ chooses `cnt` unique chains consisting of `l` chains with
        `colors` unique colors """
        l, colors = self.l, self.colors
        
        base = colors ** np.arange(l)
        
        chains, characters = [], set()
        counter = 0
        while len(chains) < cnt:
            # choose a random substrate and determine its character
            s = np.random.randint(0, colors, size=l)
            s2 = np.r_[s, s]
            character = min(np.dot(s2[k:k + l], base)
                            for k in xrange(l))
            counter += 1
            # add the substrate if it is not already in the list
            if character not in characters:
                chains.append(s)
                characters.add(character)
                counter = 0
                
            # interrupt the search if no substrate can be found
            if counter > 1000:
                raise RuntimeError('Cannot find %d chains of length %d' % 
                                   (cnt, l))
        return chains




class ChainsCollection(object):
    """ class that represents all possible collections of `cnt` distinct chains
     of length `l` """
     
    def __init__(self, cnt, l, colors=2):
        self.cnt = cnt
        self.chains = Chains(l, colors)
        
        # currently used to generate random chains
        self.l = l
        self.colors = colors
        
        
    def __repr__(self):
        return ('%s(cnt=%d, l=%d, colors=%d)' %
                (self.__class__.__name__, self.cnt, self.l, self.colors))
        
    
    def __len__(self):
        """ returns the number of possible receptor combinations """
        num = len(self.chains)
        return comb(num, self.cnt, exact=True)
               

    def __iter__(self):
        """ generates all possible receptor combinations """
        chains = self.chains.to_array()
        for chain_col in itertools.combinations(chains, self.cnt):
            yield np.array(chain_col) 


    def get_zero_chains(self):
        """ returns a list with chains of color 0 """
        return np.zeros((self.cnt, self.l), np.int)


    def get_random_chains(self):
        """ returns a random set of chains """
        # TODO: ensure that the chains are unique
        # TODO: ensure that the chains would also appear in the iteration
        return np.random.randint(0, self.colors, size=(self.cnt, self.l))
    
    
#===============================================================================
# FUNCTIONS FOR CHAIN/NECKLACE INTERACTIONS
#===============================================================================

    
class ChainsInteraction(object):
    """ class that represents the interaction between a set of substrates and a
    set of receptors.
    """
    temperature = 1  #< temperature for equilibrium binding
    threshold = 1    #< threshold above which the receptor responds

    
    def __init__(self, substrates, receptors, colors,
                 cache=None, energies=None):
        
        self.substrates = np.asarray(substrates)
        self.receptors = np.asarray(receptors)
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
        return ('%s(%d Substrates(l=%d), %d Receptors(l=%d), colors=%d)' %
                (self.__class__.__name__, cnt_s, l_s, cnt_r, l_r, self.colors))
        

    def check_consistency(self):
        """ consistency check on the number of receptors and substrates """
        # check the supplied substrates
        unique_substrates = remove_redundant_chains(self.substrates)
        redundant_count = len(self.substrates) - len(unique_substrates)
        if redundant_count:
            raise RuntimeWarning('There are %d redundant substrates' % 
                                 redundant_count)
        
        # check the supplied receptors
        cnt_r, l_r = self.receptors.shape
        chains = Chains(l_r, self.colors)
        if cnt_r > len(chains):
            raise RuntimeWarning('The number of supplied receptors is larger '
                                 'than the number of possible unique ones.')
    
        unique_receptors = remove_redundant_chains(self.receptors)
        redundant_count = len(self.receptors) - len(unique_receptors)
        if redundant_count:
            raise RuntimeWarning('There are %d redundant receptors' % 
                                 redundant_count)
    
        
    def copy(self):
        """ copies the current interaction state to allow the receptors to
        be mutated. The substrates and the cache will be shared between this
        object and its copy """
        return self.__class__(self.substrates, self.receptors.copy(),
                              self.colors, self._cache, self.energies.copy())
        
        
    @property
    def substrates2(self):
        """ return repeated substrates to implement periodic boundary
        conditions """
        try:
            return self._cache['substrates2']
        except KeyError:
            self._cache['substrates2'] = np.c_[self.substrates, self.substrates]
            return self._cache['substrates2']
        

    def update_energies(self):
        """
        updates the energies between the substrates and the receptors
        this assumes small receptors and large substrates
        TODO: lift this constraint
        """
        _get_energies(self.substrates2, self.receptors, self.energies)

           
    def get_energies(self):
        """ calculates all the energies between the substrates and the
        receptors """
        cnt_s = len(self.substrates)
        cnt_r = len(self.receptors)
        self.energies = np.empty((cnt_s, cnt_r))
        self.update_energies()
        return self.energies
                      
        
    def randomize_receptors(self):
        """ choose a completely new set of receptors """
        self.receptors = np.random.randint(0, self.colors,
                                           size=self.receptors.shape)
        self.update_energies()
    
    
    @property
    def color_alternatives(self):
        """ look-up table for changing the color of a single block """
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
            idx = random.randint(0, self.colors - 2)
            self.receptors[x, y] = clrs[idx]

        # recalculate the interaction energies of the changed receptor        
        _update_energies(self.substrates2, self.receptors[x],
                         self.energies[:, x])
        
        #assert np.sum(np.abs(self.energies - self.update_energies())) == 0
        

    def get_binding_probabilities(self):
        """
        Calculates the probability of substrates binding to receptors given
        interaction energies described by a matrix energies[substrate, receptor].
        Here, we consider the case that a single substrate must bind to any
        receptor and calculate the coverage ratio of the receptors given many
        realizations.
        """
        if self.temperature == 0:
            # determine minimal energies for each substrate
            Emin = self.energies.min(axis=1)
            # determine the receptors that are activated
            probs = (self.energies == Emin[:, np.newaxis]).astype(np.double)
            
        else:
            # calculate interaction probabilities
            probs = np.exp(self.energies/self.temperature)
            
        # normalize for each substrate across all receptors
        # => scenario in which each substrate binds to exactly one receptor
        probs /= np.sum(probs, axis=1)[:, None]
        
        return probs
    
    
    @property
    def binary_base(self):
        """ return repeated substrates to implement periodic boundary
        conditions """
        try:
            return self._cache['binary_base']
        except KeyError:
            cnt_r = len(self.receptors)
            self._cache['binary_base'] = 2 ** np.arange(cnt_r)
            return self._cache['binary_base']
        
        
    def get_output_vector(self):
        """ calculate output vector for given receptors """
        # calculate the resulting binding characteristics
        probs = self.get_binding_probabilities()
        # threshold to get the response
        output = (probs > self.threshold/self.receptor_count)
        # encode output in single integer
        return np.dot(output, self.binary_base) 
        
    
    def get_mutual_information(self):
        """ calculate the mutual information between substrates and receptors
        """
        output = self.get_output_vector()
        
        # determine the contribution from the output distribution        
        entropy_o = calc_entropy(output)
        
        cnt_s = len(output)
        return math.log(cnt_s) - entropy_o/cnt_s

    
    @property
    def substrate_count(self):
        return len(self.substrates)
    
    @property
    def receptor_count(self):
        return len(self.receptors)
    
    @property
    def output_count(self):
        """ number of different outputs """
        # the 2 is due to the binary output of the receptors
        return 2 ** len(self.receptors)
    
    @property
    def mutual_information_max(self):
        """ return upper bound for mutual information """
        # maximal mutual information restricted by the output
        MI_receptors = np.log(self.output_count)
        
        # maximal mutual information restricted by substrates
        MI_substrates = np.log(self.substrate_count)
        
        return min(MI_receptors, MI_substrates)



class ChainsInteractionCollection(object):
    """ class that represents all possible combinations of substrate and
    receptor interactions """
    
    def __init__(self, substrates, cnt_r, l_r, colors):      
        try:
            self.substrates = substrates.to_array()
        except AttributeError:
            self.substrates = np.asarray(substrates)
        self.receptors_collection = ChainsCollection(cnt_r, l_r, colors)
        self.colors = colors


    def __repr__(self):
        return ('%s(%s, cnt=%d, l=%d, colors=%d)' %
                (self.__class__.__name__, repr(self.substrates), self.cnt_r,
                 self.l_r, self.colors))
        
        
    def __len__(self):
        return len(self.receptors_collection)
    
    
    def __iter__(self):
        """ generates all possible chain interactions """
        #TODO: try to increase the performance by
        #    * improving update_energies
        #    * taking advantage of partially calculated energies
        
        # create an initial state object
        receptors = self.receptors_collection.get_zero_chains()
        state = ChainsInteraction(self.substrates, receptors, self.colors)
        
        # iterate over all receptors and update the state object
        for receptors in self.receptors_collection:
            state.receptors = receptors
            state.update_energies()
            yield state
        
    
    def get_random_state(self):
        """ returns a randomly chosen chain interaction """
        receptors = self.receptors_collection.get_random_chains()
        return ChainsInteraction(self.substrates, receptors, self.colors)
    
    
    def estimate_computation_speed(self):
        """ estimate the speed of the computation of a single iteration """
        def func():
            """ test function for estimating the speed """
            self.get_random_state().get_mutual_information()
        # call the function once to make sure that just in time compilation is
        # not timed
        func()
        
        # try different repetitions until the total run time is about 1 sec 
        number, duration = 1, 0
        while duration < 0.1:
            number *= 10
            duration = timeit.timeit(func, number=number)
            
        return duration/number
    
    
#===============================================================================
# FAST FUNCTIONS USING NUMBA
#===============================================================================


if numba:
    # TODO: monkey patch these functions into the class
    # define fast numerical functions using numba
    
    @numba.jit(nopython=True)
    def _update_energies(substrates2, receptor, out):
        """ update interaction energies of the `receptor` with the substrates
        and save them in `out` """ 
        cnt_s, l_s2 = substrates2.shape
        l_r, l_s = len(receptor), l_s2 // 2
        for s in xrange(cnt_s):
            overlap_max = -1
            for i in xrange(l_s):
                overlap = 0
                for k in xrange(l_r):
                    if substrates2[s, i + k] == receptor[k]:
                        overlap += 1
                overlap_max = max(overlap_max, overlap)
            out[s] = overlap_max


    @numba.jit(nopython=True)
    def _get_energies(substrates2, receptors, out):
        """ calculates all the interaction energies between the substrates and
        the receptors and stores them in `out` """
        cnt_s, l_s2 = substrates2.shape
        l_s = l_s2 // 2
        cnt_r, l_r = receptors.shape
        # check all substrates versus all receptors
        for x in xrange(cnt_s):
            for y in xrange(cnt_r):
                overlap_max = 0
                # find the maximum over all starting positions
                for start in xrange(l_s):
                    overlap = 0
                    # count overlap along receptor length
                    for k in xrange(l_r):
                        if substrates2[x, start + k] == receptors[y, k]:
                            overlap += 1
                    overlap_max = max(overlap_max, overlap)
                out[x, y] = overlap_max


else:
    # define the slow substitute functions working without numba
    
    def _update_energies(substrates2, receptor, out):
        """ update interaction energies of the `receptor` with the substrates
        and save them in `out` """ 
        l_s, l_r = substrates2.shape[1] // 2, len(receptor)
        out[:] = reduce(np.maximum, (
            np.sum(substrates2[:, i:i+l_r] == receptor[np.newaxis, :],
                   axis=1)
            for i in xrange(l_s)
        ))
            
            
    def _get_energies(substrates2, receptors, out):
        """ calculates all the interaction energies between the substrates and
        the receptors and stores them in `out` """
        # get dimensions
        l_s = substrates2.shape[1] // 2
        l_r = receptors.shape[1]
    
        # calculate the energies with a sliding window
        out[:] = reduce(np.maximum, (
            np.sum(substrates2[:, np.newaxis, i:i+l_r] ==
                       receptors[np.newaxis, :, :],
                   axis=2)
            for i in xrange(l_s)
        ))


