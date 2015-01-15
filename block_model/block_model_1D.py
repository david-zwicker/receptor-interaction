'''
Created on Jan 12, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import fractions
import random
import math
import itertools

from scipy.misc import comb

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


# try importing numba for speeding up calculations
try:
    import numba
except ImportError:
    numba = None
    print('Numba was not found. Slow functions will be used')



def euler_phi(n):
    """ evaluates the Euler phi function for argument `n`
    See http://en.wikipedia.org/wiki/Euler%27s_totient_function
    Implementation based on http://stackoverflow.com/a/18114286/932593
    """
    amount = 0

    for k in xrange(1, n + 1):
        if fractions.gcd(n, k) == 1:
            amount += 1

    return amount



def blocks1D_remove_redundant(blocks):
    """ removes blocks that are the same (because of periodic boundary
    conditions) """
    l_s = len(blocks[0])
    colors = blocks.max() + 1
    base = colors**np.arange(l_s)
    
    # calculate an characteristic number for each substrate
    characters = [min(np.dot(s2[k:k + l_s], base)
                      for k in xrange(l_s))
                  for s2 in np.c_[blocks, blocks]]
    
    _, idx = np.unique(characters, return_index=True)
    return blocks[idx]



def blocks1D_choose_unique(cnt, l, colors=2):
    """ chooses `cnt` unique substrates consisting of `l` blocks with
    `colors` unique colors """
    base = colors ** np.arange(l)
    
    substrates, characters = [], set()
    counter = 0
    while len(substrates) < cnt:
        # choose a random substrate and determine its character
        s = np.random.randint(0, colors, size=l)
        s2 = np.r_[s, s]
        character = min(np.dot(s2[k:k + l], base)
                        for k in xrange(l))
        counter += 1
        # add the substrate if it is not already in the list
        if character not in characters:
            substrates.append(s)
            characters.add(character)
            counter = 0
            
        # interrupt the search if no substrate can be found
        if counter > 1000:
            raise RuntimeError('Cannot find %d substrates of length %d' % 
                               (cnt, l))
    return substrates



def blocks1D_get_unique(l, colors=2):
    """ returns a list of all unique blocks of length `l` with `colors`
    possible colors per block """
    blocks = np.array(list(itertools.product(range(colors), repeat=l)))
    return blocks1D_remove_redundant(blocks)



def block1D_count(l, colors=2):
    """ returns the number of unique blocks of length l
    This number is equivalent to len(blocks1D_get_unique(l, colors))
    However, we use a more efficient mathematical solution based on
        http://en.wikipedia.org/wiki/Necklace_(combinatorics)
    """
    Nr = 0
    for d in xrange(1, l + 1):
        if l % d == 0:
            Nr += euler_phi(d) * colors**(l//d)
    Nr //= l
    
    return Nr
    
    
    
class Blocks1DInteraction(object):
    """ class that evaluates the interaction energies between a set of
    substrates and a set of receptors.
    """
    temperature = 1 #< temperature for equilibrium binding
    threshold = 0.2 #< threshold above which the receptor responds
        
    clone_cache_keys = ('energies2',)
    
    def __init__(self, substrates, receptors, colors=None, cache=None,
                 energies=None):
        
        self.substrates = np.asarray(substrates)
        self.receptors = np.asarray(receptors)
        
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


    def check_consistency(self):
        """ consistency check on the number of receptors and substrates """
        unique_substrates = blocks1D_remove_redundant(self.substrates)
        redundant_count = len(self.substrates) - len(unique_substrates)
        if redundant_count:
            raise RuntimeWarning('There are %d redundant substrates' % 
                                 redundant_count)
        
        cnt_r, l_r = self.receptors.shape
        if cnt_r > block1D_count(l_r, self.colors):
            raise RuntimeWarning('The number of supplied receptors is larger '
                                 'than the number of possible unique ones.')
    
        
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
        # calculate interaction probabilities
        probs = np.exp(self.energies/self.temperature)
        # normalize for each substrate across all receptors
        # => scenario in which each substrate binds to exactly one receptor
        probs /= np.sum(probs, axis=1)[:, None]
        return probs     
    
    
    @property
    def color_base(self):
        """ return repeated substrates to implement periodic boundary
        conditions """
        try:
            return self._cache['color_base']
        except KeyError:
            cnt_r = len(self.receptors)
            self._cache['color_base'] = self.colors**np.arange(cnt_r)
            return self._cache['color_base']
        
        
    def get_output_vector(self):
        """ calculate output vector for given receptors """
        # calculate the resulting binding characteristics
        probs = self.get_binding_probabilities()
        # threshold to get the response
        output = (probs > self.threshold)
        # encode output in single integer
        return np.dot(output, self.color_base) 
        
    
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
    def substrate_count(self):
        return len(self.substrates)
    
    @property
    def receptor_count(self):
        return len(self.receptors)
    
    @property
    def output_count(self):
        """ number of different outputs """
        return 2 ** len(self.receptors)
    
    @property
    def mutual_information_max(self):
        """ return upper bound for mutual information """
        # maximal mutual information restricted by receptors
        # there is a vector space of possible receptors, spanned
        # by the dim=min(cnt_r, l_r) basis vectors
        # => the possible number of receptors is colors^dim
        cnt_r, l_r = self.receptors.shape
        MI_receptors = min(cnt_r, l_r) * np.log(2)
        # Here, the 2 is due to the binary receptor output
        
        # maximal mutual information restricted by substrates
        cnt_s = self.substrates.shape[0]
        MI_substrates = np.log(cnt_s)
        
        return min(MI_receptors, MI_substrates)
    


class Blocks1DCollection(object):
    """ class that represents all possible combinations of blocks consisting
    of `cnt` distinct blocks of length `l` """
     
    def __init__(self, cnt, l, colors=2):
        self.cnt = cnt
        self.l = l
        self.colors = colors
        
    
    def __len__(self):
        """ returns the number of possible receptor combinations """
        num = block1D_count(self.l, self.colors)
        return comb(num, self.cnt, exact=True, repetition=False)
               

    def __iter__(self):
        """ generates all possible receptor combinations """
        blocks = blocks1D_get_unique(self.l, self.colors)
        return itertools.combinations(blocks, self.cnt)
        
        

def blocks1D_generate_interactions(substrates, cnt_r, l_r, colors):
    """ generates all possible combinations of substrate receptor interactions
    """
    #TODO: try to increase the performance by
    #    * improving update_energies
    #    * taking advantage of partially calculated energies
    receptor_collection = Blocks1DCollection(cnt_r, l_r, colors)
    for receptors in receptor_collection:
        yield Blocks1DInteraction(substrates, receptors, colors)
    
    
#===============================================================================
# DEFINE FAST FUNCTIONS USING NUMBA
#===============================================================================


if numba:
    # define fast numerical functions using numba
    
    @numba.jit('void(i8[:, :], i8[:], f8[:])', nopython=True)
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


    @numba.jit('void(i8[:, :], i8[:, :], f8[:, :])', nopython=True)
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
        out[:] = np.max([
            np.sum(substrates2[:, i:i+l_r] == receptor[np.newaxis, :],
                   axis=1)
            for i in xrange(l_s)
        ], axis=0)
        
        
    def _get_energies(substrates2, receptors, out):
        """ calculates all the interaction energies between the substrates and
        the receptors and stores them in `out` """
        # get dimensions
        l_s = substrates2.shape[1] // 2
        l_r = receptors.shape[1]

        # calculate the energies with a sliding window
        Es = np.array([
            np.sum(substrates2[:, np.newaxis, i:i+l_r] ==
                       receptors[np.newaxis, :, :],
                   axis=2)
            for i in xrange(l_s)
        ])

        out[:] = Es.max(axis=0)


