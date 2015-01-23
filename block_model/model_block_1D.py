'''
Created on Jan 12, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>

Module for handling the interaction between a set of substrates and
corresponding receptors. Both substrates and receptors are described by chains
of chains, where each chain can have a color. We consider periodic boundary
conditions, such that these chains are equivalent to necklaces in combinatorics.

Each chain is represented by an integer numpy array of length l. Sets of 
chains are generally represented by a list of chains. If all chains are of the 
same length, sets of chains may also be represented by a 2D-numpy array where 
the first dimension corresponds to different chains.
    
Special objects are used to represent the set of all unique chains and the
interaction between sets of substrate and receptor chains.
'''

from __future__ import division

import fractions
import itertools
import random
import numpy as np

from collections import Counter
from scipy.misc import comb

from .utils import classproperty


#===============================================================================
# BASIC CHAIN/NECKLACE FUNCTIONS
#===============================================================================


class Chain(np.ndarray):
    """
    class representing a single chain
    """
    
    def __new__(cls, input_array, colors=None, cyclic=False):
        """ 
        Creates the array and adds additional information
        note that `input_array` is not copied, but the data is used directly
        if `input_array` is already an numpy array
        """
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        if colors is None:
            obj.colors = max(input_array) + 1
        else:
            obj.colors = colors
        obj.cyclic = cyclic
        # Finally, we must return the newly created object:
        return obj
    
    
    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return
        self.colors = getattr(obj, 'colors', None)
        self.cyclic = getattr(obj, 'cyclic', False)
        
    
    def __str__(self):
        return ('%s(%s, colors=%d, cyclic=%s)' %
                (self.__class__.__name__, np.ndarray.__str__(self),
                 self.colors, self.cyclic))


    @property
    def character(self):
        """ return an integer representing this chain. This integer uniquely 
        identifies this chain among all chains of equal length with equal number
        of colors """
        base = self.colors ** np.arange(len(self), 0, -1)
        return np.dot(self, base)

        
    def normalized(self):
        """ returns the member of the equivalence class of necklaces that has
        the lowest character if the chain is cyclic.
        If self.cyclic=False, this just returns the chain itself """
        if self.cyclic:
            l = len(self)
            colors = self.max() + 1
            base = colors ** np.arange(l, 0, -1)
            chain2 = np.r_[self, self]
            
            # determine the chain with the lowest index
            idx = np.argmin([np.dot(chain2[k:k + l], base)
                             for k in xrange(l)])
            return chain2[idx:idx + l]
        
        else:
            return self
    
    
    def normalize(self):
        """ changes the current chain to the member of the equivalence class of
        necklaces that has the lowest character """
        if self.cyclic:
            self[:] = self.normalized()
    
        
    def get_mpl_collection(self, center=(0, 0), size=1, width=0.3, cmap=None,
                           **kwargs):
        """ create a matplotlib patch collection visualizing the chain in a 
        linear way.
            `size` is the total size of the chain
            `width` is the associated width
        """
        from matplotlib.patches import Rectangle, Wedge
        from matplotlib.collections import PatchCollection
        from matplotlib.colors import Normalize
        from matplotlib import cm

        if cmap is None:
            cmap = cm.jet
        
        if self.cyclic:        
            # create cyclic chain    
            r_min, r_max = size - width, size
            sector = 360 / len(self)
            patches = []
            for k in xrange(len(self)):
                angle = k * sector
                patches.append(Wedge(center, r_max, angle, angle + sector,
                                     width=r_max - r_min, **kwargs))
        else:            
            # create linear chain
            sector = size / len(self)
            patches = []
            for k in xrange(len(self)):
                x = center[0] - size/2 + k*sector
                y = center[1] - width/2
                patches.append(Rectangle((x, y), sector, width, **kwargs))
            
        # combine the patches in a collection
        pc = PatchCollection(patches, cmap=cmap,
                             norm=Normalize(vmin=0, vmax=self.colors))
        pc.set_array(self)
        
        return pc
    


def remove_redundant_chains(chains):
    """ removes chains that are the same (because of periodic boundary
    conditions)
    """
    if isinstance(chains, np.ndarray):
        # optimized implementation for chains of equal length
        l = len(chains[0])
        colors = chains.max() + 1
        base = colors ** np.arange(l, 0, -1)
        
        # calculate an characteristic number for each substrate
        characters = [min(np.dot(chains2[k:k + l], base)
                          for k in xrange(l))
                      for chains2 in np.c_[chains, chains]]
        
        _, idx = np.unique(characters, return_index=True)
        
        result = chains[idx]
        
    else:
        # general implementation for chains of unequal lengths
        result = []
        chains = sorted(chains, key=len)
        for l, sublist in itertools.groupby(chains, key=len):
            # handle all chains of a certain length
            arr = np.atleast_2d(list(sublist))
            result.extend(remove_redundant_chains(arr))
        
    return result



def normalize_chains(chains):
    """ picks the member of the equivalence class of necklaces that has the
    lowest character and sorts the entire list of chains"""
    result = []
    colors = max(max(chain) for chain in chains) + 1
    
    if isinstance(chains, np.ndarray):
        # copy data to not alter it in-place
        chains = chains.copy()
    else:
        # sort chains by length first
        chains = sorted(chains, key=len)
        
    for _, group in itertools.groupby(chains, key=len):
        # handle all chains of a certain length
        sublist = []
        # normalize all chains in this group
        for chain in group:
            chain = Chain(chain, colors=colors, cyclic=True)
            chain.normalize() #< normalize this chain
            sublist.append(chain)
            
        # sort the chains within this group according to their character
        sublist.sort(key=lambda c: c.character)
        result.extend(sublist)
        
    return result



class Chains(object):
    """ class that represents all chains of length l """

    colors_str = 'colors'
    
    def __init__(self, l, colors=2, cyclic=False):
        try:
            self.l_min, self.l_max = l
        except TypeError:
            self.l_min = self.l_max = l
        self.colors = colors
        self.cyclic = cyclic
        
        
    @property
    def fixed_length(self):
        return self.l_min == self.l_max
        
        
    def __repr__(self):
        if self.fixed_length:
            ls = '%d' % self.l_min
        else:
            ls = '(%d, %d)' % (self.l_min, self.l_max)
        return ('%s(l=%s, %s=%d, cyclic=%s)' %
                (self.__class__.__name__, ls, self.colors_str, self.colors,
                 self.cyclic))
        
        
    @property
    def counts(self):
        """ returns the number of unique chains of all possible lengths from
        l_min until l_max.
        The algorithm used here is based on Eq. 1 from the paper
            F. Ruskey, C. Savage, T. Min Yih Wang, J. of Algorithms, 13 (1992).
        """
        for l in xrange(self.l_min, self.l_max + 1):
            if self.cyclic:
                Nb = 0
                for d in xrange(1, l + 1):
                    Nb += self.colors ** fractions.gcd(l, d)
                Nb //= l
                yield Nb
                
            else:
                yield self.colors ** l            
            
    
    def __len__(self):
        """ returns the number of unique chains of length l
        This number is equivalent to len(list(self)), but uses a more efficient
        mathematical solution based on Eq. 1 in the paper
            F. Ruskey, C. Savage, T. Min Yih Wang, J. of Algorithms, 13 (1992).
        """
        # add up the counts from each lengths
        return sum(count for count in self.counts)
    
    
    def _iterate_fixed_length(self, l):
        """ returns a generator for all unique chains of length `l`.

        This method uses the FKM algorithm published in
        * H. Fredricksen and J. Maiorana, Necklaces of beads in k colors and
            k-ary de Bruijn sequences, Discrete Math. 23 (1978), 207-210.
        * H. Fredricksen and I. J. Kessler, An algorithm for generating 
            necklaces of beads in two colors, Discrete Math. 61 (1986), 181-188.
        """
        k = self.colors
        if self.cyclic:
            # yield all cyclic chains (necklaces)
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
        
        else:
            # yield all linear chains
            colors = range(k)
            for a in itertools.product(colors, repeat=l):
                yield np.array(a)

    
    def __iter__(self):
        """ returns a generator for all unique chains.

        Note that the return value is a view into an internal array. Use
            [v.copy() for v in self]
        instead of
            list(self)
        to create a list of all chains.
        """
        for l in xrange(self.l_min, self.l_max + 1):
            for chain in self._iterate_fixed_length(l):
                yield chain


    def to_list(self):
        """ returns an array of all unique chains of length `l` with `colors`
        possible colors per chain """
        if self.fixed_length:
            # represent all chains of a fixed length as a single numpy array
            res = np.zeros((len(self), self.l_min), np.int)
            for k, c in enumerate(self):
                res[k, :] = c
            return res
                
        else:
            return [v.copy() for v in self]
    


class ChainCollections(object):
    """ class that represents all possible collections of `cnt` distinct chains
    of length `l` """
    
    single_item_class = Chains 

     
    def __init__(self, cnt, l, colors=2, cyclic=False):
        # initialize the set of all chains
        try:
            self.l_min, self.l_max = l
        except TypeError:
            self.l_min = self.l_max = l
        self.colors = colors
        self.chains = self.single_item_class(l, colors, cyclic)
        
        if cnt == 'all':
            self.cnt = len(self.chains) 
        else:
            self.cnt = cnt
        
        
    @property
    def fixed_length(self):
        return self.l_min == self.l_max
        
        
    @classproperty
    def colors_str(cls):  # @NoSelf
        return cls.single_item_class.colors_str
    
        
    def __repr__(self):
        if self.fixed_length:
            ls = '%d' % self.l_min
        else:
            ls = '(%d, %d)' % (self.l_min, self.l_max)
        return ('%s(cnt=%d, l=%s, %s=%d, cyclic=%s)' %
                (self.__class__.__name__, self.cnt, ls,
                 self.colors_str, self.colors, self.chains.cyclic))
          

    def __len__(self):
        """ returns the number of possible receptor combinations.
        The returned number is equal to len(list(self)), but is calculated more
        efficiently. """
        num = len(self.chains)
        return comb(num, self.cnt, exact=True)
               

    def __iter__(self):
        """ generates all possible receptor combinations """
        chains = self.chains.to_list()
        if self.fixed_length:
            for chain_col in itertools.combinations(chains, self.cnt):
                yield np.array(chain_col)
                
        else: 
            for chain_col in itertools.combinations(chains, self.cnt):
                yield chain_col


    def _choose_random_fixed_length(self, cnt, l):
        """ chooses `cnt` unique chains consisting of `l` blocks with
        `colors` unique colors """
        colors = self.colors
        
        base = colors ** np.arange(l)
        
        chains, characters = [], set()
        counter = 0
        while len(chains) < cnt:
            # choose a random substrate and determine its character
            s = np.random.randint(0, colors, size=l)
            
            if self.chains.cyclic:
                s2 = np.r_[s, s]
                character = min(np.dot(s2[k:k + l], base)
                                for k in xrange(l))
            else:
                character = np.dot(s, base)
                
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
                
        chains = normalize_chains(chains)
        assert len(chains) == cnt
                
        return chains
    
    
    def choose_random(self):
        """ chooses a random representation from the current collection """
        if self.cnt == len(self.chains):
            chains = self.chains.to_list()
        
        elif self.fixed_length:
            chains = self._choose_random_fixed_length(self.cnt, self.l_min)
            chains = np.array(chains)
                    
        else:
            # choose a length distribution
            counts = list(self.chains.counts)
            lengths = Counter()
            for _ in xrange(self.cnt):
                weights = np.array(counts) / np.sum(counts)
                k = np.random.choice(len(counts), p=weights)
                lengths[k + self.l_min] += 1
                counts[k] -= 1
                
            # choose chains according to this length distribution
            chains = []
            for l, cnt in lengths.iteritems():
                chains.extend(self._choose_random_fixed_length(cnt, l))
                    
        return chains
    
    
#===============================================================================
# FUNCTIONS FOR CHAIN/NECKLACE INTERACTIONS
#===============================================================================

    
class ChainsState(object):
    """ class that represents the interaction between a set of substrates and a
    set of receptors.
    
    This code currently assumes that the substrates are cyclic chains and that
    the receptors are linear.
    """
    
    item_collection_class = ChainCollections

    
    def __init__(self, substrates, receptors, colors, cross_talk=0, 
                 interaction_range=1000, cache=None, energies=None):
        
        self.substrates = substrates
        self.receptors = receptors
        self.colors = colors
        self.cross_talk = cross_talk
        # interaction range cannot be larger than the size of the substrate
        self.interaction_range = min(interaction_range, self.substrates.shape[1])

        if cache is None:
            self._cache = {}
        else:
            self._cache = cache
            
        if energies is None:
            self.energies = np.zeros((len(substrates), len(receptors)))
            self.update_energies()
        else:
            self.energies = energies


    @classproperty
    def colors_str(cls):  # @NoSelf
        return cls.item_collection_class.colors_str
    
        
    def __repr__(self):
        return ('%s(substrates=%s, receptors=%s, %s=%d, interaction_range=%d)' %
                (self.__class__.__name__, self.substrates, self.receptors,
                 self.colors_str, self.colors, self.interaction_range))
        

    def __str__(self):
        return ('%s(%d Substrates, %d Receptors, %s=%d, interaction_range=%d)' %
                (self.__class__.__name__, len(self.substrates),
                 len(self.receptors), self.colors_str, self.colors,
                 self.interaction_range))
        

    @classmethod
    def create_test_instance(cls):
        """ creates a instance of the class with random parameters """
        coll_class = cls.item_collection_class

        # choose general parameters
        colors = random.randrange(2, 4)
        interaction_range = random.randrange(5, 10)

        # choose random substrates
        cnt_s = random.randrange(10, 20)
        l_s = random.randrange(10, 20)
        substrates = coll_class(cnt_s, l_s, colors, cyclic=True).choose_random()
        
        # choose random receptors
        cnt_r = random.randrange(10, 20)
        if random.randrange(0, 2):
            # receptors have all the same length
            l_r = random.randrange(5, 10)
        else:
            # receptors have varying length
            l_r = sorted(np.random.randint(5, 11, 2))
        receptors = coll_class(cnt_r, l_r, colors, cyclic=False).choose_random()
        
        # create object and set random parameters
        obj = cls(substrates, receptors, colors, interaction_range)
        if random.randrange(0, 2):
            obj.eps = 0.5 * np.random.random()
        else:
            obj.eps = 0
        if random.randrange(0, 2):
            obj.temperature = np.random.randint(0, 3)
        else:
            obj.temperature = 0
        obj.threshold = np.random.random()
        
        return obj
        

    def check_consistency(self):
        """ consistency check on the number of receptors and substrates """
        # TODO: check the length of the receptors and whether they are cyclic
        # or not
        
        if self.interaction_range is not None:
            assert self.interaction_range <= self.substrates.shape[1]
        
        # check the supplied substrates
        unique_substrates = remove_redundant_chains(self.substrates)
        redundant_count = len(self.substrates) - len(unique_substrates)
        if redundant_count:
            raise RuntimeWarning('There are %d redundant substrates' % 
                                 redundant_count)
        
        # check the supplied receptors
#         cnt_r, l_r = self.receptors.shape
#         chains = self.single_item_class(l_r, self.colors)
#         if cnt_r > len(chains):
#             raise RuntimeWarning('The number of supplied receptors is larger '
#                                  'than the number of possible unique ones.')
    
        unique_receptors = remove_redundant_chains(self.receptors)
        redundant_count = len(self.receptors) - len(unique_receptors)
        if redundant_count:
            raise RuntimeWarning('There are %d redundant receptors' % 
                                 redundant_count)
    
        
    def copy(self):
        """ copies the current interaction state to allow the receptors to
        be mutated. The substrates and the cache will be shared between this
        object and its copy """
        if isinstance(self.receptors, np.ndarray):
            receptors = self.receptors.copy()
        else:
            receptors = self.receptors[:]
        return self.__class__(self.substrates, receptors, self.colors,
                              self.cross_talk, self.interaction_range,
                              self._cache, self.energies.copy())
        
        
    @property
    def substrates2(self):
        """ return repeated substrates to implement periodic boundary
        conditions """
        try:
            return self._cache['substrates2']
        except KeyError:
            self._cache['substrates2'] = np.c_[self.substrates, self.substrates]
            return self._cache['substrates2']
    
    
    def update_energies_receptor(self, idx_r):
        """ updates the energy of the `idx_r`-th receptor """
        receptor = self.receptors[idx_r]
        l_s = self.substrates2.shape[1] // 2
        l_r = len(receptor) 

        # count the number of bonds
        if self.interaction_range < l_r:
            # the substrates interact with part of the receptor
            rng = self.interaction_range            
            self.energies[:, idx_r] = reduce(np.maximum, (
                np.sum(self.substrates2[:, i:i + rng] ==
                        receptor[np.newaxis, j:j + rng],
                       axis=1)
                for i in xrange(l_s)           #< try all substrate translations
                for j in xrange(l_r - rng + 1) #< try all receptor translations
            ))
            
        else:
            # the substrates interact with the full receptor
            rng = l_r
            self.energies[:, idx_r] = reduce(np.maximum, (
                np.sum(self.substrates2[:, i:i+l_r] == receptor[np.newaxis, :],
                       axis=1)
                for i in xrange(l_s)       #< try all substrate translations
            ))
            
        # account for cross-talk
        if self.cross_talk != 0:
            self.energies[:, idx_r] *= (1 - self.cross_talk)
            self.energies[:, idx_r] += rng * self.cross_talk 
        
                   
    def update_energies(self):
        """ calculates all the energies between the substrates and the
        receptors
        """
        # general implementation for receptors of unequal lengths
        for idx_r in xrange(len(self.receptors)):
            self.update_energies_receptor(idx_r)
                
                      
        
    def randomize_receptors(self):
        """ choose a completely new set of receptors """
        if isinstance(self.receptors, np.ndarray):
            # create numpy array representing the receptors
            self.receptors = np.random.randint(0, self.colors,
                                               size=self.receptors.shape)
            self.update_energies_receptor()
            
        else:
            # choose random receptors of unequal length
            raise NotImplementedError


    @property
    def num_receptors(self):
        return len(self.receptors)
    


class ChainsModel(object):
    """ class that represents all possible combinations of substrate and
    receptor interactions """
    
    state_class = ChainsState
    
    # factor determining how often the length of a receptor is kept versus
    # changing its length:
    keep_length_factor = 5
    
    
    def __init__(self, substrates, possible_receptors, **kwargs):
        self.substrates = substrates
        try:
            self.substrates_data = substrates.to_list()
        except AttributeError:
            self.substrates_data = substrates
            
        self.possible_receptors = possible_receptors
        
        # determine the parameters for single states
        state_parameters = {'colors': possible_receptors.colors,
                            'cross_talk': 0,
                            'interaction_range': 'full'}
        state_parameters.update(kwargs.pop('state_parameters', {}))
        if 'cross_talk' in kwargs:
            state_parameters['cross_talk'] = kwargs.pop('cross_talk')
        if 'interaction_range' in kwargs:
            state_parameters['interaction_range'] = kwargs.pop('interaction_range')
            
        if state_parameters['interaction_range'] == 'full':
            l_s = self.substrates_data.shape[1]
            l_r = self.possible_receptors.l_max
            state_parameters['interaction_range'] = max(l_s, l_r)
            
        if kwargs:
            raise TypeError('The following keyword parameters were not used: %s'
                            % kwargs)

        self.state_parameters = state_parameters
        
        self._cache = {}


    @property
    def colors(self):
        return self.possible_receptors.colors


    def __repr__(self):
        return ('%s(substrates=%r, receptors=%r, state_parameters=%r)' %
                (self.__class__.__name__, self.substrates,
                 self.possible_receptors, self.state_parameters))
        
        
    @property
    def num_states(self):
        return len(self.possible_receptors)
    
    
    def iterate_states(self):
        """ generates all possible chain interactions """
        #TODO: try to increase the performance by
        #    * taking advantage of partially calculated energies
        
        # create an initial state object
        receptors = self.possible_receptors.choose_random()
        state = self.state_class(self.substrates_data, receptors,
                                 **self.state_parameters)
        
        # iterate over all receptors and update the state object
        for receptors in self.possible_receptors:
            state.receptors = receptors
            state.update_energies()
            yield state
        
    
    def get_random_state(self):
        """ returns a randomly chosen chain interaction """
        receptors = self.possible_receptors.choose_random()
        obj = self.state_class(self.substrates_data, receptors,
                               **self.state_parameters)
        return obj
    
    
    @property
    def color_alternatives(self):
        """ look-up table for changing the color of a single block """
        try:
            return self._cache['color_alternatives']
        except:
            colors = [np.r_[0:c, c+1:self.colors]
                      for c in xrange(self.colors)] 
            self._cache['color_alternatives'] = colors
            return self._cache['color_alternatives']
    

    @property
    def length_change_rates(self):
        """ returns a list that contains the probabilities with which a
        receptor changes length when mutated.
        The returned array contains the accumulated probabilities of two events:
            decreasing the chain length
            increasing the chain length
        """
        try:
            return self._cache['length_change_rates']
        except KeyError:
            # calculate the rates once and store them in the cache
            
            rates = [] #< initialize for chains of zero length
            counts = list(self.possible_receptors.chains.counts)
            l_min = self.possible_receptors.l_min
            l_max = self.possible_receptors.l_max
            for l in xrange(l_max + 1):
                # count how many possible states there are
                if l >= l_min:
                    k = l - l_min #< index into the counts array
                    num_dec = 0 if l <= l_min else counts[k - 1]
                    num_nochange = counts[k]
                    num_inc = 0 if l >= l_max else counts[k + 1]
                else:
                    # this case should not happen, but we store it in the array
                    # for convenience of later access
                    num_dec = num_nochange = 0
                    num_inc = 1                    
                
                num_nochange *= self.keep_length_factor
                
                # calculate the according rates
                num_tot = num_dec + num_nochange + num_inc
                rate_dec = num_dec/num_tot
                rate_inc = num_inc/num_tot
                rates.append((rate_dec, rate_dec + rate_inc))
                
            self._cache['length_change_rates'] = rates
            return self._cache['length_change_rates']
                
        
    def _mutate_random_block(self, chain, colors):
        """ mutates a single block in a chain """
        # choose one point on the chain that will be mutated        
        block = random.randrange(0, len(chain))
        if colors == 2:
            # restricted to two colors => flip color
            chain[block] = 1 - chain[block]
        else:
            # more than two colors => use random choice
            clrs = self.color_alternatives[chain[block]]
            chain[block] = clrs[random.randrange(0, colors - 1)]
        
        
    def mutate_state(self, state):
        """ mutate a single, random receptor """
        # choose one receptor that will be mutated
        idx_r = random.randrange(0, len(state.receptors))
        receptor = state.receptors[idx_r]
        
        if self.possible_receptors.fixed_length:
            # mutate one block of the chosen receptor
            self._mutate_random_block(receptor, state.colors)
            
        else:
            # change the receptor length or mutate one block
            rates = self.length_change_rates[len(receptor)]
            rnd = random.random()
            if rnd < rates[0]:
                # decrease the receptor length
                state.receptors[idx_r] = receptor[:-1]
            elif rnd < rates[1]:
                # increase the receptor length
                block = random.randrange(0, state.colors)
                state.receptors[idx_r] = np.r_[receptor, block]
            else:
                # keep the receptor length unchanged
                self._mutate_random_block(receptor, state.colors)

        # recalculate the interaction energies of the changed receptor
        state.update_energies_receptor(idx_r)

#         energies_copy = state.energies.copy()
#         state.update_energies()
#         assert np.allclose(state.energies, energies_copy)
            

    @property
    def num_substrates(self):
        return len(self.substrates)
    
    @property
    def num_receptors(self):
        return self.possible_receptors.cnt
    
    @property
    def output_dim(self):
        """ number of different outputs """
        # the 2 is due to the binary output of the receptors
        return 2 ** self.num_receptors