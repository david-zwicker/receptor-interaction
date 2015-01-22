'''
Created on Jan 16, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>

Module that monkey patches classes in other models with equivalent, but faster
methods.
'''

from __future__ import division

import copy
import functools
import numba
import numpy as np

import model_block_1D
import model_tetris_1D
from .model_block_1D import calc_entropy  # @UnusedImport
from .utils import estimate_computation_speed



@numba.jit(nopython=True)
def ChainsInteraction_update_energies_receptor_numba(substrates2, receptor, 
                                                     interaction_range, out):
    """ update interaction energies of the `receptor` with the substrates
    and save them in `out` """ 
    cnt_s, l_s2 = substrates2.shape
    l_r, l_s = len(receptor), l_s2 // 2
    
    if interaction_range < l_r:
        # the substrates interact with part of the receptor
        rng = interaction_range            
        for s in xrange(cnt_s): #< calculate for all substrates
            overlap_max = 0
            for i in xrange(l_s): #< try all substrate translations
                for j in xrange(l_r - rng + 1): #< try all receptor translations
                    overlap = 0
                    for k in xrange(rng):
                        if substrates2[s, i + k] == receptor[j + k]:
                            overlap += 1
                    overlap_max = max(overlap_max, overlap)
            out[s] = overlap_max

    else:
        # the substrates interact with the full receptor
        for s in xrange(cnt_s): #< calculate for all substrates
            overlap_max = 0
            for i in xrange(l_s): #< try all substrate translations
                overlap = 0
                for k in xrange(l_r):
                    if substrates2[s, i + k] == receptor[k]:
                        overlap += 1
                overlap_max = max(overlap_max, overlap)
            out[s] = overlap_max
    

def ChainsInteraction_update_energies_receptor(self, idx_r):
    """ updates the energy of the `idx_r`-th receptor """
    ChainsInteraction_update_energies_receptor_numba(
        self.substrates2, self.receptors[idx_r], self.interaction_range,
        self.energies[:, idx_r]
    )



@numba.jit(nopython=True)
def ChainsInteraction_update_energies_numba(substrates2, receptors,
                                            interaction_range, out):
    """ calculates all the interaction energies between the substrates and
    the receptors and stores them in `out` """
    for idx_r in xrange(len(receptors)):
        ChainsInteraction_update_energies_receptor_numba(
            substrates2, receptors[idx_r, :], interaction_range,
            out[:, idx_r]
        )
    return

    cnt_s, l_s2 = substrates2.shape
    l_s = l_s2 // 2
    cnt_r, l_r = receptors.shape
    # check all substrates versus all receptors
    for s in xrange(cnt_s):
        for r in xrange(cnt_r):
            overlap_max = 0
            # find the maximum over all starting positions
            for start in xrange(l_s):
                overlap = 0
                # count overlap along receptor length
                for k in xrange(l_r):
                    if substrates2[s, start + k] == receptors[r, k]:
                        overlap += 1
                overlap_max = max(overlap_max, overlap)
            out[s, r] = overlap_max


def ChainsInteraction_update_energies(self):
    """ calculates all the energies between the substrates and the receptors """
#     if isinstance(self.receptors, np.ndarray):
#         # all receptors have the same length
#         ChainsInteraction_update_energies_numba(
#             self.substrates2, self.receptors, self.interaction_range,
#             self.energies
#         )
#         
#     else:
    # receptor length varies
    for idx_r, receptor in enumerate(self.receptors):
        ChainsInteraction_update_energies_receptor_numba(
            self.substrates2, receptor, self.interaction_range,
            self.energies[:, idx_r]
        )    



@numba.jit(nopython=True)
def ChainsInteraction_get_output_vector_numba(energies, temperature, threshold,
                                              out):
    """ calculate output vector for given receptors """
    cnt_s, cnt_r = energies.shape
    threshold /= cnt_r #< normalize threshold to number of receptors
     
    # iterate over all substrates
    for s in xrange(cnt_s):
        if temperature == 0:
            # only the receptors with maximal interaction energy contribute
            
            # determine maximal energies for each substrate and count how many
            # receptors have this maximal energy
            Emax = 0
            count = 0
            for r in xrange(cnt_r):
                if energies[s, r] == Emax:
                    count += 1
                elif energies[s, r] > Emax:
                    count = 1
                    Emax = energies[s, r]
            
            # encode output in single integer
            output = 0
            if 1 >= threshold*count:
                # there are few enough active receptors such that they get above
                # threshold
                base = 1
                for r in xrange(cnt_r):
                    # only consider receptors that have the maximal energy 
                    if energies[s, r] == Emax:
                        output += base
                    base *= 2
            out[s] = output
            
        else:
            # receptors contribute according to Boltzmann weights
            
            # calculate the total interaction probabilities for normalization
            total = 0
            for r in xrange(cnt_r):
                total += np.exp(energies[s, r]/temperature)
            
            # encode output in single integer
            output = 0
            base = 1
            Ethresh = temperature * np.log(threshold * total)
            for r in xrange(cnt_r):
                # only consider receptors above the threshold; test for
                #     np.exp(energies[s, r]/temperature)/total >= threshold
                if energies[s, r] >= Ethresh:
                    output += base
                base *= 2
            out[s] = output


@numba.jit(nopython=True)
def ChainsInteraction_get_mutual_information_numba(output_vector):
    """ calculates the mutual information of the sorted output_vector """
    if len(output_vector) == 0:
        return 0
    
    # calculate the entropy in the output vector
    entropy = 0
    count = 1
    last_val = output_vector[0]
    for val in output_vector[1:]:
        if val == last_val:
            count += 1
        else:
            entropy += count*np.log2(count)
            last_val = val
            count = 1
    entropy += count*np.log2(count)
    
    cnt_s = len(output_vector)
    return np.log2(cnt_s) - entropy/cnt_s
        
    
def ChainsInteraction_get_mutual_information(self):
    """ calculate output vector for given receptors """
    # create or load a temporary array to store the output vector into
    cnt_s = len(self.energies)
    try:
        output_vector = ChainsInteraction_get_mutual_information.cache[:cnt_s]
        if len(output_vector) < cnt_s:
            raise AttributeError #< to fall into exception branch
    except AttributeError:
        ChainsInteraction_get_mutual_information.cache = np.empty(cnt_s, np.int)
        output_vector = ChainsInteraction_get_mutual_information.cache
    
    # calculate the output vector
    ChainsInteraction_get_output_vector_numba(self.energies, self.temperature,
                                              self.threshold, output_vector)
    # calculate the mutual information
    output_vector.sort()
    return ChainsInteraction_get_mutual_information_numba(output_vector)



@numba.jit(nopython=True)
def TetrisInteraction_update_energies_receptor_numba(substrates2, receptor,
                                                     interaction_range, out):
    """ update interaction energies of the `receptor` with the substrates
    and save them in `out` """ 
    cnt_s, l_s2 = substrates2.shape
    l_s = l_s2 // 2
    l_r = len(receptor)
    
    if interaction_range < l_r:
        # the substrates interact with part of the receptor
        rng = interaction_range
        for s in xrange(cnt_s): #< calculate for all substrates
            overlap_max = -1
            for i in xrange(l_s): #< try all substrate translations
                for j in xrange(l_r - rng  + 1): #< try all receptor trans.
                    overlap = 0
                    dist_max = -1
                    # calculate how often the maximal distance occurs
                    for k in xrange(rng):
                        # distance between substrate and receptor at this point
                        dist = substrates2[s, i + k] + receptor[j + k]
                        if dist == dist_max:
                            # the same distance as the other ones 
                            # => increment counter
                            overlap += 1 
                        elif dist > dist_max:
                            # larger distance => reset counter
                            dist_max = dist
                            overlap = 1
                    overlap_max = max(overlap_max, overlap)
            out[s] = overlap_max
            
    else:
        # the substrates interact with the full receptor
        for s in xrange(cnt_s): #< calculate for all substrates
            overlap_max = -1
            for i in xrange(l_s): #< try all substrate translations
                overlap = 0
                dist_max = -1
                # calculate how often the maximal distance occurs
                for k in xrange(l_r):
                    # distance between substrate and receptor at this point
                    dist = substrates2[s, i + k] + receptor[k]
                    if dist == dist_max:
                        # the same distance as the other ones 
                        # => increment counter
                        overlap += 1 
                    elif dist > dist_max:
                        # larger distance => reset counter
                        dist_max = dist
                        overlap = 1
                overlap_max = max(overlap_max, overlap)
            out[s] = overlap_max
            

def TetrisInteraction_update_energies_receptor(self, idx_r):
    """ updates the energy of the `idx_r`-th receptor """
    TetrisInteraction_update_energies_receptor_numba(
        self.substrates2, self.receptors[idx_r], self.interaction_range,
        self.energies[:, idx_r]
    )
        


def check_energies(obj, (func1, func2)):
    """ checks the numba method versus the original one """
    obj.energies[:] = 0
    obj1, obj2 = obj, copy.deepcopy(obj)
    func1(obj1)
    func2(obj2)
    return np.allclose(obj1.energies, obj2.energies)


def check_return_value(obj, (func1, func2)):
    """ checks the numba method versus the original one """
    return np.allclose(func1(obj), func2(obj))



class NumbaPatcher(object):
    """ class for managing numba monkey patching in this package. This class
    only provides class methods since it is used as a singleton. """   
    
    # list of methods that have a numba equivalent
    #TODO: the arguments should be dictionaries for clearer meaning
    numba_methods = {
        'model_block_1D.ChainsInteraction.update_energies_receptor': (
            model_block_1D.ChainsInteraction.update_energies_receptor,
            ChainsInteraction_update_energies_receptor,
            check_energies, {'idx_r': 0}
        ),
        'model_block_1D.ChainsInteraction.update_energies': (
            model_block_1D.ChainsInteraction.update_energies,
            ChainsInteraction_update_energies,
            check_energies, {}
        ),
        'model_block_1D.ChainsInteraction.get_mutual_information': (
            model_block_1D.ChainsInteraction.get_mutual_information,
            ChainsInteraction_get_mutual_information,
            check_return_value, {}
        ),
        'model_tetris_1D.TetrisInteraction.update_energies_receptor': (
            model_tetris_1D.TetrisInteraction.update_energies_receptor,
            TetrisInteraction_update_energies_receptor,
            check_energies, {'idx_r': 0}
        ),
    }
    
    enabled = False #< whether numba speed-up is enabled or not


    @classmethod
    def enable(cls):
        """ enables the numba methods """
        for name, funcs in cls.numba_methods.iteritems():
            module, class_name, method_name = name.split('.')
            class_obj = getattr(globals()[module], class_name)
            setattr(class_obj, method_name, funcs[1])
        cls.enabled = True
            
            
    @classmethod
    def disable(cls):
        """ disable the numba methods """
        for name, funcs in cls.numba_methods.iteritems():
            module, class_name, method_name = name.split('.')
            class_obj = getattr(globals()[module], class_name)
            setattr(class_obj, method_name, funcs[0])
        cls.enabled = False
        
        
    @classmethod
    def toggle(cls, verbose=True):
        """ enables or disables the numba speed up, depending on the current
        state """
        if cls.enabled:
            cls.disable()
            if verbose:
                print('Numba speed-ups have been disabled.')
        else:
            cls.enable()
            if verbose:
                print('Numba speed-ups have been enabled.')
            
            
    @classmethod
    def test_consistency(cls, repeat=10, verbose=False):
        """ tests the consistency of the numba methods with their original
        counter parts """        
        problems = 0
        for name, funcs in cls.numba_methods.iteritems():
            # extract the class and the functions
            module, class_name, _ = name.split('.')
            class_obj = getattr(globals()[module], class_name)

            # extract the test function
            try:
                test_func = funcs[2]
            except IndexError:
                continue
            
            # check the functions multiple times
            for _ in xrange(repeat):
                test_obj = class_obj.create_test_instance()
                func1 = functools.partial(funcs[0], **funcs[3])
                func2 = functools.partial(funcs[1], **funcs[3])
                if not test_func(test_obj, (func1, func2)):
                    print('The numba implementation of `%s` is invalid.' % name)
                    print('Native implementation yields %s' % func1(test_obj))
                    print('Numba implementation yields %s' % func2(test_obj))
                    print('Input: %r' % test_obj)
                    problems += 1
                    break
                
            else:
                # no problems have been found
                if verbose:
                    print('`%s` has a valid numba implementation.' % name) 

        if not problems:
            print('All numba implementations are consistent.')
            
            
    @classmethod
    def test_speedup(cls, repeat=10000):
        """ tests the speed up of the supplied methods """
        for name, funcs in cls.numba_methods.iteritems():
            # extract the class and the functions
            module, class_name, func_name = name.split('.')
            class_obj = getattr(globals()[module], class_name)
            test_obj = class_obj.create_test_instance()
            func1 = functools.partial(funcs[0], **funcs[3])
            func2 = functools.partial(funcs[1], **funcs[3])
            
            # check the runtime of the original implementation
            speed1 = estimate_computation_speed(func1, test_obj)
            # check the runtime of the improved implementation
            speed2 = estimate_computation_speed(func2, test_obj)
            
            print('%s.%s: %g times' % (class_name, func_name, speed2/speed1))
            