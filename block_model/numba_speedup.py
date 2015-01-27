'''
Created on Jan 16, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>

Module that monkey patches classes in other modules with equivalent, but faster
methods.
'''

from __future__ import division

import copy
import functools
import numba
import numpy as np

from .utils import estimate_computation_speed

# these methods are used in getattr calls
import model_block_1D  # @UnusedImport
import model_tetris_1D  # @UnusedImport
import experiments  # @UnusedImport
from .utils import calc_entropy  # @UnusedImport


#===============================================================================
# NUMBA DEFINITIONS FOR MODELS
#===============================================================================


@numba.jit(nopython=True)
def ChainsState_update_energies_receptor_numba(substrates2, receptor, 
                                               interaction_range, cross_talk,
                                               out):
    """ update interaction energies of the `receptor` with the substrates
    and save them in `out` """ 
    cnt_s, l_s2 = substrates2.shape
    l_r, l_s = len(receptor), l_s2 // 2
    
    if interaction_range < l_r:
        # the substrates interact with part of the receptor
        rng = interaction_range            
        e1 = cross_talk * rng 
        e2 = 1 - cross_talk 
        for s in xrange(cnt_s): #< calculate for all substrates
            bonds_max = 0
            for i in xrange(l_s): #< try all substrate translations
                for j in xrange(l_r - rng + 1): #< try all receptor translations
                    bonds = 0
                    for k in xrange(rng):
                        if substrates2[s, i + k] == receptor[j + k]:
                            bonds += 1
                    bonds_max = max(bonds_max, bonds)
            out[s] = e1 + e2*bonds_max

    else:
        # the substrates interact with the full receptor
        e1 = cross_talk * l_r 
        e2 = 1 - cross_talk 
        for s in xrange(cnt_s): #< calculate for all substrates
            bonds_max = 0
            for i in xrange(l_s): #< try all substrate translations
                bonds = 0
                for k in xrange(l_r):
                    if substrates2[s, i + k] == receptor[k]:
                        bonds += 1
                bonds_max = max(bonds_max, bonds)
            out[s] = e1 + e2*bonds_max
    


def ChainsState_update_energies_receptor(self, idx_r):
    """ updates the energy of the `idx_r`-th receptor """
    ChainsState_update_energies_receptor_numba(
        self.substrates2, self.receptors[idx_r], self.interaction_range,
        self.cross_talk, self.energies[:, idx_r]
    )



@numba.jit(nopython=True)
def TetrisState_update_energies_receptor_numba(substrates2, receptor,
                                               interaction_range, cross_talk,
                                               out):
    """ update interaction energies of the `receptor` with the substrates
    and save them in `out` """ 
    cnt_s, l_s2 = substrates2.shape
    l_s = l_s2 // 2
    l_r = len(receptor)
    
    if interaction_range < l_r:
        # the substrates interact with part of the receptor
        rng = interaction_range
        e1 = cross_talk * rng 
        e2 = 1 - cross_talk 
        for s in xrange(cnt_s): #< calculate for all substrates
            bonds_max = -1
            for i in xrange(l_s): #< try all substrate translations
                for j in xrange(l_r - rng  + 1): #< try all receptor translat.
                    bonds = 0
                    dist_max = -1
                    # calculate how often the maximal distance occurs
                    for k in xrange(rng):
                        # distance between substrate and receptor at this point
                        dist = substrates2[s, i + k] + receptor[j + k]
                        if dist == dist_max:
                            # the same distance as the other ones 
                            # => increment counter
                            bonds += 1 
                        elif dist > dist_max:
                            # larger distance => reset counter
                            dist_max = dist
                            bonds = 1
                    bonds_max = max(bonds_max, bonds)
            out[s] = e1 + e2*bonds_max
            
    else:
        # the substrates interact with the full receptor
        e1 = cross_talk * l_r 
        e2 = 1 - cross_talk
        for s in xrange(cnt_s): #< calculate for all substrates
            bonds_max = -1
            for i in xrange(l_s): #< try all substrate translations
                bonds = 0
                dist_max = -1
                # calculate how often the maximal distance occurs
                for k in xrange(l_r):
                    # distance between substrate and receptor at this point
                    dist = substrates2[s, i + k] + receptor[k]
                    if dist == dist_max:
                        # the same distance as the other ones 
                        # => increment counter
                        bonds += 1 
                    elif dist > dist_max:
                        # larger distance => reset counter
                        dist_max = dist
                        bonds = 1
                bonds_max = max(bonds_max, bonds)
            out[s] = e1 + e2*bonds_max
            
            

def TetrisState_update_energies_receptor(self, idx_r):
    """ updates the energy of the `idx_r`-th receptor """
    TetrisState_update_energies_receptor_numba(
        self.substrates2, self.receptors[idx_r], self.interaction_range,
        self.cross_talk, self.energies[:, idx_r]
    )


#===============================================================================
# NUMBA DEFINITIONS FOR EXPERIMENTS
#===============================================================================


@numba.jit(nopython=True)
def mutual_information_from_output_numba(output_vector):
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



def mutual_information_from_output(self, output_vector):
    """ determine the mutual information from the output distribution """
    output_vector.sort()
    return mutual_information_from_output_numba(output_vector)
        
        
        
@numba.jit(nopython=True)
def DetectSingleSubstrate_get_output_vector_numba(energies, temperature,
                                                  threshold, out):
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

    
    
def DetectSingleSubstrate_get_output_vector(self, state):
    """ calculate mutual information for given state """
    # create or load a temporary array to store the output vector into
    cnt_s = len(state.energies)
    try:
        output_vector = DetectSingleSubstrate_get_output_vector.cache[:cnt_s]
        if len(output_vector) < cnt_s:
            raise AttributeError #< to fall into exception branch
    except AttributeError:
        DetectSingleSubstrate_get_output_vector.cache = np.empty(cnt_s, np.int)
        output_vector = DetectSingleSubstrate_get_output_vector.cache
    
    # calculate the output vector
    DetectSingleSubstrate_get_output_vector_numba(state.energies, self.temperature,
                                                  self.threshold, output_vector)
    return output_vector



@numba.jit(nopython=True)
def _handle_substrate_combination(substrates, weights, threshold, probs):
    """ calculates the output integer for a given list of substrates """
    # calculate interaction probabilities
    cnt_r = len(probs)
    total = 0
    for r in xrange(cnt_r):
        probs[r] = 0 #< reset probs array
        for s in substrates:
            probs[r] += weights[s, r] #< fix T=0 case
        total += probs[r]
        
    # encode output in single integer
    weights_thresh = threshold * total
    output = 0
    base = 1
    for r in xrange(cnt_r):
        # only consider receptors above the threshold; test for
        #     probs[r]/total >= threshold
        if probs[r] >= weights_thresh:
            output += base
        base *= 2
    return output        
        
        
        
@numba.jit(nopython=True)
def DetectMultipleSubstrates_get_output_vector_numba(weights, num, threshold,
                                                     probs, indices, out):
    """ calculate output vector for given receptors.
    The iteration algorithm has been adapted from itertools.combinations:
        https://docs.python.org/2/library/itertools.html#itertools.combinations
    """
    cnt_s, cnt_r = weights.shape
    if num > cnt_s:
        # can't find more substrates than there actually are
        return

    threshold /= cnt_r #< normalize threshold to number of receptors

    # iterate over all substrate combinations
    # indices = range(num) #< has been initialized outside this function
    k = 0
    out[k] = _handle_substrate_combination(indices, weights, threshold, probs)
    while True:
        k += 1
        for i in xrange(num - 1, -1, -1): #< reversed(range(num))
            if indices[i] != i + cnt_s - num:
                break
        else:
            return
        indices[i] += 1
        for j in xrange(i + 1, num):
            indices[j] = indices[j-1] + 1
        out[k] = _handle_substrate_combination(indices, weights, threshold, probs)
        
   
                      
def DetectMultipleSubstrates_get_output_vector(self, state):
    """ calculate mutual information for given state """
    # create or load a temporary array to store the output vector into
    input_dim = self.get_input_dim(state)
    try:
        output_vector = DetectMultipleSubstrates_get_output_vector.cache[:input_dim]
        if len(output_vector) < input_dim:
            raise AttributeError #< to fall into exception branch
    except AttributeError:
        DetectMultipleSubstrates_get_output_vector.cache = np.empty(input_dim, np.int)
        output_vector = DetectMultipleSubstrates_get_output_vector.cache
    
    # calculate the output vector
    probs = np.empty(state.energies.shape[1])    #< temporary array
    indices = np.arange(self.num, dtype=np.int)  #< temporary array
    if self.temperature == 0:
        Emax = state.energies.max(axis=1) #< maximal energy for each substrate
        weights = (state.energies == Emax[:, np.newaxis]).astype(np.double)
        # calculate the output vector
        DetectMultipleSubstrates_get_output_vector_numba(
            weights, self.num, self.threshold, #< input
            probs, indices, #< temporary arrays
            output_vector   #< output array
        )
        
    else:
        weights = np.exp(state.energies/self.temperature) #< Boltzmann factors
        # calculate the output vector
        DetectMultipleSubstrates_get_output_vector_numba(
            weights, self.num, self.threshold, #< input
            probs, indices, #< temporary arrays
            output_vector   #< output array
        )
        
    return output_vector


#===============================================================================
# FUNCTIONS/CLASSES INJECTING THE NUMBA ACCELERATIONS
#===============================================================================


def create_test_state():
    """ creates a random test state """
    return model_block_1D.ChainsState.create_test_instance()


def create_output_vector():
    """ create a random output vector """
    return np.random.randint(0, 10, 100)


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
    
    # register methods that have a numba equivalent
    numba_methods = {
        'model_block_1D.ChainsState.update_energies_receptor': {
            'numba': ChainsState_update_energies_receptor,
            'test_function': check_energies,
            'test_arguments': {'idx_r': 0},
        },
        'model_tetris_1D.TetrisState.update_energies_receptor': {
            'numba': TetrisState_update_energies_receptor,
            'test_function': check_energies,
            'test_arguments': {'idx_r': 0},
        },
        'experiments.DetectSingleSubstrate.mutual_information_from_output': {
            'numba': mutual_information_from_output,
            'test_function': check_return_value,
            'test_arguments': {'output_vector': create_output_vector},
        },
        'experiments.DetectSingleSubstrate.get_output_vector': {
            'numba': DetectSingleSubstrate_get_output_vector,
            'test_function': check_return_value,
            'test_arguments': {'state': create_test_state},
        },
        'experiments.DetectMultipleSubstrates.get_output_vector': {
            'numba': DetectMultipleSubstrates_get_output_vector,
            'test_function': check_return_value,
            'test_arguments': {'state': create_test_state},
        },
    }
    
    saved_original_functions = False
    enabled = False #< whether numba speed-up is enabled or not

    
    @classmethod
    def _save_original_function(cls):
        """ save the original function such that they can be restored later """
        for name, data in cls.numba_methods.iteritems():
            module, class_name, method_name = name.split('.')
            class_obj = getattr(globals()[module], class_name)
            data['original'] = getattr(class_obj, method_name)
        cls.saved_original_functions = True


    @classmethod
    def enable(cls):
        """ enables the numba methods """
        if not cls.saved_original_functions:
            cls._save_original_function()
        
        for name, data in cls.numba_methods.iteritems():
            module, class_name, method_name = name.split('.')
            class_obj = getattr(globals()[module], class_name)
            setattr(class_obj, method_name, data['numba'])
        cls.enabled = True
            
            
    @classmethod
    def disable(cls):
        """ disable the numba methods """
        for name, data in cls.numba_methods.iteritems():
            module, class_name, method_name = name.split('.')
            class_obj = getattr(globals()[module], class_name)
            setattr(class_obj, method_name, data['original'])
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
    def _prepare_functions(cls, data):
        """ prepares the arguments for the two functions that we want to test """
        # prepare the arguments
        test_args = data['test_arguments'].copy()
        for key, value in test_args.iteritems():
            if callable(value):
                test_args[key] = value()
                
        # inject the arguments
        func1 = functools.partial(data['original'], **test_args)
        func2 = functools.partial(data['numba'], **test_args)
        return func1, func2

            
            
    @classmethod
    def test_consistency(cls, repeat=10, verbose=False):
        """ tests the consistency of the numba methods with their original
        counter parts """        
        problems = 0
        for name, data in cls.numba_methods.iteritems():
            # extract the class and the functions
            module, class_name, _ = name.split('.')
            class_obj = getattr(globals()[module], class_name)

            # extract the test function
            test_func = data['test_function']
            
            # check the functions multiple times
            for _ in xrange(repeat):
                test_obj = class_obj.create_test_instance()
                func1, func2 = cls._prepare_functions(data)
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
    def test_speedup(cls, test_duration=1):
        """ tests the speed up of the supplied methods """
        for name, data in cls.numba_methods.iteritems():
            # extract the class and the functions
            module, class_name, func_name = name.split('.')
            class_obj = getattr(globals()[module], class_name)
            test_obj = class_obj.create_test_instance()
            func1, func2 = cls._prepare_functions(data)
                            
            # check the runtime of the original implementation
            speed1 = estimate_computation_speed(func1, test_obj,
                                                test_duration=test_duration)
            # check the runtime of the improved implementation
            speed2 = estimate_computation_speed(func2, test_obj,
                                                test_duration=test_duration)
            
            print('%s.%s: %g times faster' 
                  % (class_name, func_name, speed2/speed1))
            
            