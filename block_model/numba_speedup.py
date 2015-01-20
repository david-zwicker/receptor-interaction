'''
Created on Jan 16, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>

Module that monkey patches classes in other models with equivalent, but faster
methods.
'''

from __future__ import division

import copy
import numba
import numpy as np
import timeit

import model_block_1D
import model_tetris_1D
from model_block_1D import calc_entropy  # @UnusedImport



@numba.jit(nopython=True)
def ChainsInteraction_update_energies_receptor_numba(substrates2, receptor, out):
    """ update interaction energies of the `receptor` with the substrates
    and save them in `out` """ 
    cnt_s, l_s2 = substrates2.shape
    l_r, l_s = len(receptor), l_s2 // 2
    for s in xrange(cnt_s):
        overlap_max = 0
        for start in xrange(l_s):
            overlap = 0
            for k in xrange(l_r):
                if substrates2[s, start + k] == receptor[k]:
                    overlap += 1
            overlap_max = max(overlap_max, overlap)
        out[s] = overlap_max
    

def ChainsInteraction_update_energies_receptor(self, idx_r=0):
    """ updates the energy of the `idx_r`-th receptor """
    ChainsInteraction_update_energies_receptor_numba(self.substrates2,
                                                     self.receptors[idx_r],
                                                     self.energies[:, idx_r])



@numba.jit(nopython=True)
def ChainsInteraction_update_energies_numba(substrates2, receptors, out):
    """ calculates all the interaction energies between the substrates and
    the receptors and stores them in `out` """
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
    if isinstance(self.receptors, np.ndarray):
        ChainsInteraction_update_energies_numba(self.substrates2,
                                                self.receptors,
                                                self.energies)
    else:
        for idx_r, receptor in enumerate(self.receptors):
            ChainsInteraction_update_energies_receptor_numba(self.substrates2,
                                                             receptor,
                                                             self.energies[:, idx_r])    



@numba.jit(nopython=True)
def ChainsInteraction_get_output_vector_numba(energies, temperature, threshold,
                                              out, probabilities):
    """ calculate output vector for given receptors """
    cnt_s, cnt_r = energies.shape
    threshold /= cnt_r #< normalize threshold to number of receptors
     
    # iterate over all substrates
    for s in xrange(cnt_s):
        if temperature == 0:
            # determine minimal energies for each substrate
            Emin = 1000
            for r in xrange(cnt_r):
                Emin = min(Emin, energies[s, r])
                
            # determine the receptors that are activated
            normalization = 0
            for r in xrange(cnt_r):
                if energies[s, r] == Emin:
                    probabilities[r] = 1
                    normalization += 1
                else:
                    probabilities[r] = 0
            
        else:
            # calculate interaction probabilities
            normalization = 0
            for r in xrange(cnt_r):
                probabilities[r] = np.exp(energies[s, r]/temperature)
                normalization += probabilities[r]
            
        # encode output in single integer
        output = 0
        base = 1
        for r in xrange(cnt_r):
            # only consider receptors above the threshold
            if probabilities[r] > threshold*normalization:
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
    # calculate the resulting binding characteristics
    cnt_s, cnt_r = self.energies.shape
    output_vector = np.empty(cnt_s, np.int)
    tmp = np.empty(cnt_r, np.double)
    
    # calculate the output vector
    ChainsInteraction_get_output_vector_numba(self.energies, self.temperature,
                                              self.threshold, output_vector,
                                              tmp)
    # calculate the mutual information
    output_vector.sort()
    return ChainsInteraction_get_mutual_information_numba(output_vector)



@numba.jit(nopython=True)
def TetrisInteraction_update_energies_receptor_numba(substrates2, receptor, out):
    """ update interaction energies of the `receptor` with the substrates
    and save them in `out` """ 
    cnt_s, l_s2 = substrates2.shape
    l_r, l_s = len(receptor), l_s2 // 2
    
    for s in xrange(cnt_s):
        overlap_max = -1
        for i in xrange(l_s):
            overlap = 0
            dist_max = -1
            # calculate how often the maximal distance occurs
            for k in xrange(l_r):
                # distance between substrate and receptor at this point
                dist = substrates2[s, i + k] + receptor[k]
                if dist == dist_max:
                    # the same distance as the other ones => increment counter
                    overlap += 1 
                elif dist > dist_max:
                    # larger distance => reset counter
                    dist_max = dist
                    overlap = 1
            overlap_max = max(overlap_max, overlap)
        out[s] = overlap_max



def TetrisInteraction_update_energies_receptor(self, idx_r=0):
    """ updates the energy of the `idx_r`-th receptor """
    TetrisInteraction_update_energies_receptor_numba(self.substrates2,
                                                     self.receptors[idx_r],
                                                     self.energies[:, idx_r])
        


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
    numba_methods = {
        'model_block_1D.ChainsInteraction.update_energies_receptor': (
            model_block_1D.ChainsInteraction.update_energies_receptor,
            ChainsInteraction_update_energies_receptor,
            check_energies
        ),
        'model_block_1D.ChainsInteraction.update_energies': (
            model_block_1D.ChainsInteraction.update_energies,
            ChainsInteraction_update_energies,
            check_energies
        ),
        'model_block_1D.ChainsInteraction.get_mutual_information': (
            model_block_1D.ChainsInteraction.get_mutual_information,
            ChainsInteraction_get_mutual_information,
            check_return_value
        ),
        'model_tetris_1D.TetrisInteraction.update_energies_receptor': (
            model_tetris_1D.TetrisInteraction.update_energies_receptor,
            TetrisInteraction_update_energies_receptor,
            check_energies,
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
    def test_consistency(cls, repeat=10):
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
                if not test_func(test_obj, funcs[:2]):
                    print('The numba implementation of `%s` is invalid.' % name)
                    print('Native implementation yields', funcs[0](test_obj))
                    print('Numba implementation yields', funcs[1](test_obj))
                    problems += 1
                    break

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
            func1, func2 = funcs[:2]
            
            # initialize possible caches
            func1(test_obj)
            func2(test_obj)
            
            # check the runtime of the original implementation
            dur1 = timeit.timeit(lambda: func1(test_obj), number=repeat)
            # check the runtime of the improved implementation
            dur2 = timeit.timeit(lambda: func2(test_obj), number=repeat)
            
            print('%s.%s: %g times' % (class_name, func_name, dur1/dur2))
            