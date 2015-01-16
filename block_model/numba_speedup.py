'''
Created on Jan 16, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>

Module that monkey patches classes in other models with equivalent, but faster
methods.
'''

from __future__ import division

import model_block_1D
import model_tetris_1D

import copy
import numba
import numpy as np


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
    """ calculates all the energies between the substrates and the
    receptors """
    ChainsInteraction_update_energies_numba(self.substrates2, self.receptors,
                                            self.energies)
    


@numba.jit(nopython=True)
def ChainsInteraction_update_energies_receptor_numba(substrates2, receptor, out):
    """ update interaction energies of the `receptor` with the substrates
    and save them in `out` """ 
    cnt_s, l_s2 = substrates2.shape
    l_r, l_s = len(receptor), l_s2 // 2
    for s in xrange(cnt_s):
        overlap_max = 0
        for i in xrange(l_s):
            overlap = 0
            for k in xrange(l_r):
                if substrates2[s, i + k] == receptor[k]:
                    overlap += 1
            overlap_max = max(overlap_max, overlap)
        out[s] = overlap_max
    

def ChainsInteraction_update_energies_receptor(self, idx_r):
    """ updates the energy of the `idx_r`-th receptor """
    ChainsInteraction_update_energies_receptor_numba(self.substrates2,
                                                     self.receptors[idx_r],
                                                     self.energies[:, idx_r])

    


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



def TetrisInteraction_update_energies_receptor(self, idx_r):
    """ updates the energy of the `idx_r`-th receptor """
    TetrisInteraction_update_energies_receptor_numba(self.substrates2,
                                                     self.receptors[idx_r],
                                                     self.energies[:, idx_r])
        


def update_energies_check(obj, (func1, func2)):
    """ checks the numba method versus the original one """
    obj.energies[:] = 0
    obj1, obj2 = obj, copy.deepcopy(obj)
    func1(obj1)
    func2(obj2)
    return np.allclose(obj1.energies, obj2.energies)


def update_energies_receptor_check(obj, (func1, func2)):
    """ checks the numba method versus the original one """
    obj.energies[:] = 0
    obj1, obj2 = obj, copy.deepcopy(obj)
    func1(obj1, 0)
    func2(obj2, 0)
    return np.allclose(obj1.energies, obj2.energies)



class NumbaPatcher(object):
    """ class for managing numba monkey patching in this package. This class
    only provides clas methods since it is used as a singleton. """   
    
    # list of methods that have a numba equivalent
    numba_methods = {
        'model_block_1D.ChainsInteraction.update_energies': (
            model_block_1D.ChainsInteraction.update_energies,
            ChainsInteraction_update_energies,
            update_energies_check
        ),
        'model_block_1D.ChainsInteraction.update_energies_receptor': (
            model_block_1D.ChainsInteraction.update_energies_receptor,
            ChainsInteraction_update_energies_receptor,
            update_energies_receptor_check
        ),
        'model_tetris_1D.TetrisInteraction.update_energies_receptor': (
            model_tetris_1D.TetrisInteraction.update_energies_receptor,
            TetrisInteraction_update_energies_receptor,
            update_energies_receptor_check,
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
            test_func = funcs[2]
            # check the functions multiple times
            for _ in xrange(repeat):
                test_obj = class_obj.create_test_instance()
                if not test_func(test_obj, funcs[:2]):
                    print('The numba implementation of `%s` is invalid.' % name)
                    problems += 1
                    break

        if not problems:
            print('All numba implementations are consistent.')
            
            