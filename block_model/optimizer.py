'''
Created on Jan 12, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division


from simanneal import Annealer


class ReceptorOptimizerExhaustive(object):
    """ class for finding optimal receptor distribution using and exhaustive
    search """
    
    def __init__(self, possible_states):
        """ `possible_states` should return a list or a generator with all
        possible states, which can be used to iterate over """
        self.possible_states = possible_states
        self.info = {'states_considered': 0}


    def optimize(self):
        """ optimizes the receptors and returns the best receptor set together
        with the achieved mutual information.
        Extra information about the optimization procedure is stored in the
        `info` dictionary of this object """
        states_considered, multiplicity = 0, 1
        state_best, MI_best = None, 0
        for state in self.possible_states:
            MI = state.get_mutual_information()
            if MI > MI_best:
                state_best, MI_best = state, MI
                multiplicity = 1
            elif MI == MI_best:
                multiplicity += 1
            states_considered += 1
           
        self.info['states_considered'] = states_considered
        self.info['multiplicity'] = multiplicity     
        return state_best, MI_best



class ReceptorOptimizerAnnealing(Annealer):
    """ class for finding optimal receptor distribution using simulated
    annealing """
    
    Tmax =  1e2     # Max (starting) temperature
    Tmin =  1e-2    # Min (ending) temperature
    steps = 1e5     # Number of iterations
    updates = 2     # Number of outputs
    copy_strategy = 'method'


    def move(self):
        """ change a single bit in any of the receptor vectors """
        self.state.mutate_receptors()

        
    def energy(self):
        """ returns the energy of the current state """
        return -self.state.get_mutual_information()
    
    
    def optimize(self):
        """ optimizes the receptors and returns the best receptor set together
        with the achieved mutual information """
        state_best, energy_best = self.anneal()
        return state_best, -energy_best
