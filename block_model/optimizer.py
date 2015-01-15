'''
Created on Jan 12, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

from simanneal import Annealer



class ReceptorOptimizerExhaustive(object):
    """ class for finding optimal receptor distribution using and exhaustive
    search """

    def __init__(self, state_collection):
        """ `state_collection` must be a class that handles all possible states
        """
        self.state_collection = state_collection
        self.info = {'states_considered': 0}


    def optimize(self):
        """ optimizes the receptors and returns the best receptor set together
        with the achieved mutual information.
        Extra information about the optimization procedure is stored in the
        `info` dictionary of this object """
        states_considered, multiplicity = 0, 1
        state_best, MI_best = None, -1
        for state in self.state_collection:
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


    def __init__(self, possible_states):
        """ `state_collection` must be a class that handles all possible states
        """
        initial_state = possible_states.get_random_state()
        super(ReceptorOptimizerAnnealing, self).__init__(initial_state)


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



def ReceptorOptimizerAuto(state_collection, time_limit=1, verbose=True,
                          parameter_estimation=False):
    """ class that chooses the the right optimizer with the right parameters
    based on the number of receptors that have to be tested and the time limit
    that is supplied. The `time_limit` should be given in seconds.
    """

    time_per_iter = 1/10000 #< this is from a single test run
    max_iter = time_limit/time_per_iter
    
    # TODO: determine parameters for simulated annealing based on time maximum 
    
    if len(state_collection) < max_iter:
        # few steps => use brute force
        if verbose:
            print 'Brute force for %d items.' % len(state_collection) 
        optimizer = ReceptorOptimizerExhaustive(state_collection)
        
    else:
        # many steps => use simulated annealing
        if verbose:
            print 'Simulated annealing for %d items.' % len(state_collection) 
        optimizer = ReceptorOptimizerAnnealing(state_collection)
        
        if parameter_estimation:
            # automatically estimate the parameters for the simulated annealing
            params = optimizer.auto(time_limit/60)
            optimizer.Tmax = params['tmax']
            optimizer.Tmin = params['tmin']
            optimizer.steps = params['steps']
        
    return optimizer



    
    