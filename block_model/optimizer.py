'''
Created on Jan 12, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import time

from simanneal import Annealer



class ReceptorOptimizerBruteForce(object):
    """ class for finding optimal receptor distribution using and exhaustive
    search """

    def __init__(self, possible_states, output=0):
        """
        `possible_states` must be a class that handles all possible states
        `output` indicates the approximate time in seconds between progress
            output. If output <= 0, no output is done
        """
        self.possible_states = possible_states
        self.start = 0 #< start time
        self.output = output
        
        self.info = {'states_considered': 0}
        self._last_output = 0
                

    def do_output(self, step, force_output=False):
        """ outputs current progress when necessary """
        cur_time = time.time()
        if force_output or cur_time - self._last_output > self.output:
            progress = round(100 * step / self.info['states_total'])
            speed = step/(cur_time - self.start)
            print('%3d%% finished (%d iter/sec)' % (progress, speed))
            self._last_output = time.time()
        

    def optimize(self):
        """ optimizes the receptors and returns the best receptor set together
        with the achieved mutual information.
        Extra information about the optimization procedure is stored in the
        `info` dictionary of this object """
        self.info['states_total'] = len(self.possible_states)

        state_best, MI_best = None, -1
        self.start = time.time()
        for step, state in enumerate(self.possible_states):
            MI = state.get_mutual_information()
            if MI > MI_best:
                state_best, MI_best = state.copy(), MI
                multiplicity = 1
            elif MI == MI_best:
                multiplicity += 1
            if step % 1000 == 0 and self.output > 0:
                self.do_output(step)

        # finalize           
        self.do_output(step, force_output=True)
        self.info['states_considered'] = step + 1 
        self.info['total_time'] = time.time() - self.start    
        self.info['performance'] = (step + 1) / self.info['total_time']
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
        self.possible_states = possible_states
        initial_state = possible_states.get_random_state()
        super(ReceptorOptimizerAnnealing, self).__init__(initial_state)

        self.info = {}


    def move(self):
        """ change a single bit in any of the receptor vectors """
        self.possible_states.mutate_state(self.state)
        
        
    def energy(self):
        """ returns the energy of the current state """
        return -self.state.get_mutual_information()
    
    
    def optimize(self):
        """ optimizes the receptors and returns the best receptor set together
        with the achieved mutual information """
        state_best, energy_best = self.anneal()
        self.info['total_time'] = time.time() - self.start    
        self.info['states_considered'] = self.steps
        self.info['performance'] = self.steps / self.info['total_time']
        
        return state_best, -energy_best



def ReceptorOptimizerAuto(state_collection, time_limit=1, verbose=True,
                          parameter_estimation=False, output=0):
    """ factory that chooses the the right optimizer with the right parameters
    based on the number of receptors that have to be tested and the time limit
    that is supplied. The `time_limit` should be given in seconds.
    """
    # estimate how many iterations we can do per second 
    time_per_iter = state_collection.estimate_computation_speed()
    max_iter = time_limit/time_per_iter

    # determine which optimizer to use based on time constraints    
    if len(state_collection) < max_iter:
        # few steps => use brute force
        if verbose:
            print('Brute force for %d items (est. %d items/sec)'
                  % (len(state_collection), 1/time_per_iter)) 
        optimizer = ReceptorOptimizerBruteForce(state_collection, output=output)
        
    else:
        # many steps => use simulated annealing
        if verbose:
            print('Simulated annealing for %d items (est. %d items/sec)' 
                  % (len(state_collection), 1/time_per_iter)) 

        # create optimizer instance            
        optimizer = ReceptorOptimizerAnnealing(state_collection)
        
        if parameter_estimation:
            # automatically estimate the parameters for the simulated annealing
            schedule = optimizer.auto(time_limit/60)
            optimizer.set_schedule(schedule)
            
            if output != 0:
                optimizer.updates = time_limit/output
            else:
                optimizer.updates = 0
            
        elif output == 0:
            optimizer.updates = 0
        
    return optimizer

   
    