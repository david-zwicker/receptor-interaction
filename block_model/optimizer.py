'''
Created on Jan 12, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division


from simanneal import Annealer


class ReceptorOptimizer(Annealer):
    """ class for finding optimal receptor distribution """
    
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
    