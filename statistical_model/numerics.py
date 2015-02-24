'''
Created on Feb 24, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import numpy as np



class ReceptorNumerics(object):
    """ a simple energies that implements the binary glomeruli response energies,
    where each concentration field in the mucus is mapped to a set of binary
    numbers indicating whether the respective glomeruli are active or not """
    num_recep_default = 350 #< default number of receptors
    cov_frac_default = 0.33 #< default maximal fraction of glomeruli that
                            #  respond to a given odorant

    def __init__(self, energies, num_receptors, num_substrates):
        """ initialize a energies for calculation of num_odors odorants
        `energies` defines the energies used for the energies
        `num_receptors` determines the number of receptors        
        """
        self.energies = energies
        self.dist = energies.get_sensitivity_distribution()
        self.Nr = num_receptors
        self.Ns = num_substrates
        self.choose_sensitivites()


    @property
    def label(self):
        return 'Numerics ' + self.energies.name


    def choose_sensitivites(self):
        """ initialize the sensitivities of the receptors """
        self.sens = self.dist.rvs(size=(self.Nr, self.Ns))


    #===========================================================================
    # SINGLE SUBSTRATE
    #===========================================================================


    def get_activities_single(self, conc):
        """ returns an array of activity patterns for each individual odorant """
        conc = np.atleast_1d(conc)
        return (self.sens.T[:, :, None] * conc[None, None, :] > 1)
    
    
    def activity_single(self, conc):
        actv = self.get_activities_single(conc)
        frac = actv.mean(axis=0)
        return frac.mean(axis=0), frac.std(axis=0)
        

    #===========================================================================
    # MULTIPLE SUBSTRATES
    #===========================================================================


    def get_activities_mixture(self, conc):
        """ returns the receptor activity pattern for a given set of odor
        concentrations """
        assert len(conc) == self.Ns
        conc = np.asarray(conc).reshape(-1, self.num_odors)
        act = np.tensordot(self.sens, conc, axes=(-1, -1))
        return (act > 1)
