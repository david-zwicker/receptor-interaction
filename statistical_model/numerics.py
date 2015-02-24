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


    def _activities_single(self, conc):
        """ returns an array of activity patterns for each individual odorant """
        return np.multiply.outer(self.sens, conc) > 1
    
    
    def activity_single(self, conc):
        """ return statistics of the activity pattern """
        actv = self._activities_single(conc)
        frac = actv.mean(axis=0) #< get the fraction of active receptors
        # get statistics by averaging over substrates
        return frac.mean(axis=0), frac.std(axis=0)
        
        
    def delta_single(self, conc):
        """ returns the minimal concentration necessary to excite a glomerulus
        if the odorant is already present at concentration `conc` """
        # get the excitations for each receptor-substrate pair for all conc.
        exc = np.multiply.outer(self.sens, conc)
        exc[exc > 1] = 0 #< we don't care for receptors that already are excited
        # find the receptor excitation that is just below threshold 
        exc_max = np.max(exc, axis=0)
        # calculate the concentration change necessary to excite this receptor
        with np.errstate(divide='ignore'):
            delta = conc[None, :]*(1/exc_max - 1)
        # get statistics by averaging over substrates
        return delta.mean(axis=0), delta.std(axis=0)


    def resolution_single(self, conc):
        """ returns the resolution at concentration `conc` """        
        return conc/self.delta_single(conc)


    #===========================================================================
    # MULTIPLE SUBSTRATES
    #===========================================================================


    def _activities_mixture(self, cvec):
        """ returns the receptor activity pattern for a given set of odor
        concentrations """
        cvec = np.atleast_2d(cvec)
        assert cvec.shape[-1] == self.Ns
        act = np.tensordot(self.sens, cvec, axes=(-1, -1))
        return (act > 1)


    def overlap_mixture(self, cvec):
        """ returns the fraction of glomeruli that would be activated by any of 
        the odorants in the given concentration field """
        act_single = self._activities_mixture(self, cvec)
        
        return np.prod(self.activity_single(cvec), axis=-1)
