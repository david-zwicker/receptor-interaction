'''
Created on Feb 24, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import numpy as np
from scipy import stats 

from .distributions import (PartialLogNormDistribution,
                            PartialLogUniformDistribution)



class EnergiesNormal(object):
    """ represents a model in which the energies are normal distributed """
    
    def __init__(self, mean, std, max_frac=1, name=""):
        """ initialize by supplying mean and std of the energies """
        self.Emean = mean
        self.Estd = std
        self.max_frac = max_frac
        self.name = name
        

    def get_sensitivity_distribution(self):
        """ returns the sensitivity distribution that is associated with this
        energy distribution """
        if self.max_frac == 1:
            dist = stats.lognorm(scale=np.exp(self.Emean), s=self.Estd)
        else:
            dist = PartialLogNormDistribution(scale=np.exp(self.Emean),
                                              s=self.Estd,
                                              frac=self.max_frac)
        return dist
        
    
    @property
    def typical_energy(self):
        """ returns the energy at which the activity is at half its maximum """
        return self.Emean
    
    
    @property
    def typical_sensitivity(self):
        """ returns the sensitivity at which the activity is at half its maximum """
        return np.exp(self.typical_energy)
    


class EnergiesUniform(object):
    """ represents a model in which the energies are normal distributed """
    
    def __init__(self, Emin, Emax, max_frac=1, name=""):
        """ initialize by supplying the minimal and maximal energy """
        self.Emin = Emin
        self.Emax = Emax
        self.max_frac = max_frac
        self.name = name
        

    def get_sensitivity_distribution(self):
        """ returns the sensitivity distribution that is associated with this
        energy distribution """
        scale = np.exp(0.5 * (self.Emax + self.Emin))
        shape = np.exp(0.5 * (self.Emax - self.Emin))
        return PartialLogUniformDistribution(scale=scale,
                                             s=shape,
                                             frac=self.max_frac)        
    
    @property
    def typical_energy(self):
        """ returns the energy at which the activity is at half its maximum """
        return 0.5 * (self.Emax + self.Emin)
    
    
    @property
    def typical_sensitivity(self):
        """ returns the sensitivity at which the activity is at half its maximum """
        return np.exp(self.typical_energy)

