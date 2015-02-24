'''
Created on Nov 26, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import numpy as np
from scipy import optimize



class ReceptorTheory(object):
    """ a simple energies that implements the binary glomeruli response energies,
    where each concentration field in the mucus is mapped to a set of binary
    numbers indicating whether the respective glomeruli are active or not """
    num_recep_default = 350 #< default number of receptors
    cov_frac_default = 0.33 #< default maximal fraction of glomeruli that
                            #  respond to a given odorant


    def __init__(self, energies, num_receptors):
        """ initialize a energies for calculation of num_odors odorants
        `energies` defines the energies used for the energies
        `num_receptors` determines the number of receptors        
        """
        self.energies = energies
        self.dist = energies.get_sensitivity_distribution()
        self.Nr = num_receptors
    
    
    @property
    def label(self):
        return 'Theory ' + self.energies.name
    
    
    #===========================================================================
    # SINGLE SUBSTRATES
    #===========================================================================


    @property
    def max_activity_single(self):
        return self.energies.max_frac
    
    
    def activity_single(self, conc):
        """ returns the fraction of glomeruli activated by each odor """
        return self.dist.sf(1/conc) #< survival function (1 - cdf)


    def concentration_single(self, actv):
        """ returns the concentration of a single odorant given the fraction of
        excited glomeruli """
        with np.errstate(divide='ignore'):
            return 1/self.dist.ppf(1 - actv)


    def concentration_typical(self):
        """ returns a typical concentration, where half the glomeruli are 
        activated """
        return 1/self.energies.typical_sensitivity
        

    def concentration_min(self):
        """ minimal concentration at which the first glomerulus is excited """
        return self.concentration_single(1/self.Nr)


    def concentration_max(self):
        """ minimal concentration at which the first glomerulus is excited """
        return self.concentration_single(1)


    def delta_single(self, conc):
        """ returns the minimal concentration necessary to excite a glomerulus
        if the odorant is already present at concentration `conc` """
        with np.errstate(divide='ignore'):
            return conc**2/(self.Nr * self.dist.pdf(1/conc))


    def resolution_single(self, conc):
        """ returns the resolution at concentration `conc` """
        return conc/self.delta_single(conc)


    #===========================================================================
    # MULTIPLE SUBSTRATES
    #===========================================================================


    def overlap_mixture(self, cvec):
        """ returns the fraction of glomeruli that would be activated by any of 
        the odorants in the given concentration field """
        cvec = np.asarray(cvec)
        assert cvec.shape[-1] == self.Ns
        return np.prod(self.activity_single(cvec), axis=-1)


    def activity_mixture(self, cvec):
        """ returns the fraction of glomeruli activated by the given 
        concentration field """
        cvec = np.asarray(cvec)
        assert cvec.shape[-1] == self.Ns
        return 1 - np.prod(self.activity_single(cvec), axis=-1)
    

    def num_active_mixture(self, cvec):
        """ returns the number of active receptors for the given concentration
        vector """
        return self.Nr * self.activity_mixture(cvec)
        

    #===========================================================================
    # OLD METHODS
    #===========================================================================


    def theory_excitation_threshold0(self, conc=0, add_recep=1):
        """ calculates how much the concentration of component 0 must be 
        increased to excite `add_recep` additional glomeruli, given an already
        present background of odorant concentrations `conc` """
        conc = np.atleast_2d(conc)
        actv = self.theory_active_fraction_per_odor(conc)
        prod = self.num_recep * np.prod(1 - actv[..., 1:], axis=-1)
        arg = actv[..., 0] + add_recep/prod
        return self.theory_concentration_per_odor(arg) - conc[..., 0]


    def theory_discriminability(self, conc, subtract_marginal_glomeruli=False):
        """ return the number of odors that can be discriminated at a certain
        concentration level
        If subtract_marginal_glomeruli is True, the calculation corrects for
            the glomeruli that could be excited by increasing the concentration
            of already present odorants. Consequently, the discriminability
            decreases if this option is chosen

        """
        actv = self.theory_active_fraction_per_odor(conc)
        #result = (-np.log(actv) - np.log(self.num_recep))/np.log(1 - actv)
        result = 1 - (np.log(self.num_recep) + np.log(actv))/np.log(1 - actv)

        if subtract_marginal_glomeruli:
            # do numerical optimization to find the number of odorants
            def rhs(n):
                factor = self.num_recep*(1 - actv)**n - n
                return 1 + actv*factor
            result = optimize.newton(rhs, result)

        return result



    def theory_responsiveness(self):
        """ returns a responsiveness value, which indicates how sensitive
        the system is to detect a single odor """
        return 1/self.theory_concentration_per_odor([1/self.num_recep])[0]


    def theory_specificity(self):
        """ returns a specificity factor, which states how many odorants can
        be distinguished at the characteristic concentration """
        conc = self.characteristic_conc
        return self.theory_discriminability(conc)




