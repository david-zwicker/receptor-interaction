'''
Created on Nov 26, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import numpy as np
from scipy import special, optimize

from .utils import random_log_uniform



class ReceptorModel(object):
    """ a simple model that implements the binary glomeruli response model,
    where each concentration field in the mucus is mapped to a set of binary
    numbers indicating whether the respective glomeruli are active or not """
    num_recep_default = 350 #< default number of receptors
    cov_frac_default = 0.33 #< default maximal fraction of glomeruli that
                            #  respond to a given odorant


    def __init__(self, sensitivity_dist, num_receptors, cov_frac=None):
        """ initialize a model for calculation of num_odors odorants
        `sensitivity_dist` must be an instance of scipy.stats.rv_continous that
            determines the probability distribution of sensitivities
        `num_receptors` determines the number of receptors        
        """
        self.dist = sensitivity_dist
        self.Nr = num_receptors
    
    
    #===========================================================================
    # SINGLE SUBSTRATES
    #===========================================================================
    
    
    def activity_single(self, conc):
        """ returns the fraction of glomeruli activated by each odor """
        return self.dist.sf(1/conc) #< survival function (1 - cdf)


    def concentration_single(self, actv):
        """ returns the concentration of a single odorant given the fraction of
        excited glomeruli """
        raise 1/self.dist.ppf(1 - actv)


    def concentration_typical(self):
        """ returns a typical concentration, where half the glomeruli are 
        activated """
        raise NotImplementedError
        

    def concentration_min(self):
        """ minimal concentration at which the first glomerulus is excited """
        return self.concentration_single(1/self.Nr)


    def sensitivity_single(self, conc):
        """ concentration difference that is needed to excite an additional
        glomerulus """
        return self.dist.pdf(1/conc)/(self.Nr * conc**2)


    def resolution_single(self, conc):
        """ returns the minimal concentration necessary to excite a glomerulus
        if the odorant is already present at concentration `conc`
        """
        return conc/self.sensitivity_single(conc)


    #===========================================================================
    # MULTIPLE SUBSTRATES
    #===========================================================================


    def overlap_mixture(self, conc):
        """ returns the fraction of glomeruli that would be activated by any of 
        the odorants in the given concentration field """
        conc = np.asarray(conc)
        actv = self.activity_per_odor(conc)
        return np.prod(actv, axis=-1)


    def activity_mixture(self, conc):
        """ returns the fraction of glomeruli activated by the given 
        concentration field """
        conc = np.asarray(conc)
        return 1 - np.prod(self.dist.sf(1/conc), axis=-1)
    

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


'''
class ReceptorModelNormal(ReceptorModelBase):
    """ model with normal distribution of binding energies """

    S_mean_nolog_default = 1e-1 # location parameter of the sensitivity
    S_sigma_log_default = 4     # standard deviation of sensitivities in log-space


    def __init__(self, *args, **kwargs):
        self.S_sigma_log = kwargs.pop('sigma', self.S_sigma_log_default)
        self.S_mean_nolog = kwargs.pop('mean', self.S_mean_nolog_default)
        self.S_mean_log = np.log(self.S_mean_nolog)
        super(ReceptorModelNormal, self).__init__(*args, **kwargs)


    @property
    def characteristic_conc(self):
        return np.exp(-self.S_mean_log)


    def init_sensitivites(self):
        """ initialize the sensitivities of the receptors """
        # Choose the receptor response characteristics
        size = (self.num_recep, self.num_odors)
        self.sens = np.random.lognormal(self.S_mean_log, self.S_sigma_log, size)
        # switch off receptors randomly
        self.sens[np.random.random(size) > self.cov_frac] = 0


    def cdf(self, conc):
        """ cumulative probability distribution """
        erf_arg = (self.S_mean_log + np.log(conc))/(np.sqrt(2)*self.S_sigma_log)
        prob = 0.5*(1 + special.erf(erf_arg))
        return prob


    def cdf_diff(self, conc):
        """ derivative of the cumulative probability distribution """
        return (np.exp(-(self.S_mean_log + np.log(conc))**2/
                        (2*self.S_sigma_log**2))
                /(conc*np.sqrt(2*np.pi)*self.S_sigma_log))
    
    
    def cdf_inv(self, prob):
        """ inverse of the probability distribution """
    
    
    def theory_coverage_per_odor(self, conc):
        """ returns the fraction of glomeruli activated by each odor.
        This number gives the fraction relative to the maximal number of
        glomeruli that can be activated """
        erf_arg = (self.S_mean_log + np.log(conc))/(np.sqrt(2)*self.S_sigma_log)
        actv = 0.5*(1 + special.erf(erf_arg))
        return actv

    
    def theory_coverage_per_odor_diff(self, conc):


    def theory_concentration_per_odor(self, actv):
        """ returns the concentration of a single odorant given the fraction of
        excited glomeruli """
        actv = np.asarray(actv)
        conc = np.empty_like(actv) + np.nan
        idx = (0 < actv) & (actv < self.cov_frac)
        arg = special.erfinv(2*actv[idx]/self.cov_frac - 1)
        conc[idx] = np.exp(np.sqrt(2)*self.S_sigma_log*arg - self.S_mean_log)
        return conc



class ReceptorModelUniform(ReceptorModelBase):
    """ model with uniform distribution of binding energies """

    S_min = 1e-4 #< minimal sensitivity
    S_max = 1e2  #< maximal sensitivity


    @property
    def characteristic_conc(self):
        return 1/np.sqrt(self.S_min*self.S_max)


    def init_sensitivites(self):
        """ initialize the sensitivities of the receptors """
        # Choose the receptor response characteristics
        size = (self.num_recep, self.num_odors)
        self.sens = random_log_uniform(self.S_min, self.S_max, size)
        # switch off receptors randomly
        self.sens[np.random.random(size) > self.cov_frac] = 0


    def theory_coverage_per_odor(self, conc):
        """ returns the fraction of glomeruli activated by each odor """
        lo, hi = -np.log(self.S_max), -np.log(self.S_min)
        return np.clip((np.log(conc) - lo)/(hi - lo), 0, 1)


    def theory_concentration_per_odor(self, actv):
        """ returns the concentration of a single odorant given the fraction of
        excited glomeruli """
        actv = np.asarray(actv)
        conc = np.empty_like(actv) + np.nan
        idx = (0 < actv) & (actv < self.cov_frac)
        exponent = actv[idx] / self.cov_frac
        conc[idx] = 1/self.S_max * (self.S_max/self.S_min)**exponent
        return conc
'''




class ReceptorModelNumerical(object):
    """ a simple model that implements the binary glomeruli response model,
    where each concentration field in the mucus is mapped to a set of binary
    numbers indicating whether the respective glomeruli are active or not """
    num_recep_default = 350 #< default number of receptors
    cov_frac_default = 0.33 #< default maximal fraction of glomeruli that
                            #  respond to a given odorant


    def __init__(self, sensitivity_dist, num_receptors, num_substrates):
        """ initialize a model for calculation of num_odors odorants
        `sensitivity_dist` must be an instance of scipy.stats.rv_continous that
            determines the probability distribution of sensitivities
        `num_receptors` determines the number of receptors        
        """
        self.dist = sensitivity_dist
        self.Nr = num_receptors
        self.Ns = num_substrates
        self.choose_sensitivites()


    def choose_sensitivites(self):
        """ initialize the sensitivities of the receptors """
        self.sens = self.dist.rvs(size=(self.Nr, self.Ns))


    def get_activity(self, conc):
        """ returns the receptor activity pattern for a given set of odor
        concentrations """
        assert len(conc) == self.Ns
        conc = np.asarray(conc).reshape(-1, self.num_odors)
        act = np.tensordot(self.sens, conc, axes=(-1, -1))
        return (act > 1)


    def get_single_activities(self, conc):
        """ returns an array of activity patterns for each individual odorant """
        assert len(conc) == self.Ns
        conc = np.atleast_1d(conc)
        return (self.sens.T[:, :, None] * conc > 1)

