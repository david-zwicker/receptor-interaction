'''
Created on Feb 24, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import numpy as np
from scipy import stats, special



class TruncatedLogNormDistribution_gen(stats.rv_continuous):
    def _rvs(self, s, frac):
        # choose the receptor response characteristics
        res = np.exp(s * np.random.standard_normal(self._size))
        # switch off receptors randomly
        res[np.random.random(self._size) > frac] = 0
        return res
    
    
    def _pdf(self, x, s, frac):
        return frac / (s*x*np.sqrt(2*np.pi)) * np.exp(-1/2*(np.log(x)/s)**2)         
        
        
    def _cdf(self, x, s, frac): 
        return 1 + frac*(-0.5 + 0.5*special.erf(np.log(x)/(s*np.sqrt(2))))


    def _ppf(self, q, s, frac):
        q_scale = (q - (1 - frac)) / frac
        return np.where(q_scale < 0, 0, np.exp(s * special.ndtri(q_scale)))

        

TruncatedLogNormDistribution = TruncatedLogNormDistribution_gen(
    a=0, name='TruncatedLogNormDistribution'
)



class EnergiesNormal(object):
    """ represents a model in which the energies are normal distributed """
    
    def __init__(self, mean, std, max_frac=1, name=""):
        """ initialize by supplying mean and std of the energies """
        self.Emean = mean
        self.Estd = std
        self.max_frac = max_frac
        self.name = name
        

    def get_sensitivity_distribution(self):
        if self.max_frac == 1:
            dist = stats.lognorm(scale=np.exp(self.Emean), s=self.Estd)
        else:
            dist = TruncatedLogNormDistribution(scale=np.exp(self.Emean),
                                                s=self.Estd,
                                                frac=self.max_frac)
        return dist
        
    
    @property
    def typical_energy(self):
        return self.Emean
    
    
    @property
    def typical_sensitivity(self):
        return np.exp(self.Emean)
    
    
    
    

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
