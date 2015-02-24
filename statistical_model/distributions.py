'''
Created on Feb 24, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import numpy as np
from scipy import stats



def EnergiesNormal(mean, std):
    """ returns a model in which the energies are normal distributed """
    return stats.lognorm(scale=np.exp(mean), s=std)