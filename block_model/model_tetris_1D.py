'''
Created on Jan 16, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import numpy as np

from .model_block_1D import (Chain, Chains, ChainCollections, ChainsInteraction,
                             ChainsInteractionPossibilities)



class Tetris(Chain):
    """ class representing a single tetris block """
    
    @property
    def heights(self):
        return self.colors
    
    
    def to_string(self):
        """ returns Unicode string representing the tetris block """
        chars = u"\u2581\u2582\u2583\u2585\u2586\u2587"
        
        if self.heights > len(chars):
            raise ValueError('Can only visualize tetris with %d heights.' %
                             len(chars))
        
        # calculate the height factor for emphasis
        f = 2 if self.heights <= 3 else 1
        # create the string
        return ''.join(chars[f*h] for h in self)


    def get_mpl_collection(self, center=(0, 0), r_max=1, r_min=0.5, 
                           cmap=None, **kwargs):
        """ create a matplotlib patch collection visualizing the chain
        `center` denotes the center of the object
        `r_max` is the outer radius of the highest block
        `r_min` is the inner radius of all blocks
        """
        from matplotlib.patches import Wedge
        from matplotlib.collections import PatchCollection
        from matplotlib import cm
        
        if cmap is None:
            cmap = cm.jet
        
        if self.cyclic:
            # create the individual patches
            sector = 360 / len(self)
            patches = []
            for k, height in enumerate(self):
                angle = k * sector
                radius = r_min + (r_max - r_min)*(height + 1)/(self.colors + 1)
                patches.append(Wedge(center, radius, angle, angle + sector,
                                     width=radius - r_min, **kwargs))
                
            # combine the patches in a collection
            pc = PatchCollection(patches, cmap=cmap)
            pc.set_array(self)
            
        else:
            raise NotImplementedError
            
        return pc



class TetrisBlocks(Chains):
    """ class that represents all tetris blocks of length l """
        
    def __init__(self, l, heights=2, fixed_length=True, cyclic=False):
        self.heights = heights
        super(TetrisBlocks, self).__init__(l, heights, fixed_length, cyclic)
        
        
    def __repr__(self):
        return ('%s(l=%d, heights=%d, fixed_length=%s, cyclic=%s)' %
                (self.__class__.__name__, self.l, self.heights,
                 self.fixed_length, self.cyclic))



class TetrisCollections(ChainCollections):
    """ class that represents all possible collections of `cnt` distinct tetris
    blocks of length `l` """

    single_item_class = TetrisBlocks
    colors_str = 'heights'
    
    @property
    def heights(self):
        return self.colors        
        
        

class TetrisInteraction(ChainsInteraction):
    """ class that represents the interaction between a set of substrates and a
    set of receptors built of tetris blocks.
    """

    single_item_class = TetrisBlocks

    
    def __init__(self, substrates, receptors, heights,
                 cache=None, energies=None):
        self.heights = heights
        super(TetrisInteraction, self).__init__(substrates, receptors, heights,
                                                cache, energies)

        
    def __repr__(self):
        return ('%s(substrates=%s, receptors=%s, heights=%d)' %
                (self.__class__.__name__, self.substrates, self.receptors,
                 self.heights))
        

    def __str__(self):
        return ('%s(%d Substrates, %d Receptors, heights=%d)' %
                (self.__class__.__name__, len(self.substrates),
                 len(self.receptors), self.heights))
        
        
    def update_energies_receptor(self, idx_r):
        """ updates the energy of the `idx_r`-th receptor """
        receptor = self.receptors[idx_r]
        l_s, l_r = self.substrates2.shape[1] // 2, len(receptor)
        
        # iterate over all translations of the receptor over the substrates
        energies = self.energies[:, idx_r]
        energies[:] = 0
        for i in xrange(l_s):
            # calculate the distance of all blocks
            dist = self.substrates2[:, i:i+l_r] + receptor[np.newaxis, :]
            # calculate the maximal distance for each substrate
            dist_max = dist.max(axis=1)
            
            # calculate the number of contact points
            Es = np.sum(dist == dist_max[:, np.newaxis], axis=1)
            
            # take the maximum with previously calculated energies
            np.maximum(Es, energies, energies)
                
       
    def update_energies(self):
        """ calculates all the energies between the substrates and the
        receptors
        """
        # calculate the energies with a sliding window
        for idx_r in xrange(len(self.receptors)):
            self.update_energies_receptor(idx_r)

        return self.energies        
        
        
        
class TetrisInteractionPossibilities(ChainsInteractionPossibilities):        
    """ class that represents all possible combinations of substrate and
    receptor interactions """

    receptor_collection_class = TetrisCollections
    interaction_class = TetrisInteraction
    
    
    def __init__(self, substrates, possible_receptors):      
        super(TetrisInteractionPossibilities, self).__init__(substrates, 
                                                             possible_receptors)

    @property
    def heights(self):
        return self.possible_receptors.heights

        
        