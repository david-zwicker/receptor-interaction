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


    def get_mpl_collection(self, center=(0, 0), size=1, width=0.5, cmap=None,
                           **kwargs):
        """ create a matplotlib patch collection visualizing the tetris:
        
            `center` denotes the center of the object
        For a linear chain, we have
            `size` is the total length of the chain
            `width` is the associated height
        For a circular chain, we have
            `size` is the outer radius of the highest block
            `width` is the height of the highest block
        """
        from matplotlib.patches import Rectangle, Wedge
        from matplotlib.collections import PatchCollection
        from matplotlib.colors import Normalize
        from matplotlib import cm

        if cmap is None:
            cmap = cm.jet
        
        if self.cyclic:
            # create cyclic chain    
            r_min, r_max = size - width, size
            # create the individual patches
            sector = 360 / len(self)
            patches = []
            for k, height in enumerate(self):
                angle = k * sector
                radius = r_min + (r_max - r_min)*(height + 1)/(self.colors + 1)
                patches.append(Wedge(center, radius, angle, angle + sector,
                                     width=radius - r_min, **kwargs))
                
            # combine the patches in a collection
            pc = PatchCollection(patches, cmap=cmap,
                                 norm=Normalize(vmin=0, vmax=self.heights))
            pc.set_array(self)

        else:
            # create linear chain
            sector = size / len(self)
            patches = []
            for k, h in enumerate(self):
                x = center[0] - size/2 + k*sector
                y = center[1] - width/2
                h = width * (h + 1)/(self.heights + 1)
                patches.append(Rectangle((x, y), sector, h, **kwargs))
            
        # combine the patches in a collection
        pc = PatchCollection(patches, cmap=cmap,
                             norm=Normalize(vmin=0, vmax=self.heights))
        pc.set_array(self)
        
        return pc
    


class TetrisBlocks(Chains):
    """ class that represents all tetris blocks of length l """

    colors_str = 'heights'


    def __init__(self, l, heights=2, cyclic=False):
        super(TetrisBlocks, self).__init__(l, heights, cyclic)

        
    @property
    def heights(self):
        return self.colors        
        


class TetrisCollections(ChainCollections):
    """ class that represents all possible collections of `cnt` distinct tetris
    blocks of length `l` """

    single_item_class = TetrisBlocks
    
    
    def __init__(self, cnt, l, heights=2, cyclic=False):
        super(TetrisCollections, self).__init__(cnt, l, heights, cyclic)
    
    
    @property
    def heights(self):
        return self.colors        
        
        

class TetrisInteraction(ChainsInteraction):
    """ class that represents the interaction between a set of substrates and a
    set of receptors built of tetris blocks.

    This code currently assumes that the substrates are cyclic chains and that
    the receptors are linear.
    """

    item_collection_class = TetrisCollections

    
    def __init__(self, substrates, receptors, heights, interaction_range=1000,
                 cache=None, energies=None):
        super(TetrisInteraction, self).__init__(substrates, receptors, heights,
                                                interaction_range, cache,
                                                energies)

    @property
    def heights(self):
        return self.colors

        
    def update_energies_receptor(self, idx_r):
        """ updates the energy of the `idx_r`-th receptor """
        receptor = self.receptors[idx_r]
        l_s = self.substrates2.shape[1] // 2
        l_r = len(receptor) 

        energies = self.energies[:, idx_r]
        energies[:] = 0

        # count the number of bonds
        if self.interaction_range < l_r:
            # the substrates interact with part of the receptor
            rng = self.interaction_range            
            for i in xrange(l_s): #< try all substrate translations
                for j in xrange(l_r - rng + 1): #< try all receptor translations
                    # calculate the distance of all blocks
                    dist = (self.substrates2[:, i:i + rng]
                            + receptor[np.newaxis, j:j + rng])
                    # calculate the maximal distance for each substrate
                    dist_max = dist.max(axis=1)
                    
                    # calculate the number of contact points
                    Es = np.sum(dist == dist_max[:, np.newaxis], axis=1)
                    
                    # take the maximum with previously calculated energies
                    np.maximum(Es, energies, energies)
                
        else:
            # the substrates interact with the full receptor
            rng = l_r
            for i in xrange(l_s): #< try all substrate translations
                # calculate the distance of all blocks
                dist = self.substrates2[:, i:i + l_r] + receptor[np.newaxis, :]
                # calculate the maximal distance for each substrate
                dist_max = dist.max(axis=1)
                
                # calculate the number of contact points
                Es = np.sum(dist == dist_max[:, np.newaxis], axis=1)
                
                # take the maximum with previously calculated energies
                np.maximum(Es, energies, energies)

        # account for cross-talk
        if self.cross_talk != 0:
            energies *= (1 - self.cross_talk)
            energies += rng * self.cross_talk 
        
        
        
class TetrisInteractionPossibilities(ChainsInteractionPossibilities):        
    """ class that represents all possible combinations of substrate and
    receptor interactions """

    receptor_collection_class = TetrisCollections
    interaction_class = TetrisInteraction
    
    
    def __init__(self, substrates, possible_receptors,
                 interaction_range='full', **kwargs):
        super(TetrisInteractionPossibilities, self).__init__(
            substrates, possible_receptors, interaction_range, **kwargs
        )

    @property
    def heights(self):
        return self.possible_receptors.heights

        
        