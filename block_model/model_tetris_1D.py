'''
Created on Jan 16, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import numpy as np

from .model_block_1D import (Chains, ChainsCollection, ChainsInteraction,
                             ChainsInteractionCollection)



def tetris_to_string(tetris, heights=None):
    """ converts a single tetris block to a representative Unicode sequence """
    chars = u"\u2581\u2582\u2583\u2585\u2586\u2587"
    
    if heights is None:
        heights = max(tetris)
    else:
        assert heights <= max(tetris)
    
    if heights > len(chars):
        raise ValueError('Can only visualize tetris with %d heights.' %
                         len(chars))
    
    # calculate the height factor for emphasis
    f = 2 if heights <= 3 else 1
    # create the string
    return ''.join(chars[f*h] for h in tetris)



class Tetris(Chains):
    """ class that represents all tetris blocks of length l """
        
    def __init__(self, l, heights=2):
        self.heights = heights
        super(Tetris, self).__init__(l, heights)
        
        
    def __repr__(self):
        return ('%s(l=%d, heights=%d)' %
                (self.__class__.__name__, self.l, self.heights))



class TetrisCollection(ChainsCollection):
    """ class that represents all possible collections of `cnt` distinct tetris
    blocks of length `l` """

    single_item_class = Tetris 
    
     
    def __init__(self, cnt, l, heights=2):
        self.heights = heights
        super(TetrisCollection, self).__init__(cnt, l, heights)
        
        
    def __repr__(self):
        return ('%s(cnt=%d, l=%d, heights=%d)' %
                (self.__class__.__name__, self.cnt, self.l, self.heights))
        
        
        

class TetrisInteraction(ChainsInteraction):
    """ class that represents the interaction between a set of substrates and a
    set of receptors built of tetris blocks.
    """

    single_item_class = Tetris

    
    def __init__(self, substrates, receptors, heights,
                 cache=None, energies=None):
        self.heights = heights
        super(TetrisInteraction, self).__init__(substrates, receptors, heights,
                                                cache, energies)

        
    def __repr__(self):
        cnt_s, l_s = self.substrates.shape
        cnt_r, l_r = self.receptors.shape
        return ('%s(%d Substrates(l=%d), %d Receptors(l=%d), heights=%d)' %
                (self.__class__.__name__, cnt_s, l_s, cnt_r, l_r, self.heights))
        
        
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
        
        
        
class TetrisInteractionCollection(ChainsInteractionCollection):        
    """ class that represents all possible combinations of substrate and
    receptor interactions """

    receptor_collection_class = TetrisCollection
    interaction_class = TetrisInteraction
    
    
    def __init__(self, substrates, cnt_r, l_r, heights):      
        self.heights = heights
        super(TetrisInteractionCollection, self).__init__(substrates, cnt_r,
                                                          l_r, heights)


    def __repr__(self):
        return ('%s(%s, cnt_r=%d, l_r=%d, colors=%d)' %
                (self.__class__.__name__, repr(self.substrates),
                 self.receptors_collection.cnt,
                 self.receptors_collection.l, self.colors))