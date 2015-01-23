Overview
========
We study the interaction between substrates and receptors.

Class structure
---------------
* Characterization of substrates and receptors separately:
    - single blocks (only used for visualization and auxiliary methods)
    - given list of blocks (this is just a list of instances of the previous class)
    - all possible blocks of certain kind (same length, possible colors, etc.)
    - all possible collections of a fixed number of blocks
* State (interaction of substrate with receptors) :
    - interaction between two blocks (this is a special case of the next class)
    - interaction between two sets of blocks (substrates and receptors)
* Model (controller that manages all possible states)
    - interaction between a list of substrate and all possible receptors
* Experiments (determines the physical setup and how the readout is interpreted):
    - functions that calculates the mutual information for a given
        substrate/receptor combination 
* Optimization:
    - vary the _states_ of a _model_ to vary the mutual information calculated
        for a given _experiment_
    - several optimization schemes (brute-force, simulated annealing) can be
        implemented
    
 TODO
 ====
* 2D carpets as a generalization of chains
* other experiments (detecting a mixture of substrates)