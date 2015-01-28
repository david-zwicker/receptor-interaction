from model_block_1D import *
from model_tetris_1D import *
from experiments import *
from optimizer import *

# try importing numba for speeding up calculations
try:
    from numba_speedup import NumbaPatcher
    NumbaPatcher.enable() #< enable the speed-up by default
except ImportError:
    NumbaPatcher = None
    print('NumbaPatcher could not be loaded. Slow functions will be used')
    
    