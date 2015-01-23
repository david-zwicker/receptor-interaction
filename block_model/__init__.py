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
    print('Numba was not found. Slow functions will be used')
    
    

def optimize_receptors(data):
    """ convenience function for receptor optimization where all parameters
    are given as a single dictionary """
    # choose substrates
    substrates = data['collection'](
        data['cnt_s'], data['l_s'], data['colors'], cyclic=True
    ).choose_random()
        
    # choose receptors
    receptors = data['collection'](
        data['cnt_r'], data['l_r'], data['colors'], cyclic=False
    )

    # define the model
    model = data['model'](substrates, receptors, data['interaction_range'])

    # define the experiment
    experiment = data['experiment'](
        temperature=data['temperature'], threshold=data['threshold']
    )
    
    # optimize the receptors
    optimizer = ReceptorOptimizerAuto(
        experiment, model, time_limit=data['time_limit']*60,
        verbose=True, parameter_estimation=True, output=10
    )
    
    # find the best state
    state, MI_numeric = optimizer.optimize()

    # return the result    
    result = {
        'data': data,
        'MI_numeric': MI_numeric,
        'MI_theory': experiment.get_max_mutual_information(model),
        'best_state': state,
        'optimizer_info': optimizer.info
    }
    return result