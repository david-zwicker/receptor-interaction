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
    
    

def optimize_receptors(parameters):
    """ convenience function for receptor optimization where all parameters
    are given as a single dictionary """
    
    parameters_default = {
        # model parameters
        'collection': ChainCollections,
        'model': ChainsModel,
        
        'cross_talk': 0.,
        'interaction_range': 'full',
        'colors': 2,     #< different colors on the blocks
        'cnt_s': 'all',  #< substrate count
        'l_s': 6,        #< substrate length
        'cnt_r': 4,      #< number of receptors
        'l_r': 4,        #< receptor length
        
        # experiment parameters
        'experiment': DetectSingleSubstrate,
    
        'num_substrates': 1,
        'temperature': 1.,
        'threshold': 1.,
        
        # optimization parameters
        'time_limit': 1, #< in minutes
        'optimizer_output': 10,
    }
    
    data = parameters_default.copy()
    data.update(parameters)
    
    # choose substrates
    substrates = data['collection'](
        data['cnt_s'], data['l_s'], data['colors'], cyclic=True
    ).choose_random()
        
    # choose receptors
    receptors = data['collection'](
        data['cnt_r'], data['l_r'], data['colors'], cyclic=False
    )

    # define the model
    model = data['model'](
        substrates, receptors,
        cross_talk=data['cross_talk'],
        interaction_range=data['interaction_range']
    )

    # define the experiment
    if data['experiment'] == DetectMultipleSubstrates:
        experiment = DetectMultipleSubstrates(
            num_substrates=data['num_substrates'],
            temperature=data['temperature'], threshold=data['threshold']
        )
    else:
        experiment = data['experiment'](
            temperature=data['temperature'], threshold=data['threshold']
        )
    
    # setup the optimizer 
    optimizer = ReceptorOptimizerAuto(
        experiment, model, time_limit=data['time_limit']*60,
        verbose=True, parameter_estimation=True,
        output=data['optimizer_output']
    )
    
    # find the best receptor combination for the given experiment
    state, MI_numeric = optimizer.optimize()

    # return the result    
    result = {
        'parameters': data,
        'MI_theory': experiment.get_max_mutual_information(model),
        'MI_best': MI_numeric,
        'state_best': state,
        'optimizer_info': optimizer.info
    }
    return result
