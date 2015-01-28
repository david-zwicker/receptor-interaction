'''
Created on Jan 28, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import datetime
import itertools
import time
import multiprocessing as mp
import cPickle as pickle

from .model_block_1D import ChainCollections, ChainsModel
from .experiments import (DetectSingleSubstrate, DetectMultipleSubstrates,
                          MeasureMultipleSubstrates)
from .optimizer import ReceptorOptimizerAuto



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
        'concentration_range': (0.01, 1),
        'temperature': 1.,
        'threshold': 'auto',
        
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
        
    elif data['experiment'] == MeasureMultipleSubstrates:
        experiment = MeasureMultipleSubstrates(            
            num_substrates=data['num_substrates'],
            concentration_range=data['concentration_range'],
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
    
    # output progress if possible
    if 'job_data' in data:
        job = data['job_data']
        # conservative estimate of jobs that have finished and that are left
        jobs_done = max(1, job['id'] - job['processes'] + 1)
        jobs_left = job['num_jobs'] - job['id'] - 1
        # calculate quantities to show
        perc_done = 100*jobs_done/job['num_jobs']
        sec_left = jobs_left*(time.time() - job['start_time'])/jobs_done
        time_left = datetime.timedelta(seconds=int(sec_left))
        # output information
        output = ('Runtime estimate: %3d%% finished (%d/%d) - Time left: %s' %
                  (perc_done, jobs_done, job['num_jobs'], time_left))
        print('-'*len(output) + '\n' + output + '\n' + '-'*len(output))
    
    return result



def optimize_receptors_many(parameters_base, parameters_vary, result_file=None,
                            processes='auto'):
    """ optimizes receptors for multiple conditions
    `parameters_base` is a dictionary with the parameters common for all jobs
    `parameters_vary` is a dictionary with the parameters that are changed
        across the jobs. The values of this dictionary should be iterable.
    `result_file` is the name of the file where the results are written to. If
        it is None, the results are only returned from the function.
    `processes` determines how many processes to use in the calculation.
    """
    # determine how many processes to use
    if processes == 'auto':
        pool = mp.Pool()
        processes = pool._processes 
        map_func = pool.map
    elif processes == 1:
        map_func = map
    else:
        map_func = mp.Pool(processes).map
    
    # count the number of jobs
    num_jobs = 1
    parameters_vary_values = parameters_vary.values()
    for params in parameters_vary_values:
        num_jobs *= len(params) 
            
    # prepare the data for all the jobs
    jobs = []
    p_keys = parameters_vary.keys()
    for j_id, p_values in enumerate(itertools.product(*parameters_vary_values)):
        job_data = parameters_base.copy()
        for key, value in itertools.izip(p_keys, p_values):
            job_data[key] = value
        job_data['job_data'] = {'id': j_id,
                                'num_jobs': num_jobs,
                                'processes': processes,
                                'start_time': time.time(),}
        jobs.append(job_data)
    
    # run the jobs
    result = map_func(optimize_receptors, jobs)
    
    data = {'parameters_base': parameters_base,
            'parameters_vary': parameters_vary,
            'result': result,
            'runtime': time.time() - jobs[0]['job_data']['start_time']}        
        
    # save the results
    if result_file:
        pickle.dump(data, open(result_file, 'wb'))
    return data
