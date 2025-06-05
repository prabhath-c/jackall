import hashlib
import json
import pandas as pd
import matplotlib.pyplot as plt
from pyiron_atomistics import Project
import pyiron_potentialfit
from itertools import combinations

weighting_params_dict = {
        ## weights for the structures energies/forces are associated according to the distance to E_min:
        ## convex hull ( energy: convex_hull) or minimal energy per atom (energy: cohesive)
        "type": "EnergyBasedWeightingPolicy",
        ## number of structures to randomly select from the initial dataset
        "nfit": 10000,         
        ## only the structures with energy up to E_min + DEup will be selected
        "DEup": 10.0,  ## eV, upper energy range (E_min + DElow, E_min + DEup)        
        ## only the structures with maximal force on atom  up to DFup will be selected
        "DFup": 50.0, ## eV/A
        ## lower energy range (E_min, E_min + DElow)
        "DElow": 1.0,  ## eV
        ## delta_E  shift for weights, see paper
        "DE": 1.0,
        ## delta_F  shift for weights, see paper
        "DF": 1.0,
        ## 0<wlow<1 or None: if provided, the renormalization weights of the structures on lower energy range (see DElow)
        "wlow": 0.95,        
        ##  "convex_hull" or "cohesive" : method to compute the E_min
        "energy": "convex_hull",        
        ## structures types: all (default), bulk or cluster
        "reftype": "all",        
        ## random number seed
        "seed": 6169
}

def setup_job(project, df_All, atomic_symbols, hashed_key, 
              hyperparameters={'training_ratio':0.7, 
                               'cut_rad':7.0, 
                               'nradmax_by_orders':[15, 3, 2, 1], 
                               'lmax_by_orders':[0, 3, 2, 1]},
              cumulative_atoms=True):
    
    # Unpack dictionary for ease
    training_ratio = hyperparameters['training_ratio']
    cut_rad = hyperparameters['cut_rad']
    nradmax_by_orders = hyperparameters['nradmax_by_orders']
    lmax_by_orders = hyperparameters['lmax_by_orders']
    
    # Filter data as required
    filtered_df = filter_df(df_All, atomic_symbols, master_order, cumulative_atoms)
    
    # Create training and testing set
    training_set = filtered_df.sample(frac=training_ratio, random_state=6919)
    testing_set = filtered_df.drop(training_set.index)

    # Create training and testing containers
    training_container = project.create.job.TrainingContainer(f"TrainingContainer_{hashed_key}")
    training_container.include_dataset(training_set)
    training_container.run()

    testing_container = project.create.job.TrainingContainer(f"TestingContainer_{hashed_key}")
    testing_container.include_dataset(testing_set)
    testing_container.run()

    # Input for the job
    job_name = f"job_{hashed_key}"
    job = project.create.job.PacemakerJob(job_name)

    job.cutoff = cut_rad
    job.input["potential"]["functions"] = {'ALL': {'nradmax_by_orders': nradmax_by_orders, 'lmax_by_orders': lmax_by_orders}}
    job.input["fit"]['weighting'] = weighting_params_dict

    job.project_hdf5['user/hyperparameters'] = hyperparameters
    job.project_hdf5['user/hashed_key'] = hashed_key
    job.project_hdf5['user/atoms_filter'] = {'atomic_symbols':atomic_symbols,
                                            'cumulative':cumulative_atoms}
                     
    job.add_training_data(training_container)
    job.add_testing_data(testing_container)

    return job

def run_slurm(job, queue_name='cmti', num_cores=20, run_time=3600, run=False):
    job.server.queue = queue_name
    job.server.cores = num_cores
    job.server.run_time = run_time

    if run==True: 
        job.run()

    return job

def hash_hyperparams(hyperparams_dict):

    sorted_params = {k: hyperparams_dict[k] for k in sorted(hyperparams_dict)}

    for key, value in sorted_params.items():
        if isinstance(value, list):
            sorted_params[key] = tuple(value)

    params_json = json.dumps(sorted_params, sort_keys=True)

    params_bytes = params_json.encode('utf-8')
    hash_object = hashlib.sha256(params_bytes)
    hash_key = hash_object.hexdigest()

    return hash_key