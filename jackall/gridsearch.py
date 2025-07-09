import hashlib
import json
import copy
import pandas as pd
from pyiron_atomistics import Project
import pyiron_potentialfit
from itertools import product

def hash_hyperparams(hyperparams_dict, project_name):

    dict_for_hash = dict(hyperparams_dict)
    for key, value in dict_for_hash.items():
        if isinstance(value, list):
            dict_for_hash[key] = tuple(value)

    params_json = json.dumps(dict_for_hash, sort_keys=True)
    full_str = project_name + params_json
    
    hash_object = hashlib.sha256(full_str.encode('utf-8'))
    hash_key = hash_object.hexdigest()
    
    return hash_key

def generate_hyperparam_combos(pr, hyperparams_dict):
    
    nradmaxs = hyperparams_dict.get('nradmax_by_orders', None)
    lmaxs = hyperparams_dict.get('lmax_by_orders', None)
    
    default_lmax = [0, 3, 2, 1]
    default_nrad = [15, 3, 2, 1]

    if nradmaxs is not None and lmaxs is None:
        print('Using default lmax_by_orders')
        valid_pairs = [(nrad, default_lmax) for nrad in nradmaxs if len(nrad) == len(default_lmax)]
    elif lmaxs is not None and nradmaxs is None:
        print('Using default nradmax_by_orders')
        valid_pairs = [(default_nrad, lmax) for lmax in lmaxs if len(lmax) == len(default_nrad)]
    elif nradmaxs is None and lmaxs is None:
        valid_pairs = [(default_nrad, default_lmax)]
        print('Using default nradmax_by_orders and lmax_by_orders')
    else:  # both present
        valid_pairs = [(nrad, lmax) for nrad in nradmaxs for lmax in lmaxs if len(nrad) == len(lmax)]
    
    other_keys = [k for k in hyperparams_dict if k not in ('nradmax_by_orders', 'lmax_by_orders')]
    other_values = [hyperparams_dict[k] for k in other_keys]

    combos = []
    if not other_keys:
        for nrad, lmax in valid_pairs:
            combo = {'nradmax_by_orders': nrad, 'lmax_by_orders': lmax}
            combo['hashed_key'] = hash_hyperparams(combo, pr.name)
            combos.append(combo)
    else:
        for param_combo in product(*other_values):
            for nrad, lmax in valid_pairs:
                combo = dict(zip(other_keys, param_combo))
                combo['nradmax_by_orders'] = nrad
                combo['lmax_by_orders'] = lmax
                combo['hashed_key'] = hash_hyperparams(combo, pr.name)
                combos.append(combo)
    return pd.DataFrame(combos)

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

def setup_job(project, hyperparams_dict, dataset_df, testing_set, testing_container, atoms_filter):
    
    training_ratio = hyperparams_dict['training_ratio']
    cut_rad = hyperparams_dict['cut_rad']
    nradmax_by_orders = hyperparams_dict['nradmax_by_orders']
    lmax_by_orders = hyperparams_dict['lmax_by_orders']
    hashed_key = hyperparams_dict['hashed_key']

    # Create training set
    training_set_all = dataset_df.drop(testing_set.index)
    training_set= training_set_all.sample(frac=training_ratio, random_state=6919)

    # Create training and testing containers
    training_container = project.create.job.TrainingContainer(f"TrainingContainer_{hashed_key}")
    training_container.include_dataset(training_set)
    training_container.run()

    # Input for the job
    job_name = f"job_{hashed_key}"
    job = project.create.job.PacemakerJob(job_name)

    job.cutoff = cut_rad
    job.input["potential"]["functions"] = {
        'number_of_functions_per_element': 1000,
        'ALL': {'nradmax_by_orders': nradmax_by_orders, 'lmax_by_orders': lmax_by_orders}
        }
    job.input["fit"]['weighting'] = weighting_params_dict
    job.input['backend']['batch_size'] = 1000

    hyperparams_dict_copy = copy.deepcopy(hyperparams_dict)
    hyperparams_dict_copy.pop('hashed_key')
    job.project_hdf5['user/hyperparameters'] = hyperparams_dict_copy
    job.project_hdf5['user/hashed_key'] = hashed_key
    job.project_hdf5['user/atoms_filter'] = atoms_filter

    try:
        job.add_training_data(training_container)
        job.add_testing_data(testing_container)
    except:
        return job

    return job

def setup_gridsearch(project, hyperparams_df, dataset_df, atoms_filter, testing_ratio=0.5):

    # Create testing set
    testing_set = dataset_df.sample(frac=testing_ratio, random_state=6919)
    testing_container = project.create.job.TrainingContainer(f"TestingContainer")
    testing_container.include_dataset(testing_set)
    testing_container.run()

    jobs = []

    for _, row in hyperparams_df.iterrows():
        hyperparams_dict = copy.deepcopy(row.to_dict())
        job = setup_job(project, hyperparams_dict, dataset_df, testing_set, testing_container, atoms_filter)
        jobs.append(job)

    jobs_df = pd.DataFrame({'job': jobs})

    return jobs_df