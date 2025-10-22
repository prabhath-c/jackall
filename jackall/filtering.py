import pandas as pd
import numpy as np
import re
from itertools import combinations_with_replacement

def get_superset(atoms_filter):

    atomic_symbols = atoms_filter['atomic_symbols']
    alloy_order = atoms_filter['alloy_order']
    cumulative = atoms_filter['cumulative']

    superset = []

    if cumulative:
        order_range = range(1, alloy_order + 1)
    else:
        order_range = [alloy_order]
    for r in order_range:
        for combo in combinations_with_replacement(atomic_symbols, r):
            superset.append(list(combo))
    return superset

def filter_df(df, atoms_filter):

    atomic_symbols = atoms_filter['atomic_symbols']
    alloy_order = atoms_filter['alloy_order']

    if alloy_order < len(atomic_symbols):
        raise ValueError(f"alloy_order ({alloy_order}) must be greater than or equal to the "
                         f"number of atomic symbols provided ({len(atomic_symbols)}).")

    superset_atomic_symbols = get_superset(atoms_filter=atoms_filter)

    dfs = []
    for subset_atomic_symbols in superset_atomic_symbols:
        current_df = df[df['atoms'].apply(lambda x: sorted(x.get_chemical_symbols()) == sorted(subset_atomic_symbols))]
        if not current_df.empty:
            dfs.append(current_df)
    filtered_df = pd.concat(dfs) if dfs else pd.DataFrame()

    return filtered_df

def get_total_compute_time(pr, job):
    hashed_key = job.project_hdf5['user/hashed_key']
    lammps_job = pr.load(f"Time_test_{hashed_key}")
    for l in lammps_job.files["log.lammps"].list():
        if "Loop time" in l:
            match = re.search(r"Loop time of ([\d\.eE+-]+) on (\d+) procs for (\d+) steps with (\d+) atoms", l)
            if match:
                wall_time = float(match.group(1))
                n_procs = int(match.group(2))
                n_steps = int(match.group(3))
                n_atoms = int(match.group(4))
                
                if(n_steps > 0):
                    per_atom_step_time = (wall_time * n_procs) / (n_atoms * n_steps)
                    print(f"Wall time per atom per step: {per_atom_step_time:.2e} s/atom/step")
            else:
                print("No match found.")
                    
    return per_atom_step_time or None

def pyiron_table_db_filter(job_name_string=None, **kwargs):
    def filter_func(job_table):
        mask = pd.Series(True, index=job_table.index)

        if job_name_string is not None:
            mask &= job_table['job'].str.contains(job_name_string, case=False, na=False)

        for col, val in kwargs.items():
            if col not in job_table.columns:
                raise KeyError(f"Column '{col}' not found in job_table")
            if isinstance(val, (list, tuple, set)):
                mask &= job_table[col].isin(val)
            else:
                mask &= (job_table[col] == val)

        return mask
    return filter_func

def pyiron_table_add_columns(project, table, column_list):
    for col in column_list:

        # For PacemakerJob jobs
        if col == "training_ratio":
            table.add[col] = lambda job: job.project_hdf5['user/hyperparameters']['training_ratio']
        elif col == "cutoff_radius":
            table.add[col] = lambda job: job.project_hdf5['user/hyperparameters']['cut_rad']
        elif col == "nradmax_by_orders":
            table.add[col] = lambda job: job.project_hdf5['user/hyperparameters']['nradmax_by_orders']
        elif col == "lmax_by_orders":
            table.add[col] = lambda job: job.project_hdf5['user/hyperparameters']['lmax_by_orders']
        elif col == "nfuncs":
            table.add[col] = lambda job: job.project_hdf5['output']['log']['nfuncs'][0]
        elif col == "nfuncs_per_element":
            table.add[col] = lambda job: job.project_hdf5['output']['log']['nfuncs'][0]/len(np.unique(job.project_hdf5['output']['testing_data/element_arrays/symbols']))
        elif col == "loss":
            table.add[col] = lambda job: job.project_hdf5['output/log/loss'][-1]
        elif col == "rmse_epa":
            table.add[col] = lambda job: job.project_hdf5['output/log/rmse_epa'][-1]
        elif col == "rmse_f_comp":
            table.add[col] = lambda job: job.project_hdf5['output/log/rmse_f_comp'][-1]
        elif col == "loss_test":
            table.add[col] = lambda job: job.project_hdf5['output/log_test/loss'][-1]
        elif col == "rmse_epa_test":
            table.add[col] = lambda job: job.project_hdf5['output/log_test/rmse_epa'][-1]
        elif col == "rmse_f_comp_test":
            table.add[col] = lambda job: job.project_hdf5['output/log_test/rmse_f_comp'][-1] 
        elif col == "approx_compute_time":
            table.add[col] = lambda job: job.database_entry.totalcputime * job.project_hdf5['server']['cores']
        elif col == "total_compute_time":
            table.add[col] = lambda job: get_total_compute_time(project, job)
        elif col == "hashed_key":
            table.add[col] = lambda job: job.project_hdf5['user/hashed_key']

        # For Murnaghan jobs
        elif col == "lattice_parameter":
            table.add[col] = lambda job: job.project_hdf5['output/equilibrium_volume'] ** (1/3)
        elif col == "equilibrium_volume":
            table.add[col] = lambda job: job.project_hdf5['output/equilibrium_volume']
        elif col == "equilibrium_bulk_modulus":
            table.add[col] = lambda job: job.project_hdf5['output/equilibrium_bulk_modulus']

        # For Approx. Melting Point Lammps jobs
        elif col == "potential_energy":
            table.add[col] = lambda job: job.project_hdf5['output/generic']['energy_pot'][-1]
        elif col == "temperature":
            table.add[col] = lambda job: job.project_hdf5['output/generic']['temperature'][-1]
        elif col == "volume":
            table.add[col] = lambda job: job.project_hdf5['output/generic']['volume'][-1]
        elif col == "volume_per_atom":
            table.add[col] = lambda job: job.project_hdf5['output/generic']['volume'][-1]/(job.project_hdf5['output/generic']['natoms'][-1])
        elif col == "temperature_in":
            table.add[col] = lambda job: job.project_hdf5['user/temperature']
        elif col == "concentration_in":
            table.add[col] = lambda job: job.project_hdf5['user/concentration']
        else:
            raise ValueError(f"Column {col} is not supported in add_columns_to_table.")

# def slice_job_table(pr,):