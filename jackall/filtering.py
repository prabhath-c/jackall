import pandas as pd
from itertools import combinations_with_replacement

def get_superset(list_symbols, alloy_order=2, cumulative=True):
    superset = []
    if cumulative:
        order_range = range(1, alloy_order + 1)
    else:
        order_range = [alloy_order]
    for r in order_range:
        for combo in combinations_with_replacement(list_symbols, r):
            superset.append(list(combo))
    return superset

def filter_df(df, atomic_symbols, alloy_order, cumulative=True):
    if alloy_order < len(atomic_symbols):
        raise ValueError(f"alloy_order ({alloy_order}) must be greater than or equal to the "
                         f"number of atomic symbols provided ({len(atomic_symbols)}).")

    superset_atomic_symbols = get_superset(atomic_symbols, alloy_order=alloy_order, cumulative=cumulative)

    dfs = []
    for subset_atomic_symbols in superset_atomic_symbols:
        current_df = df[df['atoms'].apply(lambda x: x.get_chemical_symbols() == subset_atomic_symbols)]
        if not current_df.empty:
            dfs.append(current_df)
    filtered_df = pd.concat(dfs) if dfs else pd.DataFrame()

    return filtered_df

def pyiron_table_db_filter(status="finished", hamilton="Pacemaker2022"):
    return lambda j: (j.status == status) & (j.hamilton == hamilton)

def pyiron_table_add_columns(table, column_list):
    for col in column_list:
        if col == "training_ratio":
            table.add[col] = lambda job: job.project_hdf5['user/hyperparameters']['training_ratio']
        elif col == "cutoff_radius":
            table.add[col] = lambda job: job.project_hdf5['user/hyperparameters']['cut_rad']
        elif col == "nradmax_by_orders":
            table.add[col] = lambda job: job.project_hdf5['user/hyperparameters']['nradmax_by_orders']
        elif col == "lmax_by_orders":
            table.add[col] = lambda job: job.project_hdf5['user/hyperparameters']['lmax_by_orders']
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
        elif col == "compute_time":
            table.add[col] = lambda job: job.database_entry.totalcputime * job.project_hdf5['server']['cores']
        elif col == "hashed_key":
            table.add[col] = lambda job: job.project_hdf5['user/hashed_key']
        else:
            raise ValueError(f"Column {col} is not supported in add_columns_to_table.")
