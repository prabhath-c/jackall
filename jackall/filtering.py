import pandas as pd
import matplotlib.pyplot as plt
from pyiron_atomistics import Project
import pyiron_potentialfit
from itertools import combinations

def get_superset(list_symbols, unique=True):
    superset = []
    for r in range(len(list_symbols) + 1):
        for combo in combinations(list_symbols, r):
            if len(list(combo))>0:
                superset.append(list(combo))
    if unique==True:
        seen = set()
        superset = [sublist for sublist in superset if tuple(sublist) not in seen and (seen.add(tuple(sublist)) or True)]
        return superset
    else:
        return superset

def reorder_by_master(subset, master_order):
    return sorted(subset, key=lambda x: master_order.index(x))

def filter_df(df, atomic_symbols, master_order=None, cumulative_atoms = True):
    if master_order!=None:
        atomic_symbols = reorder_by_master(atomic_symbols, master_order)
        # print(atomic_symbols)

    filtered_df = pd.DataFrame()

    if cumulative_atoms==True:
        superset_atomic_symbols = get_superset(atomic_symbols, unique=True)
        for subset_atomic_symbols in superset_atomic_symbols:
            current_df = df[df['atoms'].apply(lambda x: x.get_chemical_symbols() == subset_atomic_symbols)]
            filtered_df = pd.concat([filtered_df, current_df])
    else:
        filtered_df = df[df['atoms'].apply(lambda x: x.get_chemical_symbols() == atomic_symbols)]
    
    return filtered_df