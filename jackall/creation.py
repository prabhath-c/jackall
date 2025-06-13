import numpy as np

def create_multicomponent_bulk(project, symbols, concentrations, structure, supercell=(4,4,4),
                               ordered=False, lattice_constant=None, seed=None):
    assert len(symbols) == len(concentrations), "Symbols and concentrations must match in length."
    assert np.isclose(sum(concentrations), 1.0), "Concentrations must sum to 1.0"

    a = lattice_constant
    atoms = project.create.structure.bulk(symbols[0], structure, a=a).repeat(supercell)
    n_sites = len(atoms)
    counts = [int(round(c * n_sites)) for c in concentrations]
    diff = n_sites - sum(counts)
    counts[0] += diff   # fix any rounding problems

    atom_list = []
    # ORDERED assignment: round-robin
    if ordered:
        idxs = []
        for i, count in enumerate(counts):
            idxs += [i]*count
        arr = np.zeros(n_sites, dtype=int)
        arr[:len(idxs)] = idxs
        arr = np.array(arr[:n_sites])
        # Sort such that species repeat in a cycle (round-robin)
        modded = np.arange(n_sites) % len(symbols)
        flat_counts = [sum(np.array(idxs)==i) for i in range(len(symbols))]
        species_map = []
        k = 0
        species_ctr = [0]*len(symbols)
        for i in range(n_sites):
            j = modded[i]
            while species_ctr[j] >= flat_counts[j]:
                j = (j+1) % len(symbols)
            species_map.append(j)
            species_ctr[j] += 1
        atom_list = [symbols[j] for j in species_map]
    else:
        # UNORDERED: random 
        for s, count in zip(symbols, counts):
            atom_list.extend([s]*count)
        np.random.seed(seed)
        np.random.shuffle(atom_list)
    atoms.set_chemical_symbols(atom_list)
    return atoms