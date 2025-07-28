import numpy as np
import spglib
import pandas as pd

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

def get_list_spacegroups():
    sg_list = []
    i = 1
    while True:
        try:
            symbol = spglib.get_spacegroup_type(i).international_short
            sg_list.append({'id': i, 'spacegroup': symbol})
            i += 1
        except Exception:
            break

    df = pd.DataFrame(sg_list)

    return df

def get_element_fractions(atoms, element=None):
    """
    Calculate element-wise fractions from an ASE Atoms object.
    Returns a dict {element: fraction}
    """
    total = len(atoms)

    if element is not None:
        fraction = {element: atoms.symbols.count(element)/total}
        return fraction
    
    elements = set(atoms.get_chemical_symbols())
    fraction = {el: atoms.symbols.count(el)/total for el in elements}
    
    return fraction

def generate_random_binary_structures(
        base_structure,
        main_element='Al',
        mixing_element='Mg',
        phase_type='fcc',
        reference_phase='solid',
        concentrations=[0, 0.5, 1],
        seed=None):
    """
    Adjusts the concentration of mixing_element to desired values by substitution.
    Initial structure may contain both elements already.
    """
    rng = np.random.default_rng(seed)
    n_sites = len(base_structure)
    structures = []
    rows = []

    for target_conc in concentrations:
        atoms = base_structure.copy()
        symbols_orig = np.array(atoms.get_chemical_symbols(), dtype=object)

        n_curr_mix = np.count_nonzero(symbols_orig == mixing_element)
        curr_conc = n_curr_mix/len(symbols_orig)
        # print(f'Current concentration of the mixing element {mixing_element} is {curr_conc}')
        n_target_mix = int(round(target_conc * n_sites))
        delta_n = n_target_mix - n_curr_mix

        main_indices = np.where(symbols_orig == main_element)[0]
        mix_indices = np.where(symbols_orig == mixing_element)[0]

        if delta_n > 0:
            # Need to substitute main_element → mixing_element
            if delta_n > len(main_indices):
                raise ValueError(f"Cannot reach concentration {target_conc}: not enough {main_element} to replace.")
            replace_indices = rng.choice(main_indices, size=delta_n, replace=False)
            symbols_orig[replace_indices] = mixing_element
        elif delta_n < 0:
            # Need to substitute mixing_element → main_element
            delta_n = abs(delta_n)
            if delta_n > len(mix_indices):
                raise ValueError(f"Cannot reach concentration {target_conc}: not enough {mixing_element} to replace.")
            replace_indices = rng.choice(mix_indices, size=delta_n, replace=False)
            symbols_orig[replace_indices] = main_element
        # else, already at target composition.

        atoms.set_chemical_symbols(symbols_orig.tolist())
        structures.append(atoms)
        # Optionally print summary per structure (remove/comment if not needed)
        # print(f"[{target_conc:.2f}] {mixing_element} count: {np.count_nonzero(symbols_orig==mixing_element)}")

        row = {
            'symbol' : atoms.get_chemical_formula(),
            'main_element' : main_element,
            'mixing_element' : mixing_element,
            'fractions' : get_element_fractions(atoms),
            'c' : get_element_fractions(atoms, element=mixing_element)[mixing_element],
            'c_in' : target_conc,
            'atoms' : atoms,
            'phase_type' : phase_type,
            'reference_phase' : reference_phase
        }
        rows.append(row)

    structures_df = pd.DataFrame(rows)

    return structures_df

def phase_list_from_landau(pr, name_tags, phases_list=[], concentrations='All', conc_range=None,  sort_by_conc=True):

    import landau.phases as ldp        

    for idx, row in pr.job_table().iterrows():
        if(row['hamilton']=='Calphy') and all(tag in row['job'] for tag in name_tags):
            temp_job = pr.load(row['job'])
            c = temp_job.project_hdf5['user/input_data']['c']
            c_in = temp_job.project_hdf5['user/input_data']['c_in']
            
            if concentrations == 'All':
                                
                temp_phase = ldp.TemperatureDependentLinePhase(
                    f"AMg_fcc_{c_in}_sol", 
                    fixed_concentration=c, 
                    temperatures=temp_job.output.temperature,
                    free_energies=temp_job.output.energy_free)
                
                phases_list.append(temp_phase)
            else:
                if c_in <=0.27 or c_in in [0.0, 1.0]:

                    print(c, c_in)


    return phases_list
def get_phase_list(pr, name_tags, phases_list=[], concentrations='All', conc_range=None,  sort_by_conc=True):

    import landau.phases as ldp

    for idx, row in pr.job_table().iterrows():
        if(row['hamilton']=='Calphy') and all(tag in row['job'] for tag in name_tags):
            temp_job = pr.load(row['job'])
            j_name = temp_job.name
            c = temp_job.project_hdf5['user/input_data']['c']
            c_in = temp_job.project_hdf5['user/input_data']['c_in']
            temperatures = temp_job.output.temperature
            free_energies = temp_job.output.energy_free
            
            if concentrations == 'All':
                print(f'Adding {j_name}')
                temp_phase = ldp.TemperatureDependentLinePhase(
                    j_name, 
                    fixed_concentration=c, 
                    temperatures=temperatures,
                    free_energies=free_energies)
                
            #     
            # else:
            #     if conc_range is not None:
            #         if not (conc_range[0] <= c_in <= conc_range[1]):
            #             if isinstance(concentrations, list):
            #                 if c_in in concentrations:
            #                     temp_phase = ldp.TemperatureDependentLinePhase(
            #                         f"AMg_fcc_{c_in}_sol", 
            #                         fixed_concentration=c, 
            #                         temperatures=temp_job.output.temperature,
            #                         free_energies=temp_job.output.energy_free)
                                
            #                     phases_list.append(temp_phase)
            #     elif isinstance(concentrations, list):
            #         if c_in in concentrations:
            #             temp_phase = ldp.TemperatureDependentLinePhase(
            #                 f"AMg_fcc_{c_in}_sol", 
            #                 fixed_concentration=c, 
            #                 temperatures=temp_job.output.temperature,
            #                 free_energies=temp_job.output.energy_free)
                        
            #             phases_list.append(temp_phase)

            phases_list.append(temp_phase)

    return phases_list