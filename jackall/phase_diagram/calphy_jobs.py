from os import name
from .. import job_handling
import numpy as np
import matplotlib.pyplot as plt

def submit_calphy_job(new_job_name, project, row, calphy_params, potential, delete_existing_job=False, slurm_params=None):

    from ase.atoms import Atoms
    from pyiron import ase_to_pyiron

    if new_job_name in project.job_table()['job']:
        raise ValueError(f"Job {new_job_name} already exists in the project.")
    
    job = project.create.job.Calphy(new_job_name, delete_existing_job=delete_existing_job)
    job.project_hdf5['user/input_data'] = row.to_dict()

    if isinstance(row['atoms'], Atoms):
        job.structure = ase_to_pyiron(row['atoms'])
    else:
        job.structure = row['atoms']

    job.potential = potential
    job.input.equilibration_control = calphy_params['equilibration_control']
    if calphy_params['melting_cycle'] == False:
        job.input.melting_cycle = False
    
    temperature=calphy_params['temp_range'][row['reference_phase']]
    print(temperature)
    print(type(temperature))

    job.calc_free_energy(
        temperature=temperature,
        pressure=0,
        reference_phase=row['reference_phase'],
        n_equilibration_steps=calphy_params['n_equilibration_steps'],
        n_switching_steps= calphy_params['n_switching_steps'],
        n_print_steps= calphy_params['n_print_steps'],
    )
    job.input.tolerance.solid_fraction = 0
    
    #print(f"Submitting job {job.name} with temperature range: {calphy_params['temp_range'][row['reference_phase']]}K")
    if slurm_params is None:
        job_handling.run_slurm(job, 'cmti', 80, 21600, run=True)
    elif slurm_params is not None:
        job_handling.run_slurm(job, **slurm_params)

def get_current_t_range(project, name_prefixes, reference_phase):
    job_table = project.job_table()
    for prefix in name_prefixes:
        matched_jobs = job_table[job_table['job'].str.startswith(prefix+'_')]
        # print(f"Checking prefix: {prefix}, matched jobs: {len(matched_jobs)}")
        if not matched_jobs.empty:
            temps_array = [
                [int(job_name.split('_')[-2]), int(job_name.split('_')[-1])]
                for job_name in matched_jobs['job']
            ]
            temps_array = np.array(temps_array)
            if reference_phase == 'solid':
                final_temps = np.min(temps_array, axis=0)
            elif reference_phase == 'liquid':
                final_temps = np.max(temps_array, axis=0)
            else:
                raise ValueError(f"Invalid reference_phase: {reference_phase}")
            return final_temps.tolist(), prefix
    # If no prefix matches
    return None, None

# def get_current_t_range(project, name_prefixes, reference_phase):
#     filtered_jobs = project.job_table()[project.job_table()['job'].str.startswith(name_prefix)]
#     if filtered_jobs.empty:
#         return None
#     temps_array = [[int(job_name.split('_')[-2]), int(job_name.split('_')[-1])] for job_name in filtered_jobs['job']]

#     if reference_phase == 'solid':
#         final_temps = np.min(temps_array, axis=0)
#     elif reference_phase == 'liquid':
#         final_temps = np.max(temps_array, axis=0)

#     # if isinstance(final_temps, np.)
#     print(type(final_temps))

#     return final_temps

def check_calphy_phase_transition_criterion(job, reference_phase, e_diff_criterion = [0.01, 0.01]):
    print(f"Checking phase transition criterion for job {job.job_name} with reference phase {reference_phase} and energy difference criterion {e_diff_criterion}")
    if reference_phase == 'solid':
        test_for = np.array(job.output.ts.forward.energy_diff).T
        test_back = np.array(job.output.ts.backward.energy_diff).T
        criterion1 = abs(test_back[0] - max(test_back))
        criterion2 = abs(test_back[-1] - test_for[0])
        print(f"Criterion values: {criterion1}, {criterion2}")
        # if (criterion1[0] > e_diff_criterion[0]) and (criterion2[0] > e_diff_criterion[1]):
        if (criterion2[0] > e_diff_criterion[1]):
            return True

    return False

def concentration_str_variants(value, conc_decimals):
    """
    Format value with the requested decimal digits. 
    Then, produce all representations by stripping trailing decimal zeros, 
    but do not remove nonzero digits. Always return at least the full representation.
    Replace the decimal dot with 'd'.
    """
    base_str = f"{value:.{conc_decimals}f}"
    if '.' in base_str:
        integer_part, decimal_part = base_str.split('.')
        variants = [integer_part + 'd' + decimal_part]
        
        # Strip only trailing zeros but never remove other digits
        trimmed = decimal_part
        while trimmed and trimmed[-1] == '0':
            trimmed = trimmed[:-1]
            if trimmed:  # Only if something remains in decimal_part
                variants.append(integer_part + 'd' + trimmed)
            else:
                variants.append(integer_part)
        return variants
    else:
        # It's an int, so just return as is
        return [base_str.replace('.', 'd')]

def check_and_resubmit_calphy(project, structures_df, potential, calphy_params, slurm_params=None, conc_decimals=4):

    from copy import deepcopy
    calphy_params = deepcopy(calphy_params)

    for idx, struct_row in structures_df.iterrows():
        elements_str = struct_row['main_element'] + struct_row['mixing_element']
        phase_type = struct_row['phase_type']
        reference_phase = struct_row['reference_phase']
        concentration = f"{struct_row['c_in']:.{conc_decimals}f}".replace('.', 'd')
        concentration_variants = concentration_str_variants(struct_row['c_in'], conc_decimals)
        name_prefix = f"{elements_str}_{phase_type}_{reference_phase}_{concentration}"
        name_prefix_variants = [f"{elements_str}_{phase_type}_{reference_phase}_{conc_var}" for conc_var in concentration_variants]

        current_t_range, name_prefix_used = get_current_t_range(project, name_prefix_variants, reference_phase)

        print('\n--------------------------------')
        if current_t_range is not None:
            name_prefix = name_prefix_used
            current_job_name = f"{name_prefix}_{current_t_range[0]}_{current_t_range[1]}"
            # print("Testing job: ", current_job_name, ' ; ', concentration_variants)
            job = project.load(current_job_name)

            if job.status in ['submitted', 'running']:
                print(f"\nJob {current_job_name} is still {job.status}. Skipping...")
                continue
            elif job.status in ['finished', 'aborted']: #FIXME: REMOVE ABORTED LATER
                print(f"\nJob {current_job_name} is {job.status}. Processing...")
                try:
                    transition_occurred = check_calphy_phase_transition_criterion(job, reference_phase, e_diff_criterion=calphy_params['e_diff_criterion'])
                    print(f"Phase transition occurred: {transition_occurred}")
                    job.project_hdf5['user/calphy/temperature_range'] = current_t_range
                    calphy_params['temp_range'][reference_phase] = current_t_range
                    
                    if transition_occurred == False:
                        job.project_hdf5['user/calphy/transition_occurred'] = False
                        print(f"Phase transition did not occur for job {current_job_name}. No resubmission needed.")
                        continue
                    elif transition_occurred == True:
                        job.project_hdf5['user/calphy/transition_occurred'] = True
                    
                        if reference_phase == 'solid':
                            new_t_range = [int(current_t_range[0]), int(current_t_range[1] - calphy_params['temp_adaptive_step'])]
                            print(f"Reference phase is {reference_phase}. Reducing upper temperature limit for solid phase from {current_t_range[1]}K to {new_t_range[1]}K")
                        elif reference_phase == 'liquid':
                            new_t_range = [int(current_t_range[0] + calphy_params['temp_adaptive_step']), int(current_t_range[1])]
                            print(f"Reference phase is {reference_phase}. Increasing lower temperature limit for liquid phase from {current_t_range[0]}K to {new_t_range[0]}K")
                        new_job_name = f"{name_prefix}_{new_t_range[0]}_{new_t_range[1]}"
                        calphy_params['temp_range'][reference_phase] = new_t_range
                        
                        submit_calphy_job(new_job_name, project, struct_row, calphy_params, potential, delete_existing_job=True, slurm_params=slurm_params)
                        print(f"Resubmitting job {new_job_name} with updated temperature range: {new_t_range}K")
        
                except Exception as e:
                    print(f"Error processing job {current_job_name}: {e}")
                    
            elif job.status == 'aborted':
                raise ValueError(f"Job {current_job_name} was aborted. Please check the job output for details.")
            
        else:
            current_t_range = calphy_params['temp_range'][reference_phase]
            current_job_name = f"{name_prefix}_{current_t_range[0]}_{current_t_range[1]}"
            submit_calphy_job(current_job_name, project, struct_row, calphy_params, potential, slurm_params=slurm_params)
            print(f"Submitting job {current_job_name} with temperature range: {current_t_range} for the first time.\n")

def check_calphy_reversible_scaling(project, job_name, fig=None, ax=None, colors_list=None):
    test = project.load(job_name)
    test_for = np.array(test.output.ts.forward.energy_diff).T
    test_back = np.array(test.output.ts.backward.energy_diff).T

    if ax == None:
        fig, ax = plt.subplots()

    if colors_list != None:
        ax.plot(test_for, label='Forward', color=colors_list[0])
        ax.plot(test_back[::-1], label='Backward', color=colors_list[1])
    else:
        ax.plot(test_for, label='Forward')
        ax.plot(test_back[::-1], label='Backward')

    ax.set_xlabel("Scaling Steps")
    ax.set_ylabel("Energy difference [eV/atom]")
    ax.set_title(f"Reversible scaling check for job\n {test.name}")
    
    # plt.legend()
    # plt.show()

    return fig, ax