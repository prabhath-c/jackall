from . import job_handling
import numpy as np

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
    job.calc_free_energy(
        temperature=calphy_params['temp_range'][row['reference_phase']], 
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

def get_current_t_range(project, name_prefix, reference_phase):
    filtered_jobs = project.job_table()[project.job_table()['job'].str.startswith(name_prefix)]
    if filtered_jobs.empty:
        return None
    temps_array = [[int(job_name.split('_')[-2]), int(job_name.split('_')[-1])] for job_name in filtered_jobs['job']]

    if reference_phase == 'solid':
        final_temps = np.min(temps_array, axis=0)
    elif reference_phase == 'liquid':
        final_temps = np.max(temps_array, axis=0)

    return final_temps

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

def check_and_resubmit_calphy(project, structures_df, potential, calphy_params, slurm_params=None):

    from copy import deepcopy
    calphy_params = deepcopy(calphy_params)

    for idx, struct_row in structures_df.iterrows():
        elements_str = struct_row['main_element'] + struct_row['mixing_element']
        phase_type = struct_row['phase_type']
        reference_phase = struct_row['reference_phase']
        concentration = f"{struct_row['c_in']:.2f}".replace('.', 'd')
        name_prefix = f"{elements_str}_{phase_type}_{reference_phase}_{concentration}"

        current_t_range = get_current_t_range(project, name_prefix, reference_phase)

        if current_t_range is not None:
            current_job_name = f"{name_prefix}_{current_t_range[0]}_{current_t_range[1]}"
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