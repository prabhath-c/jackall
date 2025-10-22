from . import job_handling

def analyse_minimized_structure(ham):
    """

    Args:
        ham (GenericJob):

    Returns:

    """
    final_structure = ham.get_structure(iteration_step=-1)
    diamond_flag = check_diamond(structure=final_structure)
    final_structure_dict = analyse_structure(
        structure=final_structure, mode="total", diamond=diamond_flag
    )
    key_max = max(final_structure_dict.items(), key=operator.itemgetter(1))[0]
    number_of_atoms = len(final_structure)
    distribution_initial = final_structure_dict[key_max] / number_of_atoms
    distribution_initial_half = distribution_initial / 2
    return (
        final_structure,
        key_max,
        number_of_atoms,
        distribution_initial_half,
        final_structure_dict,
    )

def fix_iso(job):
    """
    Add couple xyz to the fix_ensemble inside LAMMPS

    Args:
        job (LAMMPS): Lammps job object

    Returns:
        LAMMPS: Return updated job object
    """
    job.input.control["fix___ensemble"] = (
        job.input.control["fix___ensemble"] + " couple xyz"
    )
    return job

def half_velocity(job, temperature):
    """
    Rather than setting twice the kinetic energy at the beginning of a molecular dynamics simulation reduce the
    velocity to half the initial velocity. This is required to continue MD claculation.

    Args:
        job (LAMMPS): Lammps job object
        temperature (float): Temperature of the molecular dynamics calculation in K

    Returns:
        LAMMPS: Return updated job object
    """
    job.input.control["velocity"] = job.input.control["velocity"].replace(
        str(temperature * 2), str(temperature)
    )
    return job

def next_calc(project, structure, phase_type, concentration, potential, temperature, project_parameter, run_time_steps=10000):
    """
    Calculate NPT ensemble at a given temperature using the job defined in the project parameters:
    - job_type: Type of Simulation code to be used
    - project: Project object used to create the job
    - potential: Interatomic Potential
    - queue (optional): HPC Job queue to be used

    Args:
        structure (pyiron_atomistics.structure.atoms.Atoms): Atomistic Structure object to be set to the job as input sturcture
        temperature (float): Temperature of the Molecular dynamics calculation
        project_parameter (dict): Dictionary with the project parameters
        run_time_steps (int): Number of Molecular dynamics steps

    Returns:
        Final Atomistic Structure object
    """
    # ham_temp = create_job_template(
    #     job_name="temp_heating_" + str(temperature).replace(".", "_"),
    #     structure=structure,
    #     project_parameter=project_parameter,
    # )

    job_name=f"temp_heating_{phase_type}_{concentration}_" + str(temperature).replace(".", "d")
    ham_temp = project.create.job.Lammps(job_name)
    ham_temp.structure = structure
    ham_temp.potential = potential
    ham_temp.executable.version = '2024.02.07'
    ham_temp.project_hdf5['user/concentration'] = concentration
    ham_temp.project_hdf5['user/temperature'] = temperature

    ham_temp.calc_md(
        temperature=temperature,
        temperature_damping_timescale=100.0,
        pressure=0.0,
        pressure_damping_timescale=1000.0,
        n_print=run_time_steps,
        n_ionic_steps=run_time_steps,
        seed=project_parameter["seed"],
    )
    ham_temp = fix_iso(job=ham_temp)
    ham_temp = half_velocity(job=ham_temp, temperature=temperature)
    #ham_temp.run()
    job_handling.run_slurm(ham_temp, 'cmti', 20, 600, para_mode='mpi', mpi_proc=20, run=True)
    ham_temp.project.wait_for_job(ham_temp, interval_in_s=50, max_iterations=100000)
    return ham_temp.get_structure()

def analyse_structure(structure, mode="total", diamond=False):
    """
    Use either common neighbor analysis or the diamond structure detector

    Args:
        structure (pyiron_atomistics.structure.atoms.Atoms): The structure to analyze.
        mode ("total"/"numeric"/"str"): Controls the style and level
            of detail of the output.
            - total : return number of atoms belonging to each structure
            - numeric : return a per atom list of numbers- 0 for unknown,
                1 fcc, 2 hcp, 3 bcc and 4 icosa
            - str : return a per atom string of sructures
        diamond (bool): Flag to either use the diamond structure detector or
            the common neighbor analysis.

    Returns:
        (depends on `mode`)
    """
    if not diamond:
        return structure.analyse.pyscal_cna_adaptive(
            mode=mode, ovito_compatibility=True
        )
    else:
        return structure.analyse.pyscal_diamond_structure(
            mode=mode, ovito_compatibility=True
        )

def next_step_funct(
    number_of_atoms,
    key_max,
    structure_left,
    structure_right,
    temperature_left,
    temperature_right,
    distribution_initial_half,
    structure_after_minimization,
    run_time_steps,
    project_parameter,
):
    """

    Args:
        number_of_atoms:
        key_max:
        structure_left:
        structure_right:
        temperature_left:
        temperature_right:
        distribution_initial_half:
        structure_after_minimization:
        run_time_steps:
        project_parameter:

    Returns:

    """
    structure_left_dict = analyse_structure(
        structure=structure_left,
        mode="total",
        diamond=project_parameter["crystalstructure"].lower() == "diamond",
    )
    structure_right_dict = analyse_structure(
        structure=structure_right,
        mode="total",
        diamond=project_parameter["crystalstructure"].lower() == "diamond",
    )
    temperature_diff = temperature_right - temperature_left
    if (
        structure_left_dict[key_max] / number_of_atoms > distribution_initial_half
        and structure_right_dict[key_max] / number_of_atoms > distribution_initial_half
    ):
        structure_left = structure_right.copy()
        temperature_left = temperature_right
        temperature_right += temperature_diff
        structure_right = next_calc(
            project=project,
            structure=structure_after_minimization,
            phase_type=phase_type,
            concentration=concentration,
            potential=potential,
            temperature=temperature_right,
            project_parameter=project_parameter,
            run_time_steps=run_time_steps,
        )
    elif (
        structure_left_dict[key_max] / number_of_atoms
        > distribution_initial_half
        > structure_right_dict[key_max] / number_of_atoms
    ):
        temperature_diff /= 2
        temperature_left += temperature_diff
        structure_left = next_calc(
            project=project,
            structure=structure_after_minimization,
            phase_type=phase_type,
            concentration=concentration,
            potential=potential,
            temperature=temperature_left,
            project_parameter=project_parameter,
            run_time_steps=run_time_steps,
        )
    elif (
        structure_left_dict[key_max] / number_of_atoms < distribution_initial_half
        and structure_right_dict[key_max] / number_of_atoms < distribution_initial_half
    ):
        temperature_diff /= 2
        temperature_right = temperature_left
        temperature_left -= temperature_diff
        structure_right = structure_left.copy()
        structure_left = next_calc(
            project=project,
            structure=structure_after_minimization,
            phase_type=phase_type,
            concentration=concentration,
            potential=potential,
            temperature=temperature_left,
            project_parameter=project_parameter,
            run_time_steps=run_time_steps,
        )
    else:
        raise ValueError("We should never reach this point!")
    return structure_left, structure_right, temperature_left, temperature_right

def get_initial_melting_temperature_guess(
    project_parameter, ham_minimize_vol, temperature_next=None
):
    """

    Args:
        project_parameter:
        ham_minimize_vol:
        temperature_next:

    Returns:

    """
    (
        structure_after_minimization,
        key_max,
        number_of_atoms,
        distribution_initial_half,
        _,
    ) = analyse_minimized_structure(ham_minimize_vol)
    temperature_left = project_parameter["temperature_left"]
    temperature_right = project_parameter["temperature_right"]
    if temperature_next is None:
        structure_left = structure_after_minimization
        structure_right = next_calc(
            project=project,
            structure=structure_after_minimization,
            phase_type=phase_type,
            concentration=concentration,
            potential=potential,
            temperature=temperature_right,
            project_parameter=project_parameter,
            run_time_steps=project_parameter["strain_run_time_steps"],
        )
        temperature_step = temperature_right - temperature_left
        while temperature_step > 10:
            (
                structure_left,
                structure_right,
                temperature_left,
                temperature_right,
            ) = next_step_funct(
                number_of_atoms=number_of_atoms,
                key_max=key_max,
                structure_left=structure_left,
                structure_right=structure_right,
                temperature_left=temperature_left,
                temperature_right=temperature_right,
                distribution_initial_half=distribution_initial_half,
                structure_after_minimization=structure_after_minimization,
                run_time_steps=project_parameter["strain_run_time_steps"],
                project_parameter=project_parameter,
            )
            temperature_step = temperature_right - temperature_left
        temperature_next = int(round(temperature_left))
        return temperature_next, structure_left
    else:
        return temperature_next, ham_minimize_vol.get_structure()