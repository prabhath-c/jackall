import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .calphy_jobs import get_current_t_range
import landau.phases as ldp
from collections.abc import Iterable

from scipy.constants import Boltzmann, eV
import scipy.special as se

kB = Boltzmann / eV

def S(c):
    return kB * (se.entr(c) + se.entr(1 - c))

def get_phase_change_criterion_table(project, job_name_substr, unique=False):
    criterion_rows = []
    for _, row in project.job_table().iterrows():
        if(row['hamilton']=='Calphy'):
            if(job_name_substr in row['job']):
                test = project.load(row['job'])
                test_for = np.array(test.output.ts.forward.energy_diff).T
                test_back = np.array(test.output.ts.backward.energy_diff).T
                criterion1 = abs(test_back[0] - max(test_back))
                criterion2 = abs(test_back[-1] - test_for[0])
                criterion_rows.append({
                    'job': row['job'],
                    'criterion1': criterion1[0],
                    'criterion2': criterion2[0],
                    'c_in': test.project_hdf5['user/input_data']['c_in'],
                    'temp_high': float(test.name.split("_")[-1]),
                    'temp_low': float(test.name.split("_")[-2])
                })

    criterion_table = pd.DataFrame(criterion_rows)
    criterion_table = criterion_table.sort_values(by='c_in', ascending=True)

    if unique == True:
        criterion_table['c_in_rounded'] = criterion_table['c_in'].round(6)
        sorted = criterion_table.sort_values('temp_high', ascending=True)
        unique = sorted.drop_duplicates(subset='c_in_rounded', keep='first')
        unique = unique.drop(columns='c_in_rounded').reset_index(drop=True).sort_values('c_in', ascending=True)
        return unique

    return criterion_table

def aggregate_phases_with_landau(project, structures_df, verbose=False):

    import landau.phases as ldp

    line_phases_list = []

    for idx, struct_row in structures_df.iterrows():
        elements_str = struct_row['main_element'] + struct_row['mixing_element']
        phase_type = struct_row['phase_type']
        reference_phase = struct_row['reference_phase']
        concentration = f"{struct_row['c_in']:.4f}".replace('.', 'd')
        name_prefix = f"{elements_str}_{phase_type}_{reference_phase}_{concentration}"

        current_t_range = get_current_t_range(project, name_prefix, reference_phase)

        if current_t_range is not None:
            current_job_name = f"{name_prefix}_{current_t_range[0]}_{current_t_range[1]}"
            job = project.load(current_job_name)
            c = job.project_hdf5['user/input_data']['c']

            if job.status in ['submitted', 'running']:
                if verbose == True:
                    print(f"Job {current_job_name} is still {job.status}. Skipping...")
                continue
            elif job.status in ['finished']:
                if verbose == True:
                    print(f"Job {current_job_name} is {job.status}. Processing...")
                try:
                    temp_line_phase = ldp.TemperatureDependentLinePhase(name_prefix, fixed_concentration=c, temperatures=job.output.temperature, free_energies=job.output.energy_free)
                    line_phases_list.append(temp_line_phase)
                    if verbose == True:
                        print(f"Added line phase for {name_prefix} to list.")
                except Exception as e:
                    print(f"Error processing job {current_job_name}: {e}")
                    
            elif job.status == 'aborted':
                raise ValueError(f"Job {current_job_name} was aborted. Please check the job output for details.")
            
        else:
            if verbose == True:
                print(f"No jobs found for {name_prefix}. Skipping...")
    
    return line_phases_list

def get_phase_by_name(phases_list, name_str):
    for p in phases_list:
        if p.name == name_str:
            return p
    raise ValueError(f'Phase "{name_str}" not found. Available: {[q.name for q in phases_list]}')

def get_line_free_energy(
    line_phase,
    T=1000, 
    plot_excess=False,
    phases_list=None,
    parent_phase_name=None,
    end_phases_names_list=None
):
    if plot_excess==False:
        # single interpolated line free_energy
        return line_phase.line_concentration, line_phase.line_free_energy(T)

    elif plot_excess==True:
        p0 = get_phase_by_name(phases_list, end_phases_names_list[0])
        p1 = get_phase_by_name(phases_list, end_phases_names_list[1])
        
        # f0=p0.free_energy(T,0) # double interpolated free_energy
        # f1=p1.free_energy(T,1) # double interpolated free_energy

        # single interpolated line free_energies
        f0 = next((p for p in p0.phases if p.line_concentration == 0), None).free_energy(T,0)
        f1 = next((p for p in p1.phases if p.line_concentration == 1), None).free_energy(T,1)

        c=line_phase.line_concentration
        line_free_energy = line_phase.line_free_energy(T)- ((1-c)*f0 + c*f1)
        
        parent_phase = get_phase_by_name(phases_list=phases_list, name_str=parent_phase_name)
        if parent_phase.add_entropy==True:
            line_free_energy -= T * S(c)
    
        return line_phase.line_concentration, line_free_energy

def plot_diagnostics_fex_vs_c_old(phases_list, end_phases_names_list, ts_range):

    fig, axes = plt.subplots(1, len(ts_range), figsize=(15, 4))

    a_phase = get_phase_by_name(phases_list, end_phases_names_list[0])
    b_phase = get_phase_by_name(phases_list, end_phases_names_list[1])

    for idx, t in enumerate(ts_range):
        for p in phases_list:
            cs = np.linspace(*p.concentration_range, 100)
            if not isinstance(p, ldp.TemperatureDependentLinePhase):
                fa = a_phase.free_energy(t, 0)
                fb = b_phase.free_energy(t, 1)
                axes[idx].plot(cs, p.free_energy(t, cs)-(cs*fb + (1-cs)*fa), label=p.name)
                
            axes[idx].set_title(f"{t}K")
            axes[idx].set_ylabel('Free energy [eV]')
            axes[idx].set_xlabel('Concentration (c)')
            axes[idx].legend()

    plt.tight_layout()
    plt.show()

def plot_diagnostics_fex_vs_c(
    phases_list,
    end_phases_names_list,
    ts_range,
    max_cols=3,
    per_col_in=2.5,
    per_row_in=3,
    sharey=True,
    n_points=200,
    add_legend=True,
    legend_fontsize=8,
    linewidth=3,
    color_override=None,
    specific_phases=None
):
    """
    Plot excess free energy vs concentration for one or more temperatures,
    arranging subplots in a grid with up to `max_cols` columns per row.

    Parameters
    ----------
    phases_list : iterable
        Objects with attributes:
          - name (str)
          - concentration_range (tuple[min_c, max_c])
          - free_energy(T, c) -> array-like
    end_phases_names_list : list[str]
        [name_of_phase_at_c=0, name_of_phase_at_c=1]
    ts_range : float or iterable of floats
        Single temperature or a list/array of temperatures.
    max_cols : int
        Maximum number of subplot columns per row.
    per_col_in, per_row_in : float
        Size (inches) per subplot column/row to keep each axes readable.
    sharey : bool
        Share the y-axis across subplots.
    n_points : int
        Number of concentration points per curve.
    legend_fontsize : int or float
        Legend font size.

    Returns
    -------
    fig, axes : matplotlib Figure and ndarray of Axes
    """
    # Normalize temperatures to a list
    try:
        # Treat numpy scalars as scalars too
        import numpy as _np
        if _np.isscalar(ts_range):
            ts = [float(ts_range)]
        else:
            ts = list(ts_range)
    except Exception:
        ts = [float(ts_range)]
    if len(ts) == 0:
        raise ValueError("ts_range is empty.")

    # Resolve end-member phases (requires get_phase_by_name in scope)
    a_phase = get_phase_by_name(phases_list, end_phases_names_list[0])
    b_phase = get_phase_by_name(phases_list, end_phases_names_list[1])

    # Grid layout: up to max_cols per row
    n = len(ts)
    ncols = min(max_cols, n)
    nrows = int(np.ceil(n / ncols))

    # Figure size so each subplot keeps a good, fixed size
    figsize = (per_col_in * ncols, per_row_in * nrows)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=figsize,
        sharey=sharey,
        squeeze=False,
        constrained_layout=True,
    )
    axes_flat = axes.ravel()

    # Helper to skip temperature-dependent line phases if the class is available
    def _is_tdep_line(p):
        try:
            return isinstance(p, ldp.TemperatureDependentLinePhase)  # ldp must be in scope to be used
        except NameError:
            return False

    for i, T in enumerate(ts):
        ax = axes_flat[i]
        # End-member free energies at c=0 and c=1
        fa = a_phase.free_energy(T, 0.0)
        fb = b_phase.free_energy(T, 1.0)

        # Plot each phase (skip temperature-dependent line phases if applicable)
        for p in phases_list:
            if _is_tdep_line(p):
                continue
            if (specific_phases is not None) and (p not in specific_phases):
                continue
            cmin, cmax = p.concentration_range
            cs = np.linspace(cmin, cmax, n_points)
            fex = p.free_energy(T, cs) - (cs * fb + (1.0 - cs) * fa)
            if color_override is not None:
                ax.plot(cs, fex, color = color_override[p.name], label=p.name, linewidth=linewidth)
            else:
                ax.plot(cs, fex, label=p.name, linewidth=linewidth)

        ax.set_title(f"{T} K")
        ax.set_xlabel("Concentration (c)")
        if sharey:
            if i % ncols == 0:
                ax.set_ylabel("Free energy [eV]")
        else:
            ax.set_ylabel("Free energy [eV]")
        
        if(add_legend == True):
            ax.legend(fontsize=legend_fontsize)

    # Hide any unused axes if grid not completely filled
    for j in range(n, nrows * ncols):
        fig.delaxes(axes_flat[j])

    return fig, axes

def plot_diagnostics_check_conc_interpolation(
    phases,
    temperature,
    samples=50,
    plot_excess=False,
    fig=None,
    axes=None,
    max_cols=3,
    per_col_in=3.5,
    per_row_in=4,
    sharey=False
):
    if axes==None:
        n = len(phases)
        ncols = min(max_cols, n)
        nrows = int(np.ceil(n / ncols))

        figsize = (per_col_in * ncols, per_row_in * nrows)
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=figsize,
            sharey=sharey,
            squeeze=False,
            constrained_layout=True,
        )
        axes = axes.ravel()

    print("The end phases for this check are the end points of each phase, not the overall end points")

    for idx, p in enumerate(phases):
        plt.sca(axes[idx])
        p.check_concentration_interpolation(T=temperature, samples=samples, plot_excess=plot_excess)
        axes[idx].set_title(p.name)
        axes[idx].set_xlabel('Concentration (c)')
        axes[idx].set_ylabel("Free energy [eV]")

    plt.tight_layout()

    return axes

def plot_diagnostics_f_interpolation(
    phases_list,
    T=1000, 
    samples=100, 
    plot_excess=False,
    end_phases_names_list=None,
    fig=None,
    axes=None,
    max_cols=3,
    per_col_in=3.5,
    per_row_in=4,
    sharey=False
):  
    if axes==None:
        n = len(phases_list)
        ncols = min(max_cols, n)
        nrows = int(np.ceil(n / ncols))

        figsize = (per_col_in * ncols, per_row_in * nrows)
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=figsize,
            sharey=sharey,
            squeeze=False,
            constrained_layout=True,
        )
        axes = axes.ravel()

    p0 = get_phase_by_name(phases_list, end_phases_names_list[0])
    p1 = get_phase_by_name(phases_list, end_phases_names_list[1])
    
    # f0=p0.free_energy(T,0)
    # f1=p1.free_energy(T,1)

    f0 = next((p for p in p0.phases if p.line_concentration == 0), None).free_energy(T,0)
    f1 = next((p for p in p1.phases if p.line_concentration == 1), None).free_energy(T,1)

    def _get_errors(y_true, y_pred):
        y_true = np.array(y_true)   
        y_pred = np.array(y_pred)   
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        return mae, rmse
    
    for idx, p in enumerate(phases_list):
        x = np.linspace(*p.concentration_range, samples)

        if plot_excess==True and isinstance(end_phases_names_list, Iterable) and len(end_phases_names_list) == 2: 

            free_energy = p.free_energy(T, x) - ((1-x)*f0 + x*f1)

            axes[idx].plot(x, free_energy, label=p.name)
            
            interpolated_free_energies = []
            line_free_energies = []
            for line in p.phases:
                c = line.line_concentration

                interpolated_free_energies.append(p.free_energy(T, c) - ((1-c)*f0 + c*f1))
            
                line_free_energy = line.line_free_energy(T)- ((1-c)*f0 + c*f1)
                if p.add_entropy==True:
                    line_free_energy -= T * S(c)
                line_free_energies.append(line_free_energy)

                axes[idx].scatter(c, line_free_energy)
                axes[idx].set_ylabel("Excess free energy [eV/atom]")
            
            mae, rmse = _get_errors(line_free_energies, interpolated_free_energies)
            axes[idx].annotate(
                f"RMSE: {rmse:.2e}\nMAE: {mae:.2e}",
                xy=(0.65, 0.95),  # 95% along x and y in axes coordinates
                xycoords='axes fraction',
                fontsize=11,
                ha='right',
                va='top',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
            )

        else:
            free_energy = p.free_energy(T, x)
            axes[idx].plot(x, free_energy, label=p.name)
            
            interpolated_free_energies = []
            line_free_energies = []
            for line in p.phases:
                c = line.line_concentration

                interpolated_free_energies.append(p.free_energy(T, c) - ((1-c)*f0 + c*f1))

                line_free_energy = line.line_free_energy(T) 
                if p.add_entropy==True:
                    line_free_energy -= T * S(c)
                line_free_energies.append(line_free_energy)
                
                axes[idx].scatter(c, line_free_energy)
                axes[idx].set_ylabel("Free energy [eV/atom]")

            mae, rmse = _get_errors(line_free_energies, interpolated_free_energies)
            axes[idx].annotate(
                f"RMSE: {rmse:.2e}\nMAE: {mae:.2e}",
                xy=(0.65, 0.95),  # 95% along x and y in axes coordinates
                xycoords='axes fraction',
                fontsize=11,
                ha='right',
                va='top',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
            )
            
        axes[idx].set_title(p.name)
        axes[idx].set_xlabel('Concentration (c)')

    plt.tight_layout()
    # plt.show()

    return axes