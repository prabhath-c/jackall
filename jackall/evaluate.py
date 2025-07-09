import matplotlib.pyplot as plt
import pandas as pd
from . import job_handling

def get_pareto_front(df, column_headers):
    sorted_df = df.sort_values(by=column_headers[0], ascending=True).reset_index(drop=True)

    pareto_rows = [0]
    current_idx = 0

    while True:
        slope = 0
        found_better = False
        next_idx = None
        for i in range(current_idx + 1, len(sorted_df)):
            if sorted_df.loc[i, column_headers[0]] != sorted_df.loc[pareto_rows[-1], column_headers[0]]:
                temp_slope = ((sorted_df.loc[i, column_headers[1]] - sorted_df.loc[pareto_rows[-1], column_headers[1]]) /
                              (sorted_df.loc[i, column_headers[0]] - sorted_df.loc[pareto_rows[-1], column_headers[0]]))
                if temp_slope < slope:
                    slope = temp_slope
                    next_idx = i
                    found_better = True
        if not found_better:
            break
        pareto_rows.append(next_idx)
        current_idx = next_idx

    pareto_df = sorted_df.loc[pareto_rows].reset_index(drop=True)
    
    return pareto_df

def plot_from_table(df, column_headers, axis_labels=['X', 'Y'], ax=None,
                    marker_color='blue', draw_pareto=False, print_pareto=False):
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
        
    ax.scatter(df[column_headers[0]], df[column_headers[1]], c=marker_color, s=15)
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    
    if draw_pareto==True:
        column_headers.append('hashed_key')
        pareto_df = get_pareto_front(df[column_headers], column_headers)
        ax.plot(pareto_df[column_headers[0]], pareto_df[column_headers[1]], c='red', marker='^', markersize=6, linewidth=1, linestyle='dashed')

        if  print_pareto==True:
            print("Pareto points:")
            print(pareto_df)

        final_pareto_df = pd.merge(
                            pareto_df[['hashed_key']],
                            df,
                            on='hashed_key',
                            how='left'
        )

    if draw_pareto==True:
        return fig, ax, final_pareto_df
    else:
        return fig, ax
    
def setup_lammps_test(pr, structure):

    jobs = []

    for idx, row in pr.job_table().iterrows():
        if(row['hamilton']=='Pacemaker2022' and row['status']=='finished'):

            hashed_key = row['job'].split('_')[1]
            # n_atoms = len(structure)

            temp_job = pr.create.job.Lammps(f"Time_test_{hashed_key}")
            temp_job.structure = structure
            temp_job.potential = pr.load(row['id']).get_lammps_potential()
            temp_job.executable.version = '2024.02.07'
            temp_job.project_hdf5['user/hashed_key'] = hashed_key

            temp_job.calc_md(temperature=800,
                            n_ionic_steps=5000,
                            pressure=0            
            )

            temp_job.input.control

            jobs.append(temp_job)

    jobs_df = pd.DataFrame({'job': jobs})

    return jobs_df