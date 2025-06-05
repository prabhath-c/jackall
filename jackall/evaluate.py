import pandas as pd
import matplotlib.pyplot as plt
from pyiron_atomistics import Project
import pyiron_potentialfit

def get_pareto_front(x_axis_values, y_axis_values):

    df = pd.DataFrame({'x': x_axis_values.tolist(), 'y': y_axis_values.tolist()})
    df = df.sort_values(by='x', ascending=True)

    current_idx = 0
    lowest_x = df.iloc[current_idx]['x']
    y_lowest_x = df.iloc[current_idx]['y']

    pareto_x = [lowest_x]
    pareto_y = [y_lowest_x]

    while(True):
        slope = 0
        for i in range(current_idx+1, len(df['x'])):
            if(df.iloc[i]['x'] != pareto_x[-1]):
                temp_slope = (df.iloc[i]['y'] - pareto_y[-1])/(df.iloc[i]['x'] - pareto_x[-1])
                if(temp_slope < slope):
                    slope = temp_slope
                    next_x, next_y = df.iloc[i]['x'], df.iloc[i]['y']
                    next_idx = i
        if(slope==0): break

        current_idx = next_idx

        pareto_x.append(next_x)
        pareto_y.append(next_y)

    return pareto_x, pareto_y

def plot_from_table(x_axis_values, y_axis_values, x_axis_label='X', y_axis_label='Y',
                    marker_color='lightgreen', draw_pareto=False, show_figure=True):
    
    fig, ax = plt.subplots()
    ax.scatter(x_axis_values, y_axis_values, c=marker_color, s=10)
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(y_axis_label)
    
    if draw_pareto==True:
        pareto_x, pareto_y = get_pareto_front(x_axis_values, y_axis_values)
        ax.plot(pareto_x, pareto_y, c='red', marker='^', markersize=5, linewidth=0.5)

    if show_figure==True:
        plt.show()
    
    plt.close(fig)

    return fig, ax