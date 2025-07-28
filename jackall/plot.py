from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, NumeralTickFormatter, LinearColorMapper, ColorBar, BasicTicker, Legend
from bokeh.transform import linear_cmap
from . import evaluate
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from scipy.spatial import ConvexHull
import numpy as np

def default_pareto_plot_config_bokeh():
    return {
        "COLUMN_METADATA": {
            "total_compute_time": {
                "label": "Compute Time",
                "unit": "ms / (atom * time step)",
                "scale": 1e3,
                "formatter": "0.00",
            },
            "loss": {
                "label": "Loss (Training)",
                "unit": "m units",
                "scale": 1e3,
                "formatter": "0.00",
            },
            "loss_test": {
                "label": "Loss (Testing)",
                "unit": "m units",
                "scale": 1e3,
                "formatter": "0.00",
            },
                "rmse_epa": {
                "label": "RMSE (Training)",
                "unit": "meV/atom",
                "scale": 1e3,
                "formatter": "0.00",
            },
                "rmse_epa_test": {
                "label": "RMSE (Testing)",
                "unit": "meV/atom",
                "scale": 1e3,
                "formatter": "0.00",
            },
                "nfuncs": {
                "label": "Number of functions",
                "unit": '',
                "formatter": "0.00",
            },
                "nfuncs_per_element": {
                "label": "Number of functions per element",
                "unit": '',
                "formatter": "0.00",
            },
                "cutoff_radius": {
                "label": "Cutoff Radius",
                "unit": "Å",
                "formatter": "0.00",
            },
        },
        "style": {
            "scatter_color": "blue",
            "scatter_size": 8,
            "pareto_color": "red",
            "pareto_marker": "triangle",
            "pareto_size": 8,
            "plot_width": 600,
            "plot_height": 400,
            "font": "Arial",
            "axis_label_text_font_style": "normal",
            "axis_label_text_color": "black",
            "major_label_text_font_style": "normal",
            "major_label_text_color": "black",
            "legend_location": "top_right",
        }
    }

def plot_from_table_bokeh(
    df, 
    x_col, 
    y_col,
    color_col=None,
    cmap="Viridis256",
    draw_pareto=False, 
    print_pareto=False,
    config=None, 
    point_tooltip=None
):
    if config is None:
        config = default_pareto_plot_config_bokeh()
    
    # --- Retrieve config ---
    COLUMN_METADATA = config["COLUMN_METADATA"]
    style = config["style"]

    # --- Metadata lookup and scaling ---
    xmeta = COLUMN_METADATA.get(x_col, {"label": x_col, "unit": "", "scale": 1, "formatter": "0.00"})
    ymeta = COLUMN_METADATA.get(y_col, {"label": y_col, "unit": "", "scale": 1, "formatter": "0.00"})
    plot_df = df.copy().fillna("–")
    plot_df["x_scaled"] = plot_df[x_col] * xmeta["scale"]
    plot_df["y_scaled"] = plot_df[y_col] * ymeta["scale"]
    source = ColumnDataSource(plot_df)

    # --- Labels ---
    xlabel = xmeta["label"] + (f" [{xmeta['unit']}]" if xmeta["unit"] else "")
    ylabel = ymeta["label"] + (f" [{ymeta['unit']}]" if ymeta["unit"] else "")

    # --- Figure ---
    p = figure(
        width=style["plot_width"],
        height=style["plot_height"],
        x_axis_label=xlabel,
        y_axis_label=ylabel,
        tools="reset,wheel_zoom,box_zoom,pan,tap,save"
    )

    # --- Axis and tick styles ---
    p.xaxis.axis_label_text_font_style = style["axis_label_text_font_style"]
    p.xaxis.axis_label_text_color = style["axis_label_text_color"]
    p.yaxis.axis_label_text_font_style = style["axis_label_text_font_style"]
    p.yaxis.axis_label_text_color = style["axis_label_text_color"]
    p.xaxis.major_label_text_font_style = style["major_label_text_font_style"]
    p.xaxis.major_label_text_color = style["major_label_text_color"]
    p.yaxis.major_label_text_font_style = style["major_label_text_font_style"]
    p.yaxis.major_label_text_color = style["major_label_text_color"]

    # ------ Main data points ------
    if color_col is not None and color_col in plot_df.columns:
        # Create color mapper
        mapper = LinearColorMapper(palette=cmap, 
                                   low=plot_df[color_col].min(), 
                                   high=plot_df[color_col].max())
        color = {'field': color_col, 'transform': mapper}
        # Scatter with color mapping
        renderer = p.scatter(
            'x_scaled', 'y_scaled',
            source=source, size=style["scatter_size"],
            color=color, alpha=0.7, legend_label="Data"
        )
        # Add color bar
        if color_col in COLUMN_METADATA:
            colorbar_label = COLUMN_METADATA[color_col]["label"]
            colorbar_unit = COLUMN_METADATA[color_col]["unit"]
            colorbar_title = colorbar_label + (f" [{colorbar_unit}]" if colorbar_unit else "")
        else:
            colorbar_title = color_col
        color_bar = ColorBar(color_mapper=mapper, 
                             label_standoff=12,
                             ticker=BasicTicker(), 
                             location=(0,0),
                             title=colorbar_title)
        color_bar.title_text_font_style = style["major_label_text_font_style"]
        color_bar.title_text_color = style["axis_label_text_color"]

        color_bar.formatter = NumeralTickFormatter(format="0.00")
        p.add_layout(color_bar, 'right')
    else:
        # Single color fallback
        renderer = p.scatter(
            'x_scaled', 'y_scaled',
            source=source, size=style["scatter_size"], 
            color=style["scatter_color"], alpha=0.7, legend_label="Data"
        )
    legend_items = [("Data", [renderer])]

    # --- Tooltips (all columns) ---
    if point_tooltip is None:
        point_tooltip = [(col, f"@{col}") for col in df.columns]
    p.add_tools(HoverTool(renderers=[renderer], tooltips=point_tooltip))

    # --- Tick formatting ---
    p.xaxis.formatter = NumeralTickFormatter(format=xmeta["formatter"])
    p.yaxis.formatter = NumeralTickFormatter(format=ymeta["formatter"])

    # --- Pareto (if needed) ---
    final_pareto_df = None
    if draw_pareto:
        if 'hashed_key' not in plot_df:
            plot_df['hashed_key'] = plot_df.index
        pareto_df = evaluate.get_pareto_front(
            plot_df[['x_scaled', 'y_scaled', 'hashed_key']], 
            ['x_scaled', 'y_scaled', 'hashed_key']
        )
        pareto_source = ColumnDataSource(pareto_df)
        pareto_line = p.line('x_scaled', 'y_scaled', source=pareto_source, 
               line_color=style["pareto_color"], line_dash="dashed", 
               line_width=1.5, legend_label="Pareto")
        pareto_pts = p.scatter('x_scaled', 'y_scaled', source=pareto_source, color=style["pareto_color"], 
                  marker=style["pareto_marker"], size=style["pareto_size"])
        legend_items.append(("Pareto", [pareto_line, pareto_pts]))

        final_pareto_df = pd.merge(
        pareto_df[['hashed_key']],  # Only keep Pareto points, original order
        df,                         # All columns
        on='hashed_key',
        how='left'
        )

        if print_pareto:
            print("Pareto points:")
            print(pareto_df)

    legend = Legend(items=legend_items)
    p.add_layout(legend)
    p.legend.location = style["legend_location"]
    p.legend.click_policy = "hide"

    return p, final_pareto_df

def plot_mixing_energy_by_spacegroup(df, n_max=None, ax=None, plot_hull=True):
    """
    Plot mixing energy vs. composition, colored by spacegroup.
    Optionally overlay convex hull.
    """
    # Prepare color map and which spacegroups to plot
    spacegroup_counts = df['spacegroup'].value_counts()
    if n_max is not None:
        plot_sgs = spacegroup_counts.head(n_max).index.tolist()
        cmap = get_cmap('tab10', n_max if n_max > 1 else 2)
    else:
        plot_sgs = spacegroup_counts.index.tolist()
        cmap = get_cmap('tab20', len(plot_sgs) if len(plot_sgs) > 1 else 2)
    
    # Create axis if not given
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each spacegroup
    for i, sg in enumerate(plot_sgs):
        group = df[df['spacegroup'] == sg]
        # ax.plot(group['x'], group['mixing_energy'], '-', color=cmap(i), label=f'SG {sg}')
        ax.scatter(group['x'], group['mixing_energy'], color=cmap(i))
    
    ax.set_xlabel('Composition (x) of Mg')
    ax.set_ylabel('Mixing energy [eV/atom]')
    # ax.legend(title='Space group', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title(f'Mixing Energy vs Mg Composition, by spacegroup ({n_max if n_max is not None else "all"})')
    plt.tight_layout()
    
    # -------- Convex Hull Overlay --------
    if plot_hull:
        points = df[['x', 'mixing_energy']].dropna().values
        if len(points) >= 3:
            try:
                hull = ConvexHull(points)
                all_hull_vertices = hull.vertices
                all_hull_points = points[all_hull_vertices]
                lower_hull_mask = all_hull_points[:, 1] <= 0
                lower_hull_points = all_hull_points[lower_hull_mask]
                ax.plot(lower_hull_points[:, 0], lower_hull_points[:, 1], '--', color='black', lw=2, label='Convex Hull')
            
                # --- Key section for you ---
                lower_hull_df = pd.DataFrame(lower_hull_points, columns=['x', 'mixing_energy'])

                df_reset = df.reset_index()
                df_on_hull = pd.merge(df_reset, lower_hull_df, on=['x', 'mixing_energy'], how='inner')
                df_on_hull = df_on_hull.set_index('index')

            except Exception as e:
                print("Convex hull could not be computed:", e)
                df_on_hull = pd.DataFrame()
        else:
            print("Not enough points to compute convex hull.")
            df_on_hull = pd.DataFrame()

    if ax is None:
        plt.show()
    return ax, df_on_hull