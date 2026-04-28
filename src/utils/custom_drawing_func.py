import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .drawing_utils import create_directory_and_save_plot

# Description: Custom drawing functions for the PE6 project.


def axis_formatter(x, y, ax=None, **kws):
    from matplotlib.ticker import MaxNLocator

    ax = plt.gca()
    # Set the same number of ticks on both axes
    # ax.set_aspect("equal")

    ax.set_xlim([0, kws["max_lim"]])
    ax.set_ylim([0, kws["max_lim"]])

    ax.xaxis.set_major_locator(MaxNLocator(kws["max_n_locator"]))
    ax.yaxis.set_major_locator(MaxNLocator(kws["max_n_locator"]))


def corrfunc(x, y, ax=None, **kws):
    """Plot the correlation coefficient in the top left hand corner of a plot.

    Parameters:
    - x: Array-like object representing the x-values.
    - y: Array-like object representing the y-values.
    - ax: Optional matplotlib Axes object to plot on. If not provided, the current Axes instance is used.
    - **kws: Additional keyword arguments to be passed to the annotate function.

    Returns:
    None
    """
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr, spearmanr

    # Dropna for each set of x and y
    # https://stackoverflow.com/a/62787831/12327096
    indices = np.logical_not(np.logical_or(np.isnan(x), np.isnan(y)))
    indices = np.array(indices)
    x, y = x[indices], y[indices]

    r, _ = pearsonr(x, y)
    r_s, _ = spearmanr(x, y)
    ax = ax or plt.gca()
    ax.annotate(f"r = {r:.2f}", xy=(0.1, 0.95), xycoords=ax.transAxes)
    ax.annotate(f"R = {r_s:.2f}", xy=(0.1, 0.9), xycoords=ax.transAxes)
    ax.annotate(f"N = {x.shape[0]:d}", xy=(0.1, 0.85), xycoords=ax.transAxes)


def linregfuc(x, y, ax=None, **kws):
    """Plot the linear regression results in the top left hand corner of a plot."""
    import matplotlib.pyplot as plt
    import statsmodels.api as sm

    model = sm.OLS(y, x)
    result = model.fit()
    slope = result.params[0].round(2)

    ax = ax or plt.gca()
    ax.annotate(
        f"y = {slope:.2f}x",
        xy=(0.1, 0.95),
        xycoords=ax.transAxes,
    )

    return result


def draw_pairwise_scatterplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    root_path: str = "./",
    name: str = "scatterplot",
    max_lim=None,
    max_n_locator=None,
    title=None,
    figsize_width=6,
    figsize_height=6,
    sci_ticks=False,
):
    """Draw a pairwise scatterplot using the given DataFrame and column names.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        x (str): The name of the column to be plotted on the x-axis.
        y (str): The name of the column to be plotted on the y-axis.
        root_path (str, optional): The root path where the plot will be saved. Defaults to "./".
        name (str, optional): The name of the plot file. Defaults to "scatterplot".
        max_n_locator (int, optional): The maximum number of tick locators for the axes. Defaults to 8.

    Returns:
        None
    """

    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import AutoLocator, MaxNLocator

    plt.rcParams.update(
        {
            "svg.fonttype": "none",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
        }
    )
    sns.set_theme(style="ticks", font_scale=1.5)
    fig, ax = plt.subplots(figsize=(figsize_width, figsize_height))

    sns.despine(top=True, right=True)

    if x == y:  # Draw 1D histogram
        data = df[[x]].dropna()
        ax = sns.histplot(data=data, x=x, ax=ax)
    else:  # Normal case: draw scatterplot
        data = df[[x, y]].dropna()
        ax = sns.scatterplot(data=data, x=x, y=y, alpha=0.5, ax=ax, s=50, rasterized=True)

        corrfunc(df[[x, y]].dropna()[x], df[[x, y]].dropna()[y], ax=ax)
        # result = linregfuc(df[[x, y]].dropna()[x], df[[x, y]].dropna()[y], ax=ax)

        # ax.plot(df[[x, y]].dropna()[x].values, result.fittedvalues, color="black")
        ax.set_aspect("equal", "box")

        # Get the current limits of the axes
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Set the limits of the axes to be the maximum of the current limits

        if max_lim:
            ax.set_xlim([0, max_lim])
            ax.set_ylim([0, max_lim])
        else:
            ax.set_xlim([0, max(xlim[1], ylim[1])])
            ax.set_ylim([0, max(xlim[1], ylim[1])])

        # Set the same number of ticks on both axes
        ax.axes.set_aspect("equal", "box")
        if max_n_locator:
            ax.xaxis.set_major_locator(MaxNLocator(max_n_locator))
            ax.yaxis.set_major_locator(MaxNLocator(max_n_locator))
        else:
            ax.xaxis.set_major_locator(AutoLocator())
            ax.yaxis.set_major_locator(AutoLocator())

        # Format tick labels
        if sci_ticks:
            ax.axes.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))

    if title:
        ax.set_title(title)

    plt.tight_layout()

    create_directory_and_save_plot(root_path, name, fig)
    plt.show()
    plt.close(fig)


def draw_pairplot(df: pd.DataFrame, max_lim=70, max_n_locator=7, root_path="./", name="pairplot"):
    plt.rcParams.update(
        {
            "svg.fonttype": "none",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
        }
    )
    sns.set_theme(style="ticks", font_scale=1.5)
    unused_fig = plt.figure(figsize=(20, 20))

    g = sns.pairplot(
        df,
        height=5,
        diag_kind="kde",
        plot_kws={"alpha": 0.5, "rasterized": True},
    )
    g = g.map(axis_formatter, max_lim=max_lim, max_n_locator=max_n_locator)
    g = g.map_lower(corrfunc)

    plt.close(unused_fig)
    create_directory_and_save_plot(root_path, name, g.figure)
    plt.close(g.figure)


def _draw_general_activity_boxplot(box_df, names, ax, e=0.01, title=None):
    """Draw a boxplot of PE efficiency for different variants.

    Parameters:
    - box_df (pandas.DataFrame): DataFrame containing the data for the boxplot.
    - names (list): List of names of the variants. It will be displayed as the x-tick labels.
    - ax (matplotlib.axes.Axes): The matplotlib Axes object to draw the boxplot on.
    - e (float): Small constant added to the "pe efficiency" column to avoid log(0).

    Returns:
    - ax (matplotlib.axes.Axes): The matplotlib Axes object containing the boxplot.

    Example usage:
    >>> df = pd.DataFrame({'variants': ['A', 'B', 'C'], 'pe efficiency': [0.1, 0.2, 0.3]})
    >>> ax = draw_general_activity_boxplot(df)
    >>> plt.show()
    """
    import matplotlib.pyplot as plt
    import matplotlib.scale as scale
    import seaborn as sns
    from matplotlib.ticker import FormatStrFormatter

    from .drawing_variables import pe6_colormap_dict

    plt.rcParams.update(
        {
            "svg.fonttype": "none",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
        }
    )

    # Add small constant to avoid log(0)
    box_df["pe efficiency"] = box_df["pe efficiency"] + e

    # Create the boxplot
    ax = sns.boxplot(
        x="variants",
        y="pe efficiency",
        data=box_df,
        ax=ax,
        palette=pe6_colormap_dict(),
        hue="variants",
        orient="v",
        whis=(10, 90),
        fliersize=0,
    )

    # https://github.com/mwaskom/seaborn/issues/3148
    for line in ax.lines:
        if line.get_linestyle() == "None":
            ax = sns.swarmplot(
                x=[ax.get_xticklabels()[x].get_text() for x in map(int, line.get_xdata())],
                y=line.get_ydata(),
                size=0.4,
                color="#000000",
                ax=ax,
            )
            line.remove()

    # Rename the x ticks labels
    plt.xticks(rotation=90)

    ##
    ax.set_xticklabels(names)

    if title is not None:
        ax.set_title(title)

    plt.tight_layout()

    plt.xlabel("")
    plt.ylabel("Prime editing efficiency (%)")

    # Set the yscale as log
    plt.yscale(scale.LogScale(axis=ax, base=10))
    plt.ylim((0, 1e2))

    # ScalarFormatter is used to set the y-axis in numeric format
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    return ax


def draw_general_activity_ecdfplot(ecdf_df, id, output_path, log_scale=False):
    """Draw an ECDF plot of PE efficiency for different variants.

    Parameters:
    - ecdf_df (pandas.DataFrame): DataFrame containing the data for the ECDF plot.
    - id (str): Identifier for the plot. It will be used to name the output file.
    - output_path (str): Path to the directory where the output file will be saved.
    - log_scale (bool, optional): Whether to use a logarithmic scale for the y-axis. Default is False.

    Returns:
    - int: Always returns 0.

    Example usage:
    >>> df = pd.DataFrame({'variants': ['A', 'B', 'C'], 'pe efficiency': [0.1, 0.2, 0.3]})
    >>> ax = draw_general_activity_ecdfplot(df, "plot1", "/path/to/output")
    >>> plt.show()
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    from .drawing_variables import pe6_colormap_dict
    from .utils import rename_map_for_visualization

    plt.rcParams.update(
        {
            "svg.fonttype": "none",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
        }
    )

    # Rename columns and melt the DataFrame
    ecdf_df = ecdf_df.rename(columns=(rename_map_for_visualization())).melt(
        var_name="variants", value_name="pe efficiency"
    )

    # Set seaborn theme and font scale
    sns.set_theme(style="ticks", font_scale=2.0)

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(18, 6))

    # Create the ECDF plot
    ax = sns.ecdfplot(
        data=ecdf_df,
        y="pe efficiency",
        hue="variants",
        palette=pe6_colormap_dict(),
        log_scale=log_scale,
        ax=ax,
    )

    # Move the legend to the upper left corner
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    # Adjust the layout
    plt.tight_layout()

    # Set the x and y axis labels
    plt.xlabel("Cumulative probability")
    plt.ylabel("Prime editing efficiency (%)")

    # Create the output directory if it doesn't exist and save the plot
    create_directory_and_save_plot(output_path, f"{id}_ecdfplot", fig)
    plt.close(fig)

    return 0


def correlation_heatmap(
    df,
    width=20,
    height=20,
    title="Correlation matrix",
    root_path="./",
    name="correlation_heatmap",
):
    plt.rcParams.update(
        {
            "svg.fonttype": "none",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
        }
    )
    sns.set_theme(style="ticks", font_scale=2.0)
    fig, ax = plt.subplots(figsize=(width, height))
    ax = sns.heatmap(
        df,
        cmap="Blues",
        vmin=df.values.min(),
        annot=True,
        fmt=".2f",
        square=True,
        cbar_kws={"shrink": 0.6},
        annot_kws={"size": 18},
        ax=ax,
    )
    plt.title(title)
    ax.tick_params(
        axis="both",
        top=True,
        bottom=True,
        left=True,
        labeltop=True,
        labelbottom=True,
        labelleft=True,
        which="major",
        labelsize=20,
    )
    for label in ax.get_xticklabels():
        label.set_rotation(90)
    for label in ax.get_yticklabels():
        label.set_rotation(0)
    plt.tight_layout()

    create_directory_and_save_plot(root_path, name, fig)
    plt.close(fig)


def general_activity_analysis_pipeline(
    df: pd.DataFrame, features: list, names: list, id: str, output_path: str
):
    """Perform activity analysis pipeline on the given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        features (list): A list of column names to be used for analysis. (Deprecated)
        names (list): A list of names corresponding to the features.
        id (str): The identifier for the analysis.
        output_path (str): The path to save the output files.

    Returns:
        None
    """
    df = df.dropna()  # For pair comparison, drop rows with NaN values
    # Save the raw data
    df.to_csv(f"{output_path}/{id}_descriptive_data.csv", index=False)

    box_distribution_for_activity(df, names, id, output_path)


def box_distribution_for_activity(
    df, names, id, output_path, regex_col_filter=r"REF_ID|comb_vis\+"
):
    """Generate a boxplot to visualize the distribution of PE efficiency for different variants of
    an activity.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame containing the data.
    - names (list): A list of names corresponding to the variants in the DataFrame.
    - id (str): The ID of the activity.
    - output_path (str): The path to save the generated plot.

    Returns:
    None
    """

    from .utils import rename_map_for_visualization

    sns.set_theme(style="ticks", font_scale=2.0)
    fig, ax = plt.subplots(figsize=(6, 12))
    box_df = (
        df.filter(regex=regex_col_filter)
        .rename(columns=rename_map_for_visualization())
        .melt(id_vars=["REF_ID"], var_name=["variants"], value_name="pe efficiency")
    )

    ax = _draw_general_activity_boxplot(box_df, names=names, ax=ax, e=0.01, title=id)

    create_directory_and_save_plot(output_path, f"{id}_boxplot", fig)
    plt.close(fig)

    # Save the raw data
    box_df.pivot(columns="variants", index="REF_ID", values="pe efficiency").to_csv(
        f"{output_path}/{id}_data.csv", index=True
    )


def box_distribution_for_peg_properties(df, features, id, output_path):
    """Generate a boxplot to visualize the distribution of specified features in a DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    features (list): A list of column names representing the features to be plotted.
    id (str): An identifier used for naming the output plot file.
    output_path (str): The path to the directory where the plot will be saved.

    Returns:
    None
    """
    plt.rcParams.update(
        {
            "svg.fonttype": "none",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
        }
    )
    sns.set_theme(style="ticks", font_scale=2.0)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax = sns.boxplot(df[features], ax=ax)
    ax.annotate(f"N={len(df)}", xy=(0.1, 0.9), xycoords="axes fraction", ha="center", va="center")

    create_directory_and_save_plot(output_path, f"{id}_descriptive_boxplot", fig)
    plt.close(fig)

    # Save the raw data
    df.to_csv(f"{output_path}/{id}_descriptive_data.csv", index=False)


def generate_rep_trend_barplot(df, output_path, name):
    """Generate a representative trend plot based on the given DataFrame.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame containing the data.
    - output_path (str): The path to save the generated plot.
    - name (str): The name of the plot.

    Returns:
    None
    """

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    from .utils import create_directory_and_save_plot, rename_map_for_visualization

    indiv_data = df.filter(like="indiv_vis")
    kmc_data = indiv_data.filter(like="KMC")
    uar_data = indiv_data.filter(like="UAR")

    kmc_mean_data = (
        kmc_data.rename(columns=rename_map_for_visualization())
        .rename(columns=lambda x: x.replace("-KMC", ""))
        .mean()
    )
    uar_mean_data = (
        uar_data.rename(columns=rename_map_for_visualization())
        .rename(columns=lambda x: x.replace("-UAR", ""))
        .mean()
    )

    rep_trend_data = pd.DataFrame(
        [kmc_mean_data, uar_mean_data], index=["KMC", "UAR"]
    ).reset_index()
    rep_trend_data = rep_trend_data.melt(
        id_vars="index", var_name="variants", value_name="Mean PE efficiency (%)"
    )

    plt.rcParams.update(
        {
            "svg.fonttype": "none",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
        }
    )

    sns.set_theme(style="ticks", font_scale=2.0)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax = sns.stripplot(
        x="variants",
        y="Mean PE efficiency (%)",
        data=rep_trend_data,
        ax=ax,
        hue="index",
        s=10,
    )

    ax = sns.barplot(
        x="variants",
        y="Mean PE efficiency (%)",
        data=rep_trend_data.drop(columns="index").groupby("variants").mean().reset_index(),
        ax=ax,
    )

    # Rename the x ticks labels
    plt.xticks(rotation=90)
    plt.xlabel("")
    # plt.tight_layout()

    create_directory_and_save_plot(output_path, name, fig)
    plt.close(fig)


def _draw_general_bar_plot(df, x, y, hue, output_path, name):
    """Draw a bar plot using the provided DataFrame and save the plot and raw data.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the data to be plotted.
    - x (str): The column name of the x-axis variable in the DataFrame.
    - y (str): The column name of the y-axis variable in the DataFrame.
    - hue (str): The column name of the variable used for grouping and coloring the bars.
    - output_path (str): The path to the directory where the plot and raw data will be saved.
    - name (str): The name of the plot and raw data files.

    Returns:
    None
    """
    from .drawing_variables import pe6_colormap_dict

    plt.rcParams.update(
        {
            "svg.fonttype": "none",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
        }
    )
    sns.set_theme(style="ticks", font_scale=1.5)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax = sns.barplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        palette=pe6_colormap_dict(),
        ax=ax,
        errorbar=("ci", 95),
    )

    # Move lengend to the upper right corner outside the plot
    ax.legend(loc="upper right", bbox_to_anchor=(1.5, 1), title="Variants")

    create_directory_and_save_plot(output_path, name, fig)
    df.to_csv(f"{output_path}/{name}_data.csv", index=False)  # save the raw data
    plt.close(fig)


def draw_factor_analysis_barplot(df, factor, output_path, name):
    """Draw the edit type figures using the given DataFrame.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the data to be plotted.
    - output_path (str): The path to the directory where the plot and raw data will be saved.

    Returns:
    None
    """
    from .utils import rename_map_for_visualization, rename_peg_3term_motifs

    if factor == "motif":
        df[factor] = df[factor].apply(lambda x: rename_peg_3term_motifs(x))

    # Data manipulation for the edit type analysis
    df_factor = pd.concat([df[factor], df.filter(like="comb")], axis=1)
    df_factor = df_factor.rename(columns=rename_map_for_visualization())
    df_factor = df_factor.melt(
        id_vars=factor, var_name="variants", value_name="Editing efficiency (%)"
    )

    # Draw the line plot for the edit length analysis
    _draw_general_bar_plot(
        df_factor, factor, "Editing efficiency (%)", "variants", output_path, name
    )


def _draw_general_lineplot(df, x, y, hue, output_path, name):
    """Draw a line plot using the provided DataFrame and save the plot and raw data.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the data to be plotted.
    - x (str): The column name of the x-axis variable in the DataFrame.
    - y (str): The column name of the y-axis variable in the DataFrame.
    - hue (str): The column name of the variable used for grouping and coloring the lines.
    - output_path (str): The path to the directory where the plot and raw data will be saved.
    - name (str): The name of the plot and raw data files.

    Returns:
    None
    """
    from .drawing_variables import pe6_colormap_dict
    from .utils import rename_map_for_visualization

    plt.rcParams.update(
        {
            "svg.fonttype": "none",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
        }
    )

    df = df.rename(columns=rename_map_for_visualization())

    sns.set_theme(style="ticks", font_scale=1.5)
    fig, ax = plt.subplots(figsize=(12, 12))

    ax = sns.lineplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        palette=pe6_colormap_dict(),
        err_style="bars",
        errorbar=("ci", 95),
        ax=ax,
    )

    create_directory_and_save_plot(output_path, name, fig)
    df.to_csv(f"{output_path}/{name}_data.csv", index=False)  # save the raw data
    plt.close(fig)


def draw_factor_analysis_lineplot(df, factor, output_path, name):
    """Draw a line plot for factor analysis.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame containing the data.
    - factor (str): The column name of the factor to analyze.
    - output_path (str): The path to save the output plot.
    - name (str): The name of the plot.

    Returns:
    None
    """

    from .utils import rename_map_for_visualization

    df_factor = pd.concat([df[factor], df.filter(like="comb")], axis=1)
    df_factor = df_factor.rename(columns=rename_map_for_visualization())
    df_factor = df_factor.melt(
        id_vars=factor, var_name="variants", value_name="Editing efficiency (%)"
    )

    # Draw the line plot for the edit length analysis
    _draw_general_lineplot(
        df_factor, factor, "Editing efficiency (%)", "variants", output_path, name
    )


def analyze_pbs_rtt_length(df, output_path, quantity_cutoff=10):
    """Analyzes the PBS length and RT length data in the given DataFrame and generates heatmaps.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the PBS length and RT length data.
        output_path (str): The path to save the generated heatmaps.
        quantity_cutoff (int, optional): The cutoff value for quantity. Defaults to 10.

    Returns:
        None
    """

    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    from .utils import rename_map_for_visualization

    sample_rename_map = rename_map_for_visualization()

    # Data manipulation
    df_pbs_rtt_length = df.pivot_table(
        index=["PBS_length", "RT_length"],
        values=[*df.filter(like="comb").columns],
        aggfunc="mean",
    ).rename(columns=sample_rename_map)

    df_pbs_rtt_length_count = df.pivot_table(
        index=["PBS_length", "RT_length"],
        values=[*df.filter(like="comb").columns],
        aggfunc="count",
    ).rename(columns=sample_rename_map)

    # Drawing heatmaps
    sns.set_theme(style="ticks", font_scale=2.0)
    for i in df_pbs_rtt_length.columns:
        # COUNT
        d_heatmap_mask = _pbs_rtt_peg_count_heatmap(
            output_path, quantity_cutoff, cm, plt, np, sns, df_pbs_rtt_length_count, i
        )

        # EFFICIENCY
        _pbs_rtt_peg_efficiency_heatmap(
            output_path,
            quantity_cutoff,
            mcolors,
            plt,
            np,
            sns,
            df_pbs_rtt_length,
            i,
            d_heatmap_mask,
        )


def _pbs_rtt_peg_efficiency_heatmap(
    output_path, quantity_cutoff, mcolors, plt, np, sns, df_pbs_rtt_length, i, d_heatmap_mask
):
    """Generate a heatmap of efficiency values for a given quantity cutoff.

    Args:
        output_path (str): The path to save the generated plot and CSV file.
        quantity_cutoff (float): The cutoff value for the quantity.
        mcolors: The module for defining custom colors.
        plt: The module for creating plots.
        np: The module for numerical operations.
        sns: The module for statistical visualization.
        df_pbs_rtt_length (pandas.DataFrame): The DataFrame containing the efficiency values.
        i (str): The identifier for the heatmap.
        d_heatmap_mask (pandas.DataFrame): The mask for filtering the efficiency values.

    Returns:
        None
    """

    fig, ax = plt.subplots(figsize=(30, 8))
    plt.title(i)
    colormap = mcolors.LinearSegmentedColormap.from_list(
        name="custom_RdBu", colors=["#0093cf", "#ecdbdd", "#ff3500"], N=1024
    )
    colormap.set_bad("#e6e6e6")

    d_heatmap = (
        df_pbs_rtt_length[i].unstack().sort_index(ascending=False).T.sort_index(ascending=True).T
    )

    d_heatmap = d_heatmap.mask(~d_heatmap_mask, other=np.nan).dropna(how="all", axis=1)
    ax = sns.heatmap(
        d_heatmap,
        cmap=colormap,
        cbar_kws={"label": "Average efficiency (%)"},
        ax=ax,
        vmin=0,
        vmax=df_pbs_rtt_length.max().max(),
    )

    plt.xlabel("Length of RTT")
    plt.ylabel("Length of PBS")

    create_directory_and_save_plot(output_path, f"{i}_gt-{quantity_cutoff}", fig)
    d_heatmap.to_csv(
        f"{output_path}/{i}_gt-{quantity_cutoff}.csv",
        index=True,
    )

    plt.show()
    plt.close(fig)


def _pbs_rtt_peg_count_heatmap(
    output_path, quantity_cutoff, cm, plt, np, sns, df_pbs_rtt_length_count, i
):
    """Generate a heatmap of pegRNA count based on PBS-RTT length.

    Args:
        output_path (str): The path to save the generated plot and CSV files.
        quantity_cutoff (int): The cutoff value for pegRNA count.
        cm: The colormap object.
        plt: The matplotlib.pyplot module.
        np: The numpy module.
        sns: The seaborn module.
        df_pbs_rtt_length_count (pandas.DataFrame): The DataFrame containing pegRNA count data.
        i (str): The column name to plot.

    Returns:
        pandas.DataFrame: The masked DataFrame based on the quantity cutoff.
    """
    fig, ax = plt.subplots(figsize=(30, 8))
    plt.title(i)

    colormap = cm.get_cmap("Blues", 256)
    colormap.set_bad("#e6e6e6")

    # COUNT
    d_heatmap_count = (
        df_pbs_rtt_length_count[i]
        .unstack()
        .sort_index(ascending=False)
        .T.sort_index(ascending=True)
        .T
    )

    d_heatmap_mask = d_heatmap_count.transform(lambda x: x > quantity_cutoff)

    d_heatmap_count = d_heatmap_count.mask(~d_heatmap_mask, other=np.nan).dropna(how="all", axis=1)
    ax = sns.heatmap(
        d_heatmap_count,
        cmap=colormap,
        cbar_kws={"label": "pegRNA count"},
        vmin=0,
        vmax=df_pbs_rtt_length_count.max().max(),
        ax=ax,
    )
    plt.xlabel("Length of RTT")
    plt.ylabel("Length of PBS")

    create_directory_and_save_plot(output_path, f"{i}_count_gt-{quantity_cutoff}", fig)
    d_heatmap_count.to_csv(
        f"{output_path}/{i}_count_gt-{quantity_cutoff}.csv",
        index=True,
    )
    d_heatmap_mask.to_csv(
        f"{output_path}/{i}_mask_gt-{quantity_cutoff}.csv",
        index=True,
    )
    plt.show()
    plt.close(fig)
    return d_heatmap_mask


def draw_1st_ranked_proportion_with_piechart(df, masking_condition, title, root_dir, name):
    from .drawing_variables import pe6_colormap
    from .utils import (
        efficiency_filtering_and_ranking_dataframe,
        get_count_of_first_ranks,
        rename_map_for_visualization,
    )

    plt.rcParams.update(
        {
            "svg.fonttype": "none",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
        }
    )
    df = df.rename(columns=rename_map_for_visualization())
    df_ranked = efficiency_filtering_and_ranking_dataframe(df, masking_condition)
    pie_data = get_count_of_first_ranks(df_ranked)
    n = pie_data.sum()

    sns.set_theme(style="ticks", font_scale=2.5, font="Arial")
    fig, ax = plt.subplots(figsize=(20, 20))
    pie_data.plot.pie(
        ax=ax,
        startangle=90,
        counterclock=False,
        colors=pe6_colormap(),
        labels=None,
    )
    plt.title(title)
    plt.annotate(f"N = {n}", xy=(0.9, 0.1), xycoords="axes fraction", ha="center", va="center")

    labels = [f"{i} {j:1.2f} %" for i, j in zip(pie_data.index, pie_data / n * 100)]
    plt.legend(
        ax.patches,
        labels,
        loc="best",
        bbox_to_anchor=(
            -0.1,
            0.5,
        ),
    )

    plt.tight_layout()
    create_directory_and_save_plot(root_dir, name, fig)
    df_ranked.to_csv(f"{root_dir}/{name}_data.csv")
    plt.close(fig)
