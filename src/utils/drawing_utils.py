import pathlib
from functools import cache
from typing import Dict

import numpy as np
import pandas as pd
from icecream import ic


def get_count_of_first_ranks(df):
    """Calculates the count of values in each column of a DataFrame that are less than 2.

    Args:
        df (pandas.DataFrame): The input DataFrame.

    Returns:
        pandas.Series: A Series containing the count of values less than 2 for each column.
    """

    data = {}
    for col in df.columns:
        data[col] = df.loc[lambda x: x[col] == 1, col].count()

    data = pd.Series(data)
    return data


def efficiency_filtering_and_ranking_dataframe(df, mask_func):
    """Rank transform the data in the DataFrame based on the given mask.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the data to be rank transformed.
    - mask (pandas.Series): The mask to be used for the rank transformation.

    Returns:
    - df_ranked (pandas.DataFrame): The DataFrame containing the rank transformed data.

    Example usage:
    >>> df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [10, 20, 30, 40]})
    >>> mask = pd.Series([True, False, True, False])
    >>> df_ranked = efficiency_filtering_and_ranking_dataframe(df, mask)
    """

    # Rank transform the data based on the mask
    df = df.mask(df.transform(mask_func)).copy()
    df_ranked = df.rank(axis=1, ascending=False)

    return df_ranked


def hn00217129_name_mapper() -> dict:
    """Maps the original names to the corresponding new names.

    Returns:
        dict: A dictionary mapping the original names to the new names.
    """
    return {
        "MK0-0": "HEK-M-CTRL-7D-KMC",
        "MK0-1": "HEK-M-1-7D-KMC",
        "MK0-2": "HEK-M-2-7D-KMC",
        "MK0-3": "HEK-M-3-7D-KMC",
        "MK0-4": "HEK-M-4-7D-KMC",
        "MK0-5": "HEK-M-5-7D-KMC",
        "MK0-8": "HEK-M-8-7D-KMC",
        "MK0-9": "HEK-M-9-7D-KMC",
        "MU0-0": "HEK-M-CTRL-7D-UAR",
        "MU0-1": "HEK-M-1-7D-UAR",
        "MU0-2": "HEK-M-2-7D-UAR",
        "MU0-3": "HEK-M-3-7D-UAR",
        "MU0-4": "HEK-M-4-7D-UAR",
        "MU0-5": "HEK-M-5-7D-UAR",
        "MU0-6": "HEK-M-6-7D-UAR",
        "MU0-7": "HEK-M-7-7D-UAR",
        "MU0-8": "HEK-M-8-7D-UAR",
        "MU0-9": "HEK-M-9-7D-UAR",
        "SU0-0": "HEK-S-CTRL-14D-UAR",
        "SU0-12": "HEK-S-12-14D-UAR",
        "SU0-13": "HEK-S-13-14D-UAR",
        "SU0-14": "HEK-S-14-14D-UAR",
        "SU0-3": "HEK-S-3-14D-UAR",
        "SU0-4": "HEK-S-4-14D-UAR",
        "SU0-5": "HEK-S-5-14D-UAR",
        "SU0-6": "HEK-S-6-14D-UAR",
    }


def _get_barcode_category(path: pathlib.PurePath) -> str:
    """Returns the category of a barcode based on its parent directories.

    Parameters:
    path (pathlib.PurePath): The parent directories of the barcode.

    Returns:
    str: The category of the barcode. If no category is found, returns "UNKNOWN".
    """
    for p in path.parents:
        if p.stem.startswith("barcode"):
            return p.stem

    return "UNKNOWN"


def read_sequencing_batch_data(data_paths, renamer=None) -> Dict[str, pathlib.PurePath]:
    rval = {}

    try:
        for p in data_paths:
            key = "@".join(
                [
                    _get_barcode_category(p),
                    p.stem.split("+")[0] if renamer is None else renamer[p.stem.split("+")[0]],
                ]
            )
            rval[key] = p
    except Exception as e:
        ic(p)
        raise e

    return rval


# Prime editing efficiency calculation
def pe_percent_efficiency_ratio(df: pd.DataFrame, edited_col: str, unedited_col: str) -> pd.Series:
    # % of edited reads
    # No background normalization
    edited_read_counts = df[edited_col]
    wt_read_counts = df[unedited_col]
    try:
        return (edited_read_counts / (wt_read_counts + edited_read_counts) * 100).astype(float)

    except ValueError as e:
        ic(e)
        return 0.0


def annotating_pe_raw_result(datapath: pathlib.PosixPath, sample_name: str) -> pd.DataFrame:
    """Annotates the raw result of prime editing efficiency calculation.

    Args:
        datapath (pathlib.PosixPath): The path to the raw data file.
        sample_name (str): The name of the sample.

    Returns:
        pd.DataFrame: The annotated DataFrame containing the calculated prime editing efficiency.

    Raises:
        ValueError: If there is an error reading the data file.
    """

    try:
        df = pd.read_csv(datapath)
        df = df.rename(columns={"Gene": "KEY"})
        df[["REF_ID", "Library"]] = df["KEY"].str.split("@", expand=True)
        df["CATEGORY"] = df["Library"].str.split("_").str[-1]

        name = sample_name

        df_read_counts = pd.pivot_table(df, "Read_counts", index="REF_ID", columns="CATEGORY")
        df_read_counts.rename(
            columns={
                "edited": f"{name}+edited_read_counts",
                "others(barcode only)": f"{name}+others_read_counts",
                "unedited": f"{name}+unedited_read_counts",
            },
            inplace=True,
        )
        df_read_counts[f"{name}+total_read_counts"] = (
            df_read_counts[f"{name}+edited_read_counts"]
            + df_read_counts[f"{name}+unedited_read_counts"]
        )
        return df_read_counts

    except ValueError as e:
        ic(e)
        return None


def annotating_pe_normalized_result(
    df: pd.DataFrame, control: str, treatment: str, read_cutoff=200, background_cutoff=1
):
    # Background normalized prime editing efficiency calculation
    # Yu, G. et al. Prediction of efficiencies for diverse prime editing systems in multiple cell types. Cell 186, 2256–2272 (2023).

    # Dynamically calculate the background efficiency

    background_pe_efficiency = pe_percent_efficiency_ratio(
        df,
        edited_col=f"{control}+edited_read_counts",
        unedited_col=f"{control}+unedited_read_counts",
    )

    edited_read_counts = df[f"{treatment}+edited_read_counts"]
    wt_read_counts = df[f"{treatment}+unedited_read_counts"]
    total_read_counts = edited_read_counts + wt_read_counts

    try:
        # Vector ops
        df[f"Normalized+{treatment}+pe_ratio_%"] = (
            (edited_read_counts - total_read_counts * background_pe_efficiency / 100)
            / (total_read_counts - total_read_counts * background_pe_efficiency / 100)
            * 100
        )

    except AttributeError:
        # Background not found
        df[f"Normalized+{treatment}+pe_ratio_%"] = edited_read_counts / total_read_counts * 100
    # Fill NaN and Inf from division by zero with 0.0
    # df.fillna(0.0, inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Set nan for filter conditions
    df.loc[
        (total_read_counts < read_cutoff) | (background_pe_efficiency >= background_cutoff),
        f"Normalized+{treatment}+pe_ratio_%",
    ] = np.nan

    return df


def create_directory_and_save_plot(root_path, fig_name, fig):
    """Create a directory if it doesn't exist and save a plot as an SVG file.

    Parameters:
    - root_path (str): The root path where the directory will be created.
    - fig_name (str): The name of the plot file (without extension).
    - fig (matplotlib.figure.Figure): The figure object to be saved.

    Returns:
    - None
    """
    pathlib.Path(root_path).mkdir(parents=True, exist_ok=True)
    fig.savefig(
        f"{root_path}/{fig_name}.svg",
        format="svg",
        bbox_inches="tight",
        dpi=fig.dpi,
    )


@cache
def rename_map_for_visualization() -> dict:
    """Returns a dictionary that maps the original sample names to the renamed sample names for
    visualization purposes.

    Returns:
        dict: A dictionary where the keys are the original sample names and the values are the corresponding renamed sample names.
    """
    return {
        # Separated samples
        "indiv_vis+HEK-M-1-7D-KMC%": "PEmax-KMC",
        "indiv_vis+HEK-M-2-7D-KMC%": "PEmaxdRNaseH-KMC",
        "indiv_vis+HEK-M-3-7D-KMC%": "PE6a(+PEmaxCas9)-KMC",
        "indiv_vis+HEK-M-4-7D-KMC%": "PE6b(+PEmaxCas9)-KMC",
        "indiv_vis+HEK-M-5-7D-KMC%": "PE6c(+PEmaxCas9)-KMC",
        "indiv_vis+HEK-M-6-7D-KMC%": "PE6d(+PEmaxCas9)-KMC",
        "indiv_vis+HEK-M-7-7D-KMC%": "PE6e(+dRNaseH)-KMC",
        "indiv_vis+HEK-M-8-7D-KMC%": "PE6f(+dRNaseH)-KMC",
        "indiv_vis+HEK-M-9-7D-KMC%": "PE6g(+dRNaseH)-KMC",
        "indiv_vis+HEK-M-1-7D-UAR%": "PEmax-UAR",
        "indiv_vis+HEK-M-2-7D-UAR%": "PEmaxdRNaseH-UAR",
        "indiv_vis+HEK-M-3-7D-UAR%": "PE6a(+PEmaxCas9)-UAR",
        "indiv_vis+HEK-M-4-7D-UAR%": "PE6b(+PEmaxCas9)-UAR",
        "indiv_vis+HEK-M-5-7D-UAR%": "PE6c(+PEmaxCas9)-UAR",
        "indiv_vis+HEK-M-6-7D-UAR%": "PE6d(+PEmaxCas9)-UAR",
        "indiv_vis+HEK-M-7-7D-UAR%": "PE6e(+dRNaseH)-UAR",
        "indiv_vis+HEK-M-8-7D-UAR%": "PE6f(+dRNaseH)-UAR",
        "indiv_vis+HEK-M-9-7D-UAR%": "PE6g(+dRNaseH)-UAR",
        "indiv_vis+HEK-S-1-14D-KMC%": "PEmax-KMC",
        "indiv_vis+HEK-S-2-14D-KMC%": "PEmaxdRNaseH-KMC",
        "indiv_vis+HEK-S-3-14D-KMC%": "PE6a(+PEmaxCas9)-KMC",
        "indiv_vis+HEK-S-4-14D-KMC%": "PEb(+PEmaxCas9)-KMC",
        "indiv_vis+HEK-S-5-14D-KMC%": "PE6c(+PEmaxCas9)-KMC",
        "indiv_vis+HEK-S-6-14D-KMC%": "PE6d(+PEmaxCas9)-KMC",
        "indiv_vis+HEK-S-10-14D-KMC%": "PE6ec-KMC",
        "indiv_vis+HEK-S-11-14D-KMC%": "PE6fc-KMC",
        "indiv_vis+HEK-S-12-14D-KMC%": "PE6gc-KMC",
        "indiv_vis+HEK-S-13-14D-KMC%": "PE6ed-KMC",
        "indiv_vis+HEK-S-14-14D-KMC%": "PE6fd-KMC",
        "indiv_vis+HEK-S-15-14D-KMC%": "PE6gd-KMC",
        "indiv_vis+HEK-S-1-14D-UAR%": "PEmax-UAR",
        "indiv_vis+HEK-S-2-14D-UAR%": "PEmaxdRNaseH-UAR",
        "indiv_vis+HEK-S-3-14D-UAR%": "PE6a(+PEmaxCas9)-UAR",
        "indiv_vis+HEK-S-4-14D-UAR%": "PEb(+PEmaxCas9)-UAR",
        "indiv_vis+HEK-S-5-14D-UAR%": "PE6c(+PEmaxCas9)-UAR",
        "indiv_vis+HEK-S-6-14D-UAR%": "PE6d(+PEmaxCas9)-UAR",
        "indiv_vis+HEK-S-10-14D-UAR%": "PE6ec-UAR",
        "indiv_vis+HEK-S-11-14D-UAR%": "PE6fc-UAR",
        "indiv_vis+HEK-S-12-14D-UAR%": "PE6gc-UAR",
        "indiv_vis+HEK-S-13-14D-UAR%": "PE6ed-UAR",
        "indiv_vis+HEK-S-14-14D-UAR%": "PE6fd-UAR",
        "indiv_vis+HEK-S-15-14D-UAR%": "PE6gd-UAR",
        #  Combined samples
        "comb_vis+HEK-M-1-7D%": "PEmax",
        "comb_vis+HEK-M-2-7D%": "PEmaxdRNaseH",
        "comb_vis+HEK-M-3-7D%": "PE6a(+PEmaxCas9)",
        "comb_vis+HEK-M-4-7D%": "PE6b(+PEmaxCas9)",
        "comb_vis+HEK-M-5-7D%": "PE6c(+PEmaxCas9)",
        "comb_vis+HEK-M-6-7D%": "PE6d(+PEmaxCas9)",
        "comb_vis+HEK-M-7-7D%": "PE6e(+dRNaseH)",
        "comb_vis+HEK-M-8-7D%": "PE6f(+dRNaseH)",
        "comb_vis+HEK-M-9-7D%": "PE6g(+dRNaseH)",
        "comb_vis+HEK-S-1-14D%": "PEmax",
        "comb_vis+HEK-S-2-14D%": "PEmaxdRNaseH",
        "comb_vis+HEK-S-3-14D%": "PE6a(+PEmaxCas9)",
        "comb_vis+HEK-S-4-14D%": "PEb(+PEmaxCas9)",
        "comb_vis+HEK-S-5-14D%": "PE6c(+PEmaxCas9)",
        "comb_vis+HEK-S-6-14D%": "PE6d(+PEmaxCas9)",
        "comb_vis+HEK-S-10-14D%": "PE6ec",
        "comb_vis+HEK-S-11-14D%": "PE6fc",
        "comb_vis+HEK-S-12-14D%": "PE6gc",
        "comb_vis+HEK-S-13-14D%": "PE6ed",
        "comb_vis+HEK-S-14-14D%": "PE6fd",
        "comb_vis+HEK-S-15-14D%": "PE6gd",
    }


def rename_peg_3term_motifs(x) -> dict:
    motif_data = {
        "GGGTCAGGAGCCCCCCCCCTGAACCCAGGATAACCCTCAAAGTCGGGGGGC": "tmpknot",
        "CGCGGTTCTATCTAGTTACGCGTTAAACCAACTAGAA": "tevopreQ1",
    }
    return motif_data[x]


def pandas_read_csv_with_nan_path_handling(path: str, **kwargs) -> pd.DataFrame:
    """Read a CSV file into a DataFrame with handling for NaN paths.

    Parameters:
    - path (str): The path to the CSV file.
    - **kwargs: Additional keyword arguments to be passed to the `pd.read_csv` function.

    Returns:
    - pd.DataFrame: The DataFrame containing the data from the CSV file.
    """
    try:
        return pd.read_csv(path, **kwargs)
    except ValueError:
        # ic(e)
        return pd.DataFrame()
    except FileNotFoundError:
        # ic(e)
        return pd.DataFrame()


def pandas_merge_dataframe_with_empty_handling(
    left: pd.DataFrame, right: pd.DataFrame, **kwargs
) -> pd.DataFrame:
    """Merge two DataFrames with handling for empty DataFrames.

    Parameters:
    - left (pd.DataFrame): The left DataFrame to be merged.
    - right (pd.DataFrame): The right DataFrame to be merged.
    - **kwargs: Additional keyword arguments to be passed to the `pd.merge` function.

    Returns:
    - pd.DataFrame: The merged DataFrame.
    """
    if left.empty:
        return right
    elif right.empty:
        return left
    else:
        return pd.merge(left, right, **kwargs)


def add_two_columns_with_handling_a_missing_column(
    df: pd.DataFrame, x: str, y: str, z: str
) -> pd.DataFrame:
    try:
        if x not in df.columns or y not in df.columns:
            # Return the original DataFrame if either x or y is not in the columns   as there is no need to sum them
            pass
        else:
            df[z] = df[x] + df[y]
        return df

    except Exception as e:
        ic(e)
        raise e
