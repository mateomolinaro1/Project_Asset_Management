import os
import sys
import pandas as pd
import numpy as np
from typing import Union, Tuple

# def set_working_directory(folder: str, file: str):
#     """
#     Returns an absolute path to a file in a given folder relative to the main script's location.
#
#     Args:
#         folder (str): The folder containing the data.
#         file (str): The name of the data file.
#
#     Returns:
#         str: Absolute path to the target file.
#     """
#     try:
#         main_script = sys.modules["__main__"].__file__
#         base_dir = os.path.dirname(os.path.abspath(main_script))
#     except (AttributeError, KeyError):
#         base_dir = os.getcwd()  # fallback: console/debug mode
#
#     return os.path.join(base_dir, folder, file)
# def set_working_directory(folder: str, file: str) -> str:
#     """
#     Returns an absolute path to a file in a given folder relative to the main script's location.
#     """
#     try:
#         main_script = sys.modules["__main__"].__file__
#         base_dir = os.path.dirname(os.path.abspath(main_script))
#     except (AttributeError, KeyError):
#         # Debug mode or interactive mode fallback
#         print("[DEBUG] Falling back to current working directory (os.getcwd())")
#         base_dir = os.getcwd()
#
#     return os.path.join(base_dir, folder, file)
#     def set_working_directory(folder: str, file: str) -> str:
#         """
#         Returns an absolute path to a file in a given folder relative to the main script's location.
#         Handles various execution modes (Run, Debug, Console).
#         """
#         try:
#             # Try to retrieve the file of the main script
#             main_script = sys.modules["__main__"].__file__
#             base_dir = os.path.dirname(os.path.abspath(main_script))
#         except (AttributeError, KeyError):
#             # If fail (interactive console or PyCharm Debug), use le CWD
#             print("[INFO] Fallback: using current working directory (os.getcwd())")
#             base_dir = os.getcwd()
#
#         # Return the full path
#         absolute_path = os.path.join(base_dir, folder, file)
#         print(f"[DEBUG] Path resolved to: {absolute_path}")
#         return absolute_path
def find_project_root(marker_files="main.py") -> str:
    """
    Remonte dans l'arborescence pour trouver la racine du projet en cherchant un des fichiers marqueurs.
    """
    current_dir = os.getcwd()
    while True:
        if any(os.path.exists(os.path.join(current_dir, marker)) for marker in marker_files):
            return current_dir
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            # On est à la racine du disque
            raise FileNotFoundError("Impossible de trouver la racine du projet.")
        current_dir = parent_dir

def set_working_directory(folder: str, file: str) -> str:
    """
    Returns an absolute path to a file in a given folder relative to the main script's location.
    Handles various execution modes (Run, Debug, Console).
    """
    try:
        main_script = sys.modules["__main__"].__file__
        #base_dir = os.path.dirname(os.path.abspath(main_script))
        base_dir = find_project_root()
    except (AttributeError, KeyError):
        print("[INFO] Fallback: using project root via marker file.")
        base_dir = find_project_root()

    absolute_path = os.path.join(base_dir, folder, file)
    print(f"[DEBUG] Path resolved to: {absolute_path}")
    return absolute_path

def compute_percentiles(df:pd.DataFrame, percentiles:Tuple[int, int]):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas df.")
    if not ( isinstance(percentiles, tuple) and len(percentiles)==2 and all(isinstance(x, int) for x in percentiles) ):
        raise ValueError("percentiles must be a tuple of exactly two elements, containing only int.")

    upper_bound = df.apply(
        lambda row: np.nanpercentile(row, q=percentiles[1]) if not row.dropna().empty else np.nan, axis=1)
    # Format to ease comparison after
    upper_bound = pd.DataFrame(data=np.tile(upper_bound.values[:, None], (1, df.shape[1])), index=df.index,
                               columns=df.columns)

    lower_bound = df.apply(
        lambda row: np.nanpercentile(row, q=percentiles[0]) if not row.dropna().empty else np.nan, axis=1)
    # Format to ease comparison after
    lower_bound = pd.DataFrame(data=np.tile(lower_bound.values[:, None], (1, df.shape[1])), index=df.index,
                               columns=df.columns)

    # To store
    signals = pd.DataFrame(data=np.nan, index=df.index, columns=df.columns)
    signals[df >= upper_bound] = 1.0
    signals[df <= lower_bound] = -1.0

    return {'upper_bound': upper_bound,
            'lower_bound': lower_bound,
            'signals': signals}

def clean_dataframe(df:pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the DataFrame by replacing -inf or inf values by nan.

    Args:
        df (pd.DataFrame): The DataFrame to clean.

    Returns:
        pd.DataFrame: A cleaned DataFrame with NaN rows and columns removed.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas df.")

    # Replace -inf and inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    return df

def compute_zscores(df:pd.DataFrame, axis:int=1) -> pd.DataFrame:
    """
    Computes the z-scores of a DataFrame along the specified axis.

    Args:
        df (pd.DataFrame): The DataFrame to compute z-scores for.
        axis (int): The axis along which to compute z-scores. 0 for rows, 1 for columns.

    Returns:
        pd.DataFrame: A DataFrame containing the z-scores.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas df.")
    if axis not in [0, 1]:
        raise ValueError("axis must be either 0 (rows) or 1 (columns).")

    mean = df.mean(axis=axis, skipna=True)
    std = df.std(axis=axis, skipna=True)

    zscores = (df.values - mean.values[:,None]) / std.values[:,None]
    zscores = pd.DataFrame(data=zscores, index=df.index, columns=df.columns)
    return zscores

def winsorize_dataframe(df:pd.DataFrame, percentiles:Tuple[int, int]=(1,99), axis:int=1) -> pd.DataFrame:
    """
    Winsorizes the DataFrame by replacing extreme values with the specified percentiles row-wise or column-wise.

    Args:
        df (pd.DataFrame): The DataFrame to winsorize.
        percentiles (Tuple[int, int]): The lower and upper percentiles to use for winsorization.
        axis (int): The axis along which to apply winsorization.
                    0 for column-wise (apply on each column),
                    1 for row-wise (apply on each row).
    Returns:
        pd.DataFrame: A winsorized DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame.")
    if not (
            isinstance(percentiles, tuple)
            and len(percentiles) == 2
            and all(isinstance(x, int) for x in percentiles)
    ):
        raise ValueError("percentiles must be a tuple of two integers.")
    if axis not in [0, 1]:
        raise ValueError("axis must be either 0 (rows) or 1 (columns).")

    def winsorize_row_or_col(row_or_col):
        if row_or_col.isna().all():
            return row_or_col  # we keep the row or colum empty
        lower = np.nanpercentile(row_or_col, percentiles[0])
        upper = np.nanpercentile(row_or_col, percentiles[1])
        return row_or_col.clip(lower=lower, upper=upper)

    return df.apply(winsorize_row_or_col, axis=axis)