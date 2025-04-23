import pandas as pd
import numpy as np
from typing import Union, Tuple


def non_overlapping_indices_rolling_window(df: Union[pd.DataFrame, pd.Series, np.ndarray], window:int, handle_len_df_non_divisible_by_window=True):
    """
    Generates non-overlapping indices for rolling window on dataframe.
    For example, if df indices 0,1,2,...,100 and window=30 it returns:
    0,30,60,90,100 if handle_len_df_non_divisible_by_window is set to True
    because 100 is non-divisible by 30 (90+30 will be out of range).

    Args:
        df: pandas df
        window: len of the npn-overlapping rolling window
        handle_len_df_non_divisible_by_window: If True, includes a last partial window

    Returns:
        A list of indices at the beginning of each window
    """
    # Check inputs
    if not isinstance(df, (pd.DataFrame, pd.Series, np.ndarray)):
        raise ValueError("df must be a pandas dataframe.")
    if not isinstance(window, int):
        raise ValueError("window must be an int.")
    if not isinstance(handle_len_df_non_divisible_by_window, bool):
        raise ValueError("handle_len_df_non_divisible_by_window must be a bool.")

    if window>len(df):
        raise ValueError("The length of the window must be less or equal than the nb of rows of the df.")

    len_df = len(df)
    indices = list(range(0, len(df), window))

    # Check if there are elements non-covered at this end
    last_index = indices[-1]
    if handle_len_df_non_divisible_by_window and last_index < len_df:
        indices.append(len_df)

    return indices

def rolling_z_score(df: Union[pd.DataFrame, pd.Series], w:int):
    """ Function to rolling-standardize a dataframe or series with rolling window w"""
    # Check inputs
    if not isinstance(df, (pd.DataFrame, pd.Series)):
        raise ValueError("df must be either a pandas dataframe or series.")
    if not isinstance(w, int):
        raise ValueError("w must be an int.")
    if w>df.shape[0]:
        raise ValueError("The length of the window cannot be greater than the nb of rows of the df/series.")

    zscores = pd.DataFrame(np.nan, index=df.index, columns=[df.columns])
    for i_end in range(w, df.shape[0]+1):
        df_local = df.iloc[i_end-w:i_end] # upper bound not included
        zscores_temp = (df_local - df_local.mean(axis=0))/df_local.std(axis=0)
        zscores.iloc[i_end-1,:] = zscores_temp.iloc[-1,:] # because lower bound is included

    return zscores

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
