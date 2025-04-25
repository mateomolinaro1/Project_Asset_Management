import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Tuple

class Momentum:

    @staticmethod
    def compute_momentum(df: pd.DataFrame, nb_period: int, nb_period_to_exclude: Union[None, int]=None,
                                exclude_last_period: bool = False):
        """
        Computes the (m-1)-month momentum.

        This function calculates the performance between `t-m` months and `t-1` months.
        For example, `compute_momentum(df, 'm', 6)` computes the momentum over the past 5 months
        (from `t-6` to `t-1`).

        Args:
            df (pd.DataFrame): DataFrame with a datetime index and prices values.
            nb_period (int): the number of periods on which the momentum is computed. For example, on daily data,
            m = 3*20 for the 3-month mom.
            nb_period_to_exclude (None,int): number of periods to exclude for the computation of the momentum. For ex,
            when computing the 12-month mom, someone might want to remove the last month in the computation to avoid the
            short term reversal effect. This parameter is set to None if the following parameter exclude_last_period is
            set to False.
            exclude_last_period (bool): you must set this parameter to True if you want to remove nb_period_to_exclude
            periods at the end of the mom computation.


        Returns:
            mom: A float containing the computed (nb_period-nb_period_to_exclude) momentum as a time series.

        Raises:
            TypeError: If `df` is not a DataFrame or `freq` is not a string.
            ValueError: If `df` has more than one column or invalid values for `freq` or `m`.
        """
        # Check inputs
        if not isinstance(df, pd.DataFrame):
            raise TypeError("The `df` parameter must be a pandas DataFrame.")
        if not isinstance(nb_period, int):
            raise TypeError("The `m` parameter must be an integer.")
        if nb_period_to_exclude is not None:
            if nb_period_to_exclude >= nb_period:
                raise ValueError("nb_period_to_exclude must be strictly less than the nb_period")
        if nb_period >= df.shape[0]:
            raise ValueError("The `nb_period` parameter must be less than the number of rows in `df`.")

        # idx_start = (df.shape[0] - 1) - nb_period
        idx_start = df.shape[0] - 1 - nb_period

        if exclude_last_period:
            idx_end = (df.shape[0]) - nb_period_to_exclude - 1
        else:
            idx_end = df.shape[0] - 1

        # Ensure the indices are within bounds
        if idx_start < 0 or idx_end <= 0:
            raise ValueError("The `m` parameter leads to out-of-bounds indices.")

        mom = df.iloc[idx_end,:] / df.iloc[idx_start,:] - 1
        return np.array(mom)

    @staticmethod
    def rolling_momentum(df: pd.DataFrame, nb_period: int, rolling_window:int,
                         nb_period_to_exclude: Union[None, int]=None, exclude_last_period: bool = False):
        # Check inputs
        if rolling_window > df.shape[0]:
            raise ValueError("rolling_window cannot be greater than the nb of rows in df.")
        # Input types check to do

        rolling_mom = df.rolling(window=rolling_window).apply(lambda x: Momentum.compute_momentum(pd.DataFrame(x),
                                                                                    nb_period,
                                                                                    nb_period_to_exclude,
                                                                                    exclude_last_period))
        return rolling_mom

