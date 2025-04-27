import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Tuple
from sklearn.linear_model import LinearRegression
from modules.my_packages import utilities

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
    def rolling_momentum(df: pd.DataFrame,
                         nb_period: int,
                         rolling_window:int,
                         nb_period_to_exclude: Union[None, int]=None,
                         exclude_last_period: bool = False) -> pd.DataFrame:
        # Check inputs
        if rolling_window > df.shape[0]:
            raise ValueError("rolling_window cannot be greater than the nb of rows in df.")


        rolling_mom = df.rolling(window=rolling_window).apply(lambda x: Momentum.compute_momentum(pd.DataFrame(x),
                                                                                                  nb_period,
                                                                                                  nb_period_to_exclude,
                                                                                                  exclude_last_period))
        return rolling_mom

    @staticmethod
    def compute_idiosyncratic_returns(df_assets:pd.DataFrame, df_factors:pd.DataFrame, window_regression:int):
        """
        Computes the idiosyncratic returns of the assets using a rolling regression.

        Args:
            df_assets (pd.DataFrame): DataFrame of asset returns.
            df_factors (pd.DataFrame): DataFrame of factor returns.
            window_regression (int): Window size for the rolling regression.
        """
        # Check inputs
        if not isinstance(df_assets, pd.DataFrame):
            raise TypeError("The `df_assets` parameter must be a pandas DataFrame.")
        if not isinstance(df_factors, pd.DataFrame):
            raise TypeError("The `df_factors` parameter must be a pandas DataFrame.")
        if not df_assets.shape[0] == df_factors.shape[0]:
            raise ValueError("The number of rows in df_assets and df_factors must be equal.")
        if not df_assets.index.equals(df_factors.index):
            raise ValueError("The indices of df_assets and df_factors must be equal.")
        if window_regression > df_assets.shape[0]:
            raise ValueError("window_regression cannot be greater than the nb of rows in df.")

        # Perform rolling regression
        residuals = pd.DataFrame(data=np.nan, index=df_assets.index, columns=df_assets.columns)
        for col_idx,col in enumerate(df_assets.columns):
            print(f"Working on column {col} ({col_idx+1}/{df_assets.shape[1]})")
            for i in range(window_regression, df_assets.shape[0]):
                print(f"Working on row {i} ({i+1}/{df_assets.shape[0]})")
                # Get the current window of data
                y = df_assets.iloc[i-window_regression:i,col_idx]
                x = df_factors.iloc[i-window_regression:i]

                # Check presence of NaN values
                merged_yx = pd.merge(y, x, left_index=True, right_index=True, how='inner')
                merged_yx = merged_yx.dropna()
                # Check if we have enough data points for regression
                # If not, skip this iteration
                if merged_yx.shape[0] < 2:
                    continue
                y_cleaned = merged_yx.iloc[:,0].values.reshape(-1,1)
                x_cleaned = merged_yx.iloc[:,1:].values

                # Perform regression
                model = LinearRegression()
                model.fit(x_cleaned, y_cleaned)
                # Get residuals
                res = y_cleaned - model.predict(x_cleaned)
                # Store residuals
                residuals.iloc[i, col_idx] = res[-1][0]

        return residuals

    @staticmethod
    def rolling_idiosyncratic_momentum(df_assets: pd.DataFrame,
                                       df_factors: pd.DataFrame,
                                       nb_period: int,
                                       rolling_window: int,
                                       nb_period_to_exclude: Union[None, int] = None,
                                       exclude_last_period: bool = False) -> pd.DataFrame:

        # Check inputs
        if df_assets.shape[1]==0 and df_factors.shape[1]==0:
            raise ValueError("The DataFrames must not be empty.")
        if not df_assets.shape[0] == df_factors.shape[0]:
            raise ValueError("The number of rows in df_assets and df_factors must be equal.")
        if rolling_window > df_assets.shape[0]:
            raise ValueError("rolling_window cannot be greater than the nb of rows in df.")

        # Compute idiosyncratic returns
        residuals = Momentum.compute_idiosyncratic_returns(df_assets=df_assets,
                                                           df_factors=df_factors,
                                                           window_regression=rolling_window)

        rolling_mom = residuals.rolling(window=rolling_window).apply(lambda x: Momentum.compute_momentum(pd.DataFrame(x),
                                                                                                         nb_period,
                                                                                                         nb_period_to_exclude,
                                                                                                         exclude_last_period))

        return rolling_mom

    @staticmethod
    def rolling_sharpe_ratio_momentum(df_assets: pd.DataFrame,
                                      nb_period: int,
                                      rolling_window_momentum: int,
                                      rolling_window_sharpe_ratio: int,
                                      nb_period_to_exclude: Union[None, int] = None,
                                      exclude_last_period: bool = False,
                                      risk_free_rate:Union[pd.DataFrame, float] = 0.0,
                                      frequency:str = 'd') -> pd.DataFrame:

        # Check inputs
        if rolling_window_momentum > df_assets.shape[0]:
            raise ValueError("rolling_window cannot be greater than the nb of rows in df.")
        if rolling_window_sharpe_ratio > df_assets.shape[0]:
            raise ValueError("rolling_window cannot be greater than the nb of rows in df.")

        # Compute sharpe_ratios
        sharpe_ratios = utilities.rolling_sharpe_ratio(df_returns=df_assets,
                                                       rolling_window=rolling_window_sharpe_ratio,
                                                       risk_free_rate=risk_free_rate,
                                                       frequency=frequency)


        rolling_mom = sharpe_ratios.rolling(window=rolling_window_momentum).apply(
            lambda x: Momentum.compute_momentum(pd.DataFrame(x),
                                                nb_period,
                                                nb_period_to_exclude,
                                                exclude_last_period))

        return rolling_mom

    @staticmethod
    def rolling_idiosyncratic_sharpe_ratio_momentum(df_assets: pd.DataFrame,
                                                    df_factors: pd.DataFrame,
                                                    nb_period: int,
                                                    rolling_window_momentum: int,
                                                    rolling_window_sharpe_ratio: int,
                                                    rolling_window_idiosyncratic: int,
                                                    nb_period_to_exclude: Union[None, int] = None,
                                                    exclude_last_period: bool = False,
                                                    risk_free_rate: Union[pd.DataFrame, float] = 0.0,
                                                    frequency: str = 'd') -> pd.DataFrame:

        # Check inputs
        if rolling_window_momentum > df_assets.shape[0]:
            raise ValueError("rolling_window cannot be greater than the nb of rows in df_assets.")
        if rolling_window_sharpe_ratio > df_assets.shape[0]:
            raise ValueError("rolling_window cannot be greater than the nb of rows in df_assets.")
        if rolling_window_idiosyncratic > df_assets.shape[0]:
            raise ValueError("rolling_window cannot be greater than the nb of rows in df_assets.")

        # Compute idiosyncratic returns
        idiosyncratic_returns = Momentum.compute_idiosyncratic_returns(df_assets=df_assets,
                                                                       df_factors=df_factors,
                                                                       window_regression=rolling_window_idiosyncratic)
        # Compute sharpe_ratios
        sharpe_ratios = utilities.rolling_sharpe_ratio(df_returns=idiosyncratic_returns,
                                                       rolling_window=rolling_window_sharpe_ratio,
                                                       risk_free_rate=risk_free_rate,
                                                       frequency=frequency)

        rolling_mom = sharpe_ratios.rolling(window=rolling_window_momentum).apply(
            lambda x: Momentum.compute_momentum(pd.DataFrame(x),
                                                nb_period,
                                                nb_period_to_exclude,
                                                exclude_last_period))

        return rolling_mom