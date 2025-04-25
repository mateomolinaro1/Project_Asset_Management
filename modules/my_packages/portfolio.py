from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Union

class WeightingScheme(ABC):
    """Abstract class to define the interface for the weighting scheme"""
    def __init__(self, returns:pd.DataFrame, signals:pd.DataFrame, rebal_periods:Union[int, None], portfolio_type:str="long_only"):
        self.returns = returns
        self.signals = signals
        if not isinstance(rebal_periods, (int,type(None))):
            raise ValueError("rebal_periods must be an int or None.")
        self.rebal_periods = rebal_periods
        self.portfolio_type = portfolio_type
        self.weights = None
        self.rebalanced_weights = None

    @abstractmethod
    def compute_weights(self):
        """Compute the weights for the strategy"""
        pass

    @abstractmethod
    def rebalance_portfolio(self):
        """Rebalance the portfolio at the specified frequency and computes transaction costs"""
        pass



class EqualWeightingScheme(WeightingScheme):
    """Class to implement the equal weighting scheme"""
    def __init__(self, returns: pd.DataFrame, signals: pd.DataFrame, rebal_periods: int = 0,
                 portfolio_type: str = "long_only"):
        super().__init__(returns, signals, rebal_periods, portfolio_type)

    def compute_weights(self):
        """Compute the weights for the equal weighting scheme"""
        number_of_positive_signals = (self.signals == 1).sum(axis=1)
        number_of_negative_signals = (self.signals == -1).sum(axis=1)

        weights = pd.DataFrame(data=0.0, index=self.signals.index, columns=self.signals.columns)

        if self.portfolio_type == "long_only":
            # Check that there is at least one positive signal at each date
            if (number_of_positive_signals > 0).sum() == number_of_positive_signals.shape[0]:
                weights[self.signals == 1] = self.signals[self.signals == 1].div(number_of_positive_signals, axis=0)
            else:
                # Compute weights where there is at least one positive signal on the row
                mask_valid = number_of_positive_signals > 0
                weights[self.signals == 1] = self.signals[self.signals == 1].div(
                    number_of_positive_signals.where(mask_valid), axis=0)

                # Setting weights to 0 at rows where all signals are missing
                weights.loc[~mask_valid, :] = 0 # Optional as in the creation of weights, default data set to 0
                print("For some dates, there is no signals. Weights set to 0 by default.")

        elif self.portfolio_type == "short_only":
            # Check that there is at least one negative signal at each date
            if (number_of_negative_signals > 0).sum() == number_of_negative_signals.shape[0]:
                weights[self.signals == -1] = self.signals[self.signals == -1].div(number_of_negative_signals, axis=0)
            else:
                # Compute weights where there is at least one negative signal on the row
                mask_valid = number_of_negative_signals > 0
                weights[self.signals == -1] = self.signals[self.signals == -1].div(
                    number_of_negative_signals.where(mask_valid), axis=0)

                # Setting weights to 0 at rows where all signals are missing
                weights.loc[~mask_valid, :] = 0  # Optional as in the creation of weights, default data set to 0
                print("For some dates, there is no signals. Weights set to 0 by default.")
                # # Before changes
                # weights[self.signals == -1] = 0.0
                # print("For some dates, there is no signals. Weights set to 0 by default.")

        elif self.portfolio_type == "long_short":
            # Check that there is at least one positive and one negative signal at each date
            if ((number_of_positive_signals > 0).sum() == number_of_positive_signals.shape[0] and
                (number_of_negative_signals > 0).sum() == number_of_negative_signals.shape[0]):
                weights[self.signals == 1] = self.signals[self.signals == 1].div(number_of_positive_signals, axis=0)
                weights[self.signals == -1] = self.signals[self.signals == -1].div(number_of_negative_signals, axis=0)
            else:
                raise ValueError("Error division by 0. There is no positive or negative signal for some dates.")

        else:
            raise ValueError("portfolio_type not supported")

        self.weights = weights.fillna(0)
        return self.weights

    def rebalance_portfolio(self):
        """
        Rebalance the portfolio and compute weights evolution between rebalancing dates.
        Returns rebalanced weights accounting for price drift between rebalancing dates.
        """
        # First compute weights if not already done
        if self.weights is None:
            self.weights = self.compute_weights()
        
        # If no rebalancing period specified or rebalancing at every period, return original weights
        if self.rebal_periods is None or self.rebal_periods == 0:
            return self.weights
        
        # Initialize rebalanced weights
        self.rebalanced_weights = pd.DataFrame(0.0, index=self.weights.index, columns=self.weights.columns)
        
        # Set initial weights and define first rebalancing date
        first_date = self.weights.index[0]
        self.rebalanced_weights.loc[first_date] = self.weights.loc[first_date]
        
        # Create list of rebalancing dates starting from first date
        all_dates = self.weights.index
        date_position = pd.Series(range(len(all_dates)), index=all_dates)
        rebalancing_dates = all_dates[range(0, len(all_dates), self.rebal_periods)]
        
        # Loop through all dates to update weights
        for t, date in enumerate(all_dates[1:], 1):  # Start from second date
            if date in rebalancing_dates:
                # On rebalancing dates, use the computed weights
                self.rebalanced_weights.loc[date] = self.weights.loc[date]
            else:
                # Between rebalancing dates, let weights drift with returns
                prev_date = all_dates[t-1]
                prev_weights = self.rebalanced_weights.loc[prev_date]
                current_returns = self.returns.loc[date]
                
                # Calculate drifted weights
                drifted_weights = prev_weights * (1 + current_returns)
                
                # Normalize weights based on portfolio type
                if self.portfolio_type == "long_only":
                    total_weight = drifted_weights.sum()
                    if total_weight > 0:
                        drifted_weights = drifted_weights / total_weight
                
                self.rebalanced_weights.loc[date] = drifted_weights
        
        return self.rebalanced_weights.fillna(0.0)

class NaiveRiskParity(WeightingScheme):
    """Class to implement the naive risk parity weighting scheme"""

    def __init__(self, returns: pd.DataFrame, signals: pd.DataFrame, rebal_periods: int = 0, portfolio_type: str = "long_only", vol_lookback: int = 252):
        """
        Initialize the naive risk parity model.

        Parameters:
        - returns (pd.DataFrame): DataFrame of asset returns.
        - signals (pd.DataFrame): DataFrame of trading signals (1 for long, -1 for short, 0 for neutral).
        - rebal_periods (int): Rebalancing frequency (default: 0 for daily).
        - portfolio_type (str): "long_only", "short_only", or "long_short".
        - vol_lookback (int): Lookback period for volatility calculation (default: 252 days ~ 1 year).
        """
        super().__init__(returns, signals, rebal_periods, portfolio_type)
        self.vol_lookback = vol_lookback

    def compute_weights(self):
        """Compute weights using the inverse volatility method."""
        rolling_vol = self.returns.rolling(window=self.vol_lookback, min_periods=21).std()
        inv_vol = 1 / rolling_vol  # Inverse of volatility
        inv_vol.replace([np.inf, -np.inf], np.nan, inplace=True)  # Handle division by zero
        inv_vol = inv_vol.fillna(0)  # Replace NaN values

        # Mask to apply signals
        number_of_positive_signals = (self.signals == 1).sum(axis=1)
        number_of_negative_signals = (self.signals == -1).sum(axis=1)

        weights = pd.DataFrame(data=0.0, index=self.signals.index, columns=self.signals.columns)

        if self.portfolio_type == "long_only":
            valid_mask = number_of_positive_signals > 0
            masked_inv_vol = inv_vol * (self.signals == 1)
            weights = masked_inv_vol.div(masked_inv_vol.sum(axis=1), axis=0)
            weights.loc[~valid_mask, :] = 0  # Set weights to 0 if no valid signals

        elif self.portfolio_type == "short_only":
            valid_mask = number_of_negative_signals > 0
            masked_inv_vol = inv_vol * (self.signals == -1)
            weights = masked_inv_vol.div(masked_inv_vol.sum(axis=1), axis=0)
            weights.loc[~valid_mask, :] = 0

        elif self.portfolio_type == "long_short":
            valid_mask = (number_of_positive_signals > 0) & (number_of_negative_signals > 0)
            long_weights = inv_vol * (self.signals == 1)
            short_weights = inv_vol * (self.signals == -1)

            long_weights = long_weights.div(long_weights.sum(axis=1), axis=0)
            short_weights = short_weights.div(short_weights.sum(axis=1), axis=0)

            weights = long_weights - short_weights
            weights.loc[~valid_mask, :] = 0

        else:
            raise ValueError("portfolio_type not supported")

        self.weights = weights.fillna(0)
        return self.weights

    def rebalance_portfolio(self):
        if self.weights is None:
            self.weights = self.compute_weights()

        if (self.rebalanced_weights is None) and (self.rebal_periods is not None):
            self.rebalanced_weights = pd.DataFrame(data=0.0, index=self.weights.index, columns=self.weights.columns)
            for t in range(0, self.rebalanced_weights.shape[0]):
                if (t % self.rebal_periods) == 0:
                    # rebalancing date
                    self.rebalanced_weights.iloc[t,:] = self.weights.iloc[t,:]
                else:
                    # non rebalancing date, weights must derive
                    self.rebalanced_weights.iloc[t,:] = self.rebalanced_weights.iloc[t-1] * (1+self.returns.iloc[t,:])

            return self.rebalanced_weights.fillna(0.0)

        return self.weights
