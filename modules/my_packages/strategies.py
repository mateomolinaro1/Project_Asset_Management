from abc import ABC, abstractmethod
from typing import Union, Tuple
import pandas as pd
import numpy as np
import modules.my_packages.utilities as utilities

class Strategy(ABC):
    """Abstract class to define the interface for the strategy"""
    def __init__(self, prices: pd.DataFrame, returns:pd.DataFrame):
        self.prices = prices
        self.returns = returns
        self.signals_values = None
        self.signals = None

    @abstractmethod
    def compute_signals_values(self):
        """Compute the signals for the strategy"""
        pass

    @abstractmethod
    def compute_signals(self):
        """Compute the signals for the strategy"""
        pass

class BuyAndHold(Strategy):
    """Class to implement the buy and hold strategy"""
    def compute_signals_values(self):
        """Compute the signals for the buy and hold strategy"""
        self.signals_values = (~np.isnan(self.returns)).astype(int)
        return self.signals_values

    def compute_signals(self):
        """Compute the signals for the buy and hold strategy"""
        self.signals = (~np.isnan(self.returns)).astype(int)
        return self.signals

class CrossSectionalPercentiles(Strategy):
    """ Class to implement the cross-sectional percentiles strategy"""
    def __init__(self,
                 prices:pd.DataFrame,
                 returns:pd.DataFrame,
                 signal_function,
                 signal_function_inputs:dict=None,
                 percentiles_portfolios:Tuple[int, int]=(10,90),
                 percentiles_winsorization:Tuple[int, int]=(1,99)):
        """
        Initializes the CrossSectionalPercentiles strategy.

        Parameters:
        - prices: pd.DataFrame, price data for the assets.
        - returns: pd.DataFrame, return data for the assets.
        - signal_function: callable, the function to compute the signals.
        - signal_function_inputs: dict, arguments to be passed to signal_function. Keywords of the dict must match
        argument names of the function.
        - percentiles_portfolios: Tuple[int, int], percentiles to apply to signal values.
        - percentiles_winsorization: Tuple[int, int], percentiles to apply to signal values for winsorization.
        """
        super().__init__(prices, returns)
        if not callable(signal_function):
            raise ValueError("signal_function must be a callable function.")
        self.signal_function = signal_function
        self.signal_function_inputs = signal_function_inputs if signal_function_inputs is not None else {}
        if not isinstance(percentiles_portfolios, tuple) and len(percentiles_portfolios) == 2 and all(isinstance(pct, int) for pct in percentiles_portfolios):
            raise ValueError("percentiles must be a tuple of two int. (5,95) for example.")
        self.percentiles_portfolios = percentiles_portfolios
        if not isinstance(percentiles_winsorization, tuple) and len(percentiles_winsorization) == 2 and all(isinstance(pct, int) for pct in percentiles_winsorization):
            raise ValueError("percentiles must be a tuple of two int. (1,99) for example.")
        self.percentiles_winsorization = percentiles_winsorization

    def compute_signals_values(self):
        """ Compute the signals for the cross-sectional percentiles strategy"""
        # Compute signal values
        self.signals_values = self.signal_function(**self.signal_function_inputs)
        # Clean the signal values
        self.signals_values = utilities.clean_dataframe(self.signals_values)
        # Compute zscores
        self.signals_values = utilities.compute_zscores(self.signals_values, axis=1)
        # Winsorize the signal values
        self.signals_values = utilities.winsorize_dataframe(df=self.signals_values, percentiles=self.percentiles_winsorization, axis=1)
        return self.signals_values

    def compute_signals(self):
        self.signals = utilities.compute_percentiles(self.signals_values, self.percentiles_portfolios)['signals']
        return self.signals


