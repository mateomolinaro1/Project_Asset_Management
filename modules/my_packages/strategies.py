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
                 percentiles:Tuple[int, int]=(10,90)):
        """
        Initializes the CrossSectionalPercentiles strategy.

        Parameters:
        - prices: pd.DataFrame, price data for the assets.
        - returns: pd.DataFrame, return data for the assets.
        - signal_function: callable, the function to compute the signals.
        - signal_function_inputs: dict, arguments to be passed to signal_function. Keywords of the dict must match
        argument names of the function.
        - percentiles: Tuple[int, int], percentiles to apply to signal values.
        """
        super().__init__(prices, returns)
        if not callable(signal_function):
            raise ValueError("signal_function must be a callable function.")
        self.signal_function = signal_function
        self.signal_function_inputs = signal_function_inputs if signal_function_inputs is not None else {}
        if not isinstance(percentiles, tuple) and len(percentiles) == 2 and all(isinstance(pct, int) for pct in percentiles):
            raise ValueError("percentiles must be a tuple of two int. (5,95) for example.")
        self.percentiles = percentiles

    def compute_signals_values(self):
        """ Compute the signals for the cross-sectional percentiles strategy"""
        self.signals_values = self.signal_function(**self.signal_function_inputs)
        return self.signals_values

    def compute_signals(self):
        self.signals = utilities.compute_percentiles(self.signals_values, self.percentiles)['signals']
        return self.signals


