import pandas as pd
import numpy as np

class Backtest:
    """Class to backtest a strategy"""
    def __init__(self, returns:pd.DataFrame, weights:pd.DataFrame, strategy_name:str=""):
        self.returns = returns
        self.weights = weights
        self.portfolio_returns = None
        self.strategy_name = strategy_name

    def run_backtest(self):
        """Run the backtest"""
        self.portfolio_returns = (pd.DataFrame(
            (self.returns * self.weights).sum(axis=1),
        columns=[f"{self.strategy_name}"])
        )
        return self.portfolio_returns

    def get_results(self):
        """Get the backtest results"""
        if self.portfolio_returns is None:
            self.run_backtest()
        return self.portfolio_returns