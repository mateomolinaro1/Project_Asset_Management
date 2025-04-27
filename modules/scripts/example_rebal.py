### Example script to run a backtest ###
# 0 - Import necessary libraries
import os
import sys
import numpy as np
import pandas as pd

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from modules.my_packages.data import ExcelDataSource, DataManager
from modules.my_packages.strategies import CrossSectionalPercentiles
from modules.my_packages.signal_utilities import Momentum
from modules.my_packages.portfolio import EqualWeightingScheme, NaiveRiskParity
from modules.my_packages.backtest import Backtest
from modules.my_packages.analysis import PerformanceAnalyser
from modules.my_packages import utilities

# 1 - Setting working directory, Data loading and cleaning
# Setting the working directory
#file_path = os.path.join(r"C:\Users\tmont\Documents\Cours\Dauphine\Asset_manegement\Projet\Project_Asset_Management\data", "data", "data.xlsx")
file_path = utilities.set_working_directory(folder="data", file="data.xlsx")

# Data Downloading
data_source = ExcelDataSource(file_path=file_path, sheet_name="data")
data_manager = DataManager(data_source=data_source,
                           max_consecutive_nan=0, # as we work with monthly data, we'll not forward fill
                           rebase_prices=True,
                           n_implementation_lags=1 # always set to 1 because returns at time t, which are from t-1 to t, must be multiplied by returns
                           # at time t+1, which is from t to t+1 to get the strategy returns (backtest) and avoid look-ahead bias
                           )
data_manager.load_data()
data_manager.clean_data()
data_manager.compute_returns()
data_manager.account_implementation_lags()

# 2 - Strategy creation - Cross-sectional momentum
cs_mom = CrossSectionalPercentiles(prices=data_manager.cleaned_data,
                                   returns=data_manager.returns,
                                   signal_function=Momentum.rolling_momentum,
                                   signal_function_inputs={'df': data_manager.returns,
                                                           'nb_period': 12,
                                                           'rolling_window': 12+1,
                                                           'nb_period_to_exclude': 1,
                                                           'exclude_last_period': True},
                                   percentiles_portfolios=(10,90),
                                   percentiles_winsorization=(1,99)
                                   )
cs_mom.compute_signals_values()
cs_mom.compute_signals()

# 3 - Portfolio construction
# Test avec différentes périodes de rebalancement
portfolio_monthly = EqualWeightingScheme(returns=data_manager.returns,
                                         signals=cs_mom.signals,
                                         rebal_periods=1,  # Rebalancement mensuel (chaque période)
                                         portfolio_type='long_only')

portfolio_annual = EqualWeightingScheme(returns=data_manager.returns,
                                        signals=cs_mom.signals,
                                        rebal_periods=12,  # Rebalancement annuel (12 mois)
                                        portfolio_type='long_only')

# Calcul et rebalancement des portefeuilles
weights_monthly = portfolio_monthly.compute_weights()
rebalanced_weights_monthly = portfolio_monthly.rebalance_portfolio()

weights_annual = portfolio_annual.compute_weights()
rebalanced_weights_annual = portfolio_annual.rebalance_portfolio()

print(f"Monthly rebalancing - Sum of weights at last date: {rebalanced_weights_monthly.iloc[-1].sum():.4f}")
print(f"Annual rebalancing - Sum of weights at last date: {rebalanced_weights_annual.iloc[-1].sum():.4f}")

# Comparaison des performances
backtest_monthly = Backtest(returns=data_manager.aligned_returns,
                            weights=rebalanced_weights_monthly)
backtest_annual = Backtest(returns=data_manager.aligned_returns,
                           weights=rebalanced_weights_annual)

strategy_returns_monthly = backtest_monthly.run_backtest()
strategy_returns_annual = backtest_annual.run_backtest()

# Analyse des performances pour les deux stratégies
perf_monthly = PerformanceAnalyser(portfolio_returns=strategy_returns_monthly,
                                   freq='m',
                                   zscores=cs_mom.signals_values,
                                   forward_returns=data_manager.aligned_returns)
metrics_monthly = perf_monthly.compute_metrics()
for metric in metrics_monthly:
    print(f"{metric}: {metrics_monthly[metric]}")

perf_annual = PerformanceAnalyser(portfolio_returns=strategy_returns_annual,
                                  freq='m',
                                  zscores=cs_mom.signals_values,
                                  forward_returns=data_manager.aligned_returns)
metrics_annual = perf_annual.compute_metrics()
for metric in metrics_annual:
    print(f"{metric}: {metrics_annual[metric]}")


print("Monthly rebalancing dates:")
monthly_dates = rebalanced_weights_monthly.index
for i in range(0, len(monthly_dates)):
    if i == 0 or rebalanced_weights_monthly.iloc[i].equals(portfolio_monthly.weights.iloc[i]):
        print(monthly_dates[i].strftime('%Y-%m-%d'))

print("\nAnnual rebalancing dates:")
annual_dates = rebalanced_weights_annual.index
for i in range(0, len(annual_dates)):
    if i == 0 or (i % 12 == 0 and i < len(annual_dates)):
        print(annual_dates[i].strftime('%Y-%m-%d'))
        
        