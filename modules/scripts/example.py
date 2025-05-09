### Example script to run a backtest ###
# 0 - Import necessary libraries
import os
import sys

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
file_path = os.path.join(r"C:\Users\mateo\Code\AM\Projet_V2", "data", "data_short.xlsx")

# Data Downloading
# data_source = ExcelDataSource(file_path=r".\data\data_short.xlsx", sheet_name="data")
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
                                   percentiles_winsorization=(1,99)
                                   )
cs_mom.compute_signals_values()
cs_mom.compute_signals()

# 3 - Portfolio construction
portfolio = EqualWeightingScheme(returns=data_manager.returns,
                                 signals=cs_mom.signals,
                                 rebal_periods=6,
                                 portfolio_type='long_only'
                                 )
portfolio.compute_weights()
#portfolio.rebalance_portfolio()

# 4 - Backtesting
backtest =  Backtest(returns=data_manager.aligned_returns,
                     weights=portfolio.weights,
                     strategy_name="cs_mom")
strategy_returns = backtest.run_backtest()

# 5 - Performance analysis
performance_analyzer = PerformanceAnalyser(portfolio_returns=strategy_returns,
                                           freq='m',
                                           zscores=cs_mom.signals_values,
                                           forward_returns=data_manager.aligned_returns,
                                           percentiles="deciles",
                                           industries="AllIndustries",
                                           rebal_freq="Monthly")

metrics = performance_analyzer.compute_metrics()
for metric in metrics:
    print(f"{metric}: {metrics[metric]}")

performance_analyzer.plot_cumulative_performance(saving_path=r".\results\plots\cumulative_returns.png",
                                                 show=False,
                                                 blocking=False)

# import pickle
# with open(r".\results\plots\strategy_returns.pickle", "wb") as f:
#     pickle.dump(strategy_returns, f)
#
# with open(r".\results\plots\strategy_returns.pickle", "rb") as f:
#     strategy_returns = pickle.load(f)