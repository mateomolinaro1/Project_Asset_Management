### Example script to compare monthly vs annual rebalancing ###
import os
import sys

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from modules.my_packages.data import ExcelDataSource, DataManager
from modules.my_packages.strategies import CrossSectionalPercentiles
from modules.my_packages.signal_utilities import Momentum
from modules.my_packages.portfolio import EqualWeightingScheme
from modules.my_packages.analysis import PerformanceAnalyser

# 1. Data Loading and Preparation
data_source = ExcelDataSource(file_path=r".\data\data_short.xlsx", sheet_name="data")
data_manager = DataManager(data_source=data_source,
                         max_consecutive_nan=0,
                         rebase_prices=True,
                         n_implementation_lags=1)

data_manager.load_data()
data_manager.clean_data()
data_manager.compute_returns()
data_manager.account_implementation_lags()

# 2. Strategy Setup - Cross-sectional momentum
cs_mom = CrossSectionalPercentiles(
    prices=data_manager.cleaned_data,
    returns=data_manager.returns,
    signal_function=Momentum.rolling_momentum,
    signal_function_inputs={
        'df': data_manager.returns,
        'nb_period': 12,
        'rolling_window': 12+1,
        'nb_period_to_exclude': 1,
        'exclude_last_period': True
    },
    percentiles_winsorization=(1,99)
)
cs_mom.compute_signals_values()
cs_mom.compute_signals()

# 3. Portfolio Construction with Different Rebalancing Frequencies
# Monthly rebalancing
portfolio_monthly = EqualWeightingScheme(
    returns=data_manager.returns,
    signals=cs_mom.signals,
    rebal_periods=1,  # Monthly
    portfolio_type='long_only'
)

# Annual rebalancing
portfolio_annual = EqualWeightingScheme(
    returns=data_manager.returns,
    signals=cs_mom.signals,
    rebal_periods=12,  # Annual
    portfolio_type='long_only'
)

# Compute weights and get returns
weights_monthly, returns_monthly = portfolio_monthly.rebalance_portfolio(transaction_cost_bp=10)
weights_annual, returns_annual = portfolio_annual.rebalance_portfolio(transaction_cost_bp=10)

# 4. Performance Analysis
print("\n=== Performance Analysis ===")

# Monthly rebalancing analysis
monthly_perf = PerformanceAnalyser(portfolio_returns=returns_monthly, freq='m')
monthly_metrics = monthly_perf.compute_metrics()

# Annual rebalancing analysis
annual_perf = PerformanceAnalyser(portfolio_returns=returns_annual, freq='m')
annual_metrics = annual_perf.compute_metrics()

# Print results
print("\nMonthly Rebalancing:")
for metric, value in monthly_metrics.items():
    print(f"{metric}: {value:.4f}")

print("\nAnnual Rebalancing:")
for metric, value in annual_metrics.items():
    print(f"{metric}: {value:.4f}")