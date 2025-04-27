# Main script
from modules.my_packages.data import ExcelDataSource, DataManager
from modules.my_packages.strategies import CrossSectionalPercentiles
from modules.my_packages.signal_utilities import Momentum, RollingMetrics
from modules.my_packages.portfolio import EqualWeightingScheme, NaiveRiskParity
from modules.my_packages.backtest import Backtest
from modules.my_packages.analysis import PerformanceAnalyser
from modules.my_packages import utilities
import pandas as pd
import numpy as np
import os

# Data Downloading
# Assets
file_path = os.path.join(r"C:\Users\mateo\Code\AM\Project_Asset_Management", "data", "data_short.xlsx")
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

# Factors (market)
file_path = os.path.join(r"C:\Users\mateo\Code\AM\Project_Asset_Management", "data", "msci.xlsx")
# data_source_factors = ExcelDataSource(file_path=r".\data\msci.xlsx", sheet_name="data")
data_source_factors = ExcelDataSource(file_path=file_path, sheet_name="data")
data_manager_factors = DataManager(data_source=data_source_factors,
                           max_consecutive_nan=0, # as we work with monthly data, we'll not forward fill
                           rebase_prices=True,
                           n_implementation_lags=1 # always set to 1 because returns at time t, which are from t-1 to t, must be multiplied by returns
                           # at time t+1, which is from t to t+1 to get the strategy returns (backtest) and avoid look-ahead bias
                           )
data_manager_factors.load_data()
data_manager_factors.clean_data()
data_manager_factors.compute_returns()

# Industry data
np.random.seed(42)
industries = ['IT', 'Financials', 'Healthcare', 'Industrials', 'ConsumerDiscretionary',
              'ConsumerStaples', 'CommunicationServices', 'Energy', 'Materials', 'Utilities', 'RealEstate']

returns = data_manager.returns
industries_per_action = np.random.choice(industries, size=returns.shape[1])
industry_assignments = pd.DataFrame(
    np.tile(industries_per_action, (returns.shape[0], 1)),
    index=returns.index,
    columns=returns.columns
)

# Strategies setting
strats = {'cs_mom_lo': Momentum.rolling_momentum,
          'cs_idio_mom_lo': Momentum.rolling_idiosyncratic_momentum,
          'cs_reversal_lo': Momentum.rolling_momentum,
          'cs_idio_reversal_lo': Momentum.rolling_idiosyncratic_momentum,
          'cs_sr_lo' : utilities.rolling_sharpe_ratio,
          'cs_idio_sr_lo' : RollingMetrics.rolling_idiosyncratic_sharpe_ratio
          }
strats_args = {'cs_mom_lo': {'df': data_manager.returns,
                             'nb_period': 12,
                             'rolling_window': 12+1,
                             'nb_period_to_exclude': 1,
                             'exclude_last_period': True},
               'cs_idio_mom_lo': {'df_assets': data_manager.returns,
                                  'df_factors': data_manager_factors.returns,
                                  'nb_period': 12,
                                  'rolling_window': 12+1,
                                  'nb_period_to_exclude': 1,
                                  'exclude_last_period': True},
               'cs_reversal_lo': {'df': data_manager.returns,
                             'nb_period': 1,
                             'rolling_window': 1+1,
                             'nb_period_to_exclude': None,
                             'exclude_last_period': False},
               'cs_idio_reversal_lo': {'df_assets': data_manager.returns,
                                  'df_factors': data_manager_factors.returns,
                                  'nb_period': 1,
                                  'rolling_window': 1+1,
                                  'nb_period_to_exclude': None,
                                  'exclude_last_period': False},
               'cs_sr_lo' : {'df_returns': data_manager.returns,
                             'rolling_window': 12,
                             'risk_free_rate': 0.0,
                             'frequency': 'monthly'},
               'cs_idio_sr_lo' : {'df_assets': data_manager.returns,
                                  'df_factors': data_manager_factors.returns,
                                  'rolling_window_sharpe_ratio': 12,
                                  'rolling_window_idiosyncratic': 12,
                                  'risk_free_rate': 0.0,
                                  'frequency': 'monthly'}
               }

percentiles = {'deciles': (10,90),
               'quintiles': (20,80),
               'quartiles': (25,75)}

industry_segmentation = ['AllIndustries', 'BestInIndustries']

rebalancing_freqs = {'monthly': 1,
                     'quarterly': 3,
                     'semi_annually': 6,
                     'yearly': 12,
                     }

metrics = ['total_return', 'annualized_return', 'annualized_volatility', 'annualized_sharpe_ratio', 'max_drawdown']
strategies_results = {}

for strat in strats.keys():
    strategies_results[strat] = {}
    for percentile in percentiles.keys():
        strategies_results[strat][percentile] = {}
        for industry in industry_segmentation:
            strategies_results[strat][percentile][industry] = {}
            for rebalancing_freq in rebalancing_freqs.keys():
                strategies_results[strat][percentile][industry][rebalancing_freq] = {}
                strategies_results[strat][percentile][industry][rebalancing_freq]['strategy_returns'] = None
                for metric in metrics:
                    strategies_results[strat][percentile][industry][rebalancing_freq][metric] = None

# Cross-sectional strategies
for key_strat, value_signal_function in strats.items():
    print(f"working on strategy:{key_strat}")
    # Step 1 - Strategy Creation
    strategy = CrossSectionalPercentiles(prices=data_manager.cleaned_data,
                                         returns=data_manager.returns,
                                         signal_function=value_signal_function,
                                         signal_function_inputs=strats_args[key_strat],
                                         )

    strategy.compute_signals_values()

    for key_pct, value_pct in percentiles.items():
        print(f"working on percentile:{key_pct}")
        for industry in industry_segmentation:
            print(f"working on industry:{industry}")
            for key_rebalancing_freq, value_rebalancing_freq in rebalancing_freqs.items():
                print(f"working on rebalancing frequency:{key_rebalancing_freq}")

                strategy.compute_signals(percentiles_portfolios=value_pct,
                                         percentiles_winsorization=(1,99),
                                         industry_segmentation=industry_assignments if industry == 'BestInIndustries' else None)

                # Step 2 - Portfolio Construction
                portfolio = EqualWeightingScheme(returns=data_manager.returns,
                                                 signals=strategy.signals,
                                                 rebal_periods=value_rebalancing_freq,
                                                 portfolio_type='long_only'
                                                 )
                portfolio.compute_weights()
                #portfolio.rebalance_portfolio()

                # Step 3 - Backtesting
                backtest =  Backtest(returns=data_manager.aligned_returns,
                                     weights=portfolio.weights)
                strategy_returns = backtest.run_backtest()

                # Step 4 - Performance Analysis
                performance_analyzer = PerformanceAnalyser(portfolio_returns=strategy_returns,
                                                           freq='m',
                                                           zscores=strategy.signals_values,
                                                           forward_returns=data_manager.aligned_returns)
                analyzer = PerformanceAnalyser(portfolio_returns=strategy_returns,
                                               freq='m')
                metrics = analyzer.compute_metrics()

                # Step 5 - Storing results
                print(f"storing results for strategy:{key_strat}, percentile:{key_pct}, industry:{industry}, rebalancing frequency:{key_rebalancing_freq}")
                strategies_results[key_strat][key_pct][industry][key_rebalancing_freq]['strategy_returns'] = strategy_returns
                # We can also store the metrics
                for metric in metrics.keys():
                    strategies_results[key_strat][key_pct][industry][key_rebalancing_freq][metric] = metrics[metric]

