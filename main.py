# Main script
from modules.my_packages.data import ExcelDataSource, DataManager
from modules.my_packages.strategies import CrossSectionalPercentiles
from modules.my_packages.signal_utilities import Momentum
from modules.my_packages.portfolio import EqualWeightingScheme, NaiveRiskParity
from modules.my_packages.backtest import Backtest
from modules.my_packages.analysis import PerformanceAnalyser

# Data Downloading
data_source = ExcelDataSource(file_path=r".\data\data.xlsx", sheet_name="data")
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

# Strategies setting
# strats = {'cs_mom_lo': Momentum.rolling_momentum,
#                      'cs_idio_mom_lo': Momentum.rolling_idiosyncratic_momentum,
#                      'cs_reversal_lo': Momentum.rolling_momentum,
#                      'cs_idio_reversal_lo': Momentum.rolling_idiosyncratic_momentum,
#                      'cs_sr_lo' : Momentum.rolling_sharpe_ratio,
#                      'cs_idio_sr_lo' : Momentum.rolling_idiosyncratic_sharpe_ratio
#                      }
strats = {'cs_mom_lo': Momentum.rolling_momentum,
          'cs_idio_mom_lo': Momentum.rolling_momentum,
          'cs_reversal_lo': Momentum.rolling_momentum,
          'cs_idio_reversal_lo': Momentum.rolling_momentum,
          'cs_sr_lo' : Momentum.rolling_momentum,
          'cs_idio_sr_lo' : Momentum.rolling_momentum
          }
strats_args = {'cs_mom_lo': {'df': data_manager.returns,
                             'nb_period': 12,
                             'rolling_window': 12+1,
                             'nb_period_to_exclude': 1,
                             'exclude_last_period': True},
               'cs_idio_mom_lo': {'df': data_manager.returns,
                             'nb_period': 12,
                             'rolling_window': 12+1,
                             'nb_period_to_exclude': 1,
                             'exclude_last_period': True},
               'cs_reversal_lo': {'df': data_manager.returns,
                             'nb_period': 12,
                             'rolling_window': 12+1,
                             'nb_period_to_exclude': 1,
                             'exclude_last_period': True},
               'cs_idio_reversal_lo': {'df': data_manager.returns,
                             'nb_period': 12,
                             'rolling_window': 12+1,
                             'nb_period_to_exclude': 1,
                             'exclude_last_period': True},
               'cs_sr_lo' : {'df': data_manager.returns,
                             'nb_period': 12,
                             'rolling_window': 12+1,
                             'nb_period_to_exclude': 1,
                             'exclude_last_period': True},
               'cs_idio_sr_lo' : {'df': data_manager.returns,
                             'nb_period': 12,
                             'rolling_window': 12+1,
                             'nb_period_to_exclude': 1,
                             'exclude_last_period': True}
               }
# percentiles = {'deciles': 10,
#                'quintiles': 5,
#                'quartiles': 4}
percentiles = {'deciles': (10,90),
               'quintiles': (20,80),
               'quartiles': (25,75)}
industries = ['IT', 'Financials', 'Healthcare', 'Industrials', 'ConsumerDiscretionary', 'ConsumerStaples', 'CommunicationServices', 'Energy', 'Materials', 'Utilities', 'RealEstate', 'AllIndustries']
rebalancing_freqs = {'monthly': 1,
                     'quarterly': 3,
                     'semi_annually': 6,
                     'yearly': 12,
                     }

strategies_results = {}

for strat in strats.keys():
    strategies_results[strat] = {}
    for percentile in percentiles.keys():
        strategies_results[strat][percentile] = {}
        for industry in industries:
            strategies_results[strat][percentile][industry] = {}
            for rebalancing_freq in rebalancing_freqs.keys():
                strategies_results[strat][percentile][industry][rebalancing_freq] = None

# Cross-sectional momentum
for key_strat, value_signal_function in strats.items():
    for key_pct, value_pct in percentiles.items():
        for industry in industries: # we must add an industry mask to the signal function
            for key_rebalancing_freq, value_rebalancing_freq in rebalancing_freqs.items():
                # Step 1 - Strategy Creation
                strategy = CrossSectionalPercentiles(prices=data_manager.cleaned_data,
                                                   returns=data_manager.returns,
                                                   signal_function=value_signal_function,
                                                   signal_function_inputs=strats_args[key_strat],
                                                   percentiles=value_pct
                                                   )

                strategy.compute_signals_values()
                strategy.compute_signals()

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
                performance_analyzer.compute_metrics()

                # Step 5 - Storing results
                strategies_results[key_strat][key_pct][industry][key_rebalancing_freq] = strategy_returns
                # We can also store the metrics
