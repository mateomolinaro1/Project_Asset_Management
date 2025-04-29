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
import pickle
import copy

strats = {'cs_mom_lo': Momentum.rolling_momentum,
          'cs_idio_mom_lo': Momentum.rolling_idiosyncratic_momentum,
          'cs_reversal_lo': Momentum.rolling_momentum,
          'cs_idio_reversal_lo': Momentum.rolling_idiosyncratic_momentum,
          'cs_sr_lo' : utilities.rolling_sharpe_ratio,
          'cs_idio_sr_lo' : RollingMetrics.rolling_idiosyncratic_sharpe_ratio
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

with open(r"C:\Users\mateo\Code\AM\Projet_V2\results\strategies_results\strategies_results.pickle", 'rb') as handle:
    strategies_results = pickle.load(handle)

########################################################################################################################
################# Storing results in a convenient format and all strategies aligned on the same dates ##################
########################################################################################################################
# Recomputes the metrics for aligned strategies
# start_date = max(x for sublist in start_dates.values() for x in sublist)
start_date = pd.to_datetime("2009-01-01")
print(f"first date where strategy returns is available for all strategies is {start_date}")
# To store new results
strategies_results_aligned = copy.deepcopy(strategies_results)
all_series = []
all_series_by_strat = {strat: [] for strat in strats.keys()}
# all_metrics = pd.DataFrame(index=['total_return', 'annualized_return', 'annualized_volatility', 'annualized_sharpe_ratio', 'max_drawdown'])
# all_metrics_by_strat = {strat: pd.DataFrame(index=['total_return', 'annualized_return', 'annualized_volatility', 'annualized_sharpe_ratio', 'max_drawdown']) for strat in strats.keys()}
metrics_dict = {}  # Pour all_metrics
metrics_by_strat_dict = {strat: {} for strat in strats.keys()}  # Pour all_metrics_by_strat
# Now, we'll crop the strategies from this date
for key_strat, value_signal_function in strats.items():
    for key_pct, value_pct in percentiles.items():
        for industry in industry_segmentation:
            for key_rebalancing_freq, value_rebalancing_freq in rebalancing_freqs.items():
                strategies_results_aligned[key_strat][key_pct][industry][key_rebalancing_freq]['strategy_returns'] = \
                    strategies_results[key_strat][key_pct][industry][key_rebalancing_freq]['strategy_returns'].loc[
                        start_date:]

                # Recompute the metrics
                analyzer = PerformanceAnalyser(portfolio_returns=strategies_results_aligned[key_strat][key_pct][industry][key_rebalancing_freq]['strategy_returns'],
                                               freq='m',
                                               percentiles=key_pct,
                                               industries=industry,
                                               rebal_freq=key_rebalancing_freq
                                               )
                metrics = analyzer.compute_metrics()

                # Store the metrics
                for metric in metrics.keys():
                    strategies_results_aligned[key_strat][key_pct][industry][key_rebalancing_freq][metric] = metrics[metric]


                # # All metrics
                # all_metrics[f"{key_strat}_{key_pct}_{industry}_{key_rebalancing_freq}"] = pd.Series(metrics)
                # # All metrics by strat
                # all_metrics_by_strat[key_strat][f"{key_strat}_{key_pct}_{industry}_{key_rebalancing_freq}"] = pd.Series(metrics)
                col_name = f"{key_strat}_{key_pct}_{industry}_{key_rebalancing_freq}"
                metrics_dict[col_name] = metrics
                metrics_by_strat_dict[key_strat][col_name] = metrics

                # Create dataframes of all the strategies within a given strategy
                renamed_series_by_strat = strategies_results_aligned[key_strat][key_pct][industry][key_rebalancing_freq][
                    'strategy_returns'].copy()
                renamed_series_by_strat.rename(columns={f"{key_strat}": f"{key_strat}_{key_pct}_{industry}_{key_rebalancing_freq}"}, inplace=True)
                # renamed_series_by_strat.name = f"{key_strat}_{key_pct}_{industry}_{key_rebalancing_freq}"
                all_series_by_strat[key_strat].append(renamed_series_by_strat)

                # Create a dataframe of all the strategies returns to plot cumulative perf
                renamed_series = strategies_results_aligned[key_strat][key_pct][industry][key_rebalancing_freq][
                    'strategy_returns'].copy()
                renamed_series_by_strat.rename(
                    columns={f"{key_strat}": f"{key_strat}_{key_pct}_{industry}_{key_rebalancing_freq}"}, inplace=True)
                # renamed_series.name = f"{key_strat}_{key_pct}_{industry}_{key_rebalancing_freq}"
                all_series.append(renamed_series_by_strat)


# First output : a dict containing 6 dataframes with the strategies returns, one for each strategy, to plot the cumulative perf
# Concatenate all the series into a dataframe
all_strategies_returns_by_strat = {strat_key: pd.concat(all_series_by_strat[strat_key], axis=1) for strat_key in all_series_by_strat.keys()}
# Set the first line to 0.0 for each strat
for strat_key, df in all_strategies_returns_by_strat.items():
    df.iloc[0, :] = 0.0

# Second output: a unified dataframe with all the strategies returns to plot the cumulative perf
all_strategies_returns = pd.concat(all_series, axis=1)
all_strategies_returns.iloc[0,:] = 0.0 # because all the strategies must start at 0.0

# Third output: a dict containing 6 dataframes with the performance metrics, one for each strategy
all_metrics_by_strat = {strat: pd.DataFrame.from_dict(metrics_by_strat_dict[strat], orient='columns') for strat in strats.keys()}

# Fourth output: a unified dataframe with the performance metrics
all_metrics = pd.DataFrame.from_dict(metrics_dict, orient='columns')