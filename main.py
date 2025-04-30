# Main script
from modules.my_packages.data import ExcelDataSource, DataManager
from modules.my_packages.strategies import CrossSectionalPercentiles
from modules.my_packages.signal_utilities import Momentum, RollingMetrics
from modules.my_packages.portfolio import EqualWeightingScheme, NaiveRiskParity
from modules.my_packages.backtest import Backtest
from modules.my_packages.analysis import PerformanceAnalyser
from modules.my_packages import utilities
import pandas as pd
import pickle
import numpy as np
import os
import copy

########################################################################################################################
################################################# Data Downloading #####################################################
########################################################################################################################

# Assets
# file_path = os.path.join(r"C:\Users\mateo\Code\AM\Projet_V2", "data", "data_returns.xlsx")
# data_source = ExcelDataSource(file_path=file_path, sheet_name="data")
data_source = ExcelDataSource(file_path=r".\data\data_returns.xlsx", sheet_name="data")
data_manager = DataManager(data_source=data_source,
                           max_consecutive_nan=0, # as we work with monthly data, we'll not forward fill
                           rebase_prices=True,
                           n_implementation_lags=1 # always set to 1 because returns at time t, which are from t-1 to t, must be multiplied by returns
                           # at time t+1, which is from t to t+1 to get the strategy returns (backtest) and avoid look-ahead bias
                           )

data_manager.load_data()
data_manager.returns = data_manager.raw_data
data_manager.account_implementation_lags()

# Factors (market)
# file_path = os.path.join(r"C:\Users\mateo\Code\AM\Projet_v2", "data", "msci_prices.xlsx")
# data_source_factors = ExcelDataSource(file_path=file_path, sheet_name="data")
data_source_factors = ExcelDataSource(file_path=r".\data\msci_prices.xlsx", sheet_name="data")
data_manager_factors = DataManager(data_source=data_source_factors,
                           max_consecutive_nan=0, # as we work with monthly data, we'll not forward fill
                           rebase_prices=True,
                           n_implementation_lags=1 # always set to 1 because returns at time t, which are from t-1 to t, must be multiplied by returns
                           # at time t+1, which is from t to t+1 to get the strategy returns (backtest) and avoid look-ahead bias
                           )
data_manager_factors.load_data()
data_manager_factors.clean_data()
data_manager_factors.compute_returns()

# Industry
industries_classification = pd.read_excel(r".\data\industry_classification.xlsx", index_col=0, sheet_name="data")

########################################################################################################################
####################################################### Inputs #########################################################
########################################################################################################################

# Strategies setting
strats = {'cs_mom_lo': Momentum.rolling_momentum,
          'cs_idio_mom_lo': Momentum.rolling_idiosyncratic_momentum,
          'cs_reversal_lo': Momentum.rolling_momentum,
          'cs_idio_reversal_lo': Momentum.rolling_idiosyncratic_momentum,
          'cs_sr_lo' : utilities.rolling_sharpe_ratio,
          'cs_idio_sr_lo' : RollingMetrics.rolling_idiosyncratic_sharpe_ratio
          }

start_dates = {'cs_mom_lo': [],
               'cs_idio_mom_lo': [],
               'cs_reversal_lo': [],
               'cs_idio_reversal_lo': [],
               'cs_sr_lo' : [],
               'cs_idio_sr_lo' : []}

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

########################################################################################################################
################################### Backtesting and "grid search" for all strategies ###################################
########################################################################################################################

# Cross-sectional strategies
for key_strat, value_signal_function in strats.items():
    print(f"-----------------------------------------------------------------------------------------------------------")
    print(f"---------------------------working on strategy:{key_strat}-------------------------------------------------")
    # Step 1 - Strategy Creation
    strategy = CrossSectionalPercentiles(prices=data_manager.cleaned_data,
                                         returns=data_manager.returns,
                                         signal_function=value_signal_function,
                                         signal_function_inputs=strats_args[key_strat],
                                         )

    strategy.compute_signals_values()

    for key_pct, value_pct in percentiles.items():
        print(f"-------------------------working on percentile:{key_pct}-----------------------------------------------")
        for industry in industry_segmentation:
            print(f"*-----------working on industry:{industry}--------------*")
            for key_rebalancing_freq, value_rebalancing_freq in rebalancing_freqs.items():
                print(f"**------working on rebalancing frequency:{key_rebalancing_freq}**------")

                strategy.compute_signals(percentiles_portfolios=value_pct,
                                         percentiles_winsorization=(1,99),
                                         industry_segmentation=industries_classification if industry == 'BestInIndustries' else None)

                # Step 2 - Portfolio Construction
                portfolio = EqualWeightingScheme(
                    returns=data_manager.returns,
                    signals=strategy.signals,
                    rebal_periods=value_rebalancing_freq,
                    portfolio_type='long_only'
                )
                portfolio.compute_weights()
                portfolio.rebalance_portfolio()

                # Step 3 - Backtesting
                backtest = Backtest(
                    returns=data_manager.aligned_returns,
                    weights=portfolio.rebalanced_weights,
                    strategy_name=key_strat,
                    returns_after_fees=portfolio.returns_after_fees  # <-- Ajout ici
                )
                strategy_returns = backtest.run_backtest()

                # Step 4 - Performance Analysis
                analyzer = PerformanceAnalyser(portfolio_returns=strategy_returns,
                                               freq='m',
                                               percentiles=key_pct,
                                               industries=industry,
                                               rebal_freq=key_rebalancing_freq
                                               )
                metrics = analyzer.compute_metrics()

                # Step 5 - Storing results
                print(f"storing results for strategy:{key_strat}, percentile:{key_pct}, industry:{industry}, rebalancing frequency:{key_rebalancing_freq}")
                strategies_results[key_strat][key_pct][industry][key_rebalancing_freq]['strategy_returns'] = strategy_returns
                # We can also store the metrics
                for metric in metrics.keys():
                    strategies_results[key_strat][key_pct][industry][key_rebalancing_freq][metric] = metrics[metric]

                # saving start dates to align all strategies and allow comparison
                start_dates[key_strat].append((strategy_returns != 0.0).idxmax().values[0])

# Save the results
with open(r".\results\strategies_results\strategies_results.pickle", 'wb') as handle:
    pickle.dump(strategies_results, handle, protocol=pickle.HIGHEST_PROTOCOL)


########################################################################################################################
################# Storing results in a convenient format and all strategies aligned on the same dates ##################
########################################################################################################################

# Recomputes the metrics for aligned strategies
start_date = max(x for sublist in start_dates.values() for x in sublist)
print(f"first date where strategy returns is available for all strategies is {start_date}")
# To store new results
strategies_results_aligned = copy.deepcopy(strategies_results)
all_series = []
all_series_by_strat = {strat: [] for strat in strats.keys()}
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
                
                # Saving cumulative performance plot
                analyzer.plot_cumulative_performance(saving_path=fr".\results\plots\{key_strat}\cumulative_returns_{key_strat}_{key_pct}_{industry}_{key_rebalancing_freq}.png",
                                                                 show=False,
                                                                 blocking=False)
                # Store the metrics
                for metric in metrics.keys():
                    strategies_results_aligned[key_strat][key_pct][industry][key_rebalancing_freq][metric] = metrics[metric]

                # metrics_dict and metrics_by_strat_dict
                col_name = f"{key_strat}_{key_pct}_{industry}_{key_rebalancing_freq}"
                metrics_dict[col_name] = metrics
                metrics_by_strat_dict[key_strat][col_name] = metrics

                # Create dataframes of all the strategies within a given strategy
                renamed_series_by_strat = strategies_results_aligned[key_strat][key_pct][industry][key_rebalancing_freq][
                    'strategy_returns'].copy()
                # renamed_series_by_strat.name = f"{key_strat}_{key_pct}_{industry}_{key_rebalancing_freq}"
                renamed_series_by_strat.rename(
                    columns={f"{key_strat}": f"{key_strat}_{key_pct}_{industry}_{key_rebalancing_freq}"}, inplace=True)
                all_series_by_strat[key_strat].append(renamed_series_by_strat)

                # Create a dataframe of all the strategies returns to plot cumulative perf
                renamed_series = strategies_results_aligned[key_strat][key_pct][industry][key_rebalancing_freq][
                    'strategy_returns'].copy()
                # renamed_series.name = f"{key_strat}_{key_pct}_{industry}_{key_rebalancing_freq}"
                renamed_series_by_strat.rename(
                    columns={f"{key_strat}": f"{key_strat}_{key_pct}_{industry}_{key_rebalancing_freq}"}, inplace=True)
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

# Creating a plot for all unified strategies
bench = data_manager_factors.returns.loc[start_date:]
bench = bench.copy()
bench.iloc[0,:] = 0.0

utilities.plot_dataframe(df=(1+all_strategies_returns).cumprod()-1,
                         bench=(1+bench).cumprod()-1,
                         title="Cumulative returns of all strategies",
                         save_path=r".\results\final_results\cumulative_performance\all_strategies_cumulative_returns.png",
                         xlabel="Date",
                         ylabel="Cumulative returns",
                         legend=True,
                         figsize=(20,15),
                         bbox_to_anchor=(0.5,-0.05),
                         ncol=6,
                         fontsize=7,
                         show=False,
                         blocking=False
                         )

# Creating a plot for all strategies by strat
for strat_key, df in all_strategies_returns_by_strat.items():
    utilities.plot_dataframe(df=(1+df).cumprod()-1,
                             bench=(1+bench).cumprod()-1,
                             title=f"Cumulative returns of {strat_key} strategies",
                             save_path=fr".\results\final_results\cumulative_performance\{strat_key}_cumulative_returns.png",
                             xlabel="Date",
                             ylabel="Cumulative returns",
                             legend=True,
                             figsize=(20,15),
                             bbox_to_anchor=(0.5,-0.05),
                             ncol=3,
                             fontsize=9,
                             show=False,
                             blocking=False
                             )

# Saving in an Excel file the unified dataframe metrics
all_metrics.to_excel(r".\results\final_results\metrics\all_metrics.xlsx",header=True, index=True)
# Saving in an Excel file each of the strategies metrics in different sheets
with pd.ExcelWriter(r".\results\final_results\metrics\all_metrics_by_strat.xlsx") as writer:
    for strat_key, df in all_metrics_by_strat.items():
        df.to_excel(writer, sheet_name=strat_key, header=True, index=True)

# End of project
# To Do
# 0) See if the code on GitHub runs (example_multiple_starts) (Matéo and Enzo, 12:30)
# 1) See if the individual charts in plots begin with non 0.0 returns (first date is a rebalancing date) (Tristan & Matéo)
# 2) see to account for transaction costs. Transactions costs = delta weights * x bps. Deduce that from the strategy returns in backtest. (Tristan & Matéo)
# 3) See why/correct/delete the few dates at the end where we have 0.0 returns (Tristan & Matéo)
# 4) Once all above is done, run main on real data (Mateo will do it)
# 5) Write the report (Enzo?)

# 6) Clean all the scripts and all the project. Ensure all scripts run well. (Matéo will do it)
