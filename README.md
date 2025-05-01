# Project_Asset_Management
Refers to the document Asset_Management.pdf to see the project report.

This project aims was to buil an architecture to backtest multiple strats at a time and compare them. More precisely, we backtested 144 strategy-hyperparameters couples.
The 6 main strategies are: cross-sectional momentum long-only, cross-sectional idiosyncratic momentum long only, cross-sectional reversal long only, cross-sectional idiosyncratic reversal long only, cross-sectional sharpe ratio long only and cross-sectional idiosyncratic sharpe ratio long only. Data and Universe is MSCI World monthly data from April 2007 to Dec 2024. 
The hyperparameters are: 
1) percentiles: top10%, top20%, top25%
2) industry segmentation: AllIndustries (cross-sectional secuity selection across all industries) or BestInIndustries (selects best securities within each industry)
3) rebalancing frequency: monthly, quarterly, semi-annualy, yearly

The code main.py (which can be long to run ~2h) generates the results and store them in .\results.

Example of final results outputs (.\results\final_results\cumulative_performance):
![Texte alternatif](/results/final_results/cumulative_performance/all_strategies_cumulative_returns.png)

Note: excel files all_metrics and all_metrics_by_strat in .\results\final_results\metrics are generated when running the script main. all_metrics_by_strat_formatted is not but it's just a more friendly formatted version of all_mertics_by_strat which looks like this:
![Texte alternatif](/results/final_results/metrics/cs_sr_lo.png)
