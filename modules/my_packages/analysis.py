import pandas as pd
import numpy as np
from typing import Union

class PerformanceAnalyser:
    """Class to analyse the performance of a strategy"""
    def __init__(self, portfolio_returns:pd.DataFrame, freq:str, zscores:Union[None, pd.DataFrame]=None,
                 forward_returns:Union[None, pd.DataFrame]=None):
        self.portfolio_returns = portfolio_returns
        self.cumulative_performance = None
        self.equity_curve = None
        if freq not in ['d', 'w', 'm', 'y']:
            raise ValueError("freq must be in ['d', 'w', 'm', 'y'].")
        self.freq = freq

        if (zscores is not None) and isinstance(zscores, pd.DataFrame):
            self.zscores = zscores
        elif (zscores is not None) and not isinstance(zscores, pd.DataFrame):
            raise ValueError("zscores must be a pandas dataframe.")
        else:
            pass

        if (forward_returns is not None) and isinstance(forward_returns, pd.DataFrame):
            self.forward_returns = forward_returns
        elif (forward_returns is not None) and not isinstance(forward_returns, pd.DataFrame):
            raise ValueError("forward_returns must be a pandas dataframe.")
        else:
            pass


    def compute_cumulative_performance(self, compound_type:str="geometric"):
        """Compute the cumulative performance of the strategy"""
        if compound_type == "geometric":
            self.cumulative_performance = (1 + self.portfolio_returns).cumprod() - 1
        elif compound_type == "arithmetic":
            self.cumulative_performance = self.portfolio_returns.cumsum()
        else:
            raise ValueError("Compound type not supported")

        return self.cumulative_performance

    def compute_equity_curve(self):
        """Compute the equity curve of the strategy"""
        self.equity_curve = self.compute_cumulative_performance(compound_type="arithmetic")
        return self.equity_curve

    def compute_information_coefficient(self, ic_type:str='not_ranked',
                                        percentiles:Union[None, tuple]=None):
        """
        This functions returns a time-series of the information coefficient ('gross', i.e. not a percentage)
        :param ic_type: 'not_ranked' returns the ic of the values themselves whereas 'ranked' uses the ranks of the values
        :param percentiles: if set to None, uses all the data, if set to a tuple of int, e.g. (10,90) skip the "middle"
        part and only computes the ic of the extremities.
        :return: a time-series of the ic
        """
        if self.zscores is None or not isinstance(self.zscores, pd.DataFrame):
            raise ValueError("To compute the information coefficient, you must create PerformanceAnalyser object with zscores as a pandas dataframe. ")
        if self.forward_returns is None or not isinstance(self.forward_returns, pd.DataFrame):
            raise ValueError("To compute the information coefficient, you must create PerformanceAnalyser object with forward_returns as a pandas dataframe. ")
        if ic_type not in ['not_ranked', 'ranked']:
            raise ValueError("ic_type must be either 'not_ranked' or 'ranked' (string).")
        if percentiles is None:
            pass
        elif percentiles is not None:
            if not (isinstance(percentiles, tuple) and len(percentiles) == 2 and all(
                    isinstance(x, (int, float)) for x in percentiles)):
                raise ValueError("percentiles must be a tuple of exactly two elements, containing only int and float.")

        ic = pd.DataFrame(data=np.nan, index=self.zscores.index, columns=['information_coefficient'])
        first_date = self.zscores.first_valid_index()
        zs_truncated = self.zscores.loc[first_date:,:]
        for date in zs_truncated.index:
            if zs_truncated.loc[date, :].to_frame().T.isna().all().all():
                continue
            if ic_type == 'not_ranked':
                if percentiles is not None:
                    rank_temp = zs_truncated.loc[date, :].to_frame().T # we do not take the rank (this is just a variable name)
                    fwd_ret_temp = self.forward_returns.loc[date, :]

                    # Only evaluates the IC for the percentiles
                    upper_bound = np.nanpercentile(rank_temp,
                                                   q=percentiles[1]) if not rank_temp.dropna().empty else np.nan
                    # Format to ease comparison after
                    upper_bound = pd.DataFrame(data=np.tile(upper_bound, (1, rank_temp.shape[1])),
                                               index=rank_temp.index,
                                               columns=rank_temp.columns)

                    lower_bound = np.nanpercentile(rank_temp,
                                                   q=percentiles[0]) if not rank_temp.dropna().empty else np.nan
                    # Format to ease comparison after
                    lower_bound = pd.DataFrame(data=np.tile(lower_bound, (1, rank_temp.shape[1])),
                                               index=rank_temp.index,
                                               columns=rank_temp.columns)

                    # Percentiles selection for the ic computation
                    mask = (rank_temp >= upper_bound) | (rank_temp <= lower_bound)
                    cols_to_keep = mask.columns[mask.iloc[0, :]]
                    pct_rank = rank_temp[cols_to_keep].values.T
                    pct_fwd_ret = fwd_ret_temp[cols_to_keep].values[:, None]

                    # Store
                    corr_temp = np.corrcoef(pct_rank.ravel(), pct_fwd_ret.ravel())[0][1]
                    ic.loc[date, :] = corr_temp
                else:
                    corr_temp = np.corrcoef( zs_truncated.loc[date,:] , self.forward_returns.loc[date,:] )[0][1]
                    ic.loc[date,:] = corr_temp

            elif ic_type == 'ranked':
                if percentiles is not None:
                    rank_temp = zs_truncated.loc[date, :].rank(ascending=True).to_frame().T
                    fwd_ret_temp = self.forward_returns.loc[date, :]

                    # Only evaluates the IC for the percentiles
                    upper_bound = np.nanpercentile(rank_temp,
                                                   q=percentiles[1]) if not rank_temp.dropna().empty else np.nan
                    # Format to ease comparison after
                    upper_bound = pd.DataFrame(data=np.tile(upper_bound, (1, rank_temp.shape[1])),
                                               index=rank_temp.index,
                                               columns=rank_temp.columns)

                    lower_bound = np.nanpercentile(rank_temp,
                                                   q=percentiles[0]) if not rank_temp.dropna().empty else np.nan
                    # Format to ease comparison after
                    lower_bound = pd.DataFrame(data=np.tile(lower_bound, (1, rank_temp.shape[1])),
                                               index=rank_temp.index,
                                               columns=rank_temp.columns)

                    # Percentiles selection for the ic computation
                    mask = (rank_temp >= upper_bound) | (rank_temp <= lower_bound)
                    cols_to_keep = mask.columns[mask.iloc[0,:]]
                    pct_rank = rank_temp[cols_to_keep].values.T
                    pct_fwd_ret = fwd_ret_temp[cols_to_keep].values[:,None]

                    # Store
                    corr_temp = np.corrcoef(pct_rank.ravel(), pct_fwd_ret.ravel())[0][1]
                    ic.loc[date, :] = corr_temp
                else:
                    rank = zs_truncated.loc[date,:].rank(ascending=True)
                    corr_temp = np.corrcoef( rank, self.forward_returns.loc[date,:] )[0][1]
                    ic.loc[date,:] = corr_temp
            else:
                raise ValueError("ic_type must be either 'not_ranked' or 'ranked' (string).")

        return ic

    def compute_metrics(self):
        """Compute the performance metrics of the strategy"""
        freq_mapping = {'d': 252, 'w': 52, 'm': 12, 'y': 1}

        if self.freq in freq_mapping:
            freq_num = freq_mapping[self.freq]
        else:
            raise ValueError(f"Invalid frequency '{self.freq}'. Expected one of {list(freq_mapping.keys())}.")

        if self.cumulative_performance is None:
            self.compute_cumulative_performance()

        # Compute basic performance metrics
        total_return = self.cumulative_performance.iloc[-1, 0]
        annualized_return = (1 + total_return) ** (freq_num / len(self.portfolio_returns)) - 1
        volatility = self.portfolio_returns.std() * np.sqrt(freq_num)
        sharpe_ratio = annualized_return / volatility

        # Maximum drawdown
        rolling_max = self.cumulative_performance.cummax()
        drawdown = (self.cumulative_performance / rolling_max) - 1
        max_drawdown = drawdown.min()

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
