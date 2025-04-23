import pandas as pd
import numpy as np
#import modules.packages.filters as filters
from abc import ABC, abstractmethod

class DataSource(ABC):
    """Abstract class to define the interface for the data source"""
    @abstractmethod
    def fetch_data(self):
        """Retrieve gross data from the data source"""
        pass


class ExcelDataSource(DataSource):
    """Class to fetch data from an Excel file"""
    def __init__(self, file_path:str="\\data\\data.xlsx", sheet_name:str="data", index_col:int=0):
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.index_col = index_col

    def fetch_data(self):
        """Retrieve gross data from excel file"""
        data = pd.read_excel(self.file_path, sheet_name=self.sheet_name, index_col=self.index_col)
        # Convert excel date to datetime
        # data.index = pd.to_datetime("1899-12-30") + pd.to_timedelta(data.index, unit="D")
        return data


class CSVDataSource(DataSource):
    """Class to fetch data from a CSV file"""
    def __init__(self, file_path:str="//data//data.csv", index_col:int=0, date_column=None):
        self.file_path = file_path
        self.index_col = index_col
        self.date_column = date_column

    def fetch_data(self):
        """Retrieve gross data from csv file"""
        data = pd.read_csv(self.file_path, index_col=self.index_col)
        if self.date_column:
            data.index = pd.to_datetime(data.index)
        return data


class DataManager:
    """Class to manage, clean and preprocess data"""
    def __init__(self, data_source:DataSource,
                 max_consecutive_nan:int=5,
                 rebase_prices:bool=False, n_implementation_lags:int=1):
        """

        :param data_source:
        :param max_consecutive_nan:
        :param rebase_prices:
        :param n_implementation_lags:
        """
        # Data loading
        self.data_source = data_source
        # Data cleaning
        self.max_consecutive_nan = max_consecutive_nan
        self.rebase_prices = rebase_prices
        self.n_implementation_lags = n_implementation_lags
        self.raw_data = None
        self.cleaned_data = None
        self.returns = None
        # Aligned data
        self.aligned_prices = None
        self.aligned_returns = None

    def load_data(self):
        """Load data from the data source"""
        self.raw_data = self.data_source.fetch_data()
        return self.raw_data

    def clean_data(self):
        """Clean the data by filling missing values"""
        if self.raw_data is None:
            self.load_data()

        if self.rebase_prices:
            df_filled = self.raw_data.copy()
            df_filled = (df_filled / df_filled.iloc[0,:])*100
        else:
            df_filled = self.raw_data.copy()

        for col in self.raw_data.columns:
            series = self.raw_data[col]
            first_valid_index = series.first_valid_index()

            if first_valid_index is None:
                continue

            is_nan = series.isna()
            counter = 0

            for i in series.index:
                if i < first_valid_index:
                    continue

                if is_nan[i]:
                    counter += 1
                    if counter <= self.max_consecutive_nan:
                        i_idx = series.index.get_loc(i)
                        df_filled.iloc[i_idx, self.raw_data.columns.get_loc(col)] = df_filled.iloc[
                            i_idx - 1, self.raw_data.columns.get_loc(col)]

                else:
                    counter = 0

        self.cleaned_data = df_filled
        return self.cleaned_data

    def compute_returns(self):
        """Compute returns from the cleaned data"""
        if self.cleaned_data is None:
            self.clean_data()

        self.returns = self.cleaned_data.pct_change(fill_method=None)
        self.returns.fillna(0.0)
        # self.returns = self.returns.iloc[1:, :]
        return self.returns

    def account_implementation_lags(self):
        #if (self.aligned_prices is None) and (self.aligned_returns is None):
        if self.aligned_returns is None:
            self.aligned_returns = self.returns.shift(-self.n_implementation_lags)
            #self.aligned_returns = self.aligned_returns.iloc[0:-1,:]

            #self.aligned_prices = self.cleaned_data.iloc[0:-1,:]
        else:
            pass

    def get_data(self):
        """Get all data prepared"""
        if self.returns is None:
            self.compute_returns()

        return {'raw_data' : self.raw_data,
                'cleaned_data' : self.cleaned_data,
                'returns' : self.returns
                }