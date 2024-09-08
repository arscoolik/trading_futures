import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import openpyxl

import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from statsmodels.discrete.discrete_model import MNLogit
from statsmodels.stats.outliers_influence import variance_inflation_factor

from scipy.signal import argrelmax, argrelmin


class Paraming:

    def __init__(
            self, 
            currency_pair: str, 
            local_period: str,
            timeframe: str,
            extr_window: int, 
            big_candle_size: float, 
            start: str,
            end: str,
            quotes_path: str='{}_SCOPE.csv',
    ):
        self.threshold = 0.1
        self.currency_pair = currency_pair
        self.local_period = 'BME' if local_period in ['Month', 'month', 'M', 'm', 'Monthly', 'monthly'] else 'D'
        self.extr_window = extr_window
        self.big_candle_size = big_candle_size
        self.start = start
        self.end = end
        self.quotes_path = quotes_path.format(currency_pair)

        self.quotes = None
        self.data = None
        self.dd = None


    def load_quotes(self):
        '''Loads OHLCV quotes from file or MetaTrader5 terminal'''

        # self.quotes = export_mt5_quotes(ticker=f'{self.currency_pair}+', start=self.start, end=self.end, timeframe=self.timeframe, login=login, password=password, server=server)
        self.quotes = pd.read_csv(self.quotes_path, sep='\t')
        self.quotes.index = pd.to_datetime(self.quotes['<DATE>'] + ' ' + self.quotes['<TIME>'], format='%Y.%m.%d %H:%M:%S')
        self.quotes.drop(['<DATE>', '<TIME>'], axis=1, inplace=True)
        self.quotes.index.name = None
        self.quotes.columns = ['open', 'high', 'low', 'close', 'tick_volume', 'real_volume', 'spread']
        self.quotes['delta'] = self.quotes['close'] - self.quotes['open']


    def extremas(self, series: pd.Series, with_indices: bool=False) -> tuple:
        '''Returns number of extremas and max up and down price movements between them in the series'''

        imin = argrelmin(series.values, order=self.extr_window)[0]
        imax = argrelmax(series.values, order=self.extr_window)[0]

        ixtr = np.sort(np.concatenate((imax, imin)))
        n = len(ixtr)

        maxDown, maxUp = (100 * series.iloc[ixtr].pct_change().quantile(q=[0,1])).round(2)

        if with_indices:
            return imax, imin, n, maxDown, maxUp
        else:
            return n, maxDown, maxUp


    def plot_extremas(self):
        '''Plots series with corresponding extremas'''
        series = self.quotes['close']
        imax, imin, n, maxDown, maxUp = self.extremas(series, with_indices=True)
        print(n, maxDown, maxUp)
        plt.plot(series.values, label='Close', color='gray')
        plt.scatter(imax, series.iloc[imax], label='Max', marker='^', color='green', s=30, zorder=5)
        plt.scatter(imin, series.iloc[imin], label='Min', marker='v', color='red', s=30, zorder=5) 
        plt.legend()
        plt.show()   


    def big_candles(self, series: pd.Series) -> int:
        '''Returns number of big candles greater than threshold value'''

        return (np.abs(series.pct_change()) >= self.big_candle_size).sum()


    def rsd(self, series: pd.Series) -> float:
        '''Returns relative standard deviation of series'''

        return series.std() / series.mean() * 100


    def gen_df(self):
        '''Generates dataframe with profit and params'''

        params = self.quotes.groupby(pd.Grouper(freq=self.local_period)).agg({
            'close': ['min', 'max', self.rsd, self.big_candles, self.extremas],
            'tick_volume': 'mean',
            'delta': 'sum'
            })

        params.columns = params.columns.droplevel(0)
        params[['extremas_num', 'maxDown', 'maxUp']] = pd.DataFrame(params['extremas'].to_list(), index=params.index)
        params.drop('extremas', axis=1, inplace=True)
        params.rename(columns={'mean': 'tick_vol_mean'}, inplace=True)
        self.data = params.reset_index()
        self.data[['rsd', 'tick_vol_mean']] = self.data[[ 'rsd', 'tick_vol_mean']].round(2)
        self.data.rename(columns={
            'Gross_Profit_per_Net': 'Gross Profit per Net',
            'Mean_Max_DD': 'Mean Max DD',
            'rsd': 'Relative Standard Deviation',
            'tick_vol_mean': 'Mean Tick Volume'
        }, inplace=True)


    def ols_summary(self) -> dict:
        '''Returns dictionary with fitted OLS model and VIF test'''

        X = self.data.drop('Return', axis=1)
        Y = self.data['Return']

        # StandardScaler is better for models under assumption of Gaussian distribution
        scaler = StandardScaler().set_output(transform='pandas')
        X = scaler.fit_transform(X)
        X = sm.add_constant(X)

        ols = sm.OLS(Y, X).fit()

        vif_data = pd.DataFrame()
        vif_data['feature'] = X.columns
        vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

        return {
            'model': ols,
            'vif': vif_data
        }
    
    def run(self):
        self.load_quotes()
        self.gen_df()

if __name__ == '__main__':

    currency_pairs = ['XRPUSD']
    local_periods = ['Week', 'Month', 'Day']

    for currency_pair in currency_pairs:

        tables = []

        for local_period in local_periods:

            myclass = Paraming(
                currency_pair=currency_pair,
                local_period=local_period,
                timeframe='H1',
                extr_window=12,
                big_candle_size=0.003,
                start='2019-07-01',
                end='2024-07-01'
            )
            myclass.run()
            tables.append(myclass.data)
        with pd.ExcelWriter(f'{currency_pair}.xlsx', engine='openpyxl') as writer:
            for table, local_period in zip(tables, local_periods):
                table.to_excel(writer, sheet_name=local_period, index=False)


