import pandas as pd
import numpy as np
from datetime import datetime

start, end = '2007-01-01', '2007-12-31'
factors_list = ['Mkt-RF', 'SMB', 'HML']

def LoadData(path_to_data):
    path_to_prices = r'{}\prices.csv'.format(path_to_data)
    path_to_factors = r'{}\3factors.csv'.format(path_to_data)

    prices = pd.read_csv(path_to_prices)
    factors = pd.read_csv(path_to_factors)

    assets_list = list(prices['symbol'].unique())

    assets_list_cleaned = [x for x in assets_list if str(x) != 'nan']

    pivot_prices = np.round(pd.pivot_table(prices, values='close', 
                                    index='date', 
                                    columns='symbol', 
                                    aggfunc=np.mean),2)
    pivot_prices = pivot_prices.reset_index()
    pivot_prices['date'] = pd.to_datetime(pivot_prices['date'])
    factors['date'] = pd.to_datetime(factors['Date'], format="%Y%m%d")

    pivot_prices = pivot_prices.set_index('date')
    returns = pivot_prices.pct_change()
    pivot_prices = pivot_prices.reset_index()
    returns = returns.reset_index()
    returns = returns.merge(factors, on='date', how='left')
    returns = returns.drop(['Date'], axis=1)
    returns = returns.dropna()

    return returns, assets_list_cleaned, pivot_prices

def generate_date_list(data, start, end):
    start = datetime.fromisoformat(start)
    end = datetime.fromisoformat(end)

    # Train model from start_date to date
    mask = (data['date'] >= start) & (data['date'] <= end)
    data = data.loc[mask]
    return data.date.apply(lambda x: x.date()).unique().tolist()

