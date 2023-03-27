from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cvxpy as cp
from util import LoadData, generate_date_list, start, end, factors_list
from Optimizers import Optimizers, GetOptimalAllocation
from FactorModelling import GetParameterEstimates

def RunBacktest(path_to_data, opt_type, InitialValue=1000000, lookback = 30):
    returns, assets_list_cleaned, prices = LoadData(path_to_data)

    holdings = pd.DataFrame(columns=['date']+assets_list_cleaned)
    portVal = pd.DataFrame(columns=['date', 'Wealth'])

    dates = generate_date_list(returns, start=start, end=end)
    first = True

    # Make this weekly
    # Merge data structures

    for date in tqdm(dates):
        # Get Asset Prices for Today
        currentPrices = (prices[prices['date']==str(date)]
            .drop('date',axis=1)
            .values
            .flatten())
        
        # Update Portfolio Value
        if first:
            portVal.loc[len(portVal)] = [date] + [InitialValue]
            CurrentPortfolioValue = InitialValue
            first = False
        else:     
            CurrentPortfolioValue = np.dot(currentPrices,noShares)
            portVal.loc[len(portVal)] = [date] + [CurrentPortfolioValue]
            
        # We don't want the current date information, hence the lack of equality
        # Get last 30
        date = str(date)
        
        returns_lastn = returns[(returns['date'] < date)].tail(lookback)
        factor_returns = returns_lastn[factors_list]
        asset_returns = returns_lastn.drop(factors_list + ['date', 'RF'], axis=1)

        # net_train to get optimal delta
        # perform forward pass to get optimal portfolio

        mu, Q = GetParameterEstimates(asset_returns, factor_returns, log=False, bad=True)
        
        x = GetOptimalAllocation(mu, Q, opt_type)

        # Update Holdings
        holdings.loc[len(holdings)] = [date] + list(x)

        # Update shares held
        # 50% of 100k = 50k. If price is 100 we have 50,000/100=50 shares
        noShares = np.divide(x*CurrentPortfolioValue, currentPrices)
        print('Done {}'.format(date))
    
    portVal['date'] = pd.to_datetime(portVal['date'])
    portVal = portVal.merge(returns[['date','RF']], how='left', on='date')

    return holdings, portVal
