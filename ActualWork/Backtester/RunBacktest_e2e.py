from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cvxpy as cp
from util import LoadData, generate_date_list, start, end, factors_list
from Optimizers import Optimizers, GetOptimalAllocation, drrpw_net
from FactorModelling import GetParameterEstimates
import PortfolioClasses as pc
import LossFunctions as lf
from torch.autograd import Variable
import torch
from mvo_learn_norm import mvo_norm_net

from torch.utils.data import DataLoader

def RunBacktest_e2e(path_to_data, opt_type, InitialValue=1000000, lookback = 30, datatype='broad'):
    returns, assets_list_cleaned, prices, factors = LoadData(path_to_data, e2e=True, datatype=datatype)
    holdings = pd.DataFrame(columns=['date']+assets_list_cleaned)
    portVal = pd.DataFrame(columns=['date', 'Wealth'])

    dates = generate_date_list(returns, prices, start=start, end=end)
    first = True

    # Subtract 1 from n_x and n_y since we have a date column
    n_x, n_y, n_obs, perf_period = factors.shape[1] - 1, returns.shape[1] - 1, 40, 10
    lookback = 52
    print("# Factors: {}. # Assets: {}".format(n_x, n_y))

    # Hyperparameters
    lr = 0.001
    epochs_per_date = 10

    if opt_type==Optimizers.CardinalityRP:
        cardinality=10

    # For replicability, set the random seed for the numerical experiments
    set_seed = 200

    if opt_type==Optimizers.MVONormTrained:
        net = mvo_norm_net(n_x, n_y, n_obs, train_pred=True, 
                learnT=((opt_type==Optimizers.DRRPWTTrained) or (opt_type==Optimizers.MVONormTrained)), learnDelta=(opt_type==Optimizers.DRRPWDeltaTrained), 
                set_seed=set_seed, opt_layer='nominal', T_Diagonal=(opt_type==Optimizers.DRRPWTTrained_Diagonal)).double()
    else:
        net = drrpw_net(n_x, n_y, n_obs, train_pred=True, 
                learnT=(
                        (opt_type==Optimizers.DRRPWTTrained) 
                        or (opt_type==Optimizers.MVONormTrained) 
                        or (opt_type==Optimizers.DRRPWTTrained_Diagonal)), 
                learnDelta=(opt_type==Optimizers.DRRPWDeltaTrained), 
                set_seed=set_seed, opt_layer='nominal', T_Diagonal=(opt_type==Optimizers.DRRPWTTrained_Diagonal), cache_path=path_to_data).double()

    delta_trained = []
    loss_values = []
    grad_values = []
    T_diagonals = []
    T_offdiagonals = []

    for date in dates:
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
        asset_returns = returns_lastn.drop('date', axis=1)

        factor_returns = factors[(factors['date'] < date)].tail(lookback)
        factor_returns = factor_returns.drop('date', axis=1)

        train_set = DataLoader(pc.SlidingWindow(factor_returns, asset_returns, n_obs, 
                                                perf_period))

        if opt_type == Optimizers.LinearEWAndRPOptimizer:
            pass
        # net_train to get optimal delta
        net.net_train(train_set, lr=lr, epochs=epochs_per_date)

        factor_ret_tensor = Variable(torch.tensor(factor_returns.values, dtype=torch.double))
        asset_ret_tensor = Variable(torch.tensor(asset_returns.values, dtype=torch.double))

        # perform forward pass to get optimal portfolio
        x_tensor, _ = net(factor_ret_tensor, asset_ret_tensor)
        x = x_tensor.detach().numpy().flatten()
        if opt_type == Optimizers.DRRPWDeltaTrained:
            delta_val = net.delta.item()
            delta_trained.append(delta_val)
            loss_values.append(net.curr_loss)
            grad_values.append(net.curr_gradient)
        elif opt_type in [Optimizers.DRRPWTTrained, Optimizers.DRRPWTTrained_Diagonal]:
            T_val = net.T.detach().numpy()
            print(np.diag(T_val))

        # mu, Q = GetParameterEstimates(asset_returns, factor_returns, log=False, bad=True)
        # x = GetOptimalAllocation(mu, Q, opt_type)

        # Update Holdings
        holdings.loc[len(holdings)] = [date] + list(x)

        # Update shares held
        # 50% of 100k = 50k. If price is 100 we have 50,000/100=50 shares
        print("x: {}. CurrentPortfolioValue: {}. currentPrices: {}".format(x, CurrentPortfolioValue, currentPrices))
        noShares = np.divide(x*CurrentPortfolioValue, currentPrices)
        print('Done {}'.format(date))
    
    portVal['date'] = pd.to_datetime(portVal['date'])
    portVal = portVal.merge(factors[['date','RF']], how='left', on='date')

    return holdings, portVal, [delta_trained, loss_values, grad_values]
