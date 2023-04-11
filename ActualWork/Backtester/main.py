from RunBacktest import RunBacktest
from RunBacktest_e2e import RunBacktest_e2e
from Optimizers import Optimizers
from Plotter import calculateSharpe,PlotWealth
import pickle

path_to_data = r"C:\Users\Rafay\Documents\thesis\ActualWork\e2e\cache"
path_to_results = r"C:\Users\Rafay\Documents\thesis\ActualWork\Results"
InitialValue = 1000000 # $1,000,000
lookback = 30 # Number of days preceeding current date to train

datatype = 'cross_asset'

opt_try = [Optimizers.EW, Optimizers.RP, Optimizers.DRRPWDeltaTrained]
opt_try = [Optimizers.DRRPWDeltaTrained]
opt_try = [Optimizers.MVO, Optimizers.MVONormTrained, Optimizers.RobMVO]

for opt_type in opt_try:
    print(opt_type.value)
    if opt_type in [Optimizers.DRRPWDeltaTrained, Optimizers.DRRPWTTrained, Optimizers.LearnMVOAndRP, Optimizers.MVONormTrained]:
        holdings, portVal, delta_trained = RunBacktest_e2e(path_to_data, opt_type, InitialValue=1000000, lookback = 30, datatype=datatype)
        if Optimizers.DRRPWDeltaTrained:
            with open(path_to_results + '{}_deltavals_{}.pkl'.format(opt_type.value, datatype), 'wb') as f:
                pickle.dump(delta_trained, f)
    else:
        holdings, portVal = RunBacktest(path_to_data, opt_type, InitialValue=1000000, lookback = 30, datatype=datatype)


    portVal.to_pickle(path_to_results + '{}_{}_value.pkl'.format(opt_type.value, datatype))
    holdings.to_pickle(path_to_results + '{}_{}_holdings.pkl'.format(opt_type.value, datatype))

    SharpeRatio = calculateSharpe(portVal)

    print('{} SR: '.format(opt_type.value, SharpeRatio))