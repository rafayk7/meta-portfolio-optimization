from RunBacktest import RunBacktest
from RunBacktest_e2e import RunBacktest_e2e
from Optimizers import Optimizers
from Plotter import calculateSharpe,PlotWealth
import pickle

path_to_data = r"C:\Users\Rafay\Documents\thesis3\thesis\ActualWork\e2e\cache"
path_to_results = r"C:\Users\Rafay\Documents\thesis3\thesis\ActualWork\Results"
InitialValue = 1000000 # $1,000,000
lookback = 52 # Number of days preceeding current date to train

datatype = 'cross_asset'

opt_try = [Optimizers.DRRPWDeltaTrained, Optimizers.DRRPWDeltaTrained]
# opt_try = [Optimizers.DRRPWDeltaTrained]
opt_try = [Optimizers.EW, Optimizers.RP, Optimizers.DRRPWDeltaTrained]

for opt_type in opt_try:
    print(opt_type.value)
    if opt_type in [Optimizers.DRRPWDeltaTrained, Optimizers.DRRPWTTrained, Optimizers.LearnMVOAndRP, Optimizers.MVONormTrained, Optimizers.DRRPWTTrained_Diagonal]:
        holdings, portVal, hyperparams = RunBacktest_e2e(path_to_data, opt_type, InitialValue=1000000, lookback = 30, datatype=datatype)
        if Optimizers.DRRPWDeltaTrained:
            with open(path_to_results + '{}_deltavals_{}.pkl'.format(opt_type.value, datatype), 'wb') as f:
                pickle.dump(hyperparams[0], f)
            with open(path_to_results + '{}_lossvals_{}.pkl'.format(opt_type.value, datatype), 'wb') as f:
                pickle.dump(hyperparams[1], f)
            with open(path_to_results + '{}_gradvals_{}.pkl'.format(opt_type.value, datatype), 'wb') as f:
                pickle.dump(hyperparams[2], f)
    else:
        holdings, portVal = RunBacktest(path_to_data, opt_type, InitialValue=1000000, lookback = 30, datatype=datatype)


    portVal.to_pickle(path_to_results + '{}_{}_value.pkl'.format(opt_type.value, datatype, lookback))
    holdings.to_pickle(path_to_results + '{}_{}_holdings.pkl'.format(opt_type.value, datatype, lookback))

    SharpeRatio = calculateSharpe(portVal)

    print('{} SR: '.format(opt_type.value, SharpeRatio))
