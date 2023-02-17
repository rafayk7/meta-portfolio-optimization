from RunBacktest import RunBacktest
from Optimizers import Optimizers
from Plotter import calculateSharpe,PlotWealth

path_to_data = r"C:\Users\Rafay\Documents\thesis\ActualWork\Data\results"
path_to_results = r"C:\Users\Rafay\Documents\thesis\ActualWork\Results"
InitialValue = 1000000 # $1,000,000
lookback = 30 # Number of days preceeding current date to train

opt_type = Optimizers.DRRPW

holdings, portVal = RunBacktest(path_to_data, opt_type)
SharpeRatio = calculateSharpe(portVal)
PlotWealth(portVal, path_to_results+'\{}\Wealth'.format(opt_type.value))

print(SharpeRatio)