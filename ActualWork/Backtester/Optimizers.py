from enum import Enum
import numpy as np
import cvxpy as cp
import math
from util import nearestPD

import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import PortfolioClasses as pc
import LossFunctions as lf

from torch.utils.data import DataLoader


class Optimizers(Enum):
    MVO = "MVO"
    RobMVO = "RobMVO"
    RP = "RP"
    RP_Shrinkage = "RP_Shrinkage"
    DRRPW = "DRRPW"
    EW = "EW"
    DRRPWDeltaTrained = "DRRPWDeltaTrained"
    DRRPWTTrained = "DRRPWTTrained"
    CardinalityRP = "CardinalityRP"
    LearnMVOAndRP = "LearnMVOAndRP"
    MVONormTrained = "MVONormTrained"
    DRRPWTTrained_Diagonal = "DRRPWTTrained_Diagonal"
    LinearEWAndRPOptimizer = "LinearEWAndRPOptimizer"

def GetOptimalAllocation(mu, Q, technique=Optimizers.MVO, x=[]):
    if technique == Optimizers.MVO:
        return MVO(mu,Q)
    if technique in [Optimizers.RP, Optimizers.RP_Shrinkage]:
        return RP(mu, Q)
    if technique == Optimizers.DRRPW:
        return DRRPW(mu, Q)
    if technique == Optimizers.EW:
        return EW(mu, Q)
    if technique == Optimizers.RobMVO:
        return np.array(RobMVO(mu, Q, x))
    if technique == Optimizers.DRRPWDeltaTrained:
        print("Use Other Backtesting Function")

'''
Mean Variance Optimizer
Inputs: mu: numpy array, key: Symbol. value: return estimate
        Q: nxn Asset Covariance Matrix (n: # of assets)
Outputs: x: optimal allocations
'''

def EW(mu, Q):
    n = len(mu)

    return np.ones(n) / n

def MVO(mu,Q):
    
    # # of Assets
    n = len(mu)

    # Decision Variables
    w = cp.Variable(n)
    
    # Target Return for Constraint
    targetRet = np.mean(mu)
    
    constraints = [
        cp.sum(w) == 1, # Sum to 1
        mu.T @ w >= targetRet, # Target Return Constraint
        w>=0 # Disallow Short Sales
    ]

    # Objective Function
    risk = cp.quad_form(w, Q)

    prob = cp.Problem(cp.Minimize(risk), constraints=constraints)
    prob.solve()
    return w.value

import cvxpy as cp
import numpy as np
import numpy as np
from scipy.stats import chisquare
from scipy.stats import gmean
import cvxopt as opt
from cvxopt import matrix, spmatrix, sparse
from cvxopt.solvers import qp, options
from cvxopt import blas
import pandas as pd
options['show_progress'] = False
options['feastol'] = 1e-9

def RobMVO(mu,Q,x0):
    # Penalty on Turnover (very sensitive)
    c = 0
    # Penalty on variance
    lambd = 0.05
    # Pentalty on returns
    rpen = 1
    # Max weight of an asset
    max_weight = 1
    # between 0% and 200%
    turnover = 2
    #size of uncertainty set
    ep = 2

    T = np.shape(mu)[0]
    Theta = np.diag(np.diag(Q))/T
    sqrtTh = np.diag(matrix(np.sqrt(Theta)))
    n = len(Q)

    # Make Q work for abs value
    Q = matrix(np.block([[Q, np.zeros((n,n)), np.zeros((n,n))], [np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n))], [np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n))]]))

    # A and B
    b1 = np.ones([1,1])
    try:
        b2 = x0
        b = np.concatenate((b1,b2))
    except:
        b2 = matrix(x0)
        b = np.concatenate((b1,b2))


    A = matrix(np.block([[np.ones(n), c * np.ones(n), -c * np.ones(n)], [np.eye(n), np.eye(n), -np.eye(n)]]))
    

    # G and h
    G = matrix(0.0, (6 * n + 1, 3 * n))
    h = opt.matrix(0.0, (6 * n + 1, 1))
    for k in range(3 * n):
        # xi > 0 constraint
        G[k, k] = -1
    # xi > max_weight
        G[k + 3 * n, k] = 1
        h[k + 3 * n] = max_weight
    for k in range(2 * n):
        # sum dwi+ + dwi- < turnover
        G[6 * n, k + n] = 1

    h[6 * n] = turnover

    quad = lambd*Q

    r = matrix(np.block([rpen*np.array(mu) - ep*sqrtTh, -c * np.ones(2*n)]))

    return np.transpose(np.array(qp(matrix(quad), -1*matrix(r), matrix(G), matrix(h), matrix(A), matrix(b))['x'])[0:n])[0].tolist()
'''
Risk Parity Optimizer
Inputs: mu: numpy array, key: Symbol. value: return estimate
        Q: nxn Asset Covariance Matrix (n: # of assets)
Outputs: x: optimal allocations with equal risk contribution
'''

def RP(mu,Q):
    
    # # of Assets
    n = len(mu)

    # Decision Variables
    w = cp.Variable(n)

    # Kappa
    k = 2
          
    constraints = [
        w>=0 # Disallow Short Sales
    ]

    # Objective Function
    risk = cp.quad_form(w, Q)
    log_term = 0
    for i in range(n):
        log_term += cp.log(w[i])
    
    prob = cp.Problem(cp.Minimize(risk-(k*log_term)), constraints=constraints)
    
    # ECOS fails sometimes, if it does then do SCS
    try:
        prob.solve(verbose=False)
    except:
        prob.solve(solver='SCS',verbose=False)

    x = w.value
    x = np.divide(x, np.sum(x))

    # Check Risk Parity Condition is actually met
    risk_contrib = np.multiply(x, Q.dot(x))
    if not np.all(np.isclose(risk_contrib, risk_contrib[0])):
        print("RP did not work")

    return x

'''
Distributionally Robust Risk Parity With Wasserstein Distance Optimizer
Inputs: mu: numpy array, key: Symbol. value: return estimate
        Q: nxn Asset Covariance Matrix (n: # of assets)
Outputs: x: optimal allocations

Formula:
    \min_{\boldsymbol{\phi} \in \mathcal{X}} {(\sqrt{\boldsymbol{\phi}^T \Sigma_{\mathcal{P}}(R)\boldsymbol{\phi}} + \sqrt{\delta}||\boldsymbol{\phi}||_p)^2} - c\sum_{i=1}^n ln(y)

'''

def DRRPW(mu,Q):
    
    # # of Assets
    n = len(mu)

    # Decision Variables
    w = cp.Variable(n)

    # Kappa
    k = 100

    # Size of uncertainty set
    delta = 0.05

    # Norm for x
    p = 2

    constraints = [
        w>=0 # Disallow Short Sales
    ]

    # risk = cp.quad_form(w, Q)

    log_term = 0
    for i in range(n):
        log_term += cp.log(w[i])
    
    # We need to compute \sqrt{x^T Q x} intelligently because
    # cvxpy does not compute well with the \sqrt

    # To do this, I will take the Cholesky decomposition
    # Q = LL^T
    # Then, take the 2-norm of L*x

    # Idea: (L_1 * x_1)^2 = Q_1 x_1

    L = np.linalg.cholesky(Q)

    obj = cp.power(cp.norm(L@w,2) + math.sqrt(delta)*cp.norm(w, p),2)
    obj = obj - k*log_term

    prob = cp.Problem(cp.Minimize(obj), constraints=constraints)
    
    # ECOS fails sometimes, if it does then do SCS
    try:
        prob.solve(verbose=False)
    except:
        prob.solve(solver='SCS',verbose=False)
    
    x = w.value
    x = np.divide(x, np.sum(x))
    
    # Check Risk Parity Condition is actually met
    # Note: DRRPW will not meet RP, will meet a robust version of RP
    risk_contrib = np.multiply(x, Q.dot(x))
    print(risk_contrib)
    print("DRRPW Worked? {}".format(np.all(np.isclose(risk_contrib, risk_contrib[0]))))

    return x


def drrpw_nominal_learnDelta(n_y, n_obs, Q):
    # Variables
    phi = cp.Variable((n_y,1), nonneg=True)
    t = cp.Variable()
    
    # Size of uncertainty set
    delta = cp.Parameter(nonneg=True)
    # T = cp.Parameter((n_y, n_y), PSD=True)

    # Norm for x. TODO set this to be the Mahalanobis Norm
    p = 2

    # Kappa, dont need this to be trainable as the value of this doesnt really matter
    k = 2

    # We need to compute \sqrt{x^T Q x} intelligently because
    # cvxpy does not compute well with the \sqrt

    # To do this, I will take the Cholesky decomposition
    # Q = LL^T
    # Then, take the 2-norm of L*x

    # Idea: (L_1 * x_1)^2 = Q_1 x_1

    try:
        L = np.linalg.cholesky(Q)
    except:
        Q = nearestPD(Q)
        L = np.linalg.cholesky(Q)

    # Constraints
    constraints = [
        phi >= 0,
        t >= cp.power(cp.norm(L@phi, 2) + delta*cp.norm(phi, p),2)
        # t >= cp.power(cp.norm(L@phi, 2) + cp.norm(T@phi, 2),2)
    ]

    log_term = 0
    for i in range(n_y):
        log_term += cp.log(phi[i])

    # obj = cp.power(cp.norm(L@w, 2) + delta*cp.norm(w, p),2)
    # obj = cp.sum_squares(cp.norm(L@w, 2) + delta*cp.norm(w, p))
    # cp.quad_form(w, Q)
    # obj = cp.quad_form(w, Q) + 2*delta*cp.norm(w,2)*cp.norm(L@w, 2) + cp.norm(w,2)
    # print('using this one')
    # obj = 2*delta*cp.norm(w,2)*cp.norm(L@w_tilde, 2)
    obj = (t) - k*log_term

    # Objective function
    objective = cp.Minimize(obj)    

    # Construct optimization problem and differentiable layer
    problem = cp.Problem(objective, constraints=constraints)

    return CvxpyLayer(problem, parameters=[delta], variables=[phi, t])

def drrpw_nominal_learnT(n_y, n_obs, Q, isDiagonal=False):
    # Variables
    phi = cp.Variable((n_y,1), nonneg=True)
    t = cp.Variable()

    # Size of uncertainty set
    delta = cp.Parameter(nonneg=True)
    if isDiagonal:
        T_diag = cp.Parameter((n_y, 1), nonneg=True)
        T = cp.diag(T_diag)
        params = T_diag
    else:
        T = cp.Parameter((n_y, n_y), PSD=True)
        params = T

    # Norm for x. TODO set this to be the Mahalanobis Norm
    p = 2

    # Kappa, dont need this to be trainable as the value of this doesnt really matter
    k = 2

    # We need to compute \sqrt{x^T Q x} intelligently because
    # cvxpy does not compute well with the \sqrt

    # To do this, I will take the Cholesky decomposition
    # Q = LL^T
    # Then, take the 2-norm of L*x

    # Idea: (L_1 * x_1)^2 = Q_1 x_1

    try:
        L = np.linalg.cholesky(Q)
    except:
        Q = nearestPD(Q)
        L = np.linalg.cholesky(Q)

    # Constraints
    constraints = [
        phi >= 0.000000001,
        # t >= cp.power(cp.norm(L@phi, 2) + delta*cp.norm(phi, p),2)
        t >= cp.power(cp.norm(L@phi, 2) + cp.norm(T@phi, 2),2)
    ]

    log_term = 0
    for i in range(n_y):
        log_term += cp.log(phi[i])


    # obj = cp.power(cp.norm(L@w, 2) + delta*cp.norm(w, p),2)
    # obj = cp.sum_squares(cp.norm(L@w, 2) + delta*cp.norm(w, p))
    # cp.quad_form(w, Q)
    # obj = cp.quad_form(w, Q) + 2*delta*cp.norm(w,2)*cp.norm(L@w, 2) + cp.norm(w,2)
    # print('using this one')
    # obj = 2*delta*cp.norm(w,2)*cp.norm(L@w_tilde, 2)
    obj = (t) - k*log_term

    # Objective function
    objective = cp.Minimize(obj)    

    # Construct optimization problem and differentiable layer
    problem = cp.Problem(objective, constraints=constraints)

    return CvxpyLayer(problem, parameters=[params], variables=[phi, t])

class drrpw_net(nn.Module):
    """End-to-end Dist. Robust RP with Wasserstein Distance learning neural net module.
    """
    def __init__(self, n_x, n_y, n_obs, opt_layer='nominal', prisk='p_var', perf_loss='sharpe_loss',
                pred_model='linear', pred_loss_factor=0.5, perf_period=13, train_pred=True, learnT=False, learnDelta=True, set_seed=None, cache_path='cache/', T_Diagonal=False):
        """End-to-end learning neural net module

        This NN module implements a linear prediction layer 'pred_layer' and a DRO layer 
        'opt_layer' based on a tractable convex formulation from Ben-Tal et al. (2013). 'delta' and
        'gamma' are declared as nn.Parameters so that they can be 'learned'.

        Inputs
        net_train: Number of inputs (i.e., features) in the prediction model
        n_y: Number of outputs from the prediction model
        n_obs: Number of scenarios from which to calculate the sample set of residuals
        prisk: String. Portfolio risk function. Used in the opt_layer
        opt_layer: String. Determines which CvxpyLayer-object to call for the optimization layer
        perf_loss: Performance loss function based on out-of-sample financial performance
        pred_loss_factor: Trade-off between prediction loss function and performance loss function.
            Set 'pred_loss_factor=None' to define the loss function purely as 'perf_loss'
        perf_period: Number of lookahead realizations used in 'perf_loss()'
        train_pred: Boolean. Choose if the prediction layer is learnable (or keep it fixed)
        train_gamma: Boolean. Choose if the risk appetite parameter gamma is learnable
        train_delta: Boolean. Choose if the robustness parameter delta is learnable
        set_seed: (Optional) Int. Set the random seed for replicability

        Output
        drrpw_net: nn.Module object 
        """
        super(drrpw_net, self).__init__()

        # Set random seed (to be used for replicability of numerical experiments)
        if set_seed is not None:
            torch.manual_seed(set_seed)
            self.seed = set_seed

        self.n_x = n_x
        self.n_y = n_y
        self.n_obs = n_obs

        self.trainT = learnT
        self.isTDiagonal = T_Diagonal
        self.trainDelta = learnDelta

        # Upper/Lower Bound for Delta
        self.ub = 0.2
        self.lb = 0


        # Prediction loss function
        # if pred_loss_factor is not None:
        #     self.pred_loss_factor = pred_loss_factor
        #     self.pred_loss = torch.nn.MSELoss()
        # else:
        #     self.pred_loss = None
        
        self.pred_loss = None

        # Define performance loss
        self.perf_loss = lf.sharpe_loss

        # Number of time steps to evaluate the task loss
        self.perf_period = perf_period

        # Record the model design: nominal, base or DRO
        # Register 'delta' (ambiguity sizing parameter) for DR layer
        if self.trainDelta:
            self.delta = nn.Parameter(torch.FloatTensor(1).uniform_(self.lb, self.ub))
            self.delta.requires_grad = True
            self.delta_init = self.delta.item()

        self.model_type = 'dro'

        if self.trainT:                
            Sigma_k = torch.rand(self.n_y, self.n_y)
            Sigma_k = torch.mm(Sigma_k, Sigma_k.t())
            Sigma_k.add_(torch.eye(self.n_y))

            if self.isTDiagonal:
                Sigma_k = torch.rand(n_y, 1)
                # Sigma_k = torch.diag(Sigma_k)

            self.T = nn.Parameter(Sigma_k)
            self.T.requires_grad = True
            self.delta_init = 2

        # self.model_type = 'dro'

        # LAYER: Prediction model
        self.pred_model = pred_model
        if pred_model == 'linear':
            # Linear prediction model
            self.pred_layer = nn.Linear(n_x, n_y)
            self.pred_layer.weight.requires_grad = train_pred
            self.pred_layer.bias.requires_grad = train_pred
        
        # Store reference path to store model data
        self.cache_path = cache_path

        # Store initial model
        self.init_state_path = cache_path + self.model_type+'_initial_state_' + pred_model
        torch.save(self.state_dict(), self.init_state_path)

    #-----------------------------------------------------------------------------------------------
    # forward: forward pass of the e2e neural net
    #-----------------------------------------------------------------------------------------------
    def forward(self, X, Y):
        """
        Forward pass of the NN module

        The inputs 'X' are passed through the prediction layer to yield predictions 'Y_hat'. The
        residuals from prediction are then calcuclated as 'ep = Y - Y_hat'. Finally, the residuals
        are passed to the optimization layer to find the optimal decision z_star.

        Inputs
        X: Features. ([n_obs+1] x n_x) torch tensor with feature timeseries data
        Y: Realizations. (n_obs x n_y) torch tensor with asset timeseries data

        Other 
        ep: Residuals. (n_obs x n_y) matrix of the residual between realizations and predictions

        Outputs
        y_hat: Prediction. (n_y x 1) vector of outputs of the prediction layer
        z_star: Optimal solution. (n_y x 1) vector of asset weights
        """
        # Multiple predictions Y_hat from X
        # Y_hat = torch.stack([self.pred_layer(x_t) for x_t in X])

        # Calculate residuals and process them
        # y_hat = Y_hat[-1]
        # y_hat = torch.stack([])

        # Optimization solver arguments (from CVXPY for ECOS/SCS solver)
        # solver_args = {'solve_method': 'ECOS', 'max_iters': 2000000, 'abstol': 1e-7}

        solver_args = {'solve_method': 'SCS'}

        # Covariance Matrix
        Q = np.cov(Y.cpu().detach().numpy(), rowvar=False)


        # Optimization Layer
        # self.opt_layer = drrpw_nominal(n_y, n_obs, Q)

        # Optimize z per scenario
        # Determine whether nominal or dro model

        param = None
        if self.trainT:
            param = self.T
            self.opt_layer = drrpw_nominal_learnT(self.n_y, self.n_obs, Q, isDiagonal = self.isTDiagonal)
            d = 0
            
        if self.trainDelta:
            param = self.delta
            self.opt_layer = drrpw_nominal_learnDelta(self.n_y, self.n_obs, Q)
            d = 1
        z_star, _ = self.opt_layer(param, solver_args=solver_args)

        softmax = torch.nn.Softmax(dim=d)
        z_star = softmax(z_star)
        
        # z_star = np.divide(z_star, np.sum(z_star))
        
        return z_star, []

    #-----------------------------------------------------------------------------------------------
    # net_train: Train the e2e neural net
    #-----------------------------------------------------------------------------------------------
    def net_train(self, train_set, val_set=None, epochs=None, lr=None):
        """Neural net training module
        
        Inputs
        train_set: SlidingWindow object containing feaatures x, realizations y and performance
        realizations y_perf
        val_set: SlidingWindow object containing feaatures x, realizations y and performance
        realizations y_perf
        epochs: Number of training epochs
        lr: learning rate

        Output
        Trained model
        (Optional) val_loss: Validation loss
        """

        # Assign number of epochs and learning rate
        if epochs is None:
            epochs = self.epochs
        if lr is None:
            lr = self.lr

        print('training for {} epochs'.format(epochs))
        randomize = False
        if self.trainDelta and randomize:
            min_delta, max_delta = max(0, self.delta.item() - 0.1), min(1, self.delta.item() + 0.1)

            # Parameter to use existing delta param with noise, or upper/lower bound
            use_ma = True

            if use_ma:
                self.delta = nn.Parameter(torch.FloatTensor(1).uniform_(min_delta, max_delta))
            else:
                self.delta = nn.Parameter(torch.FloatTensor(1).uniform_(self.lb, self.ub))

        # Define the optimizer and its parameters
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Number of elements in training set
        n_train = len(train_set)

        loss_val = 0
        grad_val = 0

        # Train the neural network
        for epoch in range(epochs):
                
            # TRAINING: forward + backward pass
            train_loss = 0
            curr_grad = 0
            optimizer.zero_grad()
            
            for t, (x, y, y_perf) in enumerate(train_set):
                # Forward pass: predict and optimize
                z_star, y_hat = self(x.squeeze(), y.squeeze())

                # Loss function
                # print('---z_star---')
                # print(z_star)
                # print('---y_perf---')
                # print(y_perf)
                if self.trainDelta and randomize:
                    loss = (1/n_train) * self.perf_loss(z_star, y_perf.squeeze().float())
                else:
                    loss = (1/n_train) * self.perf_loss(z_star, y_perf.squeeze())
                
                # Backward pass: backpropagation
                loss.backward()

                # Accumulate loss of the fully trained model
                train_loss += loss.item()
                curr_grad += self.delta.grad
            loss_val += train_loss
            grad_val += curr_grad
            # Update parameters
            optimizer.step()

            # Ensure that gamma, delta > 0 after taking a descent step
            for name, param in self.named_parameters():
                if name=='gamma':
                    param.data.clamp_(0.0001)
                if name=='delta':
                    param.data.clamp_(min=0.0001, max=0.9999)

        self.curr_loss = loss_val
        self.curr_gradient = grad_val

        # Compute and return the validation loss of the model
        if val_set is not None:

            # Number of elements in validation set
            n_val = len(val_set)

            val_loss = 0
            with torch.no_grad():
                for t, (x, y, y_perf) in enumerate(val_set):

                    # Predict and optimize
                    z_val, y_val = self(x.squeeze(), y.squeeze())
                
                    # Loss function
                    loss = (1/n_val) * self.perf_loss(z_val, y_perf.squeeze())
                    
                    # Accumulate loss
                    val_loss += loss.item()

            return val_loss

    #-----------------------------------------------------------------------------------------------
    # net_cv: Cross validation of the e2e neural net for hyperparameter tuning
    #-----------------------------------------------------------------------------------------------
    def net_cv(self, X, Y, lr_list, epoch_list, n_val=4):
        """Neural net cross-validation module

        Inputs
        X: Features. TrainTest object of feature timeseries data
        Y: Realizations. TrainTest object of asset time series data
        epochs: number of training passes
        lr_list: List of candidate learning rates
        epoch_list: List of candidate number of epochs
        n_val: Number of validation folds from the training dataset
        
        Output
        Trained model
        """
        results = pc.CrossVal()
        X_temp = dl.TrainTest(X.train(), X.n_obs, [1, 0])
        Y_temp = dl.TrainTest(Y.train(), Y.n_obs, [1, 0])
        for epochs in epoch_list:
            for lr in lr_list:
                
                # Train the neural network
                print('================================================')
                print(f"Training E2E {self.model_type} model: lr={lr}, epochs={epochs}")
                
                val_loss_tot = []
                for i in range(n_val-1,-1,-1):

                    # Partition training dataset into training and validation subset
                    split = [round(1-0.2*(i+1),2), 0.2]
                    X_temp.split_update(split)
                    Y_temp.split_update(split)

                    # Construct training and validation DataLoader objects
                    train_set = DataLoader(pc.SlidingWindow(X_temp.train(), Y_temp.train(), 
                                                            self.n_obs, self.perf_period))
                    val_set = DataLoader(pc.SlidingWindow(X_temp.test(), Y_temp.test(), 
                                                            self.n_obs, self.perf_period))

                    # Reset learnable parameters gamma and delta
                    self.load_state_dict(torch.load(self.init_state_path))

                    if self.pred_model == 'linear':
                        # Initialize the prediction layer weights to OLS regression weights
                        X_train, Y_train = X_temp.train(), Y_temp.train()
                        X_train.insert(0,'ones', 1.0)

                        X_train = Variable(torch.tensor(X_train.values, dtype=torch.double))
                        Y_train = Variable(torch.tensor(Y_train.values, dtype=torch.double))
                    
                        Theta = torch.inverse(X_train.T @ X_train) @ (X_train.T @ Y_train)
                        Theta = Theta.T
                        del X_train, Y_train

                        with torch.no_grad():
                            self.pred_layer.bias.copy_(Theta[:,0])
                            self.pred_layer.weight.copy_(Theta[:,1:])

                    val_loss = self.net_train(train_set, val_set=val_set, lr=lr, epochs=epochs)
                    val_loss_tot.append(val_loss)

                    print(f"Fold: {n_val-i} / {n_val}, val_loss: {val_loss}")

                # Store results
                results.val_loss.append(np.mean(val_loss_tot))
                results.lr.append(lr)
                results.epochs.append(epochs)
                print('================================================')

        # Convert results to dataframe
        self.cv_results = results.df()
        self.cv_results.to_pickle(self.init_state_path+'_results.pkl')

        # Select and store the optimal hyperparameters
        idx = self.cv_results.val_loss.idxmin()
        self.lr = self.cv_results.lr[idx]
        self.epochs = self.cv_results.epochs[idx]

        # Print optimal parameters
        print(f"CV E2E {self.model_type} with hyperparameters: lr={self.lr}, epochs={self.epochs}")

    #-----------------------------------------------------------------------------------------------
    # net_roll_test: Test the e2e neural net
    #-----------------------------------------------------------------------------------------------
    def net_roll_test(self, X, Y, n_roll=4, lr=None, epochs=None):
        """Neural net rolling window out-of-sample test

        Inputs
        X: Features. ([n_obs+1] x n_x) torch tensor with feature timeseries data
        Y: Realizations. (n_obs x n_y) torch tensor with asset timeseries data
        n_roll: Number of training periods (i.e., number of times to retrain the model)
        lr: Learning rate for test. If 'None', the optimal learning rate is loaded
        epochs: Number of epochs for test. If 'None', the optimal # of epochs is loaded

        Output 
        self.portfolio: add the backtest results to the e2e_net object
        """

        # Declare backtest object to hold the test results
        portfolio = pc.backtest(len(Y.test())-Y.n_obs, self.n_y, Y.test().index[Y.n_obs:])

        # Store trained gamma and delta values 
        self.delta_trained = []

        # Store the squared L2-norm of the prediction weights and their difference from OLS weights
        if self.pred_model == 'linear':
            self.theta_L2 = []
            self.theta_dist_L2 = []

        # Store initial train/test split
        init_split = Y.split

        # Window size
        win_size = init_split[1] / n_roll

        split = [0, 0]
        t = 0
        for i in range(n_roll):

            print(f"Out-of-sample window: {i+1} / {n_roll}")

            split[0] = init_split[0] + win_size * i
            if i < n_roll-1:
                split[1] = win_size
            else:
                split[1] = 1 - split[0]

            X.split_update(split), Y.split_update(split)
            train_set = DataLoader(pc.SlidingWindow(X.train(), Y.train(), self.n_obs, 
                                                    self.perf_period))
            test_set = DataLoader(pc.SlidingWindow(X.test(), Y.test(), self.n_obs, 0))

            # Reset learnable parameters gamma and delta
            self.load_state_dict(torch.load(self.init_state_path))

            if self.pred_model == 'linear':
                # Initialize the prediction layer weights to OLS regression weights
                X_train, Y_train = X.train(), Y.train()
                X_train.insert(0,'ones', 1.0)

                X_train = Variable(torch.tensor(X_train.values, dtype=torch.double))
                Y_train = Variable(torch.tensor(Y_train.values, dtype=torch.double))
            
                Theta = torch.inverse(X_train.T @ X_train) @ (X_train.T @ Y_train)
                Theta = Theta.T
                del X_train, Y_train

                with torch.no_grad():
                    self.pred_layer.bias.copy_(Theta[:,0])
                    self.pred_layer.weight.copy_(Theta[:,1:])

            # Train model using all available data preceding the test window
            self.net_train(train_set, lr=lr, epochs=epochs)

            # Store trained values of gamma and delta
            self.delta_trained.append(self.delta.item())
            # self.delta_trained.append(2)

            # Store the squared L2 norm of theta and distance between theta and OLS weights
            if self.pred_model == 'linear':
                theta_L2 = (torch.sum(self.pred_layer.weight**2, axis=()) + 
                            torch.sum(self.pred_layer.bias**2, axis=()))
                theta_dist_L2 = (torch.sum((self.pred_layer.weight - Theta[:,1:])**2, axis=()) + 
                                torch.sum((self.pred_layer.bias - Theta[:,0])**2, axis=()))
                self.theta_L2.append(theta_L2)
                self.theta_dist_L2.append(theta_dist_L2)

            # Test model
            with torch.no_grad():
                for j, (x, y, y_perf) in enumerate(test_set):
                
                    # Predict and optimize
                    z_star, _ = self(x.squeeze(), y.squeeze())

                    # Store portfolio weights and returns for each time step 't'
                    portfolio.weights[t] = z_star.squeeze()
                    portfolio.rets[t] = y_perf.squeeze() @ portfolio.weights[t]
                    t += 1

        # Reset dataset
        X, Y = X.split_update(init_split), Y.split_update(init_split)

        # Calculate the portfolio statistics using the realized portfolio returns
        portfolio.stats()

        self.portfolio = portfolio

    #-----------------------------------------------------------------------------------------------
    # load_cv_results: Load cross validation results
    #-----------------------------------------------------------------------------------------------
    def load_cv_results(self, cv_results):
        """Load cross validation results

        Inputs
        cv_results: pd.dataframe containing the cross validation results

        Outputs
        self.lr: Load the optimal learning rate
        self.epochs: Load the optimal number of epochs
        """

        # Store the cross validation results within the object
        self.cv_results = cv_results

        # Select and store the optimal hyperparameters
        idx = cv_results.val_loss.idxmin()
        self.lr = cv_results.lr[idx]
        self.epochs = cv_results.epochs[idx]
from scipy.stats import gmean

class CardinalityLoss(nn.Module):
    def __init__(self, cardinality):
        super(CardinalityLoss, self).__init__()

        self.cardinality = cardinality

    def forward(self, output):
        penalty = 100
        a = np.array([0]*self.cardinality)
        a[-1] = -100
        b = np.array([penalty]*(len(output)-self.cardinality))
        penalty_vec = np.concatenate((b,a))
        penalty_tensor = torch.from_numpy(penalty_vec.astype('float64'))
        loss = torch.dot(output, penalty_tensor)
        return loss

class cardinality_rp(nn.Module):
    """End-to-end Dist. Robust RP with Wasserstein Distance learning neural net module.
    """
    def __init__(self, n_x, n_y, n_obs, cardinality=1, opt_layer='nominal', prisk='p_var', perf_loss='sharpe_loss',
                pred_model='linear', pred_loss_factor=0.5, perf_period=13, train_pred=True, learnT=False, learnDelta=True, set_seed=None, cache_path='cache/'):
        """End-to-end learning neural net module

        This NN module implements a linear prediction layer 'pred_layer' and a DRO layer 
        'opt_layer' based on a tractable convex formulation from Ben-Tal et al. (2013). 'delta' and
        'gamma' are declared as nn.Parameters so that they can be 'learned'.

        Inputs
        net_train: Number of inputs (i.e., features) in the prediction model
        n_y: Number of outputs from the prediction model
        n_obs: Number of scenarios from which to calculate the sample set of residuals
        prisk: String. Portfolio risk function. Used in the opt_layer
        opt_layer: String. Determines which CvxpyLayer-object to call for the optimization layer
        perf_loss: Performance loss function based on out-of-sample financial performance
        pred_loss_factor: Trade-off between prediction loss function and performance loss function.
            Set 'pred_loss_factor=None' to define the loss function purely as 'perf_loss'
        perf_period: Number of lookahead realizations used in 'perf_loss()'
        train_pred: Boolean. Choose if the prediction layer is learnable (or keep it fixed)
        train_gamma: Boolean. Choose if the risk appetite parameter gamma is learnable
        train_delta: Boolean. Choose if the robustness parameter delta is learnable
        set_seed: (Optional) Int. Set the random seed for replicability

        Output
        mvo_learnNorm: nn.Module object 
        """
        super(cardinality_rp, self).__init__()

        # Set random seed (to be used for replicability of numerical experiments)
        if set_seed is not None:
            torch.manual_seed(set_seed)
            self.seed = set_seed

        self.n_x = n_x
        self.n_y = n_y
        self.n_obs = n_obs

        self.trainT = learnT
        self.trainDelta = learnDelta

        # Prediction loss function
        # if pred_loss_factor is not None:
        #     self.pred_loss_factor = pred_loss_factor
        #     self.pred_loss = torch.nn.MSELoss()
        # else:
        #     self.pred_loss = None
        
        self.pred_loss = None

        # Define performance loss
        self.perf_loss = lf.sharpe_loss

        # Cardinality Loss
        self.cardinality_loss = CardinalityLoss(cardinality)

        # Number of time steps to evaluate the task loss
        self.perf_period = perf_period

        # Record the model design: nominal, base or DRO
        # Register 'delta' (ambiguity sizing parameter) for DR layer
        if self.trainDelta:
            ub = 0.02
            lb = 0
            self.delta = nn.Parameter(torch.FloatTensor(1).uniform_(lb, ub))
            self.delta.requires_grad = True
            self.delta_init = self.delta.item()

        self.model_type = 'dro'

        if self.trainT:
            Sigma_k = torch.rand(self.n_y, self.n_y)
            Sigma_k = torch.mm(Sigma_k, Sigma_k.t())
            Sigma_k.add_(torch.eye(self.n_y))
            
            self.T = nn.Parameter(Sigma_k)
            self.T.requires_grad = True
            self.delta_init = 2

        # self.model_type = 'dro'

        # LAYER: Prediction model
        self.pred_model = pred_model
        if pred_model == 'linear':
            # Linear prediction model
            self.pred_layer = nn.Linear(n_x, n_y)
            self.pred_layer.weight.requires_grad = train_pred
            self.pred_layer.bias.requires_grad = train_pred
        
        # Store reference path to store model data
        self.cache_path = cache_path

        # Store initial model
        self.init_state_path = cache_path + self.model_type+'_initial_state_' + pred_model
        torch.save(self.state_dict(), self.init_state_path)

    #-----------------------------------------------------------------------------------------------
    # forward: forward pass of the e2e neural net
    #-----------------------------------------------------------------------------------------------
    def forward(self, X, Y):
        """
        Forward pass of the NN module

        The inputs 'X' are passed through the prediction layer to yield predictions 'Y_hat'. The
        residuals from prediction are then calcuclated as 'ep = Y - Y_hat'. Finally, the residuals
        are passed to the optimization layer to find the optimal decision z_star.

        Inputs
        X: Features. ([n_obs+1] x n_x) torch tensor with feature timeseries data
        Y: Realizations. (n_obs x n_y) torch tensor with asset timeseries data

        Other 
        ep: Residuals. (n_obs x n_y) matrix of the residual between realizations and predictions

        Outputs
        y_hat: Prediction. (n_y x 1) vector of outputs of the prediction layer
        z_star: Optimal solution. (n_y x 1) vector of asset weights
        """
        # Multiple predictions Y_hat from X
        # Y_hat = torch.stack([self.pred_layer(x_t) for x_t in X])

        # Calculate residuals and process them
        # y_hat = Y_hat[-1]
        # y_hat = torch.stack([])

        # Optimization solver arguments (from CVXPY for ECOS/SCS solver)
        # solver_args = {'solve_method': 'ECOS', 'max_iters': 2000000, 'abstol': 1e-7}

        solver_args = {'solve_method': 'SCS'}

        # Covariance Matrix
        Q = np.cov(Y.cpu().detach().numpy(), rowvar=False)


        # Optimization Layer
        # self.opt_layer = drrpw_nominal(n_y, n_obs, Q)

        # Optimize z per scenario
        # Determine whether nominal or dro model

        param = None
        if self.trainT:
            param = self.T
            self.opt_layer = drrpw_nominal_learnT(self.n_y, self.n_obs, Q)
            d = 0
            
        if self.trainDelta:
            param = self.delta
            self.opt_layer = drrpw_nominal_learnDelta(self.n_y, self.n_obs, Q)
            d = 1
        z_star, _ = self.opt_layer(param, solver_args=solver_args)

        softmax = torch.nn.Softmax(dim=d)
        z_star = softmax(z_star)
        
        # z_star = np.divide(z_star, np.sum(z_star))
        
        return z_star, []

    #-----------------------------------------------------------------------------------------------
    # net_train: Train the e2e neural net
    #-----------------------------------------------------------------------------------------------
    def net_train(self, train_set, val_set=None, epochs=None, lr=None):
        """Neural net training module
        
        Inputs
        train_set: SlidingWindow object containing feaatures x, realizations y and performance
        realizations y_perf
        val_set: SlidingWindow object containing feaatures x, realizations y and performance
        realizations y_perf
        epochs: Number of training epochs
        lr: learning rate

        Output
        Trained model
        (Optional) val_loss: Validation loss
        """

        # Assign number of epochs and learning rate
        if epochs is None:
            epochs = self.epochs
        if lr is None:
            lr = self.lr

        print('training for {} epochs'.format(epochs))

        # Define the optimizer and its parameters
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Number of elements in training set
        n_train = len(train_set)

        # Train the neural network
        for epoch in range(epochs):
                
            # TRAINING: forward + backward pass
            train_loss = 0
            optimizer.zero_grad()
            
            for t, (x, y, y_perf) in enumerate(train_set):
                # Forward pass: predict and optimize
                z_star, y_hat = self(x.squeeze(), y.squeeze())

                # Loss function
                # print('---z_star---')
                # print(z_star)
                # print('---y_perf---')
                # print(y_perf)
                loss = (1/n_train) * self.perf_loss(z_star, y_perf.squeeze())
                
                # Backward pass: backpropagation
                loss.backward()

                # Accumulate loss of the fully trained model
                train_loss += loss.item()
        
            # Update parameters
            optimizer.step()

            # Ensure that gamma, delta > 0 after taking a descent step
            for name, param in self.named_parameters():
                if name=='gamma':
                    param.data.clamp_(0.0001)
                if name=='delta':
                    param.data.clamp_(min=0.0001, max=0.9999)

        # Compute and return the validation loss of the model
        if val_set is not None:

            # Number of elements in validation set
            n_val = len(val_set)

            val_loss = 0
            with torch.no_grad():
                for t, (x, y, y_perf) in enumerate(val_set):

                    # Predict and optimize
                    z_val, y_val = self(x.squeeze(), y.squeeze())
                
                    # Loss function
                    loss = (1/n_val) * self.perf_loss(z_val, y_perf.squeeze())
                    
                    # Accumulate loss
                    val_loss += loss.item()

            return val_loss

    #-----------------------------------------------------------------------------------------------
    # net_cv: Cross validation of the e2e neural net for hyperparameter tuning
    #-----------------------------------------------------------------------------------------------
    def net_cv(self, X, Y, lr_list, epoch_list, n_val=4):
        """Neural net cross-validation module

        Inputs
        X: Features. TrainTest object of feature timeseries data
        Y: Realizations. TrainTest object of asset time series data
        epochs: number of training passes
        lr_list: List of candidate learning rates
        epoch_list: List of candidate number of epochs
        n_val: Number of validation folds from the training dataset
        
        Output
        Trained model
        """
        results = pc.CrossVal()
        X_temp = dl.TrainTest(X.train(), X.n_obs, [1, 0])
        Y_temp = dl.TrainTest(Y.train(), Y.n_obs, [1, 0])
        for epochs in epoch_list:
            for lr in lr_list:
                
                # Train the neural network
                print('================================================')
                print(f"Training E2E {self.model_type} model: lr={lr}, epochs={epochs}")
                
                val_loss_tot = []
                for i in range(n_val-1,-1,-1):

                    # Partition training dataset into training and validation subset
                    split = [round(1-0.2*(i+1),2), 0.2]
                    X_temp.split_update(split)
                    Y_temp.split_update(split)

                    # Construct training and validation DataLoader objects
                    train_set = DataLoader(pc.SlidingWindow(X_temp.train(), Y_temp.train(), 
                                                            self.n_obs, self.perf_period))
                    val_set = DataLoader(pc.SlidingWindow(X_temp.test(), Y_temp.test(), 
                                                            self.n_obs, self.perf_period))

                    # Reset learnable parameters gamma and delta
                    self.load_state_dict(torch.load(self.init_state_path))

                    if self.pred_model == 'linear':
                        # Initialize the prediction layer weights to OLS regression weights
                        X_train, Y_train = X_temp.train(), Y_temp.train()
                        X_train.insert(0,'ones', 1.0)

                        X_train = Variable(torch.tensor(X_train.values, dtype=torch.double))
                        Y_train = Variable(torch.tensor(Y_train.values, dtype=torch.double))
                    
                        Theta = torch.inverse(X_train.T @ X_train) @ (X_train.T @ Y_train)
                        Theta = Theta.T
                        del X_train, Y_train

                        with torch.no_grad():
                            self.pred_layer.bias.copy_(Theta[:,0])
                            self.pred_layer.weight.copy_(Theta[:,1:])

                    val_loss = self.net_train(train_set, val_set=val_set, lr=lr, epochs=epochs)
                    val_loss_tot.append(val_loss)

                    print(f"Fold: {n_val-i} / {n_val}, val_loss: {val_loss}")

                # Store results
                results.val_loss.append(np.mean(val_loss_tot))
                results.lr.append(lr)
                results.epochs.append(epochs)
                print('================================================')

        # Convert results to dataframe
        self.cv_results = results.df()
        self.cv_results.to_pickle(self.init_state_path+'_results.pkl')

        # Select and store the optimal hyperparameters
        idx = self.cv_results.val_loss.idxmin()
        self.lr = self.cv_results.lr[idx]
        self.epochs = self.cv_results.epochs[idx]

        # Print optimal parameters
        print(f"CV E2E {self.model_type} with hyperparameters: lr={self.lr}, epochs={self.epochs}")

    #-----------------------------------------------------------------------------------------------
    # net_roll_test: Test the e2e neural net
    #-----------------------------------------------------------------------------------------------
    def net_roll_test(self, X, Y, n_roll=4, lr=None, epochs=None):
        """Neural net rolling window out-of-sample test

        Inputs
        X: Features. ([n_obs+1] x n_x) torch tensor with feature timeseries data
        Y: Realizations. (n_obs x n_y) torch tensor with asset timeseries data
        n_roll: Number of training periods (i.e., number of times to retrain the model)
        lr: Learning rate for test. If 'None', the optimal learning rate is loaded
        epochs: Number of epochs for test. If 'None', the optimal # of epochs is loaded

        Output 
        self.portfolio: add the backtest results to the e2e_net object
        """

        # Declare backtest object to hold the test results
        portfolio = pc.backtest(len(Y.test())-Y.n_obs, self.n_y, Y.test().index[Y.n_obs:])

        # Store trained gamma and delta values 
        self.delta_trained = []

        # Store the squared L2-norm of the prediction weights and their difference from OLS weights
        if self.pred_model == 'linear':
            self.theta_L2 = []
            self.theta_dist_L2 = []

        # Store initial train/test split
        init_split = Y.split

        # Window size
        win_size = init_split[1] / n_roll

        split = [0, 0]
        t = 0
        for i in range(n_roll):

            print(f"Out-of-sample window: {i+1} / {n_roll}")

            split[0] = init_split[0] + win_size * i
            if i < n_roll-1:
                split[1] = win_size
            else:
                split[1] = 1 - split[0]

            X.split_update(split), Y.split_update(split)
            train_set = DataLoader(pc.SlidingWindow(X.train(), Y.train(), self.n_obs, 
                                                    self.perf_period))
            test_set = DataLoader(pc.SlidingWindow(X.test(), Y.test(), self.n_obs, 0))

            # Reset learnable parameters gamma and delta
            self.load_state_dict(torch.load(self.init_state_path))

            if self.pred_model == 'linear':
                # Initialize the prediction layer weights to OLS regression weights
                X_train, Y_train = X.train(), Y.train()
                X_train.insert(0,'ones', 1.0)

                X_train = Variable(torch.tensor(X_train.values, dtype=torch.double))
                Y_train = Variable(torch.tensor(Y_train.values, dtype=torch.double))
            
                Theta = torch.inverse(X_train.T @ X_train) @ (X_train.T @ Y_train)
                Theta = Theta.T
                del X_train, Y_train

                with torch.no_grad():
                    self.pred_layer.bias.copy_(Theta[:,0])
                    self.pred_layer.weight.copy_(Theta[:,1:])

            # Train model using all available data preceding the test window
            self.net_train(train_set, lr=lr, epochs=epochs)

            # Store trained values of gamma and delta
            self.delta_trained.append(self.delta.item())
            # self.delta_trained.append(2)

            # Store the squared L2 norm of theta and distance between theta and OLS weights
            if self.pred_model == 'linear':
                theta_L2 = (torch.sum(self.pred_layer.weight**2, axis=()) + 
                            torch.sum(self.pred_layer.bias**2, axis=()))
                theta_dist_L2 = (torch.sum((self.pred_layer.weight - Theta[:,1:])**2, axis=()) + 
                                torch.sum((self.pred_layer.bias - Theta[:,0])**2, axis=()))
                self.theta_L2.append(theta_L2)
                self.theta_dist_L2.append(theta_dist_L2)

            # Test model
            with torch.no_grad():
                for j, (x, y, y_perf) in enumerate(test_set):
                
                    # Predict and optimize
                    z_star, _ = self(x.squeeze(), y.squeeze())

                    # Store portfolio weights and returns for each time step 't'
                    portfolio.weights[t] = z_star.squeeze()
                    portfolio.rets[t] = y_perf.squeeze() @ portfolio.weights[t]
                    t += 1

        # Reset dataset
        X, Y = X.split_update(init_split), Y.split_update(init_split)

        # Calculate the portfolio statistics using the realized portfolio returns
        portfolio.stats()

        self.portfolio = portfolio

    #-----------------------------------------------------------------------------------------------
    # load_cv_results: Load cross validation results
    #-----------------------------------------------------------------------------------------------
    def load_cv_results(self, cv_results):
        """Load cross validation results

        Inputs
        cv_results: pd.dataframe containing the cross validation results

        Outputs
        self.lr: Load the optimal learning rate
        self.epochs: Load the optimal number of epochs
        """

        # Store the cross validation results within the object
        self.cv_results = cv_results

        # Select and store the optimal hyperparameters
        idx = cv_results.val_loss.idxmin()
        self.lr = cv_results.lr[idx]
        self.epochs = cv_results.epochs[idx]