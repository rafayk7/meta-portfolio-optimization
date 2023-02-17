from enum import Enum
import numpy as np
import cvxpy as cp
import math

class Optimizers(Enum):
    MVO = "MVO"
    RP = "RP"
    DRRPW = "DRRPW"

def GetOptimalAllocation(mu, Q, technique=Optimizers.MVO):
    if technique == Optimizers.MVO:
        return MVO(mu,Q)
    if technique == Optimizers.RP:
        return RP(mu, Q)
    if technique == Optimizers.DRRPW:
        return DRRPW(mu, Q)


'''
Mean Variance Optimizer
Inputs: mu: numpy array, key: Symbol. value: return estimate
        Q: nxn Asset Covariance Matrix (n: # of assets)
Outputs: x: optimal allocations
'''

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
    print("RP Worked? {}".format(np.all(np.isclose(risk_contrib, risk_contrib[0]))))

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
    k = 2

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
    risk_contrib = np.multiply(x, Q.dot(x))
    print(risk_contrib)
    print("DRRPW Worked? {}".format(np.all(np.isclose(risk_contrib, risk_contrib[0]))))

    return x
