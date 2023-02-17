# Distributionally Robust Risk Parity Portfolios

The only certainty in the world of investment management is the inevitability of uncertainty in information.
There is a large body of research in the field of portfolio optimization with the goal of ingesting estimated
(and thus, uncertain) information to make optimal investment decisions.
Mean Variance Optimization (MVO), pioneered by Markowitz in 1952, was the first such attempt to create
mathematically optimal investment decisions [1]. This technique aims to balance the trade-off between
risk and reward, subject to the investor’s needs. Contrary to MVO, Risk Parity (RP) portfolios have the
objective of ensuring an equal risk contribution from each asset, with no explicit reward goal [2]. The
uncertainty of the risk and reward measures introduce significant errors making the output of RP sensitive
to input data. Recent advances in Distributionally Robust Optimization have led to Distributionally Robust
Risk Parity (DRRP), which introduces an uncertainty set around the probability distribution of historical
observations used as inputs to RP [3]. This allows the RP model to be less sensitive to errors in the input
estimates.