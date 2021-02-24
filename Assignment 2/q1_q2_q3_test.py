import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
from metrics import *

np.random.seed(42)

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))
print("Non-Vectorised")
for fit_intercept in [True, False]:
    LR = LinearRegression(fit_intercept=fit_intercept)
    LR.fit_non_vectorised(X,y,5)
    y_hat = LR.predict(X)
    print('RMSE: ', rmse(y_hat, y))
    print('MAE: ', mae(y_hat, y))
print("Vectorised")
for fit_intercept in [True, False]:
    LR = LinearRegression(fit_intercept=fit_intercept)
    LR.fit_vectorised(X, y, 5)
    y_hat = LR.predict(X)
    print('RMSE: ', rmse(y_hat, y))
    print('MAE: ', mae(y_hat, y))
print("AutoGrad")
for fit_intercept in [True, False]:
    LR = LinearRegression(fit_intercept=fit_intercept)
    LR.fit_autograd(X, y, 5)
    y_hat = LR.predict(X)
    print('RMSE: ', rmse(y_hat, y))
    print('MAE: ', mae(y_hat, y))
print("Normal Equation")
for fit_intercept in [True, False]:
    LR = LinearRegression(fit_intercept=fit_intercept)
    LR.fit_normal(X, y)
    y_hat = LR.predict(X)
    print('RMSE: ', rmse(y_hat, y))
    print('MAE: ', mae(y_hat, y))