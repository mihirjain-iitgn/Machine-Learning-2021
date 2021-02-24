import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
from metrics import *

np.random.seed(10)

n = 10
m = 5
X = np.random.randn(n, m)
y = np.random.randn(n)
col = (4*X.T[0]).reshape(-1,1)
X = np.hstack((X,col))
X = pd.DataFrame(X)
y = pd.Series(y)

try:
    LR = LinearRegression(fit_intercept=True)
    LR.fit_normal(X,y)
except:
    print("Matrix is singular, Normal Equation is not applicable.")

LR = LinearRegression(fit_intercept=True)
LR.fit_vectorised(X,y,1)
y_hat = LR.predict(X)
print(rmse(y_hat,y))
print(mae(y_hat,y))