import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from linearRegression.linearRegression import LinearRegression
from preprocessing.polynomial_features import PolynomialFeatures

np.random.seed(42)  #Setting seed for reproducibility
degree = [1,3,5,7,9]
plt.xlabel("N")
plt.ylabel("magnitude")
Mags = [[],[],[],[],[]]

for N in range(200,501):
    x = np.array([i*np.pi/180 for i in range(100,4*N,4)])
    y = 4*x + 7 + np.random.normal(0,3,len(x))
    y = pd.Series(y)
    for deg in range(len(degree)):
        poly = PolynomialFeatures(degree[deg],include_bias=True)
        X = []
        for row in range(len(x)):
            x_row = np.array([x[row]])
            x_row = poly.transform(x_row)
            X.append(x_row)
        X = pd.DataFrame(np.array(X))
        LR = LinearRegression(fit_intercept=False)
        LR.fit_normal(X,y)
        parameters = LR.coef_
        val = np.linalg.norm(parameters)
        Mags[deg].append(val)

for mag in range(len(Mags)):
    plt.plot([N for N in range(200,501)],Mags[mag],label = str(degree[mag]))
plt.legend(loc="best")
plt.savefig("./images/Q56/q6.png")