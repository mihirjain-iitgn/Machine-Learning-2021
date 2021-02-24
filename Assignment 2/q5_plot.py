import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from metrics import *
from linearRegression.linearRegression import LinearRegression
from preprocessing.polynomial_features import PolynomialFeatures


np.random.seed(10)  #Setting seed for reproducibility

x = np.array([i*np.pi/180 for i in range(60,300,4)])
y = 4*x + 7 + np.random.normal(0,3,len(x))
y = pd.Series(y)
degree = 9
magnitudes = []

for deg in range(1,degree+1):
    poly = PolynomialFeatures(deg,include_bias=True)
    X = []
    for row in range(len(x)):
        x_row = np.array([x[row]])
        x_row = poly.transform(x_row)
        X.append(x_row)
    X = pd.DataFrame(np.array(X))
    LR = LinearRegression(fit_intercept=False)
    LR.fit_normal(X,y)
    parameters = LR.coef_
    mag = np.linalg.norm(parameters)
    magnitudes.append(mag)
plt.xlabel('degree')
plt.ylabel('magnitude')
plt.plot([deg for deg in range(1,degree+1)], magnitudes)
plt.savefig("./images/Q56/q5.png")