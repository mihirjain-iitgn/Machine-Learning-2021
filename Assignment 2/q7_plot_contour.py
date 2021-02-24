import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression

np.random.seed(10)
N = 20
X = np.random.randn(N,1)
y = (50*X + 40).reshape(-1)

X = pd.DataFrame(X)
y = pd.Series(y)

LR = LinearRegression(fit_intercept=True)
LR.fit_vectorised(X,y,N,n_iter=200)

LR.plot_surface()
LR.plot_contour()
LR.plot_line_fit()