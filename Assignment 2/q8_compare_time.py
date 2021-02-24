import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from linearRegression.linearRegression import LinearRegression

np.random.seed(10)

def vary_m():
    n = 100
    gd_times = []
    ne_times = []
    for m in range(100,1001,2):
        X = pd.DataFrame(np.random.randn(n, m))
        y = pd.Series(np.random.randn(n))
        LR = LinearRegression(fit_intercept=True)
        start = time.time()
        LR.fit_normal(X,y)
        end = time.time()        
        train_time = end-start
        ne_times.append((m,train_time))
        LR = LinearRegression(fit_intercept=True)
        start = time.time()
        LR.fit_vectorised(X,y,len(X),lr=0.01)
        end = time.time()
        train_time = end-start
        gd_times.append((m,train_time))
    gd_times = pd.DataFrame(gd_times)
    ne_times = pd.DataFrame(ne_times)
    plt.plot(gd_times[0],gd_times[1],label = "Gradient Descent")
    plt.plot(ne_times[0],ne_times[1],label = "Normal Equation")
    plt.legend(loc = "best")
    plt.suptitle("Varying M, N = 100")
    plt.savefig("./images/time_compare/varym.png")
    plt.close()

def vary_n():
    m = 100
    gd_times = []
    ne_times = []
    for n in range(100,1001,2):
        X = pd.DataFrame(np.random.randn(n, m))
        y = pd.Series(np.random.randn(n))
        LR = LinearRegression(fit_intercept=True)
        start = time.time()
        LR.fit_normal(X,y)
        end = time.time()        
        train_time = end-start
        ne_times.append((n,train_time))
        LR = LinearRegression(fit_intercept=True)
        start = time.time()
        LR.fit_vectorised(X,y,len(X),lr=0.01)
        end = time.time()
        train_time = end-start
        gd_times.append((n,train_time))
    gd_times = pd.DataFrame(gd_times)
    ne_times = pd.DataFrame(ne_times)
    plt.plot(gd_times[0],gd_times[1],label = "Gradient Descent")
    plt.plot(ne_times[0],ne_times[1],label = "Normal Equation")
    plt.legend(loc = "best")
    plt.suptitle("Varying N, M = 100")
    plt.savefig("./images/time_compare/varyn.png")
    plt.close()

vary_n()
vary_m()