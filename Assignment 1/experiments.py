
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
import time

np.random.seed(42)
num_average_time = 100

"""
This file generates plots for time taken for training and testing
"""

# Discret Input Discrete Output

def di_do():
    fig, ax = plt.subplots(1, 2)
    fig.tight_layout()
    fig.suptitle("Discrete Input Discrete Output")
    ax[0].set_title("Training Time")
    ax[1].set_title("Testing Time")
    for m in range(2,8,1):
        train_times = []
        test_times = []
        for n in range(2, 51, 2):
            X = pd.DataFrame({i:pd.Series(np.random.randint(low = 0,high = 2,  size = n), dtype="category") for i in range(m)})
            y = pd.Series(np.random.randint(5, size = n), dtype = "category")
            tree_c = DecisionTree(criterion="information_gain")
            start = time.time()
            tree_c.fit(X,y)
            end = time.time()
            train_time = end-start
            start = time.time()
            tree_c.predict(X)
            end = time.time()
            test_time = end-start
            train_times.append((n,train_time))
            test_times.append((n,test_time))
        train_times = pd.DataFrame(train_times)
        test_times = pd.DataFrame(test_times)
        ax[0].plot(train_times[0], train_times[1], label = m)
        ax[1].plot(test_times[0], test_times[1], label = m)    
    ax[0].legend(loc="best")
    ax[1].legend(loc="best")
    fig.savefig("./plots/dd.png")


# Discret Input Real Output

def di_ro():
    fig, ax = plt.subplots(1, 2)
    fig.tight_layout()
    fig.suptitle("Discrete Input Real Output")
    ax[0].set_title("Training Time")
    ax[1].set_title("Testing Time")
    for m in range(2,8,1):
        train_times = []
        test_times = []
        for n in range(2, 51, 2):
            X = pd.DataFrame({i:pd.Series(np.random.randint(low = 0,high = 2,  size = n), dtype="category") for i in range(m)})
            y = pd.Series(np.random.randn(n))
            tree_c = DecisionTree(criterion="information_gain")
            start = time.time()
            tree_c.fit(X,y)
            end = time.time()
            train_time = end-start
            start = time.time()
            tree_c.predict(X)
            end = time.time()
            test_time = end-start
            train_times.append((n,train_time))
            test_times.append((n,test_time))
        train_times = pd.DataFrame(train_times)
        test_times = pd.DataFrame(test_times)
        ax[0].plot(train_times[0], train_times[1], label = m)
        ax[1].plot(test_times[0], test_times[1], label = m)    
    ax[0].legend(loc="best")
    ax[1].legend(loc="best")
    fig.savefig("./plots/dr.png")

# Real Input Discrete Output

def ri_do():
    fig, ax = plt.subplots(1, 2)
    fig.tight_layout()
    fig.suptitle("Real Input Discrete Output")
    ax[0].set_title("Training Time")
    ax[1].set_title("Testing Time")
    for m in range(2,8,1):
        train_times = []
        test_times = []
        for n in range(2, 51, 2):
            X = pd.DataFrame(np.random.randn(n, m))
            y = pd.Series(np.random.randint(5, size = n), dtype = "category")
            tree_c = DecisionTree(criterion="information_gain")
            start = time.time()
            tree_c.fit(X,y)
            end = time.time()
            train_time = end-start
            start = time.time()
            tree_c.predict(X)
            end = time.time()
            test_time = end-start
            train_times.append((n,train_time))
            test_times.append((n,test_time))
        train_times = pd.DataFrame(train_times)
        test_times = pd.DataFrame(test_times)
        ax[0].plot(train_times[0], train_times[1], label = m)
        ax[1].plot(test_times[0], test_times[1], label = m)    
    ax[0].legend(loc="best")
    ax[1].legend(loc="best")
    fig.savefig("./plots/rd.png")


# Discret Input Real Output

def ri_ro():
    fig, ax = plt.subplots(1, 2)
    fig.tight_layout()
    fig.suptitle("Real Input Real Output")
    ax[0].set_title("Training Time")
    ax[1].set_title("Testing Time")
    for m in range(2,8,1):
        train_times = []
        test_times = []
        for n in range(2, 51, 2):
            X = pd.DataFrame(np.random.randn(n, m))
            y = pd.Series(np.random.randn(n))
            tree_c = DecisionTree(criterion="information_gain")
            start = time.time()
            tree_c.fit(X,y)
            end = time.time()
            train_time = end-start
            start = time.time()
            tree_c.predict(X)
            end = time.time()
            test_time = end-start
            train_times.append((n,train_time))
            test_times.append((n,test_time))
        train_times = pd.DataFrame(train_times)
        test_times = pd.DataFrame(test_times)
        ax[0].plot(train_times[0], train_times[1], label = m)
        ax[1].plot(test_times[0], test_times[1], label = m)    
    ax[0].legend(loc="best")
    ax[1].legend(loc="best")
    fig.savefig("./plots/rr.png")



"""
Uncomment the function calls below
"""

# di_do()

# di_ro()

# ri_do()

# ri_ro()