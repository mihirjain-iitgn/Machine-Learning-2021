
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn import tree


"""
This file comapres the performance of my implementation and sklearn implementation
on a real estate (Real IO) dataset.

dataset -> ./real_estate.csv
"""

np.random.seed(42)

data = pd.read_csv("./real_estate_.csv").reset_index(drop=True)

data.drop(["No"],axis=1) # Dropping the serial number axis

data_train = data.sample(frac=0.7)
data_test = data.drop(data_train.index)

data_train.reset_index(drop=True,inplace=True)
data_test.reset_index(drop=True,inplace=True)

X_train, X_test = pd.DataFrame(data_train[["X1","X2","X3","X4","X5","X6"]]), pd.DataFrame(data_test[["X1","X2","X3","X4","X5","X6"]])
y_train, y_test = pd.Series(data_train["Y"]), pd.Series(data_test["Y"])

tree_c = DecisionTree(criterion="information_gain")

tree_c.fit(X_train,y_train)

y_hat = tree_c.predict(X_train)

print("My tree RMSE", rmse(y_hat, y_train))
print("My tree MAE", mae(y_hat, y_train))

tree_sl = tree.DecisionTreeRegressor()

tree_sl.fit(X_train,y_train)

y_hat = tree_sl.predict(X_train)

print("Sklearn Tree RMSE", rmse(y_hat, y_train))
print("Sklearn Tree MAE", mae(y_hat, y_train))

"""
A possible reason for the slight difference is that sklearn uses mse while
in my implementation, I have used varience.
Another reason could be of how python truncates floats.
"""