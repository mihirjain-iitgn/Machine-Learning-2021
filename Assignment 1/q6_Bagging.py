"""
The current code given is for the Assignment 2.
> Classification
> Regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from ensemble.bagging import BaggingClassifier
from tree.base import DecisionTree
# Or use sklearn decision tree
from sklearn import tree
# from linearRegression.linearRegression import LinearRegression


"""
This file tests Bagging on random data and on Iris dataset
"""

########### BaggingClassifier ###################

N = 30
P = 2
NUM_OP_CLASSES = 2
n_estimators = 3
X = pd.DataFrame(np.abs(np.random.randn(N, P)))
y = pd.Series(np.random.randint(NUM_OP_CLASSES, size = N), dtype="category")

tree_sl = tree.DecisionTreeClassifier(criterion="entropy")
Classifier_B = BaggingClassifier(base_estimator=tree_sl, n_estimators=n_estimators)
Classifier_B.fit(X, y)
y_hat = Classifier_B.predict(X)

[fig1, fig2] = Classifier_B.plot()
fig1.savefig("./plots/bagg1.jpg")
fig2.savefig("./plots/bagg2.jpg")

print('Accuracy: ', accuracy(y_hat, y))
for cls in y.unique():
    print('Precision: ', precision(y_hat, y, cls))
    print('Recall: ', recall(y_hat, y, cls))