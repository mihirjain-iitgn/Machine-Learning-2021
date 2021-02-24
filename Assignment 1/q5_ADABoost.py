import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from ensemble.ADABoost import AdaBoostClassifier
from tree.base import DecisionTree
from sklearn import tree
# Or you could import sklearn DecisionTree
# from linearRegression.linearRegression import LinearRegression
from sklearn import tree
from sklearn.datasets import load_iris
np.random.seed(42)


"""
This file tests Adaboost on random data and on Iris dataset
"""

########### AdaBoostClassifier on Real Input and Discrete Output ###################


def Adaboost():
    """
    Function to run Adaboost over random data
    """
    N = 30
    P = 2
    NUM_OP_CLASSES = 2
    n_estimators = 3
    X = pd.DataFrame(np.abs(np.random.randn(N, P)))
    y = pd.Series(np.random.randint(NUM_OP_CLASSES, size = N), dtype="category")
    tree_sl = tree.DecisionTreeClassifier(max_depth=1, criterion="entropy")
    Classifier_AB = AdaBoostClassifier(base_estimator=tree_sl, n_estimators=n_estimators)
    Classifier_AB.fit(X, y)
    y_hat = Classifier_AB.predict(X)
    [fig1, fig2] = Classifier_AB.plot()
    print('Accuracy: ', accuracy(y_hat, y))
    for cls in y.unique():
        print('Precision: ', precision(y_hat, y, cls))
        print('Recall: ', recall(y_hat, y, cls))

##### AdaBoostClassifier on Iris data set using the entire data set with sepal width and petal width as the two features

def Iris_Experiment():
    """
    Function to run Adaboost on Iris data with a 60-40 split
    Only two features are considered and the output classes are also virginica and not virginica
    """

    data = load_iris()

    data , datay = pd.DataFrame(data["data"]), pd.Series(data["target"])

    data = data[[1,3]]
    data.columns = [0,1]

    data.at[:,2] = datay.astype("category")

    for i in range(len(data)):
        if (data.loc[i][2]!=2):
            data.at[i,2] = 0
        else:
            data.at[i,2] = 1

    n_estimators = 25
    data_train = data.sample(frac = 0.6)
    data_test = data.drop(data_train.index)

    data_test.reset_index(inplace = True, drop = True)
    data_train.reset_index(inplace = True, drop = True)

    X_train, X_test = data_train[[0,1]], data_test[[0,1]]
    y_train, y_test = data_train[2].astype('category'), data_test[2].astype('category')

    tree_sl = tree.DecisionTreeClassifier(max_depth=1, criterion="entropy")
    Classifier_AB = AdaBoostClassifier(base_estimator=tree_sl, n_estimators=n_estimators)
    Classifier_AB.fit(X_train, y_train)

    y_hat = Classifier_AB.predict(X_test)
    [fig1, fig2] = Classifier_AB.plot()

    fig1.savefig("./plots/ada_iris1.jpg")
    fig2.savefig("./plots/ada_iris2.jpg")
    print('Accuracy: ', accuracy(y_hat, y_test))
    for cls in y_test.unique():
        print('Precision: ', precision(y_hat, y_test, cls))
        print('Recall: ', recall(y_hat, y_test, cls))


"""
> Adaboost() : Function to run Adaboost over random data
> Iris_Experiment() : Function to run Adaboost on Iris data with a 60-40 split
> nested_cross_validation() : 

Uncomment the function calls below to do the corresponding experiment.
"""

# Adaboost()
Iris_Experiment()