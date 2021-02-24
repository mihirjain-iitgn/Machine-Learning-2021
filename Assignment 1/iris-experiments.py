import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

from sklearn.datasets import load_iris
from sklearn import tree

np.random.seed(42)

def cross_validation(data,k):
    """
    performs 5-fold cross validation and prints the average accuracy over the 5 folds.
    Inputs:
    data : Dataframe of Iris Dataset
    k : parameter of k-fold cross validation
    """
    l = len(data)//k
    acc = 0
    for i in range(k-1,-1,-1):
        if (i==k-1):
            train_data = data.loc[:i*l]
            test_data = data.loc[i*l:]
        elif (i==0):
            train_data = data.loc[(i+1)*l:]
            test_data = data.loc[:(i+1)*l]
        else:
            train_data = data.loc[:i*l]
            test_data = data.loc[i*l:(i+1)*l]
            train_data1 = data.loc[(i+1)*l:]
            train_data.append(train_data1, ignore_index=True)
        train_data.reset_index(drop=True, inplace = True)
        test_data.reset_index(drop=True, inplace = True)
        tree_c = DecisionTree(criterion="information_gain")
        X_train, X_test = train_data[[0,1,2,3]], test_data[[0,1,2,3]]
        y_train, y_test = train_data[4].astype('category'), test_data[4].astype('category')
        tree_c.fit(X_train, y_train)
        tree_c.predict(X_test)
        y_hat = tree_c.predict(X_test)
        acc+=accuracy(y_hat, y_test)
    acc = acc/k
    print("Average Accuracy :", acc)


def nested_cross_validation(data,k1,k2,max_depth):
    """
    performs nested cross validation
    inputs:
    data : Dataframe of Iris Dataset
    k1 : parameter of outer loop of nestedcross validation
    k2 : parameter of inner loop of nested cross validation
    max_depth : the max height to consider for finding the optimal
    """
    l1 = len(data)//k1
    for i in range(k1-1,-1,-1):
        if (i==k1-1):
            train_data = data.loc[:i*l1]
            test_data = data.loc[i*l1:]
        elif (i==0):
            train_data = data.loc[(i+1)*l1:]
            test_data = data.loc[:(i+1)*l1]
        else:
            train_data = data.loc[:i*l1]
            test_data = data.loc[i*l1:(i+1)*l1]
            train_data1 = data.loc[(i+1)*l1:]
            train_data.append(train_data1, ignore_index=True)
        l2 = len(train_data)//k2
        train_data.reset_index(drop=True, inplace = True)
        test_data.reset_index(drop=True, inplace = True)
        heights = [0 for i in range(max_depth)]
        for j in range(k2-1,-1,-1):
            if (j==k2-1):
                train_data1 = train_data.loc[:j*l2]
                validation_data = train_data.loc[j*l2:]
            elif (j==0):
                train_data1 = train_data.loc[(j+1)*l2:]
                validation_data = train_data.loc[:(j+1)*l2]
            else:
                train_data1 = train_data.loc[:j*l2]
                validation_data = train_data.loc[j*l2:(j+1)*l2]
                train_data2 = train_data1.loc[(j+1)*l2:]
                train_data1.append(train_data2, ignore_index = True)
            train_data1.reset_index(drop = True, inplace = True)
            validation_data.reset_index(drop = True, inplace = True)
            for depth in range(1,max_depth+1):
                tree_c = DecisionTree(criterion="information_gain", max_depth=depth)
                X_train, X_test = train_data1[[0,1,2,3]], validation_data[[0,1,2,3]]
                y_train, y_test = train_data1[4].astype('category'), validation_data[4].astype('category')
                tree_c.fit(X_train, y_train)
                y_hat = tree_c.predict(X_test)
                acc = accuracy(y_hat, y_test)
                heights[depth-1]+=acc
        optimal_depth = heights.index(max(heights))+1
        print("Optimal Depth for iteration",str(k1-i) + ": ",optimal_depth)
        tree_c = DecisionTree(criterion="information_gain", max_depth=optimal_depth)
        X_train, X_test = train_data[[0,1,2,3]], test_data[[0,1,2,3]]
        y_train, y_test = train_data[4].astype('category'), test_data[4].astype('category')
        tree_c.fit(X_train, y_train)
        y_hat = tree_c.predict(X_test)
        acc = accuracy(y_hat, y_test)
        print("Accuracy on test set of optimal depth in iteration ", str(k1-i),acc)


def simple(data):
    """
    tests the tree on the iris dataset with a 70-30 split
    Input:
    data : DataFrame of Iris dataset
    """
    data_train = data.sample(frac=0.7)
    data_test = data.drop(data_train.index)

    data_train.reset_index(drop=True,inplace=True)
    data_test.reset_index(drop=True,inplace=True)

    X_train, X_test = data_train[[0,1,2,3]], data_test[[0,1,2,3]]
    y_train, y_test = data_train[4].astype('category'), data_test[4].astype('category')

    tree_c = DecisionTree(criterion="information_gain")
    tree_c.fit(X_train, y_train)

    y_hat = tree_c.predict(X_test)

    print('Accuracy: ', accuracy(y_hat, y_test))
    for cls in y_test.unique():
        print('Precision: ', precision(y_hat, y_test, cls))
        print('Recall: ', recall(y_hat, y_test, cls))






data = load_iris()

data , datay = pd.DataFrame(data["data"]), pd.Series(data["target"])
data[4] = datay

data = data.sample(frac=1)
data.reset_index(drop=True, inplace = True)

"""
> simple() : simply tests the tree on the iris dataset with a 70-30 split
> cross_validation() : performs 5-fold cross validation and prints the average accuracy over the 5 folds.
> nested_cross_validation() : 

Uncomment the function calls below to do the corresponding experiment.
"""

# simple(data)

k = 5
# cross_validation(data,k)

k1 = 5
k2 = 5
max_depth = 15
nested_cross_validation(data,k1,k2,max_depth)
