from sklearn.datasets import load_breast_cancer
from logistic_regression import LogisticRegression
import numpy as np
from numpy import arange
import pandas as pd
from utils import accuracy, rmse
from sklearn.linear_model import LogisticRegression as Lr
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold


def regularization(type_, Range):
    """
    Parameeters :
        > type_ : L1 or L2.
        > Range : Numpy Array of possible values for penality.
    """
    data = load_breast_cancer()
    scaler = MinMaxScaler()
    scaler.fit(data["data"])
    X = scaler.transform(data["data"])
    y = data["target"]
    names = data["feature_names"]
    kf1 = KFold(3, shuffle = True)
    kf1.get_n_splits(X)
    fold = 0
    print("Regularization : ", type_)
    for tv_index, test_index in kf1.split(X):
        # Train+Val-Test Split Loop
        X_tv, y_tv = X[tv_index],y[tv_index]
        X_test, y_test = X[test_index],y[test_index]
        kf2 = KFold(3, shuffle = True)
        kf2.get_n_splits(X_tv)
        Acc = {}
        for i in Range:
            Acc[i] = 0
        for train_index, val_index in kf2.split(X_tv):
            # Train-Val Split Loop
            X_train, y_train = X_tv[train_index], y_tv[train_index]
            X_val, y_val = X_tv[val_index], y_tv[val_index]
            for pf in Range:
                # Going over a range of values of penalities
                LR = LogisticRegression(use_autograd = True, penality = type_, penality_factor = pf)
                LR.fit(X_train, y_train, len(X_train),epochs = 1000, lr = 0.1)
                yv_hat = LR.predict(X_val)
                err = accuracy(y_val, yv_hat)
                Acc[pf] += err
        Best_val = float("-inf")
        for i in Acc:
            if Acc[i] > Best_val:
                Best_val = Acc[i]
                Best_key = i
        # Training Again with Train+Val Set
        LR = LogisticRegression(use_autograd = True, penality = type_, penality_factor = Best_key)
        LR.fit(X_tv, y_tv, len(X_tv), epochs = 1000, lr = 0.1)

        # Testing on the Test Set
        yt_hat = LR.predict(X_test)
        err = accuracy(y_test, yt_hat)
        print("Fold : ", fold)
        print("\t > Best Penality : ", Best_key)
        print("\t > Test Accuracy with Best Penality : ", err)
        if type_ == "L1":
            a = list(LR.coef.reshape(-1))
            b = [(abs(a[i]),i) for i in range(1,len(a))]
            b = sorted(b, reverse = True)
            print("\t Following are the Three Most Important features :")
            print("\t\t >",names[b[0][1]-1])
            print("\t\t >",names[b[1][1]-1])
            print("\t\t >",names[b[2][1]-1])
        fold += 1

regularization("L1", [i for i in arange(0.01,0.1,0.01)])
regularization("L2", [i for i in arange(0.01,0.1,0.01)])