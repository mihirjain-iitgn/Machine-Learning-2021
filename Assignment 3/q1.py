from sklearn.datasets import load_breast_cancer
from logistic_regression import LogisticRegression
import numpy as np
import pandas as pd
from utils import accuracy
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
np.seterr(over = "ignore")

def accuracy_cancer(Val):
    """
    Parameters :
        > Val : bool. if True uses AutoGrad else Direct.
    """
    data = load_breast_cancer()
    # Normalising the Data
    scaler = MinMaxScaler()
    scaler.fit(data["data"])
    X = scaler.transform(data["data"])
    y = data["target"]
    # K = 5 Folds Cross-Validation
    kf = KFold(3)
    kf.get_n_splits(X)
    acc = 0
    for train_index, test_index in kf.split(X):
        # Cross-Validation Loop
        X_train,y_train = X[train_index],y[train_index]
        X_test,y_test = X[test_index],y[test_index]
        LR = LogisticRegression(use_autograd = Val)
        LR.fit(X_train,y_train,len(X), epochs = 1000, lr = 0.1)
        y_hat = LR.predict(X_test)
        acc += accuracy(y_test, y_hat)
    return acc/3

def plot_cancer(indices):
    """
    Parameters :
        > indices : Python list of size 2. The indices of the two columns to use for plotting.
    """
    data = load_breast_cancer()
    X = data["data"]
    y = data["target"]
    X_new = []
    for i in range(len(X)):
        # Filtering Only the two given columns
        temp = []
        temp.append(X[i][indices[0]])
        temp.append(X[i][indices[1]])
        X_new.append(temp.copy())
    X_new = np.array(X_new)
    LR = LogisticRegression(use_autograd=False)
    LR.fit(X_new,y,len(X_new))
    LR.plot()


print("Testing Model on Breast Cancer Dataset\n")

print("Average accuracy over 3 Folds :")
acc = accuracy_cancer(False) # Without Autograd
print("\t > Without Autograd : ", acc)

acc = accuracy_cancer(True) # With Autograd
print("\t > With Autograd : ", acc)

indices = [2,3]
plot_cancer(indices)