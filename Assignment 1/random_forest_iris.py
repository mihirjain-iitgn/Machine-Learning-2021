import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from tree.randomForest import RandomForestClassifier
from tree.randomForest import RandomForestRegressor
from sklearn.datasets import load_iris


"""
File that tests Random Forest Classifier on the Iris Dataset.
Only two features of the Iris dataset are considered (sepal width and petal width)
"""

np.random.seed(42)

data = load_iris()

data , datay = pd.DataFrame(data["data"]), pd.Series(data["target"])

data = data[[1,3]]
data.columns = [0,1]

data.at[:,2] = datay.astype("category")

n_estimators = 3
data_train = data.sample(frac = 0.6)
data_test = data.drop(data_train.index)

data_test.reset_index(inplace = True, drop = True)
data_train.reset_index(inplace = True, drop = True)

X_train, X_test = data_train[[0,1]], data_test[[0,1]]
y_train, y_test = data_train[2].astype('category'), data_test[2].astype('category')
Classifier_RF = RandomForestClassifier(n_estimators=n_estimators)
Classifier_RF.fit(X_train, y_train)
y_hat = Classifier_RF.predict(X_test)

print("Accuracy :", accuracy(y_test,y_hat))
for cls in datay.unique():
    print('Precision: ', precision(y_hat, y_test, cls))
    print('Recall: ', recall(y_hat, y_test, cls))

[fig1, fig2] = Classifier_RF.plot()

fig1.savefig("./plots/rf1.jpg")
fig2.savefig("./plots/rf2.jpg")