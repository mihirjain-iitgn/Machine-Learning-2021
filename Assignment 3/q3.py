from sklearn.datasets import load_digits
from logistic_regression import LogisticRegression
import numpy as np
import pandas as pd
from utils import accuracy
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def precprocess(X):
    """
    Parameters :
    > X : numpy array.

    Returns :
    > Input numpy array row-wise flattened and all values normalised to [0-1)
    """
    n = len(X)
    return X.reshape((n,-1))/15


def digits_experiments(Val):
    """
    Parameeters :
        > Val : Bool. if True then Autograd otherwise direct. 
    """
    data = load_digits()
    X = precprocess(data["data"])
    y = data["target"]
    kf = KFold(4,shuffle = True)
    kf.get_n_splits(X)
    acc = 0
    fold = 0
    errors = np.zeros((10,10))
    Acc = np.zeros(10)
    for train_index, test_index in kf.split(X):
        # Test-Train Split Loop
        X_train,y_train = X[train_index],y[train_index]
        X_test,y_test = X[test_index],y[test_index]
        LR = LogisticRegression(use_autograd = Val, multi_class = True, num_classes = 10)
        LR.fit(X_train,y_train,len(X), epochs = 1000, lr = 0.01)
        y_hat = LR.predict(X_test)
        acc += accuracy(y_test, y_hat)
        print("Confusion matrix for Fold : ", fold)
        ConfusionMatrix = confusion_matrix(y_test, y_hat)
        print(ConfusionMatrix)
        for i in range(10):
            for j in range(10):
                if i!=j:
                    errors[i][j] += ConfusionMatrix[i][j] + ConfusionMatrix[j][i]
                    errors[j][i] += ConfusionMatrix[i][j] + ConfusionMatrix[j][i]
            Acc[i] += ConfusionMatrix[i][i]/np.sum(ConfusionMatrix[:,i])
        fold += 1
    return acc/4,Acc/4,errors

def qa():
    print("Testing Model on Digits Dataset\n")
    acc,Acc,Errors = digits_experiments(False) # Without Autograd
    print("Average accuracy over 4 Folds :")
    print("\t > Without Autograd : ", acc)
    print("Digit with highest average accuracy : ", np.argmax(Acc))
    nums = str(np.argmax(Errors))
    print("Digit which get confused most : ", nums[0],nums[1])
    # acc,Acc,Errors = digits_experiments(True) # With Autograd
    # print("Average accuracy over 5 Folds :")
    # print("\t > With Autograd : ", acc)
    # print("Digit with highest average accuracy : ", np.argmax(Acc))
    # print("Digit which get confused most : ", np.argmax(Errors))



def qb():
    # Reference : https://www.datacamp.com/community/tutorials/machine-learning-python
    data = load_digits()
    X = data["data"]
    y = data["target"]
    pca = PCA(n_components = 2)
    X_red = pca.fit_transform(X)
    colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
    for i in range(len(X_red)):
        plt.scatter(X_red[i][0],X_red[i][1], c = colors[y[i]])
    plt.legend(data.target_names)
    plt.savefig("./plots/pca.png")
    plt.show()


# qa()
qb()