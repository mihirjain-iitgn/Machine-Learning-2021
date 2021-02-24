from .base import DecisionTree
import copy
import random
from sklearn import tree
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import OrderedDict

class RandomForestClassifier():
    def __init__(self, n_estimators=10, criterion='entropy', max_depth=10):
        '''
        :param estimators: DecisionTree
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''
        self.n_estimators = n_estimators
        self.criteria = criterion
        self.max_depth = max_depth
        self.criteria = criterion
    

    def sample_data(self):
        data_sample = self.data.sample(n = len(self.data), replace=True)
        return data_sample

    def fit(self, X, y):
        """
        Function to train and construct the RandomForestClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        tree_sl = tree.DecisionTreeClassifier(criterion = self.criteria, max_depth = self.max_depth, max_features = math.ceil(math.sqrt(len(list(X.columns)))))
        self.models = [copy.deepcopy(tree_sl) for i in range(self.n_estimators)]
        self.X = X.copy()
        self.y = y.copy()
        self.data = X.copy()
        self.data["Output"] = y.copy()
        for i in range(self.n_estimators):
            data_sampled = self.sample_data()
            X_sample, y_sample = data_sampled.drop('Output', axis=1), data_sampled["Output"]
            self.models[i].fit(X_sample, y_sample)

    def predict(self, X):
        """
        Funtion to run the RandomForestClassifier on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        Output = pd.DataFrame()
        y = []
        for i in range(self.n_estimators):
            Output[i] = self.models[i].predict(X)
        for i in range(len(X)):
            y.append(Output.loc[i].mode()[0])
        return pd.Series(y)

    def plot(self):
        """
        Function to plot for the RandomForestClassifier.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

        2. Creates a figure showing the decision surface for each estimator

        3. Creates a figure showing the combined decision surface

        """
        # return None, None
        plot_step = 0.05
        colors = ["r", "y", "b"]
        # The 4 lines below create the input dataframe consisting of points in R2
        x_low, x_high = self.X.loc[:,0].min()-1, self.X.loc[:,0].max()+1
        y_low, y_high = self.X.loc[:,1].min()-1, self.X.loc[:,1].max()+1
        x_axis, y_axis = np.meshgrid(np.arange(x_low, x_high, plot_step),np.arange(y_low, y_high, plot_step))
        X = pd.DataFrame(np.c_[x_axis.ravel(), y_axis.ravel()])


        fig, ax = plt.subplots(1,self.n_estimators)
        for i in range(self.n_estimators):
            # Plotting the boundary for all the learnt trees
            ax[i].set_title("Iteration  : "+str(i))
            y = self.models[i].predict(X)
            y = np.array(y).reshape(y_axis.shape)
            # Below line plots the prediction to the dataframe
            ax[i].contourf(x_axis, y_axis, y, cmap = plt.get_cmap("RdBu"))
            for row in range(len(self.X)):
                # Below line plots the training points
                ax[i].scatter(self.X.loc[row][0], self.X.loc[row][1], color = colors[self.y[row]])
        fig1 = fig
        plt.show()

        # Plotting the boundary for the overall classifier
        fig, ax = plt.subplots(1,1)
        fig.suptitle("Random Forest Learnt Boundary")
        y = self.predict(X)
        y = np.array(y).reshape(y_axis.shape)
        ax.contourf(x_axis, y_axis, y, cmap = plt.get_cmap("RdBu"))
        for row in range(len(self.X)):
            ax.scatter(self.X.loc[row][0], self.X.loc[row][1], color = colors[self.y[row]])
        fig2 = fig
        plt.show()
        return fig1,fig2


class RandomForestRegressor():
    def __init__(self, n_estimators=100, criterion='mse', max_depth=None):
        '''
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''
        self.n_estimators = n_estimators
        self.criteria = criterion
    
    def sample_data(self):
        data_sample = self.data.sample(n = len(self.data), replace=True)
        return data_sample

    def fit(self, X, y):
        """
        Function to train and construct the RandomForestRegressor
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        tree_sl = tree.DecisionTreeRegressor(max_features=math.ceil(math.sqrt(len(list(X.columns)))))
        self.models = [copy.deepcopy(tree_sl) for i in range(self.n_estimators)]
        self.X = X.copy()
        self.y = y.copy()
        self.data = X.copy()
        self.data["Output"] = y.copy()
        for i in range(self.n_estimators):
            data_sampled = self.sample_data()
            X_sample, y_sample = data_sampled.drop('Output', axis=1), data_sampled["Output"]
            self.models[i].fit(X_sample, y_sample)

    def predict(self, X):
        """
        Funtion to run the RandomForestRegressor on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        Output = pd.DataFrame()
        y = []
        for i in range(self.n_estimators):
            Output[i] = self.models[i].predict(X)
        for i in range(len(X)):
            y.append(Output.loc[i].mean())
        return pd.Series(y)

    def plot(self):
        """
        Function to plot for the RandomForestClassifier.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

        2. Creates a figure showing the decision surface/estimation for each estimator. Similar to slide 9, lecture 4

        3. Creates a figure showing the combined decision surface/prediction

        """
        # Not Supposed to be Implmented
        return None, None
