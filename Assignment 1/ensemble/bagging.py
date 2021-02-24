import numpy as np
import pandas as pd
import copy
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import OrderedDict

class BaggingClassifier():
    def __init__(self, base_estimator, n_estimators=10):
        '''
        :param base_estimator: The base estimator model instance from which the bagged ensemble is built (e.g., DecisionTree(), LinearRegression()).
                               You can pass the object of the estimator class
        :param n_estimators: The number of estimators/models in ensemble.
        '''
        self.n_estimators = n_estimators
        self.models = [copy.deepcopy(base_estimator) for i in range(n_estimators)]
    

    def sample_data(self):
        data_sample = self.data.sample(n = len(self.data), replace=True)
        return data_sample

    def fit(self, X, y):
        """
        Function to train and construct the BaggingClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
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
        Funtion to run the BaggingClassifier on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        y = []
        outputs = pd.DataFrame()
        for i in range(self.n_estimators):
            outputs[i] = self.models[i].predict(X)
        for i in range(len(X)):
            y.append(outputs.loc[i].mode()[0])
        return pd.Series(y)

    def plot(self):
        """
        Function to plot the decision surface for BaggingClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns and should look similar to slide #16 of lecture
        The title of each of the estimator should be iteration number

        Figure 2 should also create a decision surface by combining the individual estimators and should look similar to slide #16 of lecture

        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

        This function should return [fig1, fig2]

        """
        plot_step = 0.05
        colors = ["r", "b"]
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
        fig.suptitle("Bagging Learnt Boundary")
        y = self.predict(X)
        y = np.array(y).reshape(y_axis.shape)
        ax.contourf(x_axis, y_axis, y, cmap = plt.get_cmap("RdBu"))
        for row in range(len(self.X)):
            ax.scatter(self.X.loc[row][0], self.X.loc[row][1], color = colors[self.y[row]])
        fig2 = fig
        plt.show()
        return fig1,fig2