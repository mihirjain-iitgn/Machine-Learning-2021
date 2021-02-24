import numpy as np
import pandas as pd
import copy
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import OrderedDict


class AdaBoostClassifier():
    def __init__(self, base_estimator, n_estimators): # Optional Arguments: Type of estimator
        '''
        :param base_estimator: The base estimator model instance from which the boosted ensemble is built (e.g., DecisionTree, LinearRegression).
                               If None, then the base estimator is DecisionTreeClassifier(max_depth=1).
                               You can pass the object of the estimator class
        :param n_estimators: The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure may be stopped early.
        '''
        self.n_estimators = n_estimators
        self.models = [copy.deepcopy(base_estimator) for i in range(self.n_estimators)]
    
    
    def update_weights(self, weights, y_hat, y):
        """
        Function to update thw weights after an iteration
        Inputs:
        weights : currents numpy array
        y_hat : Series prediction in the current iteration
        y : Series Ground Truth
        """
        err = 0
        for i in range(y.size):
            if (y_hat[i]!=y[i]):
                err+=weights[i]
        alpha = (0.5)*math.log((1-err)/err)
        for i in range(y.size):
            if (y_hat[i]==y[i]):
                weights[i] = weights[i]*pow(math.e, -1*alpha)
            else:
                weights[i] = weights[i]*pow(math.e, alpha)
        weights = weights/np.sum(weights)
        return alpha, weights

    def fit(self, X, y):
        """
        Function to train and construct the AdaBoostClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.X = X.copy()
        self.y = y.copy()
        y_m = []
        for i in range(y.size):
            if y[i]==0:
                y_m.append(-1)
            else:
                y_m.append(1)
        y_m = pd.Series(y_m)
        weights = np.full(len(X), 1/len(X))
        self.weights_plot = []
        self.alpha = np.zeros(self.n_estimators)
        for i in range(self.n_estimators):
            self.weights_plot.append(list(weights))
            self.models[i].fit(X,y_m,sample_weight = pd.Series(weights))
            y_hat = self.models[i].predict(X)
            self.alpha[i], weights = self.update_weights(weights, y_hat, y_m)


    def predict(self, X):
        """
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
            if np.dot((outputs.loc[i]).to_numpy(), self.alpha)>=0:
                y.append(1)
            else:
                y.append(0)
        return pd.Series(y)

    def plot(self):
        """
        Function to plot the decision surface for AdaBoostClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns
        The title of each of the estimator should be associated alpha (similar to slide#38 of course lecture on ensemble learning)
        Further, the scatter plot should have the marker size corresponnding to the weight of each point.

        Figure 2 should also create a decision surface by combining the individual estimators

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
            ax[i].set_title("alpha : "+str(round(self.alpha[i],4)))
            
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
        fig.suptitle("AdaBoost Learnt Boundary")
        y = self.predict(X)
        y = np.array(y).reshape(y_axis.shape)
        ax.contourf(x_axis, y_axis, y, cmap = plt.get_cmap("RdBu"))
        for row in range(len(self.X)):
            ax.scatter(self.X.loc[row][0], self.X.loc[row][1], color = colors[self.y[row]])
        fig2 = fig
        plt.show()
        return fig1,fig2