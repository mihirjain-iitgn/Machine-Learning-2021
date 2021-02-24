import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import autograd.numpy as np_a
from autograd import elementwise_grad
from mpl_toolkits import mplot3d
from matplotlib import cm

class LinearRegression():
    def __init__(self, fit_intercept=True):
        '''
        : param fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        '''
        self.fit_intercept = fit_intercept
        self.coef_ = None

    def get_batches(self, batch_size, X, y):
        """
        : batch_size : Int, Size of a batch
        : param X: Numpy Array with rows as samples and columns as features
        : param y: Numpy Array with rows corresponding to output
        > return python list with each ebtry corresponding to a batch
        """
        batches = []
        num_batches = X.shape[0]//batch_size
        if (X.shape[0]%batch_size!=0):
            num_batches += 1
        data = np.hstack((np.copy(X), np.copy(y)))
        n = len(data)
        for i in range(num_batches):
            batch = data[i*(batch_size):min(n,(i+1)*batch_size),:]
            X_t = batch[:,:-1]
            y_t = batch[:,-1].reshape(-1,1)
            batches.append((X_t,y_t))
        return batches

    def fit_non_vectorised(self, X, y, batch_size, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using non-vectorised gradient descent.
        : param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        : param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        : param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data. 
        : param n_iter: number of iterations (default: 100)
        : param lr: learning rate (default: 0.01)
        : param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number
        > return None
        '''
        self.X = X
        self.y = y
        X = X.to_numpy()
        y = y.to_numpy().reshape(-1,1)
        if (self.fit_intercept):
            bias = np.ones((X.shape[0], 1))
            X = np.append(bias, X, axis = 1)
        parameters = np.zeros((X.shape[1],1))
        lr_cur = lr
        for iter_n in range(n_iter):
            if (lr_type=="inverse"):
                lr_cur = lr_cur/(iter_n+1)
            batches = self.get_batches(batch_size,X,y)
            for batch in batches:
                X_cur ,y_cur = batch
                y_hat = np.zeros(len(y_cur))
                for row in range(len(X_cur)):
                    temp = 0
                    for index in range(len(X_cur[row])):
                        temp += (X_cur[row][index]*parameters[index])
                    y_hat[row] = temp
                for i in range(len(parameters)):
                    temp = 0
                    for row in range(len(X_cur)):
                        temp+=((y_hat[row]-y_cur[row])*X_cur[row][i])
                    temp = (2*temp)/len(X_cur)
                    parameters[i] -= lr_cur*temp
        self.coef_ = parameters
    
    def fit_vectorised(self, X, y,batch_size, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using vectorised gradient descent.
        : param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        : param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        : param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data.
        : param n_iter: number of iterations (default: 100)
        : param lr: learning rate (default: 0.01)
        : param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number
        > return None
        '''
        self.X = X
        self.y = y
        X = X.to_numpy()
        y = y.to_numpy().reshape(-1,1)
        if (self.fit_intercept):
            bias = np.ones((X.shape[0], 1))
            X = np.append(bias, X, axis = 1)
        parameters = np.zeros((X.shape[1],1))
        lr_cur = lr
        self.parameters_history = []
        for iter_n in range(n_iter):
            self.parameters_history.append(parameters.copy())
            if (lr_type=="inverse"):
                lr_cur = lr_cur/(iter_n+1)
            batches = self.get_batches(batch_size,X,y)
            for batch in batches:
                X_cur ,y_cur = batch
                y_hat = np.dot(X_cur,parameters)
                parameters += ((2/len(X_cur))*lr*(X_cur.T.dot(y_cur-y_hat)))
        self.coef_ = parameters


    def fit_autograd(self, X, y, batch_size, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using gradient descent with Autograd to compute the gradients.
        Autograd reference: https://github.com/HIPS/autograd

        : param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        : param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        : param batch_size: int specifying the  batch size. Batch size can only be between 1 and number of samples in data.
        : param n_iter: number of iterations (default: 100)
        : param lr: learning rate (default: 0.01)
        : param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number
        > return None
        '''
        self.X = X
        self.y = y
        X = X.to_numpy()
        y = y.to_numpy().reshape(-1,1)
        if (self.fit_intercept):
            bias = np.ones((X.shape[0],1))
            X = np.append(bias, X, axis = 1)
        parameters = np.zeros((X.shape[1],1))
        lr_cur = lr
        get_grad = elementwise_grad(self.cost)
        for iter_n in range(n_iter):
            if (lr_type=="inverse"):
                lr_cur = lr_cur/(iter_n+1)
            batches = self.get_batches(batch_size,X,y)
            for batch in batches:
                X_cur ,y_cur = batch
                self.X_cur = X_cur
                self.y_cur = y_cur
                grad = get_grad(parameters)
                parameters = parameters - lr_cur*grad
        self.coef_ = parameters


    def fit_normal(self, X, y):
        '''
        Function to train model using the normal equation method.
        : param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        : param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        > return None
        '''
        self.X = X
        self.y = y
        X = X.to_numpy()
        y = y.to_numpy().reshape(-1,1)
        if (self.fit_intercept):
            bias = np.ones((X.shape[0],1))
            X = np.append(bias, X, axis = 1)
        temp = np.linalg.inv(X.T.dot(X))
        self.coef_ = temp.dot(X.T.dot(y))

    def predict(self, X):
        '''
        Funtion to run the LinearRegression on a data point
        :param X: pd.DataFrame with rows as samples and columns as features
        > return: y: pd.Series with rows corresponding to output variable. The output variable in a row is the prediction for sample in corresponding row in X.
        '''
        X = X.to_numpy()
        if (self.fit_intercept):
            bias = np.ones((X.shape[0], 1))
            X = np.append(bias, X, axis = 1)
        y_hat = X.dot(self.coef_).reshape(-1)
        return pd.Series(y_hat)
    
    def cost(self, parameters):
        """
        Cost Function Used for generating gradients by AutoGrad
        : param parameters : Numpy array of the current learned parameters
        > return root mean sum of squared error
        """
        y_hat = np_a.dot(self.X_cur, parameters)
        return np_a.sum((self.y_cur-y_hat)**2/len(y_hat))
    
    def cost1(self, parameters):
        """
        Function that evaluvates the cost for the given parameters
        : param parameters : Numpy array of the current learned parameters
        > return sum of squared error
        """
        n = len(self.X)
        t1 = parameters[0]
        t2 = parameters[1]
        cost = 0
        for row in range(n):
            x = np.array(self.X.loc[row]).reshape(-1)
            cost += pow((self.y[row]-(t1 + x[0]*t2)),2)
        cost = cost/n
        return cost

    def cost_function(self, grid, X, y):
        """
        : param grid :a numpy array of values for (theta0,theta1) used for the plotting the contour
        : param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        : param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        > return numpy array , cost for each (theta0,theta1) in grid
        """
        n = len(X)
        z = []
        for point in grid:
            t1 = point[0]
            t2 = point[1]
            temp = 0
            for row in range(n):
                x = np.array(X.loc[row]).reshape(-1)
                temp += pow((y[row]-(t1 + x[0]*t2)),2)
            temp = temp/n
            z.append(temp)
        return np.array(z)

    def plot_surface(self):
        """
        Function to plot RSS (residual sum of squares) in 3D. A surface plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1 by a
        red dot. Uses self.coef_ to calculate RSS. Plot must indicate error as the title.
        """
        X = self.X
        y = self.y
        parameters_h = self.parameters_history
        minx = miny = float('inf')
        maxx = maxy = float('-inf')
        for i in range(len(parameters_h)):
            minx = min(parameters_h[i][0],minx)
            miny = min(parameters_h[i][1],miny)
            maxx = max(parameters_h[i][0],maxx)
            maxy = max(parameters_h[i][1],maxy)
        plot_step = 1
        x_axis, y_axis = np.meshgrid(np.arange(minx, 2*maxx, plot_step),np.arange(miny, 2*maxy, plot_step))
        grid = np.c_[x_axis.ravel(), y_axis.ravel()]
        Z = self.cost_function(grid,X,y).reshape(y_axis.shape)
        ax = plt.axes(projection = "3d")
        ax.view_init(45,45)
        ax.plot_surface(x_axis, y_axis, Z, cmap=cm.coolwarm, alpha = 0.7)
        for frame_no in range(0,200):
            if (frame_no%10==0):
                ax.scatter(parameters_h[frame_no][0], parameters_h[frame_no][1], self.cost1(parameters_h[frame_no]), color = "r")
                plt.savefig("./images/surface/"+str(frame_no)+".png")
        plt.close()

    def plot_line_fit(self):
        """
        Function to plot fit of the line (y vs. X plot) based on chosen value of t_0, t_1. Plot must
        indicate t_0 and t_1 as the title.
        """
        X = self.X
        y = self.y
        x_axis = X.to_numpy().reshape(-1)
        y_axis = y.to_numpy().reshape(-1)
        parameters_h = self.parameters_history
        for frame_no in range(0,200):
            plt.scatter(x_axis,y_axis,color="r")
            t0, t1 = parameters_h[frame_no][0], parameters_h[frame_no][1]
            if (frame_no%10==0):
                axes = plt.gca()
                x_vals = np.array(axes.get_xlim())
                y_vals = t0 + t1*x_vals
                plt.plot(x_vals, y_vals, '--')
                plt.savefig("./images/line/"+str(frame_no)+".png")
                plt.close()

    def plot_contour(self):
        """
        Plots the RSS as a contour plot. A contour plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1, and the
        direction of gradient steps. Uses self.coef_ to calculate RSS.
        """
        X = self.X
        y = self.y
        parameters_h = self.parameters_history
        minx = miny = float('inf')
        maxx = maxy = float('-inf')
        for i in range(len(parameters_h)):
            minx = min(parameters_h[i][0],minx)
            miny = min(parameters_h[i][1],miny)
            maxx = max(parameters_h[i][0],maxx)
            maxy = max(parameters_h[i][1],maxy)
        plot_step = 1
        x_axis, y_axis = np.meshgrid(np.arange(minx, 2*maxx, plot_step),np.arange(miny, 2*maxy, plot_step))
        grid = np.c_[x_axis.ravel(), y_axis.ravel()]
        Z = self.cost_function(grid,X,y).reshape(y_axis.shape)
        plt.contour(x_axis,y_axis,Z)
        for frame_no in range(0,200):
            if (frame_no%10==0):
                plt.scatter(parameters_h[frame_no][0], parameters_h[frame_no][1],color = "r")
                plt.savefig("./images/contour/"+str(frame_no)+".png")
        plt.close()