import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import autograd.numpy as np_a
from autograd import elementwise_grad
from autograd.scipy.special import logsumexp


class LogisticRegression():
    def __init__(self, fit_intercept = True, penality = "none", penality_factor = 0,threshold = 0.5, use_autograd = False, multi_class = False, num_classes = 2):
        """
        Parameters :
        > fit_intercept : bool. adds bias term if true.
        > penality : string. "none" : no regularization, "L1" : L1 regularization, "L2" : L2 regularised 
        > penality_factor : float. hypterparameter for regularization
        > threshold : float in (0-1). threshold for binary classification.
        > use_autograd : bool. uses autograd if true else uses hard-coded update rules.
        > multi_class : bool. if true multi-class loss function is used.
        > num_classes : int. Number of classes.
        """
        self.fit_intercept = fit_intercept
        self.penality = penality
        self.penality_factor = penality_factor
        self.threshold = threshold
        self.use_autograd = use_autograd
        self.multi_class = multi_class
        self.num_classes = num_classes
        self.coef = None
    
    def __GetBatches(self, batch_size, X, y):
        """
        Parameters :
        > batch_size : int.
        > X : numpy array.
        > y : numpy array.

        Returns :
        > batches : python list. Contains tuples of (x,y), each tuple is a batch.
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
    
    def __GetOneHot(self,y_cur):
        y_one_hot = np.zeros((len(y_cur), self.num_classes))
        for i in range(len(y_cur)):
            y_one_hot[i][int(y_cur[i])] = 1
        return y_one_hot
    
    def __sigmoid(self, x):
        """
        Parameters :
        > x : numpy array.

        Returns :
        > numpy array of element wise sigmoid of x.
        """
        return 1/(1+np.exp(-x))
    

    def __softmax(self, x):
        """
        Parameters :
        > x : numpy array.

        Returns :
        > numpy array of row-wise sigmoid of x.
        """
        exp = np.exp(x)
        row_sums = np.sum(exp,axis = 1)
        z = exp / row_sums[:, np.newaxis]
        return z
    
    def __cost(self, parameters, X_cur, y_cur):
        """
        Parameters :
        > parameters : numpy array. Current set of wights
        > X_cur : numpy array. Current Batch Input.
        > y_cur : numpy array. Current Batch Output.

        Returns :
        > cur_cost : float. Cross entropy loss with or without regulatization.
        """
        a = np_a.dot(X_cur, parameters)
        z = 1 /(1 + np_a.exp(-a))
        m = len(X_cur)
        cur_cost =  -np_a.sum(y_cur*np_a.log(z) + (1-y_cur)*(np_a.log(1-z)))/m
        if self.penality == "L2":
            cur_cost = cur_cost + self.penality_factor*np_a.dot(parameters.T,parameters)
        elif self.penality == "L1":
            cur_cost = cur_cost + self.penality_factor*np_a.sum(np_a.absolute(parameters))
        return cur_cost
    

    def __cost_multi(self, parameters, X_cur, y_cur):
        """
        Parameters :
        > parameters : numpy array. Current set of wights
        > X_cur : numpy array. Current Batch Input.
        > y_cur : numpy array. Current Batch Output.

        Returns :
        > cur_cost : float. Multi-class Logistic Regression loss with or without regulatization.
        """
        z = np_a.dot(X_cur, parameters)
        a = z - logsumexp(z, axis=1, keepdims=True)
        n = len(X_cur)
        cost = 0
        for i in range(n):
            j = int(y_cur[i][0])
            cost = cost + a[i][j]
        cost = -cost/n
        return cost
    
    def __StepFunction(self,y):
        """
        Parameters :
        > y : numpy array.

        Returns :
        > All entries of y that are less than threshold are made 0,rest 1.
        """
        y[y >= self.threshold] = 1
        y[self.threshold > y] = 0
        return y
    
    def fit(self, X, y, batch_size, epochs=500, lr=0.01, lr_type="constant"):
        """
        Parameters :
        > X : numpy array. Training Data.
        > y : numpy array. Training Data labels.
        > batch_size : int. Number of batches.
        > epochs : int. Number of epochs.
        > lr : float.  Learning Rate.
        > lr_type : string. if "constant" learning rate remains constant throughout the learning process
                            elif "inverse" learning rate decreases inverly with the epochs.
        """
        self.X = X.copy()
        self.y = y.copy().reshape(-1,1)
        if (self.fit_intercept):
            bias = np.ones((X.shape[0], 1))
            self.X = np.append(bias, self.X, axis = 1)
        
        if self.use_autograd:
            if self.penality == "L1" or self.penality == "L2" or self.penality == "none":
                self.__fit2(batch_size,epochs,lr,lr_type)
            else:
                print("No such penality")
                exit(0)
        else:
            if self.penality == "none":
                self.__fit1(batch_size,epochs,lr,lr_type)
            elif self.penality == "L1" or self.penality == "L2":
                print("L1 and L2 penality are only avaliable with Autograd")
                exit(0)
            else:
                print("No such penality")
                exit(0)

    
    def __fit1(self,batch_size, epochs, lr, lr_type):
        """
        Parameters :
        > batch_size : int. Number of batches.
        > epochs : int. Number of epochs.
        > lr : float.  Learning Rate.
        > lr_type : string. if "constant" learning rate remains constant throughout the learning process
                            elif "inverse" learning rate decreases inverly with the epochs.
        """
        X = self.X
        y = self.y
        lr_cur = lr
        if self.multi_class:
            parameters = np.ones((X.shape[1],self.num_classes))
        else:
            parameters = np.ones((X.shape[1],1))
        batches = self.__GetBatches(batch_size,X,y)
        for iter_n in range(epochs):
            if (lr_type=="inverse"):
                lr_cur = lr_cur/(iter_n+1)
            for batch in batches:
                X_cur ,y_cur = batch
                if self.multi_class:
                    y_hat = self.__softmax(np.dot(X_cur,parameters))
                    y_one_hot = self.__GetOneHot(y_cur)
                    parameters -= (lr_cur*(1/len(X_cur))*(X_cur.T.dot(y_hat-y_one_hot)))
                else:
                    y_hat = self.__sigmoid(np.dot(X_cur,parameters))
                    parameters -= (lr_cur*(1/len(X_cur))*(X_cur.T.dot(y_hat-y_cur)))
        self.coef = parameters
    
    def __fit2(self,batch_size, epochs, lr, lr_type):
        """
        Parameters :
        > batch_size : int. Number of batches.
        > epochs : int. Number of epochs.
        > lr : float.  Learning Rate.
        > lr_type : string. if "constant" learning rate remains constant throughout the learning process
                            elif "inverse" learning rate decreases inverly with the epochs.
        """
        X = self.X
        y = self.y
        lr_cur = lr
        if self.multi_class:
            parameters = np.ones((X.shape[1],self.num_classes))
            get_grad = elementwise_grad(self.__cost_multi,0)
        else:
            parameters = np.ones((X.shape[1],1))
            get_grad = elementwise_grad(self.__cost,0)
        batches = self.__GetBatches(batch_size,X,y)
        for iter_n in range(epochs):
            if (lr_type == "inverse"):
                lr_cur = lr_cur/(iter_n+1)
            for batch in batches:
                X_cur ,y_cur = batch
                grad = get_grad(parameters,X_cur,y_cur)
                parameters -= (lr_cur*grad)
        self.coef = parameters

    def predict(self, X):
        """
        Parameters :
        > X : numpy array. Test Data.

        Returns :
        > ouput : numpy array. Predictions from the learned model.
        """
        if (self.fit_intercept):
            bias = np.ones((X.shape[0], 1))
            X = np.append(bias, X, axis = 1)
        if self.multi_class:
            a = np.dot(X, self.coef)
            z = self.__softmax(a)
            output = np.argmax(z, axis = 1)
        else:
            y_hat = self.__sigmoid(X.dot(self.coef).reshape(-1))
            output = self.__StepFunction(y_hat)
        return output
        
        
    def plot(self):
        """
        Plots the decison boundary of the learned model.
        Note : The code will break if the number of columns in X is more than 2.
        """
        X = self.X
        y = self.y.reshape(-1)
        colors = "rb"
        minx = miny = float('inf')
        maxx = maxy = float('-inf')
        for i in range(len(X)):
            minx = min(X[i][1],minx)
            miny = min(X[i][2],miny)
            maxx = max(X[i][1],maxx)
            maxy = max(X[i][2],maxy)
        plot_step = 0.1
        x_axis, y_axis = np.meshgrid(np.arange(minx, maxx, plot_step),np.arange(miny, maxy, plot_step))
        grid = np.c_[x_axis.ravel(), y_axis.ravel()]
        Z = self.predict(grid).reshape(y_axis.shape)
        fig, ax = plt.subplots(1,1)
        fig.suptitle("Logistic Regression Decison Boundary")
        ax.contourf(x_axis, y_axis, Z, cmap = plt.get_cmap("RdBu"))
        for row in range(len(X)):
            ax.scatter(X[row][1], X[row][2], color = colors[y[row]])
        plt.savefig("./plots/decison_surface.png")
        plt.show()