import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from .utils import entropy, information_gain, gini_index, information_gain1, gini_gain, gini_gain1

np.random.seed(42)

class DecisionTree():
    def __init__(self, criterion, max_depth=10):
        """
        Initialization of the tree
        Inputs:
        criterion : {"information_gain", "gini_index"} # criterion won't be used for regression
        max_depth : The maximum depth the tree can grow to
        Other Parameters:
        Graph : A dictionary that will store all the nodes and edges of the tree
        type : An integer that will indicate the type of IO, 0 for D Input R Output, 1 for D Input R Output, 2 for R Input D Output, 3 for R Input D Output
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.type = -1
        self.Graph = {}
        """
        Structure of self.Graph
        Discrete Input
            {root : {attr_name : {val1 : {...}, val2 : {...}}}}
            > root is a dummy starting point.
            > attr_name is the attribute
            > val1, val2 are values of attr_name
            > the same patterm repeats
        Real Input
            {root : {attr_name: {(val,0) : {...}, (val,1) : {...}}}}
            > root is a dummy starting point.
            > attr_name is the attribute
            > val is the threashold value of the attribute
            > (val,0) and (val,1) goes to the subtree when value is less or high than threashold respectively.
        """

    def fit(self, X, y):
        """
        Function to train and construct the decision tree
        Inputs:
        X : pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y : pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.X = X.copy()
        self.y = y.copy()
        attributes = list(self.X.columns)

        # Adding an extra column in X for the output variable
        X.loc[:,"Output"] = self.y

        if (X.dtypes[0].name=="category"):
            if (y.dtype.name=="category"):
                # Discrete Input Discrete Output
                self.type = 0
            else:
                # Discrete Input Real Output
                self.criterion = "information_gain"
                self.type = 1
            
            # "root" is the name of the dummy root created for the tree
            self.Graph["root"] = self.di(X,attributes,0)
        else:
            if (y.dtype.name=="category"):
                # Real Input Discrete Output
                self.type = 2
            else:
                # Real Input Real Output
                self.criterion = "information_gain"
                self.type = 3

            # "root" is the name of the dummy root created for the tree
            self.Graph["root"] = self.ri(X,attributes,0)
        
        X.drop(["Output"],axis = 1,inplace = True)

    def predict(self, X):
        """
        Funtion to run the decision tree on a data point
        Input:
        X : pd.DataFrame with rows as samples and columns as features
        Output:
        y : pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        # The code below walks down the tree untill a leaf is reached
        # For exact details about the structure of self.Graph refer description in init function.

        if self.type==0 or self.type==1:
            # Discrete Input Discrete Output/Real Output
            y = []
            for i in range(len(X)):
                attr = self.Graph["root"]
                while(type(attr)==dict):
                    attr_name = list(attr.keys())[0]
                    attr = attr[attr_name][X.loc[i][attr_name]]
                # leaf is reached
                y.append(attr)
            return pd.Series(y)
        else:
            # Discrete Input Discrete Output/Real Output
            y = []
            for i in range(len(X)):
                attr = self.Graph["root"]
                while(type(attr)==dict):
                    attr_name = list(attr.keys())[0]
                    val = list(attr[attr_name].keys())[0][0]
                    if (X.loc[i][attr_name]>=val):
                        attr = attr[attr_name][(val,1)]
                    else:
                        attr = attr[attr_name][(val,0)]
                # leaf is reached
                y.append(attr)
            return pd.Series(y)
    
    def subplot_1(self, Graph, n_tabs):
        """
        Recursive function to print the tree for discrete input

        Inputs:
        Graph :  The current subtree in the traversal
        n_tabs : The number of tabs for this level in the tree.
        """
        # The code below walks does a pre-order traversal of the tree
        # For exact details about the structure of self.Graph refer  description in init function.

        attr_name = list(Graph.keys())[0]
        print("\t"*(n_tabs),"feature name :",attr_name)
        for val in list(Graph[attr_name].keys()):
            print("\t"*(n_tabs+1),"feature value :",val)
            sub_graph = Graph[attr_name][val]
            if (type(sub_graph)==dict):
                self.subplot_1(sub_graph, n_tabs+2)
            else:
                print("\t"*(n_tabs+2),"class :", sub_graph)
    
    def subplot_2(self, Graph, n_tabs):
        """
        Recursive function to print the tree for discrete input
        
        Inputs:
        Graph : The current subtree in the traversal
        n_tabs : The number of tabs for this level in the tree.
        """
        # The code below walks does a pre-order traversal of the tree
        # For exact details about the structure of self.Graph refer description in init function.

        attr_name = list(Graph.keys())[0]
        print("\t"*(n_tabs),"feature name :",attr_name)
        for val in list(Graph[attr_name].keys()):
            if (val[1]==1):
                des = "greater"
            else:
                des = "lower"
            print("\t"*(n_tabs+1),"feature threashold :", val[0]," ",des)
            sub_graph = Graph[attr_name][val]
            if (type(sub_graph)==dict):
                self.subplot_2(sub_graph, n_tabs+2)
            else:
                print("\t"*(n_tabs+2), "prediction :",sub_graph)

    def plot(self):
        """
        Function to plot the tree
        """
        attr = self.Graph["root"]
        if (self.type == 0 or self.type == 1):
            self.subplot_1(attr, 0)
        else:
            self.subplot_2(attr, 0)

    def best_split(self, X, y, attributes):
        """
        function to find the best split in the case of discrete input
        Inputs:
        X : DataFrame of Current Data
        y : Series of Output corresponding the current Datafrane
        attributes : Python list of the remaining attributes
        Outputs:
        attr : the attributes upon which the split should be made
        """
        if (self.criterion=="information_gain"):
            global_if = float('-inf') # the highest value of information gain/gini gain seen so far
            attr = None
            for attribute in attributes:
                attr_val = X[attribute].copy()
                cur_if = information_gain(y,attr_val,self.type)
                if (cur_if>global_if):
                    # Update when a better split is receieved
                    global_if = cur_if
                    attr = attribute
            return attr
        else:
            global_if = float('inf')
            attr = None
            for attribute in attributes:
                attr_val = X[attribute].copy()
                cur_if = gini_gain(y,attr_val)
                if (global_if>cur_if):
                    # Update when a better split is receieved
                    global_if = cur_if
                    attr = attribute
            return attr

    def best_split1(self,X,attributes):
        """
        function to find the best split in the case of real input
        Inputs:
        X : DataFrame of Current Data
        y > Series of Output corresponding the current Datafrane
        attributes > Python list of the remaining attributes
        """
        if (self.criterion=="information_gain"):
            global_if = float('-inf') # the highest value of varience seen so far
            attr , val = None, None
            for attribute in attributes[::-1]:
                attr_val = pd.Series(X[attribute].unique()).sort_values(ignore_index=True)
                last_val = attr_val[0]
                for i in range(1,attr_val.size):
                    cur_val = attr_val[i]
                    valc = round((last_val+cur_val)/2,4)
                    last_val = cur_val
                    cur_if = information_gain1(valc,X[attribute],X["Output"],self.type)
                    if (cur_if>global_if):
                        global_if,attr,val = cur_if,attribute,valc
            return attr,val
        else:
            global_if = float('inf') # the lowest value of varience seen so far
            attr , val = None, None
            for attribute in attributes[::-1]:
                attr_val = pd.Series(X[attribute].unique()).sort_values(ignore_index=True)
                last_val = attr_val[0]
                for i in range(1,attr_val.size):
                    cur_val = attr_val[i]
                    valc = round((last_val+cur_val)/2,4)
                    last_val = cur_val
                    cur_if = gini_gain1(X["Output"],X[attribute], valc)
                    if (global_if>cur_if):
                        global_if,attr,val = cur_if,attribute,valc
            return attr,val


    def di(self, X, attributes, depth):
        """function to learn the tree in the case of discrete input
        inputs:
        X: DataFrame of the current data
        attributes: list of Remaining attributes
        depth: current depth
         Output:
        Graph : A dictionary respresenting the substree
        """
        y = X["Output"]
        if (y.unique().size>1):
            if (len(attributes)>=1):
                next_attr = self.best_split(X,y,attributes)
                attributes.remove(next_attr)
                Graph = {next_attr : {}}
                for value in self.X[next_attr].unique():
                    X_new = X.loc[X[next_attr]==value]
                    X_new.reset_index(drop=True,inplace=True)
                    if (len(X_new)==0):
                        # No data points corresponding to this attributes in the current data
                        if (self.type==0):
                            # discrete output
                            Graph[next_attr][value] = y.mode()[0]
                        else:
                            # real output
                            Graph[next_attr][value] = y.mean()
                    elif (self.max_depth==depth+1):
                        # Max depth is reached
                        if (self.type==0):
                            # discrete output
                            Graph[next_attr][value] = X_new["Output"].mode()[0]
                        else:
                            # real output
                            Graph[next_attr][value] = X_new["Output"].mean()
                    else:
                        Graph[next_attr][value] = self.di(X_new,attributes.copy(),depth+1)
                return Graph
            else:
                # No attributes are left
                if (self.type==0):
                    # Discrete Output
                    return y.mode()[0]
                else:
                    # Real Output
                    return y.mean()
        else:
            # All Output classes are same
            return y[0]
    


    def ri(self,X,attributes,depth):
        """function to learn the tree in the case of discrete input
        inputs:
        X: DataFrame of the current data
        attributes: list of Remaining attributes
        depth: current depth
        Output:
        Graph : A dictionary respresenting the substree
        """
        y = X["Output"]
        if (y.unique().size>1):
            next_attr,val = self.best_split1(X,attributes)
            if (next_attr==None):
                # This handles a very rare corner case
                # when all input columns have the same value 
                # but the outputs are different
                if (self.type==2):
                    return y.mode()[0]
                else:
                    return y.mean()
            else:
                Graph = {next_attr:{}}
                X1 = X.loc[X[next_attr]<val]
                X2 = X.loc[X[next_attr]>=val]
                if (len(X1)==0 or len(X2)==0):
                    if (self.type==2):
                        return y.mode()[0]
                    else:
                        return y.mean()
                elif (self.max_depth>depth+1):
                    X1.reset_index(drop=True, inplace = True)
                    # left subtree
                    Graph[next_attr][(val,0)] = self.ri(X1,attributes,depth+1)
                    X2.reset_index(drop=True, inplace = True)
                    # right subtree
                    Graph[next_attr][(val,1)] = self.ri(X2,attributes,depth+1)
                else:
                    # Max Depth is reached
                    if (self.type==2):
                        # discrete output
                        Graph[next_attr][(val,0)] = X1["Output"].mode()[0]
                        Graph[next_attr][(val,1)] = X2["Output"].mode()[0]
                    else:
                        # real output
                        Graph[next_attr][(val,0)] = X1["Output"].mean()
                        Graph[next_attr][(val,1)] = X2["Output"].mean()
                return Graph
        else:
            # All Outputs are the same
            return y[0]