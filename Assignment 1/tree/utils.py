import math
import numpy as np
import pandas as pd
from collections import defaultdict

def entropy(Y):
    """
    Function to calculate the entropy 

    Inputs:
    > Y: pd.Series of Labels
    Outpus:
    > Returns the entropy as a float
    """
    n = Y.size
    count = defaultdict(int)
    for i in Y:
        count[i] += 1
    entropy = 0
    for label in count.keys():
        p_label = count[label]/n
        entropy += (p_label*math.log2(p_label))
    entropy = entropy*-1
    return entropy


def varience(Y):
    """
    Function to calculate the varience
    Inputs:
    > Y: pd.Series of Labels
    Outpus:
    > Returns the varience as a float
    """
    return np.var(Y.to_numpy())

def gini_index(Y):
    """
    Function to calculate the gini index
    Inputs:
    > Y: pd.Series of Labels
    Outpus:
    > Returns the gini index as a float
    """
    n = Y.size
    count = defaultdict(int)
    for i in Y:
        count[i] += 1
    gini = 1
    for label in count.keys():
        p_label = count[label]/n
        gini -= (pow(p_label,2))
    return gini

def gini_gain(Y, attr):
    """
    Function to calculate the gini gain for discrete input
    Inputs:
    > Y: pd.Series of Labels
    > attr: pd.Series of attribute at which the gain should be calculated
    Outputs:
    > Return the information gain as a float
    """
    X = pd.concat([attr,Y],axis=1, ignore_index=True)
    gini_gain = 0
    for value in attr.unique():
        y_new = X.loc[X[0]==value][1]
        gini_gain += ((y_new.size/Y.size)*gini_index(Y))
    return gini_gain

def gini_gain1(Y, attr, val):
    """
    Function to calculate the gini gain for real input
    Inputs:
    > Y: pd.Series of Labels
    > attr: pd.Series of attribute at which the gain should be calculated
    Outputs:
    > Return the information gain as a float
    """
    X = pd.concat([attr,Y],axis=1, ignore_index=True)
    gini = 0
    y1 = X[X[0]<val][1]
    y2 = X[X[0]>=val][1]
    gini = ((y1.size/Y.size)*gini_index(y1) + (y2.size/Y.size)*gini_index(y2))
    return gini


def information_gain(Y, attr, cd):
    """
    Function to calculate the information gain for discrete input
    
    Inputs:
    > Y: pd.Series of Labels
    > attr: pd.Series of attribute at which the gain should be calculated
    Outputs:
    > Return the information gain as a float
    """
    X = pd.concat([attr,Y],axis=1, ignore_index=True)
    if (cd==0):
        entropy_ttl = entropy(Y)
        for value in attr.unique():
            y_new = X.loc[X[0]==value][1]
            entropy_ttl-=((y_new.size/Y.size)*entropy(y_new))
        return entropy_ttl
    else:
        varience_ttl = varience(Y)
        for value in attr.unique():
            y_new = X.loc[X[0]==value][1]
            varience_ttl-=((y_new.size/Y.size)*varience(y_new))
        return varience_ttl


def information_gain1(val,attr,Y,cd):
    """
    Function to calculate the information gain for real input
    
    Inputs:
    > Y: pd.Series of Labels
    > attr: pd.Series of attribute at which the gain should be calculated
    Outputs:
    > Return the information gain as a float
    """
    x = pd.concat([attr,Y],axis=1, ignore_index=True)
    if (cd==2):
        entropy_ttl = entropy(Y)
        y1 = x.loc[x[0]<val][1]
        y2 = x.loc[x[0]>=val][1]
        entropy_ttl-=(((y1.size/Y.size)*entropy(y1))+((y2.size/Y.size)*entropy(y2)))
        return entropy_ttl
    else:
        varience_ttl = varience(Y)
        y1 = x[x[0]<val][1]
        y2 = x[x[0]>=val][1]
        varience_ttl-=(((y1.size/Y.size)*entropy(y1))+((y2.size/Y.size)*entropy(y2)))
        return varience_ttl