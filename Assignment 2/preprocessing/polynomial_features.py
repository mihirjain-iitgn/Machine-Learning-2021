''' In this file, you will utilize two parameters degree and include_bias.
    Reference https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PolynomialFeatures():
    def __init__(self, degree=2,include_bias=True):
        """
        : param degree : (int) max degree of polynomial features
        : param include_bias : (boolean) specifies wheter to include bias term in returned feature array.
        """
        self.degree = degree
        self.include_bias = include_bias
    
    def transform(self,X):
        """
        Transform data to polynomial features
        Generate a new feature matrix consisting of all polynomial combinations of the features with degree less than or equal to the specified degree. 
        For example, if an input sample is  np.array([a, b]), the degree-2 polynomial features with "include_bias=True" are [1, a, b, a^2, ab, b^2].
        : param X : (np.array) Dataset to be transformed
        > returns (np.array) Tranformed dataset.
        """
        n = len(X)
        degree = self.degree
        if (self.include_bias):
            X_modified = np.ones(n*degree+1)
        else:
            X_modified = np.ones(n*degree)
        for col in range(n):
            for deg in range(1,degree+1):
                pos = (degree*col) + (deg-1)
                if (self.include_bias):
                    pos += 1
                X_modified[pos] = pow(X[col],deg)
        return X_modified