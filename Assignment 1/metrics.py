import pandas as pd

def accuracy(y_hat, y):
    """
    Function to calculate the accuracy

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the accuracy as float
    """
    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    acc = 0
    for i in range(y.size):
        if (y[i]==y_hat[i]):
            acc+=1
    return acc/y.size

def precision(y_hat, y, Class):
    """
    Function to calculate the precision

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the precision as float
    """
    n = y.size
    pre = 0
    ttl = 0
    for i in range(n):
        if (y_hat[i]==Class):
            ttl+=1
            if (y_hat[i]==y[i]):
                pre+=1
    if (ttl>0):
        return pre/ttl
    else:
        return "Not Defined"

def recall(y_hat, y, Class):
    """
    Function to calculate the recall

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the recall as float
    """
    n = y.size
    rec = 0
    ttl = 0
    for i in range(n):
        if (y[i]==Class):
            ttl+=1
            if (y_hat[i]==y[i]):
                rec+=1
    if (ttl>0):
        return rec/ttl
    else:
        return "Not Defined"

def rmse(y_hat, y):
    """
    Function to calculate the root-mean-squared-error(rmse)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the rmse as float
    """
    n = y.size
    rmse = 0
    for i in range(n):
        rmse += pow(y_hat[i]-y[i],2)
    rmse = pow(rmse/y.size,0.5)
    return rmse

def mae(y_hat, y):
    """
    Function to calculate the mean-absolute-error(mae)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the mae as float
    """
    n = y.size
    mae = 0
    for i in range(n):
        mae += abs(y_hat[i]-y[i])
    mae = mae/n
    return mae
