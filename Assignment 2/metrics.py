import pandas as pd
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