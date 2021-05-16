def rmse(y, y_hat):
  """
  Parameters :
    > y : Ground Truth.
    > y_hat : Prediction.
  
  Returns :
    > Root Mean squared error.
  """
  n = len(y)
  rmse = 0
  for i in range(n):
    rmse += pow(y_hat[i]-y[i],2)
  rmse = pow(rmse/y.size,0.5)
  return rmse

def accuracy(y,y_hat):
  """
  Parameters :
    > y : Ground Truth.
    > y_hat : Prediction.
  Returns :
  > Accuracy.
  """
  acc = 0
  for i in range(y.size):
    if (y[i]==y_hat[i]):
      acc+=1
  return acc/y.size