############################################################
# Main function: Carry out model evaluation, that is, the definition of evaluation index function.
############################################################
import numpy as np

# Mean square error
def MSE(y_true, y_pred):
    n = len(y_true)
    mse = sum(np.square(y_true - y_pred))/n
    return mse

# Mean absolute percentage error
def MAPE(y_true, y_pred):
    n = len(y_true)
    mape = sum(np.abs((y_true - y_pred)/y_true))/n*100
    return mape

# Mean absolute error
def MAE(y_true, y_pred):
    n = len(y_true)
    mae = sum(np.abs(y_true - y_pred))/n
    return mae