#####################################
# Main functions: Model training and prediction
#####################################
import warnings
warnings.filterwarnings("ignore")
from preprocess import data_preprocess
import numpy as np
import pandas as pd
import eval
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

###############################################
# Trained function
# data_path: Path to read in data
# month: Month, value range1， 2， 3， 4
# model: Trained model
# model_name: The name of the trained model
###############################################
def train(data_path, month, model,  model_name):

    # Call the data preprocessing method to get the processed data set
    X = data_preprocess(data_path, month)

    # Take 70% of the data as the training set and 30% of the data as the test set
    # The input features are from the second column to the last column
    # The output label is the first column
    train_size = int(X.shape[0] * 0.7)
    x_train = X.iloc[:train_size, 2:]
    y_train = X.iloc[:train_size, 1:2]
    x_test = X.iloc[train_size:, 2:]
    y_test = X.iloc[train_size:, 1:2]

    # Convert data format
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # Training model, predictive model
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    # Output and save the prediction results
    pre_out = np.array(pred).reshape(-1, 1)
    tru_out = np.array(y_test).reshape(-1, 1)

    out = np.concatenate((tru_out, pre_out), axis=1)
    out = pd.DataFrame(out, columns=['true', 'pred'])
    out.to_csv('%s_%s.csv' % (model_name, str(month)))

    # Visualize the results
    plt.figure(figsize=(12, 3))
    plt.plot(y_test[:100])
    plt.plot(pred[:100])
    plt.show()

    # Use mse, mae, mape to evaluate model performance
    mse = eval.MSE(y_test, np.array(pred).reshape(-1,1))
    mae = eval.MAE(y_test, np.array(pred).reshape(-1,1))
    mape = eval.MAPE(y_test, np.array(pred).reshape(-1,1))
    print('mse: ', mse)
    print('mae: ', mae)
    print('mape: ', mape)

# Initialize the model
DT_model = DecisionTreeRegressor(max_depth=20)
LR_model = LinearRegression()

model_list = [DT_model, LR_model]
model_name_list = ['DT_model', 'LR_model']

#Train and predict
# for i in range(len(model_list)):
#     print("==========================================================================")
#     for month in range(1, 5):
#         print(model_name_list[i] + ' ' + str(month) + '月')
#         train('data/green_tripdata_2020-0' + str(month) + '.csv', month, model_list[i], model_name_list[i])

train('data/green_tripdata_2020-02.csv', 2, LR_model, model_name_list[1])