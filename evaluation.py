#Importing Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os 

#Importing the Trained Models for Predictions 
from src.lstm import lstm_train_predict
from src.randomForest import randomForest_train_predict
from src.svr import svr_train_predict

#1) IMPORTING DATA ----------------------------------------------------------------------

#CODE HERE:
data = pd.read_csv('data/stocks_data.csv')

df = pd.DataFrame(data)
#2) EVALUATION FUNCTION -----------------------------------------------------------------

#CODE HERE:
def evaluate_model(model, stock, Y_test, Y_predictions):
    print(f"{model} evaluation for {stock}:")

    mse = mean_squared_error(Y_test, Y_predictions)
    r2 = r2_score(Y_test, Y_predictions)

    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    return mse, r2

#3) PREDICTING STOCK --------------------------------------------------------------------

#CODE HERE:
stocks = data["Name"].unique() #Extracting unique stock names from the data

# Features & Target
features = ['open', 'high', 'low', 'volume', 'return', 'rolling_mean', 'rolling_std'] #Features to be used for prediction
target = 'close'

results = {'lstm_r2' : [], 'lstm_mse' : [],
           'randomForest_r2' : [], 'randomForest_mse' : [],
          'svr_r2' : [], 'svr_mse' : []}

for stock in stocks: #Iterating over each stock to be predicted by each model

    # ---- Stock Data Preparation ----

    #CODE HERE:
    stock_data = data[data['Name'] == stock]

    X = stock_data[features]
    y = stock_data[target]
    dates = stock_data['date']

    X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(X, y, dates, test_size=0.2, random_state= 42)

    # Sort Test Data by Date
    X_test_sorted = X_test.sort_index()
    y_test_sorted = y_test.sort_index()
    dates_test_sorted = pd.to_datetime(dates_test.sort_values())

    # ---- LSTM Model ----

    #CODE HERE:
    Y_predictions = lstm_train_predict(X_train, y_train, X_test_sorted)
    mse, r2 = evaluate_model('LSTM', stock, y_test_sorted, Y_predictions)

    results['lstm_r2'].append(r2)
    results['lstm_mse'].append(mse)

    plt.figure(figsize=(10, 6))
    plt.plot(dates_test_sorted, y_test_sorted.values, label='True Values', color='green')
    plt.plot(dates_test_sorted, Y_predictions, label='LSTM Predictions', color='purple')
    plt.title(f'{stock} - LSTM')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.savefig(os.path.join('Workshop 4/result/lstm_plots', f'{stock}_lstm.png'))
    plt.close()

    # ---- Random Forest Model ----

    #CODE HERE:

    Y_predictions = randomForest_train_predict(X_train, y_train, X_test_sorted)
    mse, r2 = evaluate_model('Random Forest', stock, y_test_sorted, Y_predictions)

    results['randomForest_r2'].append(r2)
    results['randomForest_mse'].append(mse)

    plt.figure(figsize=(10, 6))
    plt.plot(dates_test_sorted, y_test_sorted.values, label='True Values', color='green')
    plt.plot(dates_test_sorted, Y_predictions, label='Random Forest Predictions', color='purple')
    plt.title(f'{stock} - Random Forest')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.savefig(os.path.join('Workshop 4/result/randomForest_plots', f'{stock}_randomForest.png'))
    plt.close()

    # ---- SVR (Support Vector Regressor) Model ----

    #CODE HERE:
    Y_predictions = svr_train_predict(X_train, y_train, X_test_sorted)
    mse, r2 = evaluate_model('SVR', stock, y_test_sorted, Y_predictions)

    results['svr_r2'].append(r2)
    results['svr_mse'].append(mse)

    plt.figure(figsize=(10, 6))
    plt.plot(dates_test_sorted, y_test_sorted.values, label='True Values', color='green')
    plt.plot(dates_test_sorted, Y_predictions, label='SVR Predictions', color='purple')
    plt.title(f'{stock} - SVR')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.savefig(os.path.join('Workshop 4/result/svr_plots', f'{stock}_svr.png'))
    plt.close()

#4) PRINTING RESULTS --------------------------------------------------------------------
print("Results:")

# LSTM Average Results 
l 