import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

'''
Terminal Commands:
"python3.10 -m venv myenv"

Windows: "myenv\Scripts\activate"
MacOS/Linux: "source myenv/bin/activate"

Library Installation:
pip install pandas 
pip install scikit-learn
pip install matplotlib
pip install numpy
pip install tensorflow
'''

def svr_train_predict(x_train, y_train, x_test):

    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(x_train)
    X_test_scaled = scaler_X.transform(x_test)

    Y_train_scaled = scaler_Y.fit_transform(y_train.values.reshape(-1, 1))

    # Create the SVR model
    model = SVR()

    # Define the hyperparameters to tune
    param_grid = {
        'kernel' : ['linear', 'rbf', 'sigmoid'],
        'C' : [1, 10],
        'gamma' : ['scale']
    }

    # Grid search 
    gridSearch = GridSearchCV(estimator= model, param_grid= param_grid,
                              cv= 3, n_jobs= -1, verbose= 2)
    
    # Fit the model
    gridSearch.fit(X_train_scaled, Y_train_scaled.ravel())

    model = gridSearch.best_estimator_ # Get the best model

    # Make predictions
    predictions_scaled = model.predict(X_test_scaled)

    predictions = scaler_Y.inverse_transform(predictions_scaled.reshape(-1, 1))

    return predictions