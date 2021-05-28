from sklearn.linear_model import LinearRegression 
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from Preprocessing import PreProcessing
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
class LinearRegressor:
    def __init__(self):
        prePro=PreProcessing()
        X_train, X_test, y_train, y_test=prePro.PreProcess()
        lm = LinearRegression() 
        lm.fit(X_train,y_train)
        print(lm.intercept_)
        print(lm.coef_)
        predictions = lm.predict(X_test)
        plt.scatter(np.array(y_test),predictions)
        sns.distplot((y_test-predictions),bins=50)
        print("Accureccy of LinearRegression")
        print('R-Square:',r2_score(y_test, lm.predict(X_test)))
        print('MAE:', metrics.mean_absolute_error(y_test, predictions))
        print('MSE:', metrics.mean_squared_error(y_test, predictions))
        print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
        fig, ax = plt.subplots() 
        ax.scatter(np.array(y_test),predictions,edgecolors=(0, 0, 0))
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title("Actual vs Predicted")
        plt.show()