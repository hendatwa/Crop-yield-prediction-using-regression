from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from Preprocessing import PreProcessing
from sklearn.metrics import r2_score
#Edit: added second square bracket above to fix the ValueError problem
class Polynomial:
    def __init__(self):
        poly = PolynomialFeatures(degree=4)
        prePro=PreProcessing()
        X_train,X_test, y_train, y_test=prePro.PreProcess()
        
        X_Poly= poly.fit_transform(X_train)
        poly.fit(X_Poly,y_train)
        X_test_poly= poly.fit_transform(X_test)
        lin=linear_model.LinearRegression()
        lin.fit(X_Poly, y_train)
        plt.scatter(np.array(y_test),lin.predict(poly.fit_transform(X_test)))
        sns.distplot((y_test-lin.predict(poly.fit_transform(X_test))),bins=50);
        print("Accurecy of Polynomial Regression with 4 degree")
        print('R-Square:',r2_score(y_test, lin.predict(poly.fit_transform(X_test))))
        print('MAE:', metrics.mean_absolute_error(y_test, lin.predict(poly.fit_transform(X_test))))
        print('MSE:', metrics.mean_squared_error(y_test, lin.predict(poly.fit_transform(X_test))))
        print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, lin.predict(poly.fit_transform(X_test)))))
        fig, ax = plt.subplots() 
        
        ax.scatter(np.array(y_test),lin.predict(poly.fit_transform(X_test)),edgecolors=(0, 0, 0))
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title("Actual vs Predicted")
        plt.show()