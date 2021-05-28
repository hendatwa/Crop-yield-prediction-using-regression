from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy
from Preprocessing import PreProcessing
from sklearn.metrics import r2_score
import pandas as pd
#turn the simple  DataFrame to A quadretic
class PureQuadretic:
    def __init__(self):
        prePro=PreProcessing()
        X_train,X_test, y_train, y_test=prePro.PreProcess()
        X=self.pureQuadretic(X_train)
        x_test=self.pureQuadretic(X_test)
        lin=linear_model.LinearRegression()
        lin.fit(X, y_train)
        plt.scatter(np.array(y_test),lin.predict(x_test))
        sns.distplot(y_test-lin.predict((x_test)),bins=50);
        print("Accurecy of Pure Quadratic Regression")
        print('R-Square:',r2_score(y_test, lin.predict(x_test)))
        print('MAE:', metrics.mean_absolute_error(y_test, lin.predict(x_test)))
        print('MSE:', metrics.mean_squared_error(y_test, lin.predict(x_test)))
        print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, lin.predict(x_test))))
        fig, ax = plt.subplots() 
        
        ax.scatter(np.array(y_test),lin.predict(self.pureQuadretic(X_test)),edgecolors=(0, 0, 0))
        
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title("Actual vs Predicted")
        plt.show()
    def pureQuadretic(self,df):
            poly = PolynomialFeatures(degree=2)
            x=np.array(df)
            Q=[]
            df= poly.fit_transform(df)
            for i in range(x.shape[0]):
                temp=[]
                for j in range(x.shape[1]):
                        temp.append(x[i][j]*x[i][j])
                        temp.append(x[i][j])
                Q.append(temp)
            Q=pd.DataFrame(Q)
            Q.insert(0,'0', 1)
            Q=np.array(Q)
            return Q
        

    