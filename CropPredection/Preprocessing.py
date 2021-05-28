import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
#-----------------preProcessing---------------------------------------------
#data = pd.read_csv("yield_df.csv")
#print(data)
##NewDataFrame=data.loc[data['Item'].isin(['Wheat'])]
#NewDataFrame=pd.DataFrame(NewDataFrame)
#NewDataFrame.to_csv(r'WheatDataSet2.csv', index=False)
class PreProcessing:
    def PreProcess(self):
       df=pd.read_csv("WheatDataSet2.csv")
        #print(df)
        #print(Data)
#reScale the columns needed 
       scale= StandardScaler()
       df = scale.fit_transform(df)
#Normailizing Data
       df = preprocessing.normalize(df)
       df=pd.DataFrame(df)
        #print(Data)
#spliting data
       cols = df.shape[1]
       X = df.iloc[:,2:cols-1]
       y = df.iloc[:,cols-1:cols]
# convert to matrices 
       X = np.matrix(X.values)
       y=  np.matrix(y.values)
        #split data (30% of dataset for testing)
       x_train, x_test, y_train, y_test = train_test_split( X, y, test_size=0.3)
       return x_train, x_test, y_train, y_test
