# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 11:57:29 2020

@author: gaura
"""
import numpy as np
import pandas as pd
import pickle

dataset=pd.read_csv('corona.csv')
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,5].values


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,random_state=0)

'''
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
'''

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)

pickle.dump(regressor,open('model.pkl','wb'))

model=pickle.load(open('model.pkl','rb'))


'''
y_pred_ans=regressor.predict([[104,0,100,0,1]])
y_pred_ans=(np.round(y_pred_ans*100))
'''

