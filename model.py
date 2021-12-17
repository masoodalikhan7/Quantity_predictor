# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 19:22:58 2021

@author: masoodalikhan.k
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv("C:/Users/masoodalikhan.k/Documents/New folder (3)/CHVW 4861.csv")

from datetime import datetime
data['Posting Date'] = pd.to_datetime(data['Posting Date'])

data['Quantity'] = data['Quantity'].str.replace(',', '').str.replace('%', '').astype(float)

df = data.loc[data['Movement Type'] == 101]
df

df['month'] = df['Posting Date'].dt.month_name(locale = 'English')
df['month']


df1 = df.loc[df['Base Unit of Measure'] == 'KG']
df1

df2 = df.loc[df['Base Unit of Measure'] == 'LB']
df2

df2['Quantity'] = df2['Quantity']*2.2046

df3 = df.loc[df['Base Unit of Measure'] == 'MT']
df3


df3['Quantity'] = df3['Quantity']*0.001


dff = pd.concat([df1, df2, df3], axis=0)



dff = dff.dropna(subset=['Vendor'])


X = dff[['Material', 'month', 'Vendor']]
Y = dff['Quantity']

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.10, random_state = 0)





pipe = Pipeline([('oe', OrdinalEncoder(handle_unknown = "ignore")), ('Random_Forest', RandomForestRegressor(n_estimators = 100, bootstrap=True, criterion='mse', max_depth=None, random_state=0))], verbose = True)

pipe.fit(X_train, Y_train)


Y_pred = pipe.predict(X_test)


import pickle
saved_model_c = open('saved_model_c.sav', 'wb')
pickle.dump(pipe, saved_model_c)

filename = 'saved_model_c.sav'
saved_pred = pickle.load(open(filename, 'rb'))







