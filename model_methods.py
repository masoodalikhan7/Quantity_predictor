# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 20:21:28 2021

@author: masoodalikhan.k
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle
def predict(arr):
    # Load the model
    filename = 'saved_model_c.sav'
    saved_pred = pickle.load(open(filename, 'rb'))
    # return prediction as well as class probabilities
    preds = saved_pred.predict([arr])[0]
    return ([np.argmax(preds)], preds)