# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 20:45:41 2021

@author: masoodalikhan.k
"""

import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")

sns.set()


from model_methods import predict

st.title("Prediction of Quantity")
st.markdown('**Objective** : Given details the model predicts the Quantity.')
st.markdown('The model can predict the Quantity by taking an input from a particular : **Material, month and Vendor** ')


def predict_class():
    data = list([Material,month,Vendor])
    result = predict(data)
    st.write("The predicted Quantity is ",result)

st.markdown("**Please enter the details **")
    
Material = st.text_input('Enter Material', '')
month = st.text_input('Enter month', '')
Vendor = st.text_input('Enter Vendor', '')
if st.button("Make Prediction"):
    predict_class()