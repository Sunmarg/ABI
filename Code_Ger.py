# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 13:30:16 2020

@author: Sunmarg Das
"""
############............install anaconda.....
###...install xgboost ....
####....run "pip install xgboost" in anaconda prompt....

## import librares
import pandas as pd
import matplotlib.pylab as plt
from datetime import date
import xlsxwriter
from statsmodels.tsa.seasonal import seasonal_decompose 
import numpy as np
import math
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()


## loading data set
new_data= pd.read_excel('data_Ger.xlsx')
new_data1=new_data.copy()
label_encoder=label_encoder.fit(new_data['Month'])
new_data.dropna(subset=['Total_amount_of_delayed_invoice'], how='all', inplace=True)


workbook1 = xlsxwriter.Workbook('Predictions_Ger.xlsx')
excl = workbook1.add_worksheet()
excl.write(0, 0,'Month')     
excl.write(0, 1, 'Year')  
excl.write(0, 2,'Forecasted_Outlier')
excl.write(0, 3,'Forecasted_Normal')
    
def train_norm(train_till,test_start,test_end,pred_start,pred_end,m):
    
    X_train=new_data1[['Month', 'Year','Industrial_Production_Manufacturing_Index_quarter',
        
       'Target'
    ]].iloc[:train_till,:]
    X_train['Month'] = label_encoder.transform(X_train['Month'])
    X_test=new_data1[['Month','Year','Industrial_Production_Manufacturing_Index_quarter',
        
       'Target'
           ]].iloc[test_start:test_end,:]
  
    X_test['Month'] = label_encoder.transform(X_test['Month'])
    y_train=new_data1[['Total_amount_of_delayed_invoice']].iloc[:train_till,:]
    y_train.reset_index(inplace=True)
    y_test=new_data1[['Total_amount_of_delayed_invoice']].iloc[test_start:test_end,:]
    y_test.reset_index(inplace=True)
    y_train.drop(["index"],axis=1,inplace=True)
    y_test.drop(["index"],axis=1,inplace=True)
    reg = xgb.XGBRegressor(n_estimators=100)
    
    
    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            early_stopping_rounds=50, 
            verbose=False)

    X_test1=new_data1[['Month','Year','Industrial_Production_Manufacturing_Index_quarter',
        
       'Target']].iloc[pred_start:pred_end+m,:]
    X_test1.reset_index(inplace=True)
    X_test1.drop(["index"],axis=1,inplace=True)

    for i in range(0,m):
        excl.write(i+1,0,X_test1["Month"][i])
        excl.write(i+1,1,X_test1["Year"][i])
    X_test1['Month'] = label_encoder.transform(X_test1['Month'])
    
    predictions=reg.predict(X_test1)
  
    for i in range(0,m):
        excl.write(i+1,2,predictions[i])
    
def train_out(train_till,test_start,test_end,pred_start,pred_end,m):
    
    X_train=new_data1[['Month', 'Year','Industrial_Production_Manufacturing_Index_quarter',
        
       'Target'
    ]].iloc[:train_till,:]
    X_train['Month'] = label_encoder.transform(X_train['Month'])
    X_test=new_data1[['Month','Year','Industrial_Production_Manufacturing_Index_quarter',
        
       'Target'
           ]].iloc[test_start:test_end,:]
  
    X_test['Month'] = label_encoder.transform(X_test['Month'])
    y_train=new_data1[['Total_amount_of_delayed_invoice']].iloc[:train_till,:]
    y_train.reset_index(inplace=True)
    y_test=new_data1[['Total_amount_of_delayed_invoice']].iloc[test_start:test_end,:]
    y_test.reset_index(inplace=True)
    y_train.drop(["index"],axis=1,inplace=True)
    y_test.drop(["index"],axis=1,inplace=True)
    reg = xgb.XGBRegressor(n_estimators=100)
    
    
    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            early_stopping_rounds=50, 
            verbose=False)

    X_test1=new_data1[['Month','Year','Industrial_Production_Manufacturing_Index_quarter',
        
       'Target']].iloc[pred_start:pred_end+m,:]
    X_test1.reset_index(inplace=True)
    X_test1.drop(["index"],axis=1,inplace=True)

    
    X_test1['Month'] = label_encoder.transform(X_test1['Month'])
    
    predictions=reg.predict(X_test1)
    
    for i in range(0,m):
        excl.write(i+1,3,predictions[i])
    workbook1.close()


m=6  ##forecast ahead
train_till=new_data["Total_amount_of_delayed_invoice"].size #train till the data when prediction is poor
test_start=new_data["Total_amount_of_delayed_invoice"].size-3 # start testing the data 3 months earlier
test_end=new_data["Total_amount_of_delayed_invoice"].size #end testing till the last data
pred_start=new_data["Total_amount_of_delayed_invoice"].size #predict for future values
pred_end=new_data["Total_amount_of_delayed_invoice"].size+1  #predict till the month you need
train_norm(train_till,test_start,test_end,pred_start,pred_end,m)

train_till=new_data["Total_amount_of_delayed_invoice"].size-1 #train till the data when prediction is poor
test_start=new_data["Total_amount_of_delayed_invoice"].size-1 # start testing the data 3 months earlier
test_end=new_data["Total_amount_of_delayed_invoice"].size #end testing till the last data
pred_start=new_data["Total_amount_of_delayed_invoice"].size #predict for future values
pred_end=new_data["Total_amount_of_delayed_invoice"].size+1 #predict tillthe month you need
train_out(train_till,test_start,test_end,pred_start,pred_end,m)

