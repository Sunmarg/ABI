# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 16:16:39 2020

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
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()


## loading dataset
new_data = pd.read_excel("data_UK.xlsx") 
new_data1=new_data.copy()
new_data.dropna(subset=['Total_amount_of_delayed_invoice'], how='all', inplace=True)
label_encoder=label_encoder.fit(new_data1['Month'])


workbook1 = xlsxwriter.Workbook('Predictions_UK_2.xlsx')
excl = workbook1.add_worksheet()
excl.write(0, 0,'Month')     
excl.write(0, 1, 'Year')  
excl.write(0, 2,'Forecasted_Outlier')
excl.write(0, 3,'Forecasted_Normal')

def train_covid(train_till,test_start,test_end,pred_start,pred_end,m):
   
    for i in range(0,m):
        if i==0:
            X_train1=new_data1[['Month','Year','Unemployment_Rate_quarter',"Off_Trade_Volumes","CLI"
        
            ]].iloc[:train_till,:]
            X_train1['Month'] = label_encoder.transform(X_train1['Month'])
            X_test1=new_data1[['Month','Year','Unemployment_Rate_quarter',"Off_Trade_Volumes","CLI"
        
                   ]].iloc[test_start:test_end,:]
          
            X_test1['Month'] = label_encoder.transform(X_test1['Month'])
            y_train1=new_data1[['Total_amount_of_delayed_invoice']].iloc[:train_till,:]
            y_train1.reset_index(inplace=True)
            y_test1=new_data1[['Total_amount_of_delayed_invoice']].iloc[test_start:test_end,:]
            y_test1.reset_index(inplace=True)
            y_train1.drop(["index"],axis=1,inplace=True)
            y_test1.drop(["index"],axis=1,inplace=True)
            reg = xgb.XGBRegressor(n_estimators=100)
            
        
        reg.fit(X_train1, y_train1,
                eval_set=[(X_train1, y_train1), (X_test1, y_test1)],
                early_stopping_rounds=50, #stop if 50 consequent rounds without decrease of error
                verbose=False)
      #  sp=input("Give Share Price")
        X_test_1=new_data1[['Month','Year','Unemployment_Rate_quarter',"Off_Trade_Volumes","CLI"]].iloc[pred_start:pred_end,:]
        X_test_1['Month'] = label_encoder.transform(X_test_1['Month'])
        X_test_1.reset_index(inplace=True)
        X_test_1.drop(["index"],axis=1,inplace=True)
       # X_test1["Share_price"][X_test1["Share_price"].size-1]=sp 
        X_norm=reg.predict(X_test_1)
    
        
        if i==0:
            X_train2=new_data1[['Month','Year','Unemployment_Rate_quarter',"Off_Trade_Volumes","Share_price"
        
            ]].iloc[:train_till,:]
            X_train2['Month'] = label_encoder.transform(X_train2['Month'])
            X_test2=new_data1[['Month','Year','Unemployment_Rate_quarter',"Off_Trade_Volumes","Share_price"
        
                   ]].iloc[test_start:test_end,:]
          
            X_test2['Month'] = label_encoder.transform(X_test2['Month'])
            y_train2=new_data1[['Total_amount_of_delayed_invoice']].iloc[:train_till,:]
            y_train2.reset_index(inplace=True)
            y_test2=new_data1[['Total_amount_of_delayed_invoice']].iloc[test_start:test_end,:]
            y_test2.reset_index(inplace=True)
            y_train2.drop(["index"],axis=1,inplace=True)
            y_test2.drop(["index"],axis=1,inplace=True)
            reg = xgb.XGBRegressor(n_estimators=100)
            
        
        reg.fit(X_train2, y_train2,
                eval_set=[(X_train2, y_train2), (X_test2, y_test2)],
                early_stopping_rounds=50, #stop if 50 consequent rounds without decrease of error
                verbose=False)
      #  sp=input("Give Share Price")
        X_test_2=new_data1[['Month','Year','Unemployment_Rate_quarter',"Off_Trade_Volumes","Share_price"]].iloc[pred_start:pred_end,:]
        excl.write(i+1,0,X_test_2["Month"][X_train2["Share_price"].size])
        excl.write(i+1,1,X_test_2["Year"][X_train2["Share_price"].size])
        X_test_2['Month'] = label_encoder.transform(X_test_2['Month'])
        X_test_2.reset_index(inplace=True)
        X_test_2.drop(["index"],axis=1,inplace=True)
       # X_test1["Share_price"][X_test1["Share_price"].size-1]=sp 
        X_norm2=reg.predict(X_test_2)
        predictions=(X_norm+X_norm2)/2
        
        excl.write(i+1,2,predictions[0])
        
    
        
        X_train1=X_train1.append(X_test_1)
        df = pd.DataFrame({'Total_amount_of_delayed_invoice':predictions})
        y_train1=y_train1.append(df)
        X_train1.reset_index(inplace=True)
        X_train1.drop(["index"],axis=1,inplace=True)
        y_train1.reset_index(inplace=True)
        y_train1.drop(["index"],axis=1,inplace=True)
       
        X_test1=X_test1.append(X_test_1)
        X_test1.reset_index(inplace=True)
        X_test1.drop(["index"],axis=1,inplace=True)
        y_test1=y_test1.append(df)
        y_test1.reset_index(inplace=True)
        
        y_test1.drop(["index"],axis=1,inplace=True)    
        X_test1=X_test1.iloc[-3:,:]
        y_test1=y_test1.iloc[-3:,:]
        
        
        
        
        X_train2=X_train2.append(X_test_2)
        df = pd.DataFrame({'Total_amount_of_delayed_invoice':predictions})
        y_train2=y_train2.append(df)
        X_train2.reset_index(inplace=True)
        X_train2.drop(["index"],axis=1,inplace=True)
        y_train2.reset_index(inplace=True)
        y_train2.drop(["index"],axis=1,inplace=True)
       
        X_test2=X_test2.append(X_test_2)
        X_test2.reset_index(inplace=True)
        X_test2.drop(["index"],axis=1,inplace=True)
        y_test2=y_test2.append(df)
        y_test2.reset_index(inplace=True)
        
        y_test2.drop(["index"],axis=1,inplace=True)    
        X_test2=X_test2.iloc[-3:,:]
        y_test2=y_test2.iloc[-3:,:]
        pred_start=X_train2['Share_price'].size
        pred_end=X_train2['Share_price'].size +1

    
def train_norm(train_till,test_start,test_end,pred_start,pred_end,m):
    
    X_train=new_data[['Month','Year','Unemployment_Rate_quarter',"Off_Trade_Volumes","CLI"
    ]].iloc[:train_till,:]
    X_train['Month'] = label_encoder.transform(X_train['Month'])
    X_test=new_data1[['Month','Year','Unemployment_Rate_quarter',"Off_Trade_Volumes","CLI"
           ]].iloc[test_start:test_end,:]
  
    X_test['Month'] = label_encoder.transform(X_test['Month'])
    y_train=new_data[['Total_amount_of_delayed_invoice']].iloc[:train_till,:]
    y_train.reset_index(inplace=True)
    y_test=new_data[['Total_amount_of_delayed_invoice']].iloc[test_start:test_end,:]
    y_test.reset_index(inplace=True)
    y_train.drop(["index"],axis=1,inplace=True)
    y_test.drop(["index"],axis=1,inplace=True)
    reg = xgb.XGBRegressor(n_estimators=100)
    
    
    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            early_stopping_rounds=50, #stop if 50 consequent rounds without decrease of error
            verbose=False)
  #  sp=input("Give Share Price")
    X_test1=new_data1[['Month','Year','Unemployment_Rate_quarter',"Off_Trade_Volumes","CLI"]].iloc[pred_start:pred_end+m,:]
   
    X_test1.reset_index(inplace=True)
    X_test1.drop(["index"],axis=1,inplace=True)
  
    X_test1['Month'] = label_encoder.transform(X_test1['Month'])
    predictions=reg.predict(X_test1)
    for i in range(0,m):
        excl.write(i+1,3,predictions[i])
    workbook1.close()

m=5##predict till the month you need
train_till=new_data["Total_amount_of_delayed_invoice"].size
test_start=new_data["Total_amount_of_delayed_invoice"].size-3
test_end=new_data["Total_amount_of_delayed_invoice"].size
pred_start=new_data["Total_amount_of_delayed_invoice"].size
pred_end=new_data["Total_amount_of_delayed_invoice"].size+1 
train_covid(train_till,test_start,test_end,pred_start,pred_end,m)


train_till=new_data["Total_amount_of_delayed_invoice"].size
test_start=new_data["Total_amount_of_delayed_invoice"].size-3
test_end=new_data["Total_amount_of_delayed_invoice"].size
pred_start=new_data["Total_amount_of_delayed_invoice"].size
pred_end=new_data["Total_amount_of_delayed_invoice"].size+1
train_norm(train_till,test_start,test_end,pred_start,pred_end,m)


