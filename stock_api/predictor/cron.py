import tensorflow
KERAS_BACKEND=tensorflow
import keras
import os
import joblib

from . models import Predictions
import numpy as np
from keras.models import Sequential,load_model
from keras.layers import Dense, LSTM
#from pandas.testing import assert_frame_equal
import pandas as pd
import yfinance as yf

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.preprocessing import MinMaxScaler
from datetime import date
import math
#import pandas as pd
#import horovod.keras as hvd



def predictionfunction():
    symbol_details=pd.read_csv('company.csv')

    sym=symbol_details['symbol']
    #sym=["AAPL","MSFT","AMZN","BA","TSLA","GOOGL"]	

    for i in range(len(sym)):
      if i%16==0:
        predictor(sym[i])



def predictor(sym):
    model_path="/home/ubuntu/stonks/stonks/stock_api/all_instances/"+sym+".h5"
    print(model_path)
    end_date=date.today()


    new_model = load_model(model_path, custom_objects={
    'Adam': lambda **kwargs: hvd.DistributedOptimizer(keras.optimizers.Adam(**kwargs))
    })

    try:
      
      stock_sym=sym
      stock = yf.Ticker(stock_sym)
      df_after2008=stock.history( start='2009-08-01', end=end_date)
      df_after2008=stock.history( start='2002-01-01', end='2007-06-30')
      df_new=pd.concat([df_after2008,df_before2007])
      df = df_new[df_new.columns[:4]]

    except:
        try:
          df_new=stock.history( start='2018-01-01', end=end_date)
          df = df_new[df_new.columns[:4]]
        except:
          try:
            df_new=stock.history( start='2019-01-01', end=end_date)
            df = df_new[df_new.columns[:4]]
          except:
            df_new=stock.history( start='2020-04-01', end=end_date)
            df = df_new[df_new.columns[:4]]
    
    scaler = MinMaxScaler(feature_range=(0, 1)) 

    data = df.filter(['Close'])#Converting the dataframe to a numpy array
    dataset = data.values#Get /Compute the number of rows to train the model on
    training_data_len = math.ceil( len(dataset) *.6)
    


    
    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    scaled_data = scaler.fit_transform(dataset)


    
   
 
    try:
      data_for_prediction_ =stock.history(start='2020-01-01', end=end_date)
      data_for_prediction=data_for_prediction_[data_for_prediction_.columns[:4]]
    except:
      data_for_prediction_ =stock.history(start='2020-04-01', end=end_date)
      data_for_prediction=data_for_prediction_[data_for_prediction_.columns[:4]]

    #Create a new dataframe

    try:
      new_df = data_for_prediction.filter(['Close'])#Get the last 60 day closing price 
      last_60_days = new_df[-60:].values#Scale the data to be values between 0 and 1
      last_60_days_scaled = scaler.transform(last_60_days)#Create an empty list
      X_test = []#Append teh past 60 days
      X_test.append(last_60_days_scaled)#Convert the X_test data set to a numpy array
      X_test = np.array(X_test)#Reshape the data
      X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))#Get the predicted scaled price
      pred_price = new_model.predict(X_test)#undo the scaling 
      pred_price = scaler.inverse_transform(pred_price)
    except:
      new_df = data_for_prediction.filter(['Close'])#Get the last 60 day closing price 
      last_30_days = new_df[-30:].values#Scale the data to be values between 0 and 1
      last_30_days_scaled = scaler.transform(last_30_days)#Create an empty list
      X_test = []#Append teh past 60 days
      X_test.append(last_30_days_scaled)#Convert the X_test data set to a numpy array
      X_test = np.array(X_test)#Reshape the data
      X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))#Get the predicted scaled price
      pred_price = new_model.predict(X_test)#undo the scaling 
      pred_price = scaler.inverse_transform(pred_price)

    #print(sym)
    #print(pred_price)

    
    save_to_database(sym,pred_price)

def save_to_database(sym,pred_price):

    print(sym)
    print(pred_price)
    '''
    predictor_symbol=Predictions(
      sym_name=sym,
      sym_prediction=pred_price
    )
    predictor_symbol.save(force_insert=True)
    '''

predictionfunction()



