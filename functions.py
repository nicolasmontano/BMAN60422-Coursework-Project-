import zipfile
import os
from os.path import  isfile, join
import pandas as pd
import numpy as np
import math
import datetime as dt
import matplotlib.pyplot as plt
import math
import datetime as dt
#Plot
import seaborn as sns
#Model
from sklearn import  metrics
from sklearn.model_selection import TimeSeriesSplit

from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
#models
from sklearn.linear_model import Ridge,Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
#model selection
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import  RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_squared_error as mse
from sklearn.model_selection import ShuffleSplit 
import warnings
warnings.filterwarnings('ignore')



def load_data(file):
    #Load Data
    #Load Train and stores data
    train_file=pd.read_csv(file)
    train_file['Date']=pd.to_datetime(train_file['Date'],format='%Y-%m-%d')
    rows=train_file.shape[0]
    train_file['StateHoliday'].replace(0,'0', inplace = True) 
    print(len(train_file))

    train_file.sort_values(by='Date',inplace=True)
    train_file.reset_index(drop=True,inplace=True)

    #Preprocess
    train_file['interval_month'].fillna(0,inplace=True)
    train_file['PromoInterval'].fillna('null',inplace=True)
    
    return train_file

#Time series split
def time_series_split(data,train_per=0.9, valid_per=0):
  test_per=1-train_per-valid_per

  days=(data['Date'].max()-data['Date'].min()).days+1
  train_len=(days*train_per)//1
  valid_len=(days*valid_per)//1
  test_len=(days*test_per)//1
  missing=days-sum([train_len,valid_len,test_len])
  train_len+=missing
  
  #train
  min_date_train=data['Date'].min()
  train=data[(data['Date']>=min_date_train) & (data['Date']<=min_date_train+dt.timedelta(train_len-1))]
  #validation
  min_date_valid=train['Date'].max() #it is the max date
  val=data[(data['Date']>min_date_valid) & (data['Date']<=min_date_valid+dt.timedelta(valid_len))]

  #test
  if valid_per>0:
    min_date_test=val['Date'].max()
  else:
    min_date_test=train['Date'].max() 

  test=data[(data['Date']>min_date_test) & (data['Date']<=min_date_test+dt.timedelta(test_len))]

  train_days= (train['Date'].max()-train['Date'].min()).days+1
  val_days=(val['Date'].max()-val['Date'].min()).days+1
  test_days=(test['Date'].max()-test['Date'].min()).days+1
  data_days=(data['Date'].max()-data['Date'].min()).days+1

  print('Train min {} and max {}, days:{}'.format(train['Date'].min(), train['Date'].max(), train_days))
  print('Val min {} and max {}, days:{}'.format(val['Date'].min(), val['Date'].max(), val_days))
  print('Test min {} and max {}, days:{}'.format(test['Date'].min(), test['Date'].max(), test_days))
  print('Data min {} and max {}, days:{}'.format(data['Date'].min(), data['Date'].max(), data_days))
  print('train : {:2.2%}, val  {:2.2%} and test  {:2.2%}'.format(train_days/data_days,val_days/data_days,test_days/data_days))


  return train, val, test

#Time series split for grid search
def time_series_split_cv(data,train_per=0.9, splits=1):
    '''
    Split for cv
    '''

    test_per = 1 - train_per
    days = (data['Date'].max() - data['Date'].min()).days + 1

    train_len = (days * train_per) // 1

    test_len = (days * test_per) // 1
    missing = days - sum([train_len, test_len])
    train_len += missing
    test_per

    # train
    min_date_train = data['Date'].min()
    train = data[(data['Date'] >= min_date_train) & (data['Date'] <= min_date_train + dt.timedelta(train_len - 1))]
    # test
    min_date_test = train['Date'].max()  # it is the max date
    test = data[(data['Date'] > min_date_test) & (data['Date'] <= min_date_test + dt.timedelta(test_len))]

    for i in range(splits):
        train_index = train.index.values
        test_index = test.index.values

        tr = train[train.index.values == train_index].loc[:, 'Date']
        ts = test[test.index.values == test_index].loc[:, 'Date']

        train_days = (tr.max() - tr.min()).days + 1
        test_days = (ts.max() - ts.min()).days + 1
        data_days = (data['Date'].max() - data['Date'].min()).days + 1

        print('Train: min {} and max {}, days:{}'.format(tr.min(), tr.max(), train_days))
        print('Test: min {} and max {}, days:{}'.format(ts.min(), ts.max(), test_days))
        print('Data: min {} and max {}, days:{}'.format(data['Date'].min(), data['Date'].max(), data_days))
        print('train : {:2.2%}, test  {:2.2%}'.format(train_days / data_days, test_days / data_days))

        yield (np.array(train_index), np.array(test_index))


#Pipeline
def def_pipeline(model,numeric,categoric):
  '''
  Return the model pipeline for gridsearch
  '''
  numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

  categorical_transformer = OneHotEncoder(handle_unknown='error',drop='first')

  preprocessor = ColumnTransformer(
  transformers=[
    ('num', numeric_transformer, numeric),
    ('cat', categorical_transformer, categoric)])

  # Append classifier to preprocessing pipeline.
  # Now we have a full prediction pipeline.
  reg = Pipeline(steps=[('preprocessor', preprocessor),
                  ('regressor', model)])
  return reg


#Metrics
def rmspe(y, y_pred):
    '''
    Compute Root Mean Square Percentage Error between two arrays.
    '''
    loss = np.sqrt(np.mean(np.square(y_pred-y_test)))
    return loss

def calculate_metrics(y_pred,y_test):
  y_res=pd.concat([pd.DataFrame(y_test).reset_index(drop=True),pd.DataFrame(y_pred).reset_index(drop=True)],axis=1)
  y_res.columns=['y_test','y_pred']
  rmse=np.sqrt(mse(y_res['y_test'],y_res['y_pred']))

  percentage= y_res[y_res['y_test']!=0]
  rmspe =  np.sqrt(np.mean(((percentage['y_pred']-percentage['y_test'])/percentage['y_test'])**2))
  mape=np.mean(np.abs((percentage['y_pred']-percentage['y_test'])/percentage['y_test'])) 

  return {'RMSE':rmse,'RMSPE':rmspe,'MAPE':mape} 

def downcast_dtypes(df):
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype in ["int64", "int32"]]
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int32)
    return df
'''
-----------------FEATURE ENGINEERING-----
'''
#works only for this 
def add_lags(cols,lags,df):
  initial_length=len(df)
  cols.append('Store')
  df['remove']='no'
  #sort by store and date
  df=df.sort_values(['Store','Date']).reset_index(drop=True)

  for col in cols:
    for lag in lags:
      df[col+'_'+str(lag)]=df[col].shift(lag)

      if col=='Store':
        df['remove']=df.apply(lambda row: np.nan if (row[col]!=row[col+'_'+str(lag)]) else row['remove'],axis=1)
        df.drop(col+'_'+str(lag),axis=1,inplace=True)

  date_removed=df[df['remove'].isnull()].loc[:,'Date'].unique()
  count_removed=df[df['remove'].isnull()].loc[:,'Date'].count()

  #drop all nulls
  df=df[df['remove']=='no']

  #drop addtional cols
  df.drop('remove',axis=1,inplace=True)
  
  final_length=len(df)
  print('additional deleted rows:{}\n \nDates removed:{}'.format(initial_length-final_length-count_removed,date_removed))
  return df
