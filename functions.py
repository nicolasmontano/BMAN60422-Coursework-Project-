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
import xgboost as xgb
#model selection
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import  RandomizedSearchCV
from sklearn.metrics import make_scorer
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
def time_series_split_cv(train_per=0.9, data=train_file, splits=1):
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




