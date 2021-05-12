# -*- coding: utf-8 -*-


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
import sklearn
from sklearn import  metrics
from sklearn.model_selection import TimeSeriesSplit

import warnings

#change_dir
os.chdir('C:\\Users\\User\\Desktop\\models\\git\\data')
import functions as func

train_file=func.load_data('train_3.csv')
train_file=train_file[train_file['Date']>='2015-07-01']
test_file=func.load_data('test_3.csv')

len(train_file)

dates_test=list(set(test_file['Date']))
dates_test=sorted(dates_test)
dates_test

test_file.columns==train_file.columns

full_file=pd.concat([train_file,test_file],axis=0)



cols=['Open','Promo','Sales']
lags=[1,2,3,4,5,6,7,8,9,10,11,12,13]


import pickle
with open('xgboost.pkl', 'rb') as file:
    xgboost= pickle.load(file)

#Variables
numeric=['CompetitionDistance',
         'days_competitve',
         'days_promo2',
         'interval_month',
         'Sales_1',
         'Sales_2',
         'Sales_3', 
         'Sales_4',
       'Sales_5','Sales_6', 'Sales_7', 'Sales_8', 'Sales_9', 'Sales_10','Sales_11', 'Sales_12', 'Sales_13']
       
categoric=['Open',
           'DayOfWeek', 
           'Promo',
           'StateHoliday', 
           'SchoolHoliday',
           'StoreType', 
           'Assortment',
           'Promo2',
           'PromoInterval',
           'month',
           'Open_1',
           'Open_2',
           'Open_3', 
           'Open_4', 
           'Open_5','Open_6', 'Open_7', 'Open_8', 'Open_9', 'Open_10', 'Open_11', 'Open_12','Open_13',
           'Promo_1',
           'Promo_2', 
           'Promo_3', 
           'Promo_4',
           'Promo_5','Promo_6', 'Promo_7', 'Promo_8', 'Promo_9', 'Promo_10', 'Promo_11','Promo_12', 'Promo_13']

#numeric=['CompetitionDistance', 'days_competitve']
#categoric=['DayOfWeek','StateHoliday']

variables_used=numeric+categoric
random_state=1234

full_file=full_file.sort_values(['Store','Date']).reset_index(drop=True)

full_file['part']=full_file['Sales'].apply(lambda x: 'test' if math.isnan(x) else 'train')

y_pred_full=[]

for date in dates_test:
  print(date)
  df=full_file[(full_file['Date']<=date) & (full_file['Date']>date+np.timedelta64(-16,'D'))]
  indexes=df.index.to_numpy()
  df=func.add_lags(cols,lags,df)
  
  df2=df.loc[:,variables_used+['Date']]
  print(df2['Date'].min(),df2['Date'].max())
  indexes=df2.index.to_numpy()

  y_pred=xgboost.predict(df2)
  print(len(y_pred_full))
  y_pred_full=np.concatenate([y_pred_full,y_pred])
  #print(len(y_pred),len(full_file_2[full_file_2.index==df2.index.to_numpy()]))



results=pd.concatenate([full_file[full_file['part']='test']y_pred_full],axis=1).loc[:['Store','Date']

# load the model from disk
import pickle
loaded_model = pickle.load(open('xgboost.pkl', 'rb'))

indexes=df2.index.to_numpy()
len(full_file_2[full_file_2.index.isin(indexes)])

len(y_pred)

