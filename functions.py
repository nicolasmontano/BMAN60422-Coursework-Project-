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