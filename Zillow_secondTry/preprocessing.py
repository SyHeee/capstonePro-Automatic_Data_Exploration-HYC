#Ref. https://www.kaggle.com/zusmani/scrpt/code
#http://datascience.ibm.com/blog/missing-data-
#conundrum-exploration-and-imputation-techniques/

### Importing Libraries or Packages that are needed throughout the Program
import numpy as np
import pandas as pd
import gc 
from sklearn.preprocessing import Imputer 
from sklearn.preprocessing import LabelEncoder 
#import fancyimpute#.simple_fill import SimpleFill
#from fancyimpute.mice import MICE
#import datetime as dt

import seaborn as sns #python visualization library
color = sns.color_palette()

### Read in Raw Data ###

print( "\nReading data from disk ...")
properties = pd.read_csv('../input/properties_2016.csv')
train = pd.read_csv('../input/train_2016_v2.csv' , parse_dates=["transactiondate"])
test = pd.read_csv('../input/sample_submission.csv')
test= test.rename(columns={'ParcelId': 'parcelid'})

### Type Converting the DataSet ###
# The processing of some of the algorithms can be made quick if data representation is 
#made in int/float32 instead of int/float64. Therefore, in order to make sure that all
# of our columns types are in float32, we are implementing the following lines of code #
for c, dtype in zip(properties.columns, properties.dtypes):
    if dtype == np.float64:
        properties[c] = properties[c].astype(np.float32)
    if dtype == np.int64:
        properties[c] = properties[c].astype(np.int32)
for column in test.columns:
    if test[column].dtype == int:
        test[column] = test[column].astype(np.int32)
    if test[column].dtype == float:
        test[column] = test[column].astype(np.float32)
### Feature Engineering
#living area proportions
properties['living_area_prop'] = properties['calculatedfinishedsquarefeet'] / properties['lotsizesquarefeet']
#tax value ratio
properties['value_ratio'] = properties['taxvaluedollarcnt'] / properties['taxamount']
#tax value proportions
properties['value_prop'] = properties['structuretaxvaluedollarcnt'] / properties['landtaxvaluedollarcnt']

###Merging the Datasets ###
df_train = train.merge(properties, how='left', on='parcelid')
df_test = test.merge(properties, how='left', on='parcelid')

### Remove Previous Variables to Keep Some Memory
del properties, train
gc.collect();
df_train[['latitude', 'longitude']] /= 1e6
df_test[['latitude', 'longitude']] /= 1e6
df_train['censustractandblock'] /= 1e12
df_test['censustractandblock'] /= 1e12

### Rearranging the DataSets ###
x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc',
'propertycountylandusecode', ], axis=1)
x_test = df_test.drop(['parcelid', 'propertyzoningdesc',
'propertycountylandusecode', '201610', '201611',
'201612', '201710', '201711', '201712'], axis = 1)
x_train = x_train.values
y_train = df_train['logerror'].values

### Handling Missing Data ###
lbl = LabelEncoder()
for c in df_train.columns:
    df_train[c]=df_train[c].fillna(0)
    if df_train[c].dtype == 'object':
        lbl.fit(list(df_train[c].values))
        df_train[c] = lbl.transform(list(df_train[c].values))

for c in df_test.columns:
    df_test[c]=df_test[c].fillna(0)
    if df_test[c].dtype == 'object':
        lbl.fit(list(df_test[c].values))
        df_test[c] = lbl.transform(list(df_test[c].values)) 
        
########## zhe duan bu work bu ran bu yong le ###############
#Method1: Mean, Median and Mode Imputation                 ##
#mean_imp = Imputer(missing_values='NaN', strategy='mean', axis=0)#, verbose=0, copy=True)
#x_train1 = mean_imp.fit(df_train)                         ##
#x_test1 =  mean_imp.fit(df_test)                          ##
#lbl = LabelEncoder()                                      ##
#for c in df_train.columns:                                ##
#    x = np.empty(shape= df_train[c].shape)                ##
#    df_train[c]=x.fillna(x[~np.isnan(x)].mean())          ##
#    if df_train[c].dtype == 'object':                     ##
#        lbl.fit(list(df_train[c].values))
#        df_train[c] = lbl.transform(list(df_train[c].values))
#
#for c in df_test.columns:
#    y = np.empty(shape= df_test[c].shape)
#    df_test[c]=y.fillna(y[~np.isnan(x)].mean())
#    if df_test[c].dtype == 'object':
#        lbl.fit(list(df_test[c].values))                  ##
#        df_test[c] = lbl.transform(list(df_test[c].values)) 
#Method2: MICE (Multiple Imputation by Chained Equations)  ##
#x_train = fancyimpute.MICE().complete(df_train)           ##
                                                           ##
#############################################################

### Removing the Outliers
log_errors = df_train['logerror']
# df_train = df_train[df_train.logerror < np.percentile(log_errors, 99.5)]
# df_train = df_train[df_train.logerror > np.percentile(log_errors, 0.5)]

### Rearranging the DataSets ###
x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 
                         'propertycountylandusecode' ], axis=1)

x_test = df_test.drop(['parcelid', 'propertyzoningdesc',
                       'propertycountylandusecode', '201610', '201611', 
                       '201612', '201710', '201711', '201712'], axis = 1) 

x_train = x_train.values
y_train = df_train['logerror'].values

np.savetxt("../intermediate/x_allData.txt", x_train)
np.savetxt("../intermediate/y_allData.txt", y_train)

print('\nDone...')