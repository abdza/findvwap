#!/usr/bin/env python

import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime,timedelta

import autokeras as ak
from sklearn.preprocessing import MinMaxScaler


starttest = datetime.now()
raw_data = pd.read_csv('raw_data_range_20231203.csv')
topop = ['ticker','date','day','Big Reverse','Bottom After Noon','Bottom Before Noon','Bottom Lunch','Peak After Noon','Peak Before Noon','Peak Lunch','diff','profitable','performance']
for tp in topop:
    raw_data.pop(tp)
scaler = MinMaxScaler()
train_size = int(raw_data.shape[0] * 0.9)
train_data = pd.DataFrame(raw_data[:train_size])
test_data = pd.DataFrame(raw_data[train_size:])
train_data[['yavg','yyavg','1range','1body','gap','marks']] = scaler.fit_transform(train_data[['yavg','yyavg','1range','1body','gap','marks']])
y_data = train_data.pop('diff_level')

reg = ak.StructuredDataRegressor(
    overwrite=True, max_trials=15
)  # It tries 3 different models.

reg.fit(x=train_data,y=y_data,verbose=1)
y_test = test_data.pop('diff_level')
print("Evaluate:",reg.evaluate(x=test_data,y=y_test))
model = reg.export_model()
model.summary()
try:
    model.save("model_diff_level", save_format="tf")
except Exception:
    model.save("model_diff_level.h5")

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

raw_data = pd.read_csv('raw_data_range_20231203.csv')
topop = ['ticker','date','day','Big Reverse','Bottom After Noon','Bottom Before Noon','Bottom Lunch','Peak After Noon','Peak Before Noon','Peak Lunch','diff','diff_level','performance']
for tp in topop:
    raw_data.pop(tp)
train_size = int(raw_data.shape[0] * 0.9)
train_data = pd.DataFrame(raw_data[:train_size])
test_data = pd.DataFrame(raw_data[train_size:])
train_data[['yavg','yyavg','1range','1body','gap','marks']] = scaler.fit_transform(train_data[['yavg','yyavg','1range','1body','gap','marks']])
y_data = raw_data.pop('profitable')
# y_data = labelencoder.fit_transform(y_data)
reg = ak.StructuredDataClassifier(
    overwrite=True, max_trials=15
)  # It tries 3 different models.
reg.fit(x=raw_data,y=y_data,verbose=1)
y_test = test_data.pop('profitable')
print("Evaluate:",reg.evaluate(x=test_data,y=y_test))
model = reg.export_model()
model.summary()
try:
    model.save("model_profitable", save_format="tf")
except Exception:
    model.save("model_profitable.h5")
endtest = datetime.now()
print("Start:",starttest)
print("End:",endtest)
print("Time:",endtest-starttest)
