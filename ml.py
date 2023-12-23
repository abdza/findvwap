#!/usr/bin/env python

import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime,timedelta

import autokeras as ak
from sklearn.preprocessing import MinMaxScaler
from props import *

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

starttest = datetime.now()
scaler = MinMaxScaler()
raw_data = pd.read_csv(os.path.join(script_dir,'raw_data_perc_marks.csv'))
topop = ['ticker','date','day','diff','diff_level','performance']
for tp in topop:
    raw_data.pop(tp)
for tp in ignore_prop:
    raw_data.pop(tp)
# todrop = ['prev_marks','opening_marks','late_marks','marks','gap']
todrop = ['gap']
for tp in todrop:
    raw_data.pop(tp)
raw_data = raw_data.dropna()
train_size = int(raw_data.shape[0] * 0.9)
print("Train size:",train_size)
train_data = pd.DataFrame(raw_data[:train_size])
print("Keys:",train_data.columns)
test_data = pd.DataFrame(raw_data[train_size:])
y_data = train_data.pop('profitable')

column_types = {}
for column_name in train_data.columns:
    if column_name in prop_list:
        column_types[column_name] = 'categorical'
        column_types['Perc ' + column_name] = 'numerical'
    else:
        column_types[column_name] = 'numerical'
print("Ct size:",len(column_types))
print("Column types:",column_types)
reg = ak.StructuredDataClassifier(
    overwrite=True, max_trials=5, column_types=column_types, objective="val_accuracy", loss='categorical_crossentropy'
)  # It tries 3 different models.
print("Train shape:",train_data.shape)
reg.fit(x=train_data,y=y_data,verbose=1,epochs=100)
y_test = test_data.pop('profitable')
print("Evaluate:",reg.evaluate(x=test_data,y=y_test))
model = reg.export_model()
model.summary()
try:
    model.save(os.path.join(script_dir,"model_profitable"), save_format="tf")
except Exception:
    model.save(os.path.join(script_dir,"model_profitable.h5"))
endtest = datetime.now()
print("Start:",starttest)
print("End:",endtest)
print("Time:",endtest-starttest)
