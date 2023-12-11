#!/usr/bin/env python

import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime,timedelta

import autokeras as ak
from sklearn.preprocessing import MinMaxScaler

prop_list = [
'Big Reverse',
'Bottom After Noon',
'Bottom Before Noon',
'Bottom Lunch',
'Consecutive Early Green',
'Consecutive Early Red',
'Consecutive FVG',
'Consecutive Green',
'Consecutive Late Green',
'Consecutive Late Red',
'Consecutive Negative FVG',
'Consecutive Negative Volume Gap',
'Consecutive Red',
'Consecutive Volume Gap',
'Continue Higher High',
'Continue Higher Low',
'Continue Lower High',
'Continue Lower Low',
'First Green',
'First Hammer',
'First Red',
'First Reverse Hammer',
'FVG First',
'FVG Second',
'Gap Down Above Average',
'Gap Down Above 2 Day Average',
'Gap Down Below Prev Min',
'Gap Down',
'Gap Up Above Average',
'Gap Up Above 2 Day Average',
'Gap Up Above Prev Max',
'Gap Up',
'Higher High',
'Higher Low',
'Lower High',
'Lower Low',
'Negative FVG First',
'Negative FVG Second',
'Negative Volume Gap First',
'Negative Volume Gap Second',
'Open Higher Than 2 Prev Max',
'Open Higher Than Prev Max Plus Average',
'Open Higher Than Prev Max',
'Open Lower Than 2 Prev Max',
'Open Lower Than Prev Min Minus Average',
'Open Lower Than Prev Min',
'Peak After Noon',
'Peak Before Noon',
'Peak Lunch',
'Range Above 2 Day Average',
'Range Above Average',
'Range Lower 2 Day Average',
'Range Lower Average',
'Range More Than Gap Down',
'Range More Than Gap Up',
'Second Green',
'Second Hammer',
'Second Long',
'Second Red',
'Second Reverse Hammer',
'Second Short',
'Third Green',
'Third Hammer',
'Third Long',
'Third Red',
'Third Reverse Hammer',
'Third Short',
'Two Small Reverse',
'Volume Gap First',
'Volume Gap Second',
'Volume Higher Than Average',
'Volume Lower Than Average',
'Volume Open Higher',
'Volume Open Lower',
'Volume Open Excedingly High',
'Volume Open Excedingly Low',
'Volume Open After High',
'Volume Open After Low',
'Early Top Level',
'Late Top Level',
'Top Level',
'Second Range Shorter',
'Second Range Longer',
'Third Range Longer',
'Third Range Shorter',
'Consecutive Shorter Range',
'Consecutive Longer Range',
'Second Range Very Shorter',
'Second Range Very Longer',
'Third Range Very Longer',
'Third Range Very Shorter',
'Consecutive Very Shorter Range',
'Consecutive Very Longer Range',
'Second Volume Lower',
'Second Volume Higher',
'Third Volume Higher',
'Third Volume Lower',
'Consecutive Lower Volume',
'Consecutive Higher Volume',
'Limp Second Diff',
'Limp Third Diff',
'Consecutive Limp Diff',
'Tiny Range',
'Second Tiny Range',
'Third Tiny Range',
'Consecutive Early Tiny Range',
'Consecutive Late Tiny Range',
'Consecutive Tiny Range',
'Huge Range',
'Second Huge Range',
'Third Huge Range',
'Consecutive Early Huge Range',
'Consecutive Late Huge Range',
'Consecutive Huge Range',
'Huge Negative Range',
'Second Huge Negative Range',
'Third Huge Negative Range',
'Consecutive Early Huge Negative Range',
'Consecutive Late Huge Negative Range',
'Consecutive Huge Negative Range',
'Max After Min',
'Min After Max',
'Yesterday End In Red',
'Yesterday End Volume Above Average',
'Volume Above 5 Time Average',
'Volume Above 10 Time Average',
'Volume Above 5 Time Before Average',
'Volume Above 10 Time Before Average',
'Volume Consecutive Above 5 Time Average',
'Volume Consecutive Above 10 Time Average',
'New IPO',
'Fairly New IPO',
'Sluggish Ticker',
'Continue Sluggish Ticker',
    ]

ignore_prop = [
'Big Reverse',
'Two Small Reverse',
'Bottom After Noon',
'Bottom Before Noon',
'Bottom Lunch',
'Peak After Noon',
'Peak Before Noon',
'Peak Lunch',
'Min After Max',
'Max After Min',
]


starttest = datetime.now()
raw_data = pd.read_csv('raw_data_20231209.csv')
topop = ['ticker','date','day','diff','profitable','performance']
for tp in topop:
    raw_data.pop(tp)
for tp in ignore_prop:
    raw_data.pop(tp)

print("Nan data:",np.count_nonzero(np.isnan(raw_data)))
raw_data = raw_data.dropna()
print("After drop Nan data:",np.count_nonzero(np.isnan(raw_data)))
scaler = MinMaxScaler()
train_size = int(raw_data.shape[0] * 0.9)
train_data = pd.DataFrame(raw_data[:train_size])
test_data = pd.DataFrame(raw_data[train_size:])
train_data[['yavg','yyavg','1range','1body','gap','marks']] = scaler.fit_transform(train_data[['yavg','yyavg','1range','1body','gap','marks']])
y_data = train_data.pop('diff_level')

column_types = {}
for column_name in train_data.columns:
    if column_name in prop_list:
        column_types[column_name] = 'categorical'
    else:
        column_types[column_name] = 'numerical'

reg = ak.StructuredDataRegressor(
    overwrite=True, max_trials=100, objective='val_loss', column_types=column_types
)  # It tries 3 different models.

reg.fit(x=train_data,y=y_data,verbose=1,use_multiprocessing=True,epochs=1000)
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

raw_data = pd.read_csv('raw_data_20231209.csv')
topop = ['ticker','date','day','diff','diff_level','performance']
for tp in topop:
    raw_data.pop(tp)
for tp in ignore_prop:
    raw_data.pop(tp)
topop = ['yavg','yyavg','1range','1body','gap','marks']
for tp in topop:
    raw_data.pop(tp)
print("Nan data:",np.count_nonzero(np.isnan(raw_data)))
raw_data = raw_data.dropna()
print("After drop Nan data:",np.count_nonzero(np.isnan(raw_data)))
train_size = int(raw_data.shape[0] * 0.9)
train_data = pd.DataFrame(raw_data[:train_size])
print("Keys:",train_data.columns)
test_data = pd.DataFrame(raw_data[train_size:])
# train_data[['yavg','yyavg','1range','1body','gap','marks']] = scaler.fit_transform(train_data[['yavg','yyavg','1range','1body','gap','marks']])
# y_data = labelencoder.fit_transform(y_data)
y_data = train_data.pop('profitable')
column_types = {}
for column_name in train_data.columns:
    if column_name in prop_list:
        column_types[column_name] = 'categorical'
    else:
        column_types[column_name] = 'numerical'
print("Ct size:",len(column_types))
print("Column types:",column_types)
reg = ak.StructuredDataClassifier(
    overwrite=True, max_trials=100, column_types=column_types, objective="val_accuracy", loss='categorical_crossentropy'
)  # It tries 3 different models.
print("Train shape:",train_data.shape)
reg.fit(x=train_data,y=y_data,verbose=1,epochs=1000)
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
