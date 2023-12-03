#!/usr/bin/env python

import numpy as np
import pandas as pd
import tensorflow as tf

import autokeras as ak

raw_data = pd.read_csv('raw_data_20231203.csv')
# topop = ['Big Reverse','Bottom After Noon','Bottom Before Noon','Bottom Lunch','Peak After Noon','Peak Before Noon','Peak Lunch','diff','performance','marks']
topop = ['ticker','date','day','Big Reverse','Bottom After Noon','Bottom Before Noon','Bottom Lunch','Peak After Noon','Peak Before Noon','Peak Lunch','diff','profitable','performance','marks']
for tp in topop:
    raw_data.pop(tp)
train_size = int(raw_data.shape[0] * 0.9)
raw_data[:train_size].to_csv("train.csv", index=False)
raw_data[train_size:].to_csv("eval.csv", index=False)
train_file_path = "train.csv"
test_file_path = "eval.csv"
y_data = raw_data.pop('diff_level')

# Initialize the structured data regressor.
reg = ak.StructuredDataRegressor(
# reg = ak.StructuredDataClassifier(
    overwrite=True, max_trials=15
)  # It tries 3 different models.

reg.fit(x=raw_data,y=y_data,verbose=0)
# Feed the structured data regressor with training data.
# reg.fit(
#     # The path to the train.csv file.
#     train_file_path,
#     # The name of the label column.
#     "profitable",
#     # epochs=10,
# )
# Predict with the best model.
# test_data = pd.read_csv(test_file_path)
# test_data['predicted'] = reg.predict(test_file_path)
# test_data.to_csv('predicted.csv')
# Evaluate the best model with testing data.
print("Evaluate:",reg.evaluate(test_file_path, "diff_level"))
model = reg.export_model()
model.summary()
try:
    model.save("model_diff_level", save_format="tf")
except Exception:
    model.save("model_diff_level.h5")
