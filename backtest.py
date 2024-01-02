#!/usr/bin/env python

from logging import lastResort
import pandas as pd 
import os
import json
import sys
import csv
import getopt
from datetime import datetime,timedelta
from pandas.core.indexing import convert_missing_indexer
import yahooquery as yq
import numpy as np
import math
from tabulate import tabulate
from numerize import numerize
from sklearn.cluster import KMeans
from ta.trend import EMAIndicator
import streamlit as st
from streamlit_calendar import calendar
from props import *
from sklearn.preprocessing import MinMaxScaler

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

starttest = datetime.now()

result_perc = pd.read_csv(os.path.join(script_dir,'raw_data_perc.csv'))

dates = result_perc['date'].unique()
dateperc = pd.DataFrame()
global_marks = pd.read_csv(os.path.join(script_dir,'analyze_global.csv'))
for cdate in dates:
    daytrade = result_perc[result_perc['date']==cdate]
    percdict = {}
    percdict['date'] = cdate
    for prop in prop_list:
        dayprop = daytrade[daytrade[prop]==1]
        propperc = round(len(dayprop)/len(daytrade),4)
        cgmark = global_marks[global_marks['Prop']==prop]
        if len(cgmark)>0:
            percdict['Corr ' + prop] = cgmark.iloc[0]['Corr']
            percdict['Profitable ' + prop] = cgmark.iloc[0]['Profitable']
            percdict['Good ' + prop] = cgmark.iloc[0]['Good']
            percdict['Great ' + prop] = cgmark.iloc[0]['Great']
        else:
            percdict['Corr ' + prop] = 0
            percdict['Profitable ' + prop] = 0
            percdict['Good ' + prop] = 0
            percdict['Great ' + prop] = 0
    percdf = pd.DataFrame.from_dict(percdict,orient='index').T
    dateperc = pd.concat([dateperc,percdf])
result_perc = result_perc.set_index('date').join(dateperc.set_index('date'))
result_perc.to_csv(os.path.join(script_dir,'raw_data_corr.csv'))

result_perc = calc_marks(result_perc)
result_perc = result_perc.reset_index()

fieldnames = ['date','ticker','diff_level','performance','profitable','marks','prev_marks','opening_marks','late_marks','hour_marks','daily_marks','gap']
minuscolumns = list(set(result_perc.columns.to_list()) - set(fieldnames))
finalcolumns = fieldnames + sorted(minuscolumns)

result_perc = result_perc[finalcolumns]

result_perc.to_csv(os.path.join(script_dir,'raw_data_perc_marks.csv'),index=False)
endtest = datetime.now()
print("Start:",starttest)
print("End:",endtest)
print("Time:",endtest-starttest)
