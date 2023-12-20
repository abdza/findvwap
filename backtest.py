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

result_perc = pd.read_csv('raw_data_perc.csv')

global_marks = pd.read_csv(os.path.join(script_dir,'analyze_global.csv'))
global_fail = pd.read_csv(os.path.join(script_dir,'analyze_global_fail.csv'))
highest_prop = pd.read_csv(os.path.join(script_dir,'analyze_highest_prop.csv'))

median_highest = highest_prop['count'].median()

result_perc.reset_index(inplace=True)
# result_perc.set_index(['date','ticker'])

result_perc['prev_marks'] = 1
result_perc['neg_prev_marks'] = 1
result_perc['final_prev_marks'] = 1
for prop in prev_prop_list:
    # cgmark = global_marks[global_marks['Prop']==prop]
    highest = highest_prop[highest_prop['prop']==prop]
    if highest.iloc[0]['count']>median_highest:
        result_perc.loc[result_perc[prop]==1,'prev_marks'] += 1
    # if len(cgmark):
    #     if cgmark.iloc[0]['Marks']>0:
    #         # result_perc.loc[result_perc[prop]==1,'prev_marks'] *= cgmark.iloc[0]['Marks']
    #         result_perc.loc[result_perc[prop]==1,'prev_marks'] += result_perc[prop] * cgmark.iloc[0]['Marks']
    #     # result_perc.loc[result_perc['Perc ' + prop]>cgmark.iloc[0]['Median'],'prev_marks'] *= (1 + cgmark.iloc[0]['Median'])
    # cgmark = global_fail[global_fail['Prop']==prop]
    # if len(cgmark):
    #     result_perc.loc[result_perc[prop]==1,'neg_prev_marks'] += result_perc[prop] * cgmark.iloc[0]['Marks']
# result_perc['final_prev_marks'] = result_perc['prev_marks'] - result_perc['neg_prev_marks']

result_perc['opening_marks'] = 1
result_perc['neg_opening_marks'] = 1
result_perc['final_opening_marks'] = 1
for prop in opening_prop_list:
    highest = highest_prop[highest_prop['prop']==prop]
    if highest.iloc[0]['count']>median_highest:
        result_perc.loc[result_perc[prop]==1,'opening_marks'] += 1
#     cgmark = global_marks[global_marks['Prop']==prop]
#     if len(cgmark):
#         if cgmark.iloc[0]['Marks']>0:
#             # result_perc.loc[result_perc[prop]==1,'opening_marks'] *= cgmark.iloc[0]['Marks']
#             result_perc.loc[result_perc[prop]==1,'opening_marks'] += result_perc[prop] * cgmark.iloc[0]['Marks']
#         # result_perc.loc[result_perc['Perc ' + prop]>cgmark.iloc[0]['Median'],'opening_marks'] *= (1 + cgmark.iloc[0]['Median'])
#     cgmark = global_fail[global_fail['Prop']==prop]
#     if len(cgmark):
#         result_perc.loc[result_perc[prop]==1,'neg_opening_marks'] += result_perc[prop] * cgmark.iloc[0]['Marks']
# result_perc['final_opening_marks'] = result_perc['opening_marks'] - result_perc['neg_opening_marks']

result_perc['late_marks'] = 1
result_perc['neg_late_marks'] = 1
result_perc['final_late_marks'] = 1
for prop in late_prop_list:
    highest = highest_prop[highest_prop['prop']==prop]
    if highest.iloc[0]['count']>median_highest:
        result_perc.loc[result_perc[prop]==1,'late_marks'] += 1
#     cgmark = global_marks[global_marks['Prop']==prop]
#     if len(cgmark):
#         if cgmark.iloc[0]['Marks']>0:
#             # result_perc.loc[result_perc[prop]==1,'late_marks'] *= cgmark.iloc[0]['Marks']
#             result_perc.loc[result_perc[prop]==1,'late_marks'] += result_perc[prop] * cgmark.iloc[0]['Marks']
#         # result_perc.loc[result_perc['Perc ' + prop]>cgmark.iloc[0]['Median'],'late_marks'] *= (1 + cgmark.iloc[0]['Median'])
#     cgmark = global_fail[global_fail['Prop']==prop]
#     if len(cgmark):
#         result_perc.loc[result_perc[prop]==1,'neg_late_marks'] += result_perc[prop] * cgmark.iloc[0]['Marks']
# result_perc['final_late_marks'] = result_perc['late_marks'] - result_perc['neg_late_marks']
        

    # print(tabulate(corr,headers='keys'))
#     dayprop = daytrade[daytrade[prop]==1]
#     print(corr.columns)

scaler = MinMaxScaler()
result_perc[['prev_marks','neg_prev_marks','final_prev_marks','opening_marks','neg_opening_marks','final_opening_marks','late_marks','neg_late_marks','final_late_marks']] = scaler.fit_transform(result_perc[['prev_marks','neg_prev_marks','final_prev_marks','opening_marks','neg_opening_marks','final_opening_marks','late_marks','neg_late_marks','final_late_marks']])
result_perc['early_marks'] = result_perc['final_prev_marks'] + result_perc['final_opening_marks']
result_perc['marks'] = result_perc['final_prev_marks'] + result_perc['final_opening_marks'] + result_perc['final_late_marks']
result_perc[['early_marks','marks']] = scaler.fit_transform(result_perc[['early_marks','marks']])


prop_corr = pd.read_csv(os.path.join(script_dir,'analyze_global_corr.csv'))
print("prop:",prop_corr.columns)
result_perc['corr_marks'] = 1
for prop in prop_list:
    print("Corr for ",prop)
    filtered = result_perc[result_perc['Perc ' + prop]>0.5]
    filtered = filtered[filtered[prop]==1]
    if prop not in ignore_prop:
        cgmark = global_marks[global_marks['Prop']==prop]
        for tp in cgmark['All CP']:
            # print("Tp:",tp)
            intp = eval(tp)
            for iii in intp:
                print("In tp:",iii)
                # if iii['CorrRatio']>0.3:
                filtered = filtered[filtered[iii['Prop']]==1]
                result_perc.loc[filtered.index,'corr_marks'] += result_perc['corr_marks'] * iii['CorrRatio']
                print("We've got ratio")
        # corr = prop_corr[prop_corr[prop]>0.5]
        # filtered = result_perc[result_perc['Perc ' + prop]>0.5]
        # filtered = filtered[filtered[prop]==1]
        # for cp in corr['Prop']:
        #     filtered = filtered[filtered[cp]==1]
        #     print("P:",cp," Filter size:",len(filtered))
        # if len(filtered)>0:
        #     result_perc.loc[filtered.index,'corr_marks'] += 1

result_perc.loc[result_perc['corr_marks']==1,'corr_marks'] = 0

result_perc[['corr_marks']] = scaler.fit_transform(result_perc[['corr_marks']])

result_perc['final_marks'] = result_perc['marks'] + result_perc['corr_marks']
result_perc[['final_marks']] = scaler.fit_transform(result_perc[['final_marks']])

fieldnames = ['date','ticker','diff_level','performance','profitable','final_marks','marks','prev_marks','neg_prev_marks','final_prev_marks','opening_marks','neg_opening_marks','final_opening_marks','late_marks','neg_late_marks','final_late_marks','early_marks','corr_marks','gap']
minuscolumns = list(set(result_perc.columns.to_list()) - set(fieldnames))
finalcolumns = fieldnames + sorted(minuscolumns)

result_perc = result_perc[finalcolumns]

result_perc.to_csv(os.path.join(script_dir,'raw_data_perc_marks.csv'),index=False)
endtest = datetime.now()
print("Start:",starttest)
print("End:",endtest)
print("Time:",endtest-starttest)
