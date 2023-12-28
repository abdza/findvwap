#!/usr/bin/env python

import pandas as pd 
import os
import sys
import csv
from datetime import datetime,timedelta
import numpy as np
import math
from statistics import mean
from tabulate import tabulate
from sklearn.preprocessing import MinMaxScaler
from props import *

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

datas = pd.read_csv(os.path.join(script_dir,'raw_data_perc.csv'))

maxheight = 20
todisp = datas.sort_values('diff_level',ascending=False)
todisp = todisp[todisp['profitable']==1]
pcount = []
for p in prop_list:
    if p not in ignore_prop:
        pcount.append({'prop':p,'count':todisp[p].sum()})
pcdf = pd.DataFrame(pcount)
pcdf.sort_values('count',inplace=True,ascending=False)
pcdf.to_csv(os.path.join(script_dir,"analyze_highest_prop.csv"),index=False)

tominus = datas.sort_values('diff_level')
tominus = tominus[:maxheight]
mpcount = []
for p in prop_list:
    if p not in ignore_prop:
        mpcount.append({'prop':p,'count':tominus[p].sum()})
mpcdf = pd.DataFrame(mpcount)
mpcdf.sort_values('count',inplace=True,ascending=False)
mpcdf.to_csv(os.path.join(script_dir,"analyze_lowest_prop.csv"),index=False)

global_prop = []
global_fail = {}
global_negate = {}

dates = datas['date'].unique()
print("Dates:",dates)

fieldnames = ['diff_level']
fieldnames += prop_list
for prop in prop_list:
    fieldnames.append('Perc ' + prop)
# profitable = datas[datas['profitable']==1]
# profitable = profitable[profitable['First Green']==1]
# profitable = profitable[profitable['Gap Up']==1]
profitable = datas.copy()
propsprofitable = profitable[fieldnames]
corrprofit = propsprofitable.corr()
# corrprofit.index.names = ['Prop']
upper_corr_mat = corrprofit.where( 
    np.triu(np.ones(corrprofit.shape), k=1).astype(bool)) 
  
# Convert to 1-D series and drop Null values 
unique_corr_pairs = upper_corr_mat.unstack().dropna() 
  
# Sort correlation pairs 
sorted_mat = unique_corr_pairs.sort_values(ascending=False) 
most_corr_prop = {}
for index,value in sorted_mat.items():
    if index[1]=='diff_level':
        if index[0] in prop_list:
            most_corr_prop[index[0]] = value*100
        print("Index:",index," Value:",value)
corrprofit.to_csv(os.path.join(script_dir,"analyze_global_corr.csv"))
print("Most corr:",most_corr_prop)

propsprofitable = datas[['profitable'] + fieldnames]
# propsprofitable = propsprofitable[propsprofitable['First Green']==1]
for prop in prop_list:
    propsprofitable['Profit ' + prop] = 0
    appear = propsprofitable[propsprofitable[prop]==1]
    appearprofit = appear[appear['profitable']==1]
    if len(appearprofit):
        propsprofitable.loc[appearprofit.index,'Profit ' + prop] = 1

group_by_diff = propsprofitable.groupby('diff_level').sum()
group_by_diff.reset_index(inplace=True)
group_by_diff.to_csv(os.path.join(script_dir,"analyze_global_group_numbers.csv"),index=False)
group_by_diff = group_by_diff.corr()
group_by_diff.index.names = ['Prop']
group_by_diff.to_csv(os.path.join(script_dir,"analyze_global_group_corr.csv"))

tickerprice = datas.copy()
tickerprice = tickerprice[['date','ticker','diff']]
pivot_tickerprice = tickerprice.pivot(index='date',columns='ticker',values='diff')
# tickerprice['ticker'] = tickerprice['ticker'].astype('category')
tickercoor = pivot_tickerprice.corr()
tickercoor.to_csv(os.path.join(script_dir,"analyze_ticker_coor.csv"))

pair_prop = [
['First Red','First Green'],
['Yesterday Loss','Yesterday Profitable'],
['Volume Open Lower','Volume Open Higher'],
['Volume Lower Than Average','Volume Higher Than Average'],
['Tiny Range','Huge Range'],
['Third Volume Lower','Third Volume Higher'],
['Third Tiny Range','Third Huge Range'],
['Third Red','Third Green'],
['Second Volume Lower','Second Volume Higher'],
['Second Tiny Range','Second Huge Range'],
['Second Red','Second Green'],
['Yesterday Positive Morning Range','Yesterday Negative Morning Range'],
['2 Days Ago Negative Morning Range','2 Days Ago Positive Morning Range'],
['Yesterday Negative Afternoon Range','Yesterday Positive Afternoon Range'],
['2 Days Ago Negative Afternoon Range','2 Days Ago Positive Afternoon Range'],
['Yesterday Morning Range Larger','Yesterday Afternoon Range Larger'],
['2 Days Ago Morning Range Larger','2 Days Ago Afternoon Range Larger'],
['Hour End In Green','Hour End In Red'],
['Hour Last Bottom After Last Peak','Hour Last Bottom Before Last Peak'],
['Range Larger Than Hourly Average','Range Smaller Than Hourly Average'],
['Range Larger Than Daily Average','Range Smaller Than Daily Average'],
['Yesterday Range Larger Than Hourly Average','Yesterday Range Smaller Than Hourly Average'],
['Yesterday Range Larger Than Daily Average','Yesterday Range Smaller Than Daily Average'],
['2 Days Ago Range Larger Than Hourly Average','2 Days Ago Range Smaller Than Hourly Average'],
['2 Days Ago Range Larger Than Daily Average','2 Days Ago Range Smaller Than Daily Average'],
]

for pl in prop_list:
    if pl not in ignore_prop:
        currow = {}
        currow['Prop'] = pl
        if pl in prev_prop_list:
            currow['Category'] = 'Prev'
        elif pl in opening_prop_list:
            currow['Category'] = 'Opening'
        elif pl in late_prop_list:
            currow['Category'] = 'Late'
        elif pl in hour_prop_list:
            currow['Category'] = 'Hour'
        elif pl in daily_prop_list:
            currow['Category'] = 'Daily'
        proprows = datas[datas[pl]==1]
        currow['Count'] = len(proprows)
        # inpair = False
        # for pp in pair_prop:
        #     if pp[0]==pl or pp[1]==pl:
        #         inpair = True
        #         if pl in most_corr_prop.keys():
        #             pc0 = most_corr_prop[pp[0]]
        #             pc1 = most_corr_prop[pp[1]]
        #             if pc0>pc1:
        #                 if pp[0]==pl:
        #                     profitable = proprows[proprows['profitable']==1]
        #                 else:
        #                     profitable = []
        #             else:
        #                 if pp[1]==pl:
        #                     profitable = proprows[proprows['profitable']==1]
        #                 else:
        #                     profitable = []
        #         else:
        #             pp0 = len(datas[datas[pp[0]]==1])
        #             pp1 = len(datas[datas[pp[1]]==1])
        #             if pp0>pp1:
        #                 if pp[0]==pl:
        #                     profitable = proprows[proprows['profitable']==1]
        #                 else:
        #                     profitable = []
        #             else:
        #                 if pp[1]==pl:
        #                     profitable = proprows[proprows['profitable']==1]
        #                 else:
        #                     profitable = []
        # if not inpair:
        profitable = proprows[proprows['profitable']==1]
        great = proprows[proprows['performance']=='Great']
        good = proprows[proprows['performance']=='Good']
        currow['Profitable'] = len(profitable)
        currow['Count Profitable'] = len(proprows[proprows['profitable']==1])
        currow['Great'] = len(great)
        currow['Good'] = len(good)
        if pl in most_corr_prop.keys():
            currow['Corr'] = most_corr_prop[pl]
        global_prop.append(currow)

outdata = pd.DataFrame.from_dict(global_prop)
outdata.to_csv(os.path.join(script_dir,'analyze_global.csv'),index=False)
# outdata = pd.DataFrame(list(global_negate.items()), columns=['Prop','Marks'])
# outdata.to_csv(os.path.join(script_dir,'analyze_global_negate.csv'),index=False)
outdata = pd.DataFrame(list(global_fail.items()), columns=['Prop','Marks'])
outdata.to_csv(os.path.join(script_dir,'analyze_global_fail.csv'),index=False)


# for prop in prop_list:
#     sigprop = corrprofit[corrprofit[prop]>0.5]
#     print("For prop ",prop)
#     print(sigprop.index)
    # print(tabulate(sigprop,headers='keys'))
