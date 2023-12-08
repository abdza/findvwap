#!/usr/bin/env python

import pandas as pd 
import os
import sys
import csv
from datetime import datetime,timedelta
import numpy as np
import math
from tabulate import tabulate
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
]

datas = pd.read_csv('raw_data_20231205.csv')

global_prop = {}

def get_props(rows,props):
    halfnum = math.floor(len(rows)/2)
    newprops= {}
    single_props = {}
    toret = {}
    for pl in prop_list:
        if rows[pl].sum()>0 and pl not in ignore_prop:
            newprops[pl] = rows[pl].sum()
    single_props = [ (x,newprops[x]) for x in newprops.keys() if newprops[x]>halfnum ]
    for sp,sn in single_props:
        if sp not in props:
            toret[sp] = sn

    return toret


def analyze_performance(performance):
    rows = datas[datas['performance']==performance]
    props = {}
    level = 0
    curprops = get_props(rows,props)
    props = curprops
    props['Dummy'] = 0
    scaler = MinMaxScaler()
    dfprops = pd.DataFrame(list(props.items()),columns=['Prop','Occurance'])
    dfprops['Scale'] = scaler.fit_transform(dfprops['Occurance'].values.reshape(-1,1))
    dfprops.to_csv('analyze_' + performance + '.csv',index=False)

def add_props(props,mark):
    for i in range(len(props)):
        curprop = props.iloc[i]
        if curprop['Prop'] in global_prop:
            global_prop[curprop['Prop']] += mark
        else:
            global_prop[curprop['Prop']] = mark


analyze_performance('Great')
analyze_performance('Good')
analyze_performance('Fair')
analyze_performance('Fail')

great_data = pd.read_csv('analyze_Great.csv')
good_data = pd.read_csv('analyze_Good.csv')
fair_data = pd.read_csv('analyze_Fair.csv')
fail_data = pd.read_csv('analyze_Fail.csv')
# positif_data = pd.concat([great_data,good_data])
# positif_data.to_csv('analyze_positive.csv',index=False)

add_props(great_data,3)
add_props(good_data,2)
add_props(fair_data,1)

for i in range(len(fail_data)):
    curdat = fail_data.iloc[i]
    marks = -6
    got_great = great_data[great_data['Prop']==curdat['Prop']]
    if len(got_great)>0:
        marks += 3
    print("Got great:",got_great)
    got_good = good_data[good_data['Prop']==curdat['Prop']]
    if len(got_good)>0:
        marks += 2
    print("Got good:",got_good)
    got_fair = fair_data[fair_data['Prop']==curdat['Prop']]
    if len(got_fair)>0:
        marks += 1
    print("Got fair:",got_fair)
    if curdat['Prop'] in global_prop:
        global_prop[curdat['Prop']] += marks
    else:
        global_prop[curdat['Prop']] = marks


outdata = pd.DataFrame(list(global_prop.items()), columns=['Prop','Marks'])
outdata.to_csv('analyze_global.csv',index=False)
