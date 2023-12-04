#!/usr/bin/env python

import pandas as pd 
import os
import sys
import csv
from datetime import datetime,timedelta
import numpy as np
import math
from tabulate import tabulate

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

datas = pd.read_csv('raw_data.csv')

def analyze_performance(performance):
    rows = datas[datas['performance']==performance]
    halfnum = math.floor(len(rows)/2)
    props = {}
    for pl in prop_list:
        if rows[pl].sum()>0 and pl not in ignore_prop:
            props[pl] = rows[pl].sum()
    single_props = [ (x,props[x]) for x in props.keys() if props[x]>halfnum ]
    single_props = sorted(single_props,key=lambda x:x[1],reverse=True)
    # print("Single Props:",single_props)
    print(tabulate(single_props,headers="keys",tablefmt="grid"))
    with open('analyze_' + performance + '.csv', 'a') as f:
        # create the csv writer
        writer = csv.writer(f)

        for srow in single_props:
        # write a row to the csv file
            writer.writerow(srow)

    dprops = {}
    for i in range(len(prop_list)):
        for j in range(i,len(prop_list)):
            if prop_list[i]!=prop_list[j] and prop_list[i] not in ignore_prop and prop_list[j] not in ignore_prop:
                for rowi in range(len(rows)):
                    curdata = rows.iloc[rowi]
                    if curdata[prop_list[i]]==1 and curdata[prop_list[j]]==1:
                        curkey = prop_list[i] + ':' + prop_list[j]
                        if curkey in dprops:
                            dprops[curkey] += 1
                        else:
                            dprops[curkey] = 1
    # dprops = sorted(dprops.items(),key=lambda x:x[1],reverse=True)
    fprops = [ (x,dprops[x]) for x in dprops.keys() if dprops[x]>halfnum ]
    fprops = sorted(fprops,key=lambda x:x[1],reverse=True)
    # print("Double Props:",dprops)
    print(tabulate(fprops,headers="keys",tablefmt="grid"))
    with open('analyze_' + performance + '.csv', 'a') as f:
        # create the csv writer
        writer = csv.writer(f)

        for srow in fprops:
        # write a row to the csv file
            writer.writerow(srow)

    tprops = {}
    for i in range(len(prop_list)):
        for j in range(i,len(prop_list)):
            for k in range(j,len(prop_list)):
                if prop_list[i]!=prop_list[j] and prop_list[j]!=prop_list[k] and prop_list[i]!=prop_list[k] and prop_list[i] not in ignore_prop and prop_list[j] not in ignore_prop and prop_list[k] not in ignore_prop:
                    for rowi in range(len(rows)):
                        curdata = rows.iloc[rowi]
                        if curdata[prop_list[i]]==1 and curdata[prop_list[j]]==1 and curdata[prop_list[k]]==1:
                            curkey = prop_list[i] + ':' + prop_list[j] + ':' + prop_list[k]
                            if curkey in tprops:
                                tprops[curkey] += 1
                            else:
                                tprops[curkey] = 1
    ftprops = [ (x,tprops[x]) for x in tprops.keys() if tprops[x]>halfnum ]
    ftprops = sorted(ftprops,key=lambda x:x[1],reverse=True)
    with open('analyze_' + performance + '.csv', 'a') as f:
        # create the csv writer
        writer = csv.writer(f)

        for srow in ftprops:
        # write a row to the csv file
            writer.writerow(srow)
    # print("Double Props:",tprops)
    print(tabulate(ftprops,headers="keys",tablefmt="grid"))

analyze_performance('Great')
