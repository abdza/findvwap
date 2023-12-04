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

datas = pd.read_csv('raw_data.csv')

def check_prop(rows,key):
    tkns = key.split(':')
    gotdat = rows[rows[tkns[0]]==1]
    if len(tkns)>1 and len(gotdat):
        return check_prop(gotdat,':'.join(tkns[1:]))
    else:
        return len(gotdat)

def get_props(rows,props,level):
    halfnum = math.floor(len(rows)/2)
    newprops= {}
    single_props = {}
    toret = {}
    if len(props)>0:
        print("Got ",len(props)," props level ",level)
        for pk,pval in props.items():
            prevp = pk.split(':')
            if len(prevp)==level:
                for pl in prop_list:
                    if pl not in ignore_prop and pl not in prevp:
                        if pk<pl:
                            newkey = pk + ':' + pl
                        else:
                            newkey = pl + ':' + pk
                        if not newkey in toret:
                            rowfound = check_prop(rows[rows[pl]==1],newkey)
                            if rowfound>halfnum:
                                toret[newkey] = rowfound
    else:
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
    curprops = get_props(rows,props,level)
    while(len(curprops)>0 and level<3):
        level += 1
        props = curprops | props
        curprops = get_props(rows,curprops,level)

    props = curprops | props
    props['Dummy'] = 0
    scaler = MinMaxScaler()
    dfprops = pd.DataFrame(list(props.items()),columns=['Prop','Occurance'])
    dfprops['Scale'] = scaler.fit_transform(dfprops['Occurance'].values.reshape(-1,1))
    dfprops.to_csv('analyze_' + performance + '.csv',index=False)

# analyze_performance('Good')
# analyze_performance('Fail')
# analyze_performance('Great')

great_data = pd.read_csv('analyze_Great.csv')
good_data = pd.read_csv('analyze_Good.csv')
positif_data = pd.concat([great_data,good_data])
positif_data.to_csv('analyze_positive.csv',index=False)
