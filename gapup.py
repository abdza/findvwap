#!/usr/bin/env python

from logging import lastResort
import pandas as pd 
import os
import sys
import csv
import getopt
from datetime import datetime,timedelta
from pandas.core.frame import AnyAll
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
from tensorflow.keras.models import load_model
import tensorflow as tf
import autokeras as ak

def green_candle(candle):
    if candle['open']<candle['close']:
        return True
    else:
        return False

def red_candle(candle):
    if candle['open']>candle['close']:
        return True
    else:
        return False

def clean_bull(first,second):   # second is higher than first
    score = 0
    if first['low']<second['low']:
        score += 1
        if first['high']<second['high']:
            score += 1
            if first['open']<second['open']:
                score += 1
                if first['close']<second['close']:
                    score += 1
    return score

def clean_bear(first,second):   # second is lower than first
    score = 0
    if first['low']>second['low']:
        score += 1
        if first['high']>second['high']:
            score += 1
            if first['open']>second['open']:
                score += 1
                if first['close']>second['close']:
                    score += 1
    return score

def bull(first,second):   # second is higher than first
    score = 0
    if first['low']<second['low']:
        score += 1
    if first['high']<second['high']:
        score += 1
    if first['open']<second['open']:
        score += 1
    if first['close']<second['close']:
        score += 1
    return score

def bear(first,second):   # second is lower than first
    score = 0
    if first['low']>second['low']:
        score += 1
    if first['high']>second['high']:
        score += 1
    if first['open']>second['open']:
        score += 1
    if first['close']>second['close']:
        score += 1
    return score

def find_bounce(candles):
    reversed_candles = candles.iloc[::-1]
    bounce = 0
    pullback = 0
    for i in range(len(reversed_candles.index)-2):
        curcandle = reversed_candles.iloc[i]
        prevcandle = reversed_candles.iloc[i+1]
        if bounce==0:     # currently in pullback mode
            if green_candle(curcandle) and bull(prevcandle,curcandle)>2:
                pullback += 1
            else:     # curcandle is red, so now in bounce mode
                bounce += 1
        else:
            if red_candle(curcandle) and bear(prevcandle,curcandle)>2:
                bounce += 1
            else:
                break
    return pullback,bounce

def range_avg(candles):
    p_range = candles['high'] - candles['low']
    return p_range.mean()

def find_levels(candles):
    start_date = candles.iloc[0]['date']
    end_date = candles.iloc[-1]['date']
    response = "Levels:"
    min = candles['low'].min()
    max = candles['high'].max()
    p_range = candles['high'] - candles['low']
    range_avg = p_range.mean()
    vol_avg = candles['volume'].mean()
    response += "\nStart: " + str(start_date)
    response += "\nEnd: " + str(end_date)
    response += "\nMin: " + str(min)
    response += "\nMax: " + str(max)
    response += "\nRange Avg: " + str(numerize.numerize(range_avg))
    response += "\nVol Avg: " + str(numerize.numerize(vol_avg))

    datarange = max - min
    if datarange < 50:
        kint = int(datarange / 0.5)
    else:
        kint = int(datarange % 20)

    datalen = len(candles)

    highlevels = np.array(candles['high'])
    resistancelevels = {}
    donecluster = []
    finalreslevels = {}
    dresponse = ""
    try:
        kmeans = KMeans(n_clusters=kint,n_init=10).fit(highlevels.reshape(-1,1))
        highclusters = kmeans.predict(highlevels.reshape(-1,1))
        for cidx in range(datalen):
            curcluster = highclusters[cidx]
            if curcluster not in resistancelevels:
                resistancelevels[curcluster] = 1
            else:
                resistancelevels[curcluster] += 1
        for cidx in range(datalen):
            candle = candles.iloc[cidx]
            curcluster = highclusters[cidx]
            if resistancelevels[curcluster] > 2:
                if curcluster not in donecluster:
                    donecluster.append(curcluster)
                    finalreslevels[curcluster] = {'level':candle['high'],'count':1}
                else:
                    finalreslevels[curcluster] = {'level':(finalreslevels[curcluster]['level'] + candle['high'])/2,'count':finalreslevels[curcluster]['count']+1}
    except:
        print("Got error for levels")

    response += "\n\nResistance levels:"
    sortedreslevels = []
    for lvl,clstr in sorted(finalreslevels.items(),key=lambda x: x[1]['level']):
        sortedreslevels.append(clstr)
        response += "\n" + str(clstr['level']) + " : " + str(clstr['count'])

    if datarange < 50:
        kint = int(datarange / 0.5)
    else:
        kint = int(datarange % 20)
    lowlevels = np.array(candles['low'])
    supportlevels = {}
    donecluster = []
    finalsuplevels = {}
    dresponse = ""
    try:
        kmeans = KMeans(n_clusters=kint,n_init=10).fit(lowlevels.reshape(-1,1))
        lowclusters = kmeans.predict(lowlevels.reshape(-1,1))
        for cidx in range(datalen):
            curcluster = lowclusters[cidx]
            if curcluster not in supportlevels:
                supportlevels[curcluster] = 1
            else:
                supportlevels[curcluster] += 1
        for cidx in range(datalen):
            candle = candles.iloc[cidx]
            curcluster = lowclusters[cidx]
            if supportlevels[curcluster] > 2:
                if curcluster not in donecluster:
                    donecluster.append(curcluster)
                    finalsuplevels[curcluster] = {'level':candle['low'],'count':1}
                else:
                    finalsuplevels[curcluster] = {'level':(finalsuplevels[curcluster]['level'] + candle['low'])/2,'count':finalsuplevels[curcluster]['count']+1}
    except:
        print("Got error for levels")

    response += "\n\nSupport levels:"
    sortedsuplevels = []
    for lvl,clstr in sorted(finalsuplevels.items(),key=lambda x: x[1]['level']):
        sortedsuplevels.append(clstr)
        response += "\n" + str(clstr['level']) + " : " + str(clstr['count'])
    
    response += "\n\n" + dresponse
    # return sortedreslevels, sortedsuplevels
    return sortedsuplevels

def body_top(candle):
    if candle['open']>candle['close']:
        return candle['open']
    else:
        return candle['close']

def body_bottom(candle):
    if candle['open']<candle['close']:
        return candle['open']
    else:
        return candle['close']

def body_length(candle):
    return body_top(candle) - body_bottom(candle)

def is_peak_body(candles,c_pos,dlen=1):
    if c_pos>0 and c_pos<len(candles)-dlen:
        before = False
        cloop = dlen
        while cloop>0 and not before:
            before = body_top(candles.iloc[c_pos])>body_top(candles.iloc[c_pos-cloop])
            cloop -= 1
        after = False
        cloop = dlen
        while cloop>0 and not after:
            after = body_top(candles.iloc[c_pos])>body_top(candles.iloc[c_pos+cloop])
            cloop -= 1
        return before and after
    else:
        return False

def is_bottom_body(candles,c_pos,dlen=1):
    if c_pos>0 and c_pos<len(candles)-dlen:
        before = False
        cloop = dlen
        while cloop>0 and not before:
            before = body_bottom(candles.iloc[c_pos])<body_bottom(candles.iloc[c_pos-cloop])
            cloop -= 1
        after = False
        cloop = dlen
        while cloop>0 and not after:
            after = body_bottom(candles.iloc[c_pos])<body_bottom(candles.iloc[c_pos+cloop])
            cloop -= 1
        return before and after
    else:
        return False

def gather_range_body(candles):
    peaks = []
    bottoms = []
    peakindex = []
    bottomindex = []
    for i in range(len(candles)):
        if is_peak_body(candles,i):
            peaks.append(candles.iloc[i])
        if is_bottom_body(candles,i):
            bottoms.append(candles.iloc[i])
    if len(peaks)==0:
        for i in range(len(candles)):
            if is_peak_body(candles,i,2) and i-1 not in peakindex:
                peaks.append(candles.iloc[i])
                peakindex.append(i)
    if len(bottoms)==0:
        for i in range(len(candles)):
            if is_bottom_body(candles,i,2) and i-1 not in bottomindex:
                bottoms.append(candles.iloc[i])
                bottomindex.append(i)

    return peaks,bottoms

def is_peak_unit(unit,candles,c_pos,dlen=1):
    if c_pos>0 and c_pos<len(candles)-dlen:
        before = False
        cloop = dlen
        while cloop>0 and not before:
            before = candles.iloc[c_pos][unit]>candles.iloc[c_pos-cloop][unit]
            cloop -= 1
        after = False
        cloop = dlen
        while cloop>0 and not after:
            after = candles.iloc[c_pos][unit]>candles.iloc[c_pos+cloop][unit]
            cloop -= 1
        return before and after
    else:
        return False

def is_bottom_unit(unit,candles,c_pos,dlen=1):
    if c_pos>0 and c_pos<len(candles)-dlen:
        before = False
        cloop = dlen
        while cloop>0 and not before:
            before = candles.iloc[c_pos][unit]<candles.iloc[c_pos-cloop][unit]
            cloop -= 1
        after = False
        cloop = dlen
        while cloop>0 and not after:
            after = candles.iloc[c_pos][unit]<candles.iloc[c_pos+cloop][unit]
            cloop -= 1
        return before and after
    else:
        return False

def gather_range_unit(unit,candles):
    peaks = []
    bottoms = []
    for i in range(len(candles)):
        if is_peak_unit(unit,candles,i):
            peaks.append(candles.iloc[i])
        if is_bottom_unit(unit,candles,i):
            bottoms.append(candles.iloc[i])
    if len(peaks)==0:
        for i in range(len(candles)):
            if is_peak_unit(unit,candles,i,2):
                peaks.append(candles.iloc[i])
    if len(bottoms)==0:
        for i in range(len(candles)):
            if is_bottom_unit(unit,candles,i,2):
                bottoms.append(candles.iloc[i])

    return peaks,bottoms

def is_peak(candles,c_pos,dlen=1):
    if c_pos==0 and len(candles)>1:
        after = candles.iloc[c_pos]['high']>candles.iloc[c_pos+1]['high']
        return after
    elif c_pos>0 and c_pos<len(candles)-dlen:
        before = False
        cloop = dlen
        while cloop>0 and not before:
            before = candles.iloc[c_pos]['high']>candles.iloc[c_pos-cloop]['high']
            cloop -= 1
        after = False
        cloop = dlen
        while cloop>0 and not after:
            after = candles.iloc[c_pos]['high']>candles.iloc[c_pos+cloop]['high']
            cloop -= 1
        return before and after
    elif c_pos==len(candles)-dlen:
        before = candles.iloc[c_pos]['high']>candles.iloc[c_pos-1]['high']
        return before
    else:
        return False

def is_bottom(candles,c_pos,dlen=1):
    if c_pos==0 and len(candles)>1:
        after = candles.iloc[c_pos]['low']<candles.iloc[c_pos+1]['low']
        return after
    elif c_pos>0 and c_pos<len(candles)-dlen:
        before = False
        cloop = dlen
        while cloop>0 and not before:
            before = candles.iloc[c_pos]['low']<candles.iloc[c_pos-cloop]['low']
            cloop -= 1
        after = False
        cloop = dlen
        while cloop>0 and not after:
            after = candles.iloc[c_pos]['low']<candles.iloc[c_pos+cloop]['low']
            cloop -= 1
        return before and after
    elif c_pos==len(candles)-dlen:
        before = candles.iloc[c_pos]['low']<candles.iloc[c_pos-1]['low']
        return before
    else:
        return False

def gather_range(candles):
    peaks = []
    bottoms = []
    for i in range(len(candles)):
        if is_peak(candles,i):
            peaks.append(candles.iloc[i])
        if is_bottom(candles,i):
            bottoms.append(candles.iloc[i])
    if len(peaks)==0:
        for i in range(len(candles)):
            if is_peak(candles,i,2):
                peaks.append(candles.iloc[i])
    if len(bottoms)==0:
        for i in range(len(candles)):
            if is_bottom(candles,i,2):
                bottoms.append(candles.iloc[i])

    return peaks,bottoms

def min_bottom(bottoms,exclude=None):
    curbottom = None
    for candle in bottoms:
        goon = True
        if exclude is not None:
            for intest in exclude:
                if intest['date']==candle['date']:
                    goon = False
        if curbottom is None and goon:
            curbottom = candle
        else:
            if goon and candle['low'] < curbottom['low']:
                curbottom = candle
    return curbottom

def max_peak(peaks,exclude=None):
    curpeak = None
    for candle in peaks:
        goon = True
        if exclude is not None:
            for intest in exclude:
                if intest['date']==candle['date']:
                    goon = False
        if curpeak is None and goon:
            curpeak = candle
        else:
            if goon and candle['high'] > curpeak['high']:
                curpeak = candle
    return curpeak

inputfile = 'stocks.csv'
outfile = 'shorts.csv'
instockdate = None
openrangelimit = 1
purchaselimit = 300
completelist = False
trackunit = None
manualstocks = None
perctarget = 10
opts, args = getopt.getopt(sys.argv[1:],"i:o:d:r:p:c:u:x:s:",["input=","out=","date=","range=","purchaselimit=","complete=","unit=","perctarget=","stocks="])
for opt, arg in opts:
    if opt in ("-i", "--input"):
        inputfile = arg
    if opt in ("-o", "--out"):
        outfile = arg
    if opt in ("-d", "--date"):
        instockdate = datetime.strptime(arg + ' 23:59:59', '%Y-%m-%d %H:%M:%S')
    if opt in ("-r", "--range"):
        openrangelimit = float(arg)
    if opt in ("-x", "--perctarget"):
        perctarget = float(arg)
    if opt in ("-p", "--purchaselimit"):
        purchaselimit = float(arg)
    if opt in ("-c", "--complete"):
        completelist = True
    if opt in ("-u", "--unit"):
        if arg in ['close','high','low','open']:
            trackunit = arg
    if opt in ("-s", "--stocks"):
        manualstocks = arg.split(',')

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

prop_marks = [
    {'prop':'Volume Above 5 Time Average','marks':10},
    {'prop':'Volume Above 10 Time Average','marks':10},
    {'prop':'Volume Consecutive Above 5 Time Average','marks':10},
    {'prop':'Volume Consecutive Above 10 Time Average','marks':10},
    {'prop':'Huge Range','marks':3},
    {'prop':'Second Huge Range','marks':3},
    {'prop':'Third Huge Range','marks':3},
    {'prop':'Consecutive Early Huge Range','marks':3},
    {'prop':'Consecutive Late Huge Range','marks':3},
    {'prop':'Consecutive Huge Range','marks':3},
    {'prop':'Third Green','marks':3},
    {'prop':'Second Green','marks':3},
    {'prop':'Third Long','marks':3},
    {'prop':'Second Long','marks':3},
    {'prop':'Lower High','marks':3},
    {'prop':'Max After Min','marks':3},
    {'prop':'Volume Open Lower','marks':2},
    {'prop':'Second Range Shorter','marks':2},
    {'prop':'Higher Low','marks':2},
    {'prop':'Open Lower Than 2 Prev Max','marks':2},
    {'prop':'Range Lower Average','marks':2},
    {'prop':'Third Range Shorter','marks':2},
    {'prop':'Continue Higher Low','marks':1},
    {'prop':'Range Lower 2 Day Average','marks':1},
    {'prop':'Continue Lower High','marks':-1},
    {'prop':'Third Red','marks':-1},
    {'prop':'First Red','marks':-1},
    {'prop':'Third Short','marks':-1},
    {'prop':'Third Range Shorter','marks':-1},
    {'prop':'Third Range Very Shorter','marks':-1},
    {'prop':'Second Red','marks':-1},
    {'prop':'Second Short','marks':-1},
    {'prop':'Lower Low','marks':-1},
    {'prop':'Continue Lower Low','marks':-1},
    {'prop':'Third Reverse Hammer','marks':-1},
    {'prop':'Second Reverse Hammer','marks':-1},
    {'prop':'First Reverse Hammer','marks':-1},
    {'prop':'Early Top Level','marks':-1},
    {'prop':'Second Volume Lower','marks':-2},
    {'prop':'Third Volume Lower','marks':-2},
    {'prop':'Consecutive Lower Volume','marks':-2},
    {'prop':'Limp Second Diff','marks':-2},
    {'prop':'Limp Third Diff','marks':-2},
    {'prop':'Consecutive Limp Diff','marks':-2},
    {'prop':'Top Level','marks':-3},
    {'prop':'Tiny Range','marks':-3},
    {'prop':'Second Tiny Range','marks':-3},
    {'prop':'Third Tiny Range','marks':-3},
    {'prop':'Consecutive Early Tiny Range','marks':-3},
    {'prop':'Consecutive Late Tiny Range','marks':-3},
    {'prop':'Consecutive Tiny Range','marks':-3},
    {'prop':'Huge Negative Range','marks':-3},
    {'prop':'Second Huge Negative Range','marks':-3},
    {'prop':'Third Huge Negative Range','marks':-3},
    {'prop':'Consecutive Early Huge Negative Range','marks':-3},
    {'prop':'Consecutive Late Huge Negative Range','marks':-3},
    {'prop':'Min After Max','marks':-3},
    {'prop':'Yesterday End In Red','marks':-3},
    {'prop':['Yesterday End In Red','Yesterday End Volume Above Average'],'marks':-3},
    {'prop':['Third Range Longer','Third Red'],'marks':-3},
    {'prop':['Second Range Longer','Second Red'],'marks':-3},
    {'prop':['Second Green','Second Long','Second Huge Range','Third Red','Third Range Very Shorter','Late Top Level'],'marks':-5},
]

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
if manualstocks:
    stocks = pd.DataFrame({'Ticker':manualstocks})
else:
    stocks = pd.read_csv(os.path.join(script_dir,inputfile),header=0)

def minute_test(peaks,bottoms):
    if len(bottoms)>0 and len(peaks)>0:
        if bottoms[0]['date']<peaks[0]['date']:
            return True
    return False

minprofit = 0.5

def profit_test(peaks,bottoms):
    if len(bottoms)>0 and len(peaks)>0:
        maxp = max_peak(peaks)
        minb = min_bottom(bottoms)
        if minb['date']<maxp['date'] and maxp['high'] - minb['low']>minprofit:
            return maxp['high'] - minb['low']
    return False

def minute_profit_test(peaks,bottoms):
    if len(bottoms)>0 and len(peaks)>0:
        maxp = max_peak(peaks)
        minb = min_bottom(bottoms)
        if bottoms[0]['date']<peaks[0]['date'] and minb['date']<maxp['date'] and maxp['high'] - minb['low']>minprofit:
            return maxp['high'] - minb['low']
    return False

def hammer_pattern(candle):
    max_range = candle['range']/2.5
    return body_length(candle) < max_range and body_bottom(candle) > candle['low'] + max_range

def reverse_hammer_pattern(candle):
    max_range = candle['range']/2.5
    return body_length(candle) < max_range and body_top(candle) < candle['low'] + max_range

def pattern_test(today):
    minrange = 0.15
    candle1 = today.iloc[0]
    if red_candle(candle1):
        return False
    either_green = True
    if len(today)>2:
        either_green = green_candle(today.iloc[1]) or green_candle(today.iloc[2])
    first_hammer = hammer_pattern(candle1)
    if not (either_green or first_hammer):
        return False
    if len(today)>1:
        if reverse_hammer_pattern(today.iloc[1]):
            return False
        if today.iloc[1]['range'] < body_length(today.iloc[0])/3:
            return False
    if len(today)>2:
        if candle1['range']<minrange and today.iloc[1]['range']<minrange and today.iloc[2]['range']<minrange:
            return False
        for i in range(2):
            if body_length(today.iloc[i+1]) > body_length(today.iloc[i]) * 0.7 and red_candle(today.iloc[i+1]):
                return False
        if len(today)>3:
            if body_top(today.iloc[3]) < body_bottom(today.iloc[2]):
                return False
            for i in range(3):
                if today.iloc[i+1]['high']<today.iloc[i]['low']:
                    return False
                if today.iloc[i+1]['range']>today.iloc[i]['range']*0.75 and red_candle(today.iloc[i+1]):
                    return False
            if len(today)>4:
                for i in range(4):
                    if red_candle(today.iloc[i]) and red_candle(today.iloc[i+1]):
                        return False
    return True

def prev_avg_test(today,yesterday,target_multiple=4):
    y_avg = range_avg(yesterday)
    if len(today)>2:
        for i in range(2):
            if today.iloc[i]['range']>y_avg*target_multiple:
                return True
        early_range = today.iloc[1]['high'] - today.iloc[0]['low']
        if early_range>y_avg*target_multiple:
            return True
        late_range = today.iloc[2]['high'] - today.iloc[0]['low']
        if late_range>y_avg*target_multiple:
            return True
    return False

def append_hash_set(hashdata,key,value):
    if not key in hashdata:
        hashdata[key] = []
        hashdata[key].append(value)
    else:
        if not value in hashdata[key]:
            hashdata[key].append(value)
    return hashdata

def set_params(ticker,proptext,prop_data,tickers_data,all_props):
    if not proptext in prop_list:
        raise Exception("Proptext '" + proptext + "' does not exists in prop_list")
    prop_data = append_hash_set(prop_data,proptext,ticker)
    tickers_data = append_hash_set(tickers_data,ticker,proptext)
    all_props.append(proptext)
    return prop_data, tickers_data, all_props

def findgap():
    end_date = datetime.now()
    if instockdate:
        if isinstance(instockdate,datetime):
            end_date = instockdate
        else:
            end_date = datetime.strptime(instockdate + ' 23:59:59', '%Y-%m-%d %H:%M:%S')
    print("Got end date:",end_date)
    start_date = end_date - timedelta(days=365)
    tickers = []
    tickers_data = {}
    prop_data = {}
    ticker_marks = {}
    latest_price = {}
    latest_date = {}
    first_price = {}
    max_price = {}
    maxmovement = {}
    levels = {}
    all_props = []
    end_of_trading = False

    for i in range(len(stocks.index)):
    # for i in range(5):
        if isinstance(stocks.iloc[i]['Ticker'], str):
            ticker = stocks.iloc[i]['Ticker'].upper()
            dticker = yq.Ticker(ticker)
            candles = dticker.history(start=start_date,end=end_date,interval='1d')
            candles = candles.loc[(candles['volume']>0)]
            print("Processing ",ticker, " got ",len(candles))
        else:
            continue

        if len(candles):
            candles = candles.reset_index(level=[0,1])
            candles['range'] = candles['high'] - candles['low']
            candles['body_length'] = candles['close'] - candles['open']
            curcandle = candles.iloc[-1]
            if isinstance(curcandle['date'],datetime):
                curkey = str(curcandle['date'].date())
            else:
                curkey = str(curcandle['date'])
            minute_end_date = datetime.strptime(curkey + ' 23:59:59', '%Y-%m-%d %H:%M:%S')
            minute_start_date = minute_end_date - timedelta(days=5)
            full_minute_candles = dticker.history(start=minute_start_date,end=minute_end_date,interval='15m')
            full_minute_candles['range'] = full_minute_candles['high'] - full_minute_candles['low']
            full_minute_candles['body_length'] = full_minute_candles['close'] - full_minute_candles['open']
            peaks = []
            bottoms = []
            if len(full_minute_candles)>1:
                tickers.append(ticker)
                full_minute_candles = full_minute_candles.reset_index(level=[0,1])
                minutelastcandle = full_minute_candles.iloc[-2]
                ldate = str(minutelastcandle['date'].date())
                fdate = str(datetime.date(minutelastcandle['date'])+timedelta(days=1))
                minute_candles = full_minute_candles.loc[(full_minute_candles['date']>ldate)]
                minute_candles = minute_candles.loc[(minute_candles['date']<fdate)]
                nowdate = str(datetime.now().date())
                if curkey!=nowdate:
                    end_of_trading = True

                if manualstocks:
                    print("Minute Candles:",minute_candles)

                datediff = 1
                bdate = str(datetime.date(minutelastcandle['date'])-timedelta(days=datediff))
                bminute_candles = full_minute_candles.loc[(full_minute_candles['date']>bdate)]
                bminute_candles = bminute_candles.loc[(bminute_candles['date']<ldate)]
                while len(bminute_candles)==0 and datediff<=5:
                    datediff += 1
                    bdate = str(datetime.date(minutelastcandle['date'])-timedelta(days=datediff))
                    bminute_candles = full_minute_candles.loc[(full_minute_candles['date']>bdate)]
                    bminute_candles = bminute_candles.loc[(bminute_candles['date']<ldate)]

                datediff += 1
                bbdate = str(datetime.date(minutelastcandle['date'])-timedelta(days=datediff))
                bbminute_candles = full_minute_candles.loc[(full_minute_candles['date']>bbdate)]
                bbminute_candles = bbminute_candles.loc[(bbminute_candles['date']<bdate)]
                while len(bbminute_candles)==0 and datediff<=5:
                    datediff += 1
                    bbdate = str(datetime.date(minutelastcandle['date'])-timedelta(days=datediff))
                    bbminute_candles = full_minute_candles.loc[(full_minute_candles['date']>bbdate)]
                    bbminute_candles = bbminute_candles.loc[(full_minute_candles['date']<bdate)]

                peaks,bottoms = gather_range(minute_candles)

                if len(bminute_candles)==0 and len(bbminute_candles)==0:
                    prop_data, tickers_data, all_props = set_params(ticker,'New IPO',prop_data,tickers_data,all_props)
                if len(bbminute_candles)==0:
                    prop_data, tickers_data, all_props = set_params(ticker,'Fairly New IPO',prop_data,tickers_data,all_props)
                if green_candle(minute_candles.iloc[0]):
                    prop_data, tickers_data, all_props = set_params(ticker,'First Green',prop_data,tickers_data,all_props)
                if len(minute_candles)>1 and green_candle(minute_candles.iloc[1]):
                    prop_data, tickers_data, all_props = set_params(ticker,'Second Green',prop_data,tickers_data,all_props)
                    if 'First Green' in tickers_data[ticker]:
                        prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Early Green',prop_data,tickers_data,all_props)
                if len(minute_candles)>2 and green_candle(minute_candles.iloc[2]):
                    prop_data, tickers_data, all_props = set_params(ticker,'Third Green',prop_data,tickers_data,all_props)
                    if 'Consecutive Early Green' in tickers_data[ticker]:
                        prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Green',prop_data,tickers_data,all_props)
                    elif 'Second Green' in tickers_data[ticker]:
                        prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Late Green',prop_data,tickers_data,all_props)
                if red_candle(minute_candles.iloc[0]):
                    prop_data, tickers_data, all_props = set_params(ticker,'First Red',prop_data,tickers_data,all_props)
                if len(minute_candles)>1 and red_candle(minute_candles.iloc[1]):
                    prop_data, tickers_data, all_props = set_params(ticker,'Second Red',prop_data,tickers_data,all_props)
                    if 'First Red' in tickers_data[ticker]:
                        prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Early Red',prop_data,tickers_data,all_props)
                if len(minute_candles)>2 and red_candle(minute_candles.iloc[2]):
                    prop_data, tickers_data, all_props = set_params(ticker,'Third Red',prop_data,tickers_data,all_props)
                    if 'Consecutive Early Red' in tickers_data[ticker]:
                        prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Red',prop_data,tickers_data,all_props)
                    elif 'Second Red' in tickers_data[ticker]:
                        prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Late Red',prop_data,tickers_data,all_props)
                y_avg = bminute_candles['range'].mean()
                yy_avg = bbminute_candles['range'].mean()
                avg_multiple = 3
                latest_price = append_hash_set(latest_price,ticker,minute_candles.iloc[-1]['close'])
                if len(minute_candles)>1:
                    first_price = append_hash_set(first_price,ticker,body_top(minute_candles.iloc[1]))
                else:
                    first_price = append_hash_set(first_price,ticker,minute_candles.iloc[0]['open'])
                if len(bottoms)>0:
                    minb = min_bottom(bottoms,[minute_candles.iloc[0]])
                    if minb is not None:
                        if str(minb['date'].time())<'12:00:00':
                            prop_data, tickers_data, all_props = set_params(ticker,'Bottom Before Noon',prop_data,tickers_data,all_props)
                        elif str(minb['date'].time())>'13:00:00':
                            prop_data, tickers_data, all_props = set_params(ticker,'Bottom After Noon',prop_data,tickers_data,all_props)
                        else:
                            prop_data, tickers_data, all_props = set_params(ticker,'Bottom Lunch',prop_data,tickers_data,all_props)

                if len(peaks)>0:
                    maxp = max_peak(peaks,[minute_candles.iloc[0]])
                    minp = min_bottom(bottoms,[minute_candles.iloc[0]])
                    if maxp is not None:
                        if minp is not None:
                            if maxp['date']>minp['date']:
                                prop_data, tickers_data, all_props = set_params(ticker,'Max After Min',prop_data,tickers_data,all_props)
                            else:
                                prop_data, tickers_data, all_props = set_params(ticker,'Min After Max',prop_data,tickers_data,all_props)
                                
                        max_price = append_hash_set(max_price,ticker,maxp['high'])
                        if str(maxp['date'].time())<'12:00:00':
                            prop_data, tickers_data, all_props = set_params(ticker,'Peak Before Noon',prop_data,tickers_data,all_props)
                        elif str(maxp['date'].time())>'13:00:00':
                            prop_data, tickers_data, all_props = set_params(ticker,'Peak After Noon',prop_data,tickers_data,all_props)
                        else:
                            prop_data, tickers_data, all_props = set_params(ticker,'Peak Lunch',prop_data,tickers_data,all_props)
                    else:
                        max_price = append_hash_set(max_price,ticker,minute_candles.iloc[-1]['high'])
                    if len(minute_candles)>2:
                        for i in range(len(minute_candles)):
                            if red_candle(minute_candles.iloc[i]) and green_candle(minute_candles.iloc[i-1]) and minute_candles.iloc[i]['range'] > minute_candles.iloc[i-1]['range']*0.6:
                                prop_data, tickers_data, all_props = set_params(ticker,'Big Reverse',prop_data,tickers_data,all_props)
                            if red_candle(minute_candles.iloc[i]) and red_candle(minute_candles.iloc[i-1]) and green_candle(minute_candles.iloc[i-2]) and minute_candles.iloc[i]['range'] + minute_candles.iloc[i-1]['range'] > minute_candles.iloc[i-2]['range']*0.6:
                                prop_data, tickers_data, all_props = set_params(ticker,'Two Small Reverse',prop_data,tickers_data,all_props)

                else:
                    max_price = append_hash_set(max_price,ticker,minute_candles.iloc[-1]['high'])

                if minute_candles.iloc[0]['range'] > y_avg*avg_multiple:
                    prop_data, tickers_data, all_props = set_params(ticker,'Range Above Average',prop_data,tickers_data,all_props)
                    if minute_candles.iloc[0]['range'] > yy_avg*avg_multiple:
                        prop_data, tickers_data, all_props = set_params(ticker,'Range Above 2 Day Average',prop_data,tickers_data,all_props)
                if minute_candles.iloc[0]['range'] < y_avg*avg_multiple:
                    prop_data, tickers_data, all_props = set_params(ticker,'Range Lower Average',prop_data,tickers_data,all_props)
                    if minute_candles.iloc[0]['range'] < yy_avg*avg_multiple:
                        prop_data, tickers_data, all_props = set_params(ticker,'Range Lower 2 Day Average',prop_data,tickers_data,all_props)
                if len(minute_candles)>1:

                    if minute_candles.iloc[0]['low']<minute_candles.iloc[1]['low']:
                        prop_data, tickers_data, all_props = set_params(ticker,'Higher Low',prop_data,tickers_data,all_props)
                        if len(minute_candles)>2:
                            if minute_candles.iloc[1]['low']<minute_candles.iloc[2]['low']:
                                prop_data, tickers_data, all_props = set_params(ticker,'Continue Higher Low',prop_data,tickers_data,all_props)
                    if minute_candles.iloc[0]['low']>minute_candles.iloc[1]['low']:
                        prop_data, tickers_data, all_props = set_params(ticker,'Lower Low',prop_data,tickers_data,all_props)
                        if len(minute_candles)>2:
                            if minute_candles.iloc[1]['low']>minute_candles.iloc[2]['low']:
                                prop_data, tickers_data, all_props = set_params(ticker,'Continue Lower Low',prop_data,tickers_data,all_props)
                    if minute_candles.iloc[0]['high']<minute_candles.iloc[1]['high']:
                        prop_data, tickers_data, all_props = set_params(ticker,'Higher High',prop_data,tickers_data,all_props)
                        if len(minute_candles)>2:
                            if minute_candles.iloc[1]['high']<minute_candles.iloc[2]['high']:
                                prop_data, tickers_data, all_props = set_params(ticker,'Continue Higher High',prop_data,tickers_data,all_props)
                    if minute_candles.iloc[0]['high']>minute_candles.iloc[1]['high']:
                        prop_data, tickers_data, all_props = set_params(ticker,'Lower High',prop_data,tickers_data,all_props)
                        if len(minute_candles)>2:
                            if minute_candles.iloc[1]['high']>minute_candles.iloc[2]['high']:
                                prop_data, tickers_data, all_props = set_params(ticker,'Continue Lower High',prop_data,tickers_data,all_props)
                    if minute_candles.iloc[0]['body_length']*0.4<minute_candles.iloc[1]['body_length']:
                        prop_data, tickers_data, all_props = set_params(ticker,'Second Long',prop_data,tickers_data,all_props)
                    if minute_candles.iloc[0]['body_length']*0.2>minute_candles.iloc[1]['body_length']:
                        prop_data, tickers_data, all_props = set_params(ticker,'Second Short',prop_data,tickers_data,all_props)
                    if len(minute_candles)>2:
                        if minute_candles.iloc[1]['body_length']*0.4<minute_candles.iloc[2]['body_length']:
                            prop_data, tickers_data, all_props = set_params(ticker,'Third Long',prop_data,tickers_data,all_props)
                        if minute_candles.iloc[1]['body_length']*0.2>minute_candles.iloc[2]['body_length']:
                            prop_data, tickers_data, all_props = set_params(ticker,'Third Short',prop_data,tickers_data,all_props)
                if len(bbminute_candles)>1:
                    if minute_candles.iloc[0]['low']>bbminute_candles['high'].max():
                        prop_data, tickers_data, all_props = set_params(ticker,'Open Higher Than 2 Prev Max',prop_data,tickers_data,all_props)
                    if minute_candles.iloc[0]['low']<bbminute_candles['high'].max():
                        prop_data, tickers_data, all_props = set_params(ticker,'Open Lower Than 2 Prev Max',prop_data,tickers_data,all_props)
                if len(bminute_candles)>1:
                    if red_candle(bminute_candles.iloc[-1]):
                        prop_data, tickers_data, all_props = set_params(ticker,'Yesterday End In Red',prop_data,tickers_data,all_props)
                    if bminute_candles.iloc[-1]['range'] > y_avg:
                        prop_data, tickers_data, all_props = set_params(ticker,'Yesterday End Volume Above Average',prop_data,tickers_data,all_props)
                    if body_top(minute_candles.iloc[0]) < body_bottom(bminute_candles.iloc[-1]):
                        prop_data, tickers_data, all_props = set_params(ticker,'Gap Down',prop_data,tickers_data,all_props)
                        if minute_candles.iloc[0]['range'] > body_bottom(bminute_candles.iloc[-1]) - body_top(minute_candles.iloc[0]):
                            prop_data, tickers_data, all_props = set_params(ticker,'Range More Than Gap Down',prop_data,tickers_data,all_props)
                        if body_bottom(minute_candles.iloc[0]) < bminute_candles['low'].min():
                            prop_data, tickers_data, all_props = set_params(ticker,'Gap Down Below Prev Min',prop_data,tickers_data,all_props)
                    if body_top(minute_candles.iloc[0]) + y_avg < body_bottom(bminute_candles.iloc[-1]):
                        prop_data, tickers_data, all_props = set_params(ticker,'Gap Down Above Average',prop_data,tickers_data,all_props)
                    if body_top(minute_candles.iloc[0]) + yy_avg < body_bottom(bminute_candles.iloc[-1]):
                        prop_data, tickers_data, all_props = set_params(ticker,'Gap Down Above 2 Day Average',prop_data,tickers_data,all_props)
                    if body_bottom(minute_candles.iloc[0]) > body_top(bminute_candles.iloc[-1]):
                        prop_data, tickers_data, all_props = set_params(ticker,'Gap Up',prop_data,tickers_data,all_props)
                        if minute_candles.iloc[0]['range'] > body_bottom(minute_candles.iloc[0]) - body_top(bminute_candles.iloc[-1]):
                            prop_data, tickers_data, all_props = set_params(ticker,'Range More Than Gap Up',prop_data,tickers_data,all_props)
                        if body_bottom(minute_candles.iloc[0]) > bminute_candles['high'].max():
                            prop_data, tickers_data, all_props = set_params(ticker,'Gap Up Above Prev Max',prop_data,tickers_data,all_props)
                    if body_bottom(minute_candles.iloc[0]) > body_top(bminute_candles.iloc[-1]) + y_avg:
                        prop_data, tickers_data, all_props = set_params(ticker,'Gap Up Above Average',prop_data,tickers_data,all_props)
                    if body_bottom(minute_candles.iloc[0]) > body_top(bminute_candles.iloc[-1]) + yy_avg:
                        prop_data, tickers_data, all_props = set_params(ticker,'Gap Up Above 2 Day Average',prop_data,tickers_data,all_props)
                    if minute_candles.iloc[0]['open']>bminute_candles['high'].max():
                        prop_data, tickers_data, all_props = set_params(ticker,'Open Higher Than Prev Max',prop_data,tickers_data,all_props)
                    if minute_candles.iloc[0]['open']>bminute_candles['high'].max() + y_avg:
                        prop_data, tickers_data, all_props = set_params(ticker,'Open Higher Than Prev Max Plus Average',prop_data,tickers_data,all_props)
                    if minute_candles.iloc[0]['open']<bminute_candles['low'].min() :
                        prop_data, tickers_data, all_props = set_params(ticker,'Open Lower Than Prev Min',prop_data,tickers_data,all_props)
                    if minute_candles.iloc[0]['open']<bminute_candles['low'].min() - y_avg :
                        prop_data, tickers_data, all_props = set_params(ticker,'Open Lower Than Prev Min Minus Average',prop_data,tickers_data,all_props)
                    if minute_candles.iloc[0]['volume']>bminute_candles.iloc[-1]['volume']:
                        prop_data, tickers_data, all_props = set_params(ticker,'Volume Open Higher',prop_data,tickers_data,all_props)
                    if minute_candles.iloc[0]['volume']<bminute_candles.iloc[-1]['volume']:
                        prop_data, tickers_data, all_props = set_params(ticker,'Volume Open Lower',prop_data,tickers_data,all_props)
                    if minute_candles.iloc[0]['volume']>bminute_candles['volume'].mean()*1.5:
                        prop_data, tickers_data, all_props = set_params(ticker,'Volume Higher Than Average',prop_data,tickers_data,all_props)
                        if minute_candles.iloc[0]['volume']>bminute_candles['volume'].mean()*5:
                            prop_data, tickers_data, all_props = set_params(ticker,'Volume Above 5 Time Average',prop_data,tickers_data,all_props)
                        if minute_candles.iloc[0]['volume']>bbminute_candles['volume'].mean()*5:
                            prop_data, tickers_data, all_props = set_params(ticker,'Volume Above 5 Time Before Average',prop_data,tickers_data,all_props)
                            if 'Volume Above 5 Time Average' in tickers_data[ticker]:
                                prop_data, tickers_data, all_props = set_params(ticker,'Volume Consecutive Above 5 Time Average',prop_data,tickers_data,all_props)
                        if minute_candles.iloc[0]['volume']>bminute_candles['volume'].mean()*10:
                            prop_data, tickers_data, all_props = set_params(ticker,'Volume Above 10 Time Average',prop_data,tickers_data,all_props)
                        if minute_candles.iloc[0]['volume']>bbminute_candles['volume'].mean()*10:
                            prop_data, tickers_data, all_props = set_params(ticker,'Volume Above 10 Time Before Average',prop_data,tickers_data,all_props)
                            if 'Volume Above 10 Time Average' in tickers_data[ticker]:
                                prop_data, tickers_data, all_props = set_params(ticker,'Volume Consecutive Above 10 Time Average',prop_data,tickers_data,all_props)
                        if 'Volume Open Higher' in tickers_data[ticker]:
                            prop_data, tickers_data, all_props = set_params(ticker,'Volume Open Excedingly High',prop_data,tickers_data,all_props)
                        else:
                            prop_data, tickers_data, all_props = set_params(ticker,'Volume Open After High',prop_data,tickers_data,all_props)
                    if minute_candles.iloc[0]['volume']<bminute_candles['volume'].mean()*0.5:
                        prop_data, tickers_data, all_props = set_params(ticker,'Volume Lower Than Average',prop_data,tickers_data,all_props)
                        if 'Volume Open Lower' in tickers_data[ticker]:
                            prop_data, tickers_data, all_props = set_params(ticker,'Volume Open Excedingly Low',prop_data,tickers_data,all_props)
                        else:
                            prop_data, tickers_data, all_props = set_params(ticker,'Volume Open After Low',prop_data,tickers_data,all_props)
                if hammer_pattern(minute_candles.iloc[0]):
                    prop_data, tickers_data, all_props = set_params(ticker,'First Hammer',prop_data,tickers_data,all_props)
                if reverse_hammer_pattern(minute_candles.iloc[0]):
                    prop_data, tickers_data, all_props = set_params(ticker,'First Reverse Hammer',prop_data,tickers_data,all_props)
                if minute_candles.iloc[0]['range']<0.05:
                    prop_data, tickers_data, all_props = set_params(ticker,'Tiny Range',prop_data,tickers_data,all_props)
                if minute_candles.iloc[0]['range']>0.3 and green_candle(minute_candles.iloc[0]):
                    prop_data, tickers_data, all_props = set_params(ticker,'Huge Range',prop_data,tickers_data,all_props)
                if minute_candles.iloc[0]['range']>0.3 and red_candle(minute_candles.iloc[0]):
                    prop_data, tickers_data, all_props = set_params(ticker,'Huge Negative Range',prop_data,tickers_data,all_props)
                if len(minute_candles)>1:
                    if minute_candles.iloc[1]['range']<0.05:
                        prop_data, tickers_data, all_props = set_params(ticker,'Second Tiny Range',prop_data,tickers_data,all_props)
                        if 'Tiny Range' in tickers_data[ticker]:
                            prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Early Tiny Range',prop_data,tickers_data,all_props)
                    if minute_candles.iloc[1]['range']>0.3 and green_candle(minute_candles.iloc[1]):
                        prop_data, tickers_data, all_props = set_params(ticker,'Second Huge Range',prop_data,tickers_data,all_props)
                        if 'Huge Range' in tickers_data[ticker]:
                            prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Early Huge Range',prop_data,tickers_data,all_props)
                    if minute_candles.iloc[1]['range']>0.3 and red_candle(minute_candles.iloc[1]):
                        prop_data, tickers_data, all_props = set_params(ticker,'Second Huge Negative Range',prop_data,tickers_data,all_props)
                        if 'Huge Negative Range' in tickers_data[ticker]:
                            prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Early Huge Negative Range',prop_data,tickers_data,all_props)
                    if minute_candles.iloc[0]['range']<minute_candles.iloc[1]['range']:
                        prop_data, tickers_data, all_props = set_params(ticker,'Second Range Longer',prop_data,tickers_data,all_props)
                        if minute_candles.iloc[0]['range']*2<minute_candles.iloc[1]['range']:
                            prop_data, tickers_data, all_props = set_params(ticker,'Second Range Very Longer',prop_data,tickers_data,all_props)
                    else:
                        prop_data, tickers_data, all_props = set_params(ticker,'Second Range Shorter',prop_data,tickers_data,all_props)
                        if minute_candles.iloc[0]['range']>minute_candles.iloc[1]['range']*2:
                            prop_data, tickers_data, all_props = set_params(ticker,'Second Range Very Shorter',prop_data,tickers_data,all_props)
                    if minute_candles.iloc[0]['volume']>minute_candles.iloc[1]['volume']:
                            prop_data, tickers_data, all_props = set_params(ticker,'Second Volume Lower',prop_data,tickers_data,all_props)
                    else:
                            prop_data, tickers_data, all_props = set_params(ticker,'Second Volume Higher',prop_data,tickers_data,all_props)
                    if minute_candles.iloc[1]['high']-minute_candles.iloc[0]['high']<0.20:
                            prop_data, tickers_data, all_props = set_params(ticker,'Limp Second Diff',prop_data,tickers_data,all_props)
                    if hammer_pattern(minute_candles.iloc[1]):
                        prop_data, tickers_data, all_props = set_params(ticker,'Second Hammer',prop_data,tickers_data,all_props)
                    if reverse_hammer_pattern(minute_candles.iloc[1]):
                        prop_data, tickers_data, all_props = set_params(ticker,'Second Reverse Hammer',prop_data,tickers_data,all_props)
                    if body_bottom(minute_candles.iloc[1])-body_top(minute_candles.iloc[0])>0.1:
                        prop_data, tickers_data, all_props = set_params(ticker,'FVG First',prop_data,tickers_data,all_props)
                    if minute_candles.iloc[1]['low']-minute_candles.iloc[0]['high']>0.1:
                        prop_data, tickers_data, all_props = set_params(ticker,'Volume Gap First',prop_data,tickers_data,all_props)
                    if abs(minute_candles.iloc[1]['high'] - minute_candles.iloc[0]['high']) < 0.05:
                        prop_data, tickers_data, all_props = set_params(ticker,'Early Top Level',prop_data,tickers_data,all_props)
                    if body_bottom(minute_candles.iloc[0])-body_top(minute_candles.iloc[1])>0.1:
                        prop_data, tickers_data, all_props = set_params(ticker,'Negative FVG First',prop_data,tickers_data,all_props)
                    if minute_candles.iloc[0]['low']-minute_candles.iloc[1]['high']>0.1:
                        prop_data, tickers_data, all_props = set_params(ticker,'Negative Volume Gap First',prop_data,tickers_data,all_props)
                if len(minute_candles)>2:
                    if minute_candles.iloc[2]['range']<0.05:
                        prop_data, tickers_data, all_props = set_params(ticker,'Third Tiny Range',prop_data,tickers_data,all_props)
                        if 'Second Tiny Range' in tickers_data[ticker]:
                            prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Late Tiny Range',prop_data,tickers_data,all_props)
                        if 'Consecutive Early Tiny Range' in tickers_data[ticker] and 'Consecutive Late Tiny Range' in tickers_data[ticker]:
                            prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Tiny Range',prop_data,tickers_data,all_props)
                    if minute_candles.iloc[2]['range']>0.3 and green_candle(minute_candles.iloc[2]):
                        prop_data, tickers_data, all_props = set_params(ticker,'Third Huge Range',prop_data,tickers_data,all_props)
                        if 'Second Huge Range' in tickers_data[ticker]:
                            prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Late Huge Range',prop_data,tickers_data,all_props)
                        if 'Consecutive Early Huge Range' in tickers_data[ticker] and 'Consecutive Late Huge Range' in tickers_data[ticker]:
                            prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Huge Range',prop_data,tickers_data,all_props)
                    if minute_candles.iloc[2]['range']>0.3 and red_candle(minute_candles.iloc[2]):
                        prop_data, tickers_data, all_props = set_params(ticker,'Third Huge Negative Range',prop_data,tickers_data,all_props)
                        if 'Second Huge Negative Range' in tickers_data[ticker]:
                            prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Late Huge Negative Range',prop_data,tickers_data,all_props)
                        if 'Consecutive Early Huge Negative Range' in tickers_data[ticker] and 'Consecutive Late Huge Negative Range' in tickers_data[ticker]:
                            prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Huge Negative Range',prop_data,tickers_data,all_props)
                    if minute_candles.iloc[1]['range']<minute_candles.iloc[2]['range']:
                        prop_data, tickers_data, all_props = set_params(ticker,'Third Range Longer',prop_data,tickers_data,all_props)
                        if minute_candles.iloc[1]['range']*2<minute_candles.iloc[2]['range']:
                            prop_data, tickers_data, all_props = set_params(ticker,'Third Range Very Longer',prop_data,tickers_data,all_props)
                    else:
                        prop_data, tickers_data, all_props = set_params(ticker,'Third Range Shorter',prop_data,tickers_data,all_props)
                        if minute_candles.iloc[1]['range']>minute_candles.iloc[2]['range']*2:
                            prop_data, tickers_data, all_props = set_params(ticker,'Third Range Very Shorter',prop_data,tickers_data,all_props)
                    if minute_candles.iloc[1]['volume']>minute_candles.iloc[2]['volume']:
                            prop_data, tickers_data, all_props = set_params(ticker,'Third Volume Lower',prop_data,tickers_data,all_props)
                            if 'Second Volume Lower' in tickers_data[ticker]:
                                prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Lower Volume',prop_data,tickers_data,all_props)
                    else:
                            prop_data, tickers_data, all_props = set_params(ticker,'Third Volume Higher',prop_data,tickers_data,all_props)
                            if 'Second Volume Higher' in tickers_data[ticker]:
                                prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Higher Volume',prop_data,tickers_data,all_props)
                    if minute_candles.iloc[2]['high']-minute_candles.iloc[1]['high']<0.20:
                            prop_data, tickers_data, all_props = set_params(ticker,'Limp Third Diff',prop_data,tickers_data,all_props)
                            if 'Limp Second Diff' in tickers_data[ticker]:
                                prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Limp Diff',prop_data,tickers_data,all_props)
                    if hammer_pattern(minute_candles.iloc[2]):
                        prop_data, tickers_data, all_props = set_params(ticker,'Third Hammer',prop_data,tickers_data,all_props)
                    if reverse_hammer_pattern(minute_candles.iloc[2]):
                        prop_data, tickers_data, all_props = set_params(ticker,'Third Reverse Hammer',prop_data,tickers_data,all_props)
                    if abs(minute_candles.iloc[2]['high'] - minute_candles.iloc[1]['high']) < 0.05:
                        prop_data, tickers_data, all_props = set_params(ticker,'Late Top Level',prop_data,tickers_data,all_props)
                        if 'Early Top Level' in tickers_data[ticker]:
                            prop_data, tickers_data, all_props = set_params(ticker,'Top Level',prop_data,tickers_data,all_props)
                    if body_bottom(minute_candles.iloc[2])-body_top(minute_candles.iloc[1])>0.1:
                        prop_data, tickers_data, all_props = set_params(ticker,'FVG Second',prop_data,tickers_data,all_props)
                        if 'FVG First' in tickers_data[ticker]:
                            prop_data, tickers_data, all_props = set_params(ticker,'Consecutive FVG',prop_data,tickers_data,all_props)
                    if minute_candles.iloc[2]['low']- minute_candles.iloc[1]['high']>0.1:
                        prop_data, tickers_data, all_props = set_params(ticker,'Volume Gap Second',prop_data,tickers_data,all_props)
                        if 'Volume Gap First' in tickers_data[ticker]:
                            prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Volume Gap',prop_data,tickers_data,all_props)
                    if body_bottom(minute_candles.iloc[1])-body_top(minute_candles.iloc[2])>0.1:
                        prop_data, tickers_data, all_props = set_params(ticker,'Negative FVG Second',prop_data,tickers_data,all_props)
                        if 'Negative FVG First' in tickers_data[ticker]:
                            prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Negative FVG',prop_data,tickers_data,all_props)
                    if minute_candles.iloc[1]['low']- minute_candles.iloc[2]['high']>0.1:
                        prop_data, tickers_data, all_props = set_params(ticker,'Negative Volume Gap Second',prop_data,tickers_data,all_props)
                        if 'Negative Volume Gap First' in tickers_data[ticker]:
                            prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Negative Volume Gap',prop_data,tickers_data,all_props)
                    if 'Second Range Shorter' in tickers_data[ticker] and 'Third Range Shorter' in tickers_data[ticker]:
                        prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Shorter Range',prop_data,tickers_data,all_props)
                    if 'Second Range Longer' in tickers_data[ticker] and 'Third Range Longer' in tickers_data[ticker]:
                        prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Longer Range',prop_data,tickers_data,all_props)
                    if 'Second Range Very Shorter' in tickers_data[ticker] and 'Third Range Very Shorter' in tickers_data[ticker]:
                        prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Very Shorter Range',prop_data,tickers_data,all_props)
                    if 'Second Range Very Longer' in tickers_data[ticker] and 'Third Range Very Longer' in tickers_data[ticker]:
                        prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Very Longer Range',prop_data,tickers_data,all_props)
                if len(candles)>100:
                    levels[ticker] = find_levels(candles)
                else:
                    levels[ticker] = []
                latest_date[ticker] = minute_candles.iloc[-1]['date']
                if manualstocks:
                    print("Prop:",tickers_data[ticker])
                tickers_data = append_hash_set(tickers_data,ticker,'------------')
                maxmovement[ticker] = minute_candles['high'].max() - minute_candles['low'].min()
            if ticker in tickers_data:
                # for tm in prop_marks:
                #     if isinstance(tm['prop'],str):
                #         if tm['prop'] in tickers_data[ticker]:
                #             # print("Updating marks for ",tm['prop']," with ",tm['marks'])
                #             if ticker in ticker_marks:
                #                 ticker_marks[ticker] += tm['marks']
                #             else:
                #                 ticker_marks[ticker] = tm['marks']
                #             # print("Updated marks:",ticker_marks[ticker])
                #     else:
                #         rulecount = 0
                #         for pitem in tm['prop']:
                #             if pitem in tickers_data[ticker]:
                #                 rulecount += 1
                #         if rulecount==len(tm['prop']):
                #             # print("Updating marks for ",tm['prop']," with ",tm['marks'])
                #             if ticker in ticker_marks:
                #                 ticker_marks[ticker] += tm['marks']
                #             else:
                #                 ticker_marks[ticker] = tm['marks']
                #             # print("Updated marks:",ticker_marks[ticker])

                global_marks = pd.read_csv('analyze_global.csv')
                for i in range(len(global_marks)):
                    curprop = global_marks.iloc[i]
                    breakup = curprop['Prop'].split(':')
                    target = len(breakup)
                    curmark = 0
                    for p in breakup:
                        if p in tickers_data[ticker]:
                            curmark += 1
                    if curmark==target:
                        if ticker in ticker_marks:
                            ticker_marks[ticker] += curprop['Marks']
                        else:
                            ticker_marks[ticker] = curprop['Marks']

                # positive_marks = pd.read_csv('analyze_positive.csv')
                # for i in range(len(positive_marks)):
                #     curprop = positive_marks.iloc[i]
                #     breakup = curprop['Prop'].split(':')
                #     target = len(breakup)
                #     curmark = 0
                #     for p in breakup:
                #         if p in tickers_data[ticker]:
                #             curmark += 1
                #     if curmark==target:
                #         if ticker in ticker_marks:
                #             ticker_marks[ticker] += curprop['Scale']
                #         else:
                #             ticker_marks[ticker] = curprop['Scale']
                # negative_marks = pd.read_csv('analyze_Fail.csv')
                # for i in range(len(negative_marks)):
                #     curprop = negative_marks.iloc[i]
                #     breakup = curprop['Prop'].split(':')
                #     target = len(breakup)
                #     curmark = 0
                #     for p in breakup:
                #         if p in tickers_data[ticker]:
                #             curmark += 1
                #     if curmark==target:
                #         if ticker in ticker_marks:
                #             ticker_marks[ticker] -= curprop['Scale']
                #         else:
                #             ticker_marks[ticker] = 0 - curprop['Scale']
                        
                # if not ticker in ticker_marks:
                #     ticker_marks[ticker] = 0
                # if maxmovement[ticker] > 0.3:
                #     adjust = round(maxmovement[ticker],1) * 5
                #     # print("Max movement:",maxmovement[ticker]," Adjusting final mark with:",str(adjust))
                #     ticker_marks[ticker] += adjust
                # print("Final marks:",ticker_marks[ticker])

                with open(os.path.join(script_dir,'gapup_raw_data.csv'), 'a') as f:
                    curdiff = max_price[ticker][0] - first_price[ticker][0]
                    if curdiff < 0:
                        tcat = 'Fail'
                    elif curdiff < 1:
                        tcat = 'Fair'
                    elif curdiff < 5:
                        tcat = 'Good'
                    else:
                        tcat = 'Great'
                    if curdiff > 0.5:
                        profitable = 1
                    else:
                        profitable = 0
                    dlvl = str(round(curdiff,1))
                    fieldnames = ['ticker','date','day','diff','diff_level','performance','profitable','marks','yavg','yyavg','1range','1body','gap']
                    try:
                        gap = minute_candles.iloc[0]['open']-bminute_candles.iloc[-1]['close']
                    except:
                        gap = 0
                    row = {'ticker':ticker,'date':ldate,'day':datetime.strptime(ldate,'%Y-%m-%d').strftime('%A'),'diff':curdiff,'diff_level':dlvl,'performance':tcat,'profitable':profitable,'marks':ticker_marks[ticker],'yavg':y_avg,'yyavg':yy_avg,'1range':minute_candles.iloc[0]['range'],'1body':minute_candles.iloc[0]['body_length'],'gap':gap}
                    for pp in prop_list:
                        fieldnames.append(pp)
                        if pp in tickers_data[ticker]:
                            row[pp] = 1
                        else:
                            row[pp] = 0
                    writer = csv.DictWriter(f,fieldnames=fieldnames,extrasaction='ignore')
                    writer.writerow(row)

    print("End date:",end_date)
    tckr_diff = {}
    with_price = []
    for ctckr in ticker_marks.keys():
        tckr_diff[ctckr] = max_price[ctckr][0] - first_price[ctckr][0]
        with_price.append({'date':latest_date[ctckr].strftime("%d/%m %H:%M"),'ticker':ctckr,'marks':ticker_marks[ctckr],'open':first_price[ctckr][0],'price':latest_price[ctckr][0],'max':max_price[ctckr][0],'diff':tckr_diff[ctckr],'prop':"\n".join(tickers_data[ctckr]),'levels':"\n".join([ str(lvl['level']) + ' --- ' + str(lvl['count']) for lvl in levels[ctckr] ])})

    common_props = set(all_props)
    fail_common_props = set(all_props)
    prop_count = {}
    fail_prop_count = {}
    price_levels = {}
    diff_levels = {}
    outstanding = {}
    top_prop = {}
    for pinfo in with_price:
        if not math.isnan(pinfo['price']):
            plvl = str(round(pinfo['price'],0))
            if not plvl in price_levels:
                price_levels[plvl] = 1
            else:
                price_levels[plvl] += 1
        if not math.isnan(pinfo['diff']):
            dlvl = str(round(pinfo['diff'],1))
            if pinfo['diff']>1:
                outstanding[pinfo['ticker']] = dlvl
                for td in tickers_data[pinfo['ticker']]:
                    if not td in top_prop:
                        top_prop[td] = 1
                    else:
                        top_prop[td] += 1
            if not dlvl in diff_levels:
                diff_levels[dlvl] = 1
            else:
                diff_levels[dlvl] += 1
        if not math.isnan(pinfo['diff']) and pinfo['diff']>0.3:
            common_props = common_props.intersection(tickers_data[pinfo['ticker']])
            for td in tickers_data[pinfo['ticker']]:
                if not td in prop_count:
                    prop_count[td] = 1
                else:
                    prop_count[td] += 1
        else:
            fail_common_props = fail_common_props.intersection(tickers_data[pinfo['ticker']])
            for td in tickers_data[pinfo['ticker']]:
                if not td in fail_prop_count:
                    fail_prop_count[td] = 1
                else:
                    fail_prop_count[td] += 1

    print("Common props:",common_props)
    print(tabulate(dict(sorted(prop_count.items(),key=lambda item: item[1],reverse=True)).items(),headers=['Prop','Count'],tablefmt="github"))
    print("Fail Common props:",fail_common_props)
    print(tabulate(dict(sorted(fail_prop_count.items(),key=lambda item: item[1],reverse=True)).items(),headers=['Prop','Count'],tablefmt="github"))
    result=sorted(with_price,key=lambda x:x['diff'])
    if len(result)>0:
        print("Props of top ticker ",result[-1]['ticker'])
        print("\n".join(tickers_data[result[-1]['ticker']]))
    else:
        print("No results found")
    return with_price,end_of_trading

starttest = datetime.now()
with open(os.path.join(script_dir,'gapup_raw_data.csv'), 'w') as f:
    fieldnames = ['ticker','date','day','diff','diff_level','performance','profitable','marks','yavg','yyavg','1range','1body','gap']
    for pp in prop_list:
        fieldnames.append(pp)
    writer = csv.DictWriter(f,fieldnames=fieldnames,extrasaction='ignore')
    writer.writeheader()
# result=sorted(findgap(),key=lambda x:x['diff'])
result,endtrading = findgap()
result=sorted(result,key=lambda x:x['marks'])
result = pd.DataFrame.from_dict(result)
result.to_csv(os.path.join(script_dir,'results.csv'),index=False)
# loaded_model = load_model("model_autokeras", custom_objects=ak.CUSTOM_OBJECTS)
diff_model = load_model(os.path.join(script_dir,"model_diff_level"), custom_objects=ak.CUSTOM_OBJECTS)
profitable_model = load_model(os.path.join(script_dir,"model_profitable"), custom_objects=ak.CUSTOM_OBJECTS)
# [print('Fd:',i,i.shape, i.dtype) for i in loaded_model.inputs]
tocsv = pd.read_csv(os.path.join(script_dir,'gapup_raw_data.csv'))
profitablecsv = tocsv.copy()
diffcsv = tocsv.copy()


topop = ['ticker','date','day','Big Reverse','Bottom After Noon','Bottom Before Noon','Bottom Lunch','Peak After Noon','Peak Before Noon','Peak Lunch','diff','diff_level','performance','profitable']
for tp in topop:
    profitablecsv.pop(tp)
profitablefloat = np.asarray(profitablecsv).astype(np.float32)
tocsv['predicted_profitable'] = profitable_model.predict(profitablefloat)

topop = ['ticker','date','day','Big Reverse','Bottom After Noon','Bottom Before Noon','Bottom Lunch','Peak After Noon','Peak Before Noon','Peak Lunch','diff','profitable','performance','diff_level']
for tp in topop:
    diffcsv.pop(tp)
difffloat = np.asarray(diffcsv).astype(np.float32)
tocsv['predicted_diff'] = diff_model.predict(difffloat)

tocsv.sort_values(by=['predicted_profitable','predicted_diff'],ascending=False,inplace=True)
tocsv.to_csv(os.path.join(script_dir,'gapup_raw_data.csv'),index=False)
todisp = tocsv[['ticker','date','profitable','predicted_profitable','diff','diff_level','predicted_diff','performance']]
print(tabulate(todisp[:10],headers="keys",tablefmt="grid"))
toresult = tocsv.iloc[:10][['ticker','date','profitable','predicted_profitable','diff','diff_level','predicted_diff','performance','marks']]
toresult.to_csv(os.path.join(script_dir,'results_predicted.csv'))
print("End trading:",endtrading)
endtest = datetime.now()
print("Start:",starttest)
print("End:",endtest)
print("Time:",endtest-starttest)
