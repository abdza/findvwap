#!/usr/bin/env python

from logging import lastResort
import pandas as pd 
import os
import json
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
outfile = 'backtest.json'
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

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
if manualstocks:
    stocks = pd.DataFrame({'Ticker':manualstocks})
else:
    stocks = pd.read_csv(os.path.join(script_dir,inputfile),header=0)

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
'Early Top Level',
'Late Top Level',
'Top Level'
    ]

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
    latest_price = {}
    latest_date = {}
    first_price = {}
    max_price = {}
    levels = {}
    all_props = []
    end_of_trading = False
    ldate = None

    for i in range(len(stocks.index)):
    # for i in range(10):
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
            # print("Candles:",candles)
            curcandle = candles.iloc[-1]
            if isinstance(curcandle['date'],datetime):
                curkey = str(curcandle['date'].date())
            else:
                curkey = str(curcandle['date'])
            print("Curkey:",curkey," with ticker:",ticker)
            if ticker in latest_date and latest_date[ticker]==curkey:
                print("Latest date for ",ticker," is already ",curkey)
                continue
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
                # print("Last candle date:",minutelastcandle['date'].time())
                # print("End of trading:",end_of_trading)
                ldate = str(minutelastcandle['date'].date())
                fdate = str(datetime.date(minutelastcandle['date'])+timedelta(days=1))
                minute_candles = full_minute_candles.loc[(full_minute_candles['date']>ldate)]
                minute_candles = minute_candles.loc[(minute_candles['date']<fdate)]
                nowdate = str(datetime.now().date())
                latest_date[ticker] = minute_candles.iloc[-1]['date']
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

                # print("BMinute Candles:",bminute_candles)

                datediff += 1
                bbdate = str(datetime.date(minutelastcandle['date'])-timedelta(days=datediff))
                bbminute_candles = full_minute_candles.loc[(full_minute_candles['date']>bbdate)]
                bbminute_candles = bbminute_candles.loc[(bbminute_candles['date']<bdate)]
                while len(bbminute_candles)==0 and datediff<=5:
                    datediff += 1
                    bbdate = str(datetime.date(minutelastcandle['date'])-timedelta(days=datediff))
                    bbminute_candles = full_minute_candles.loc[(full_minute_candles['date']>bbdate)]
                    bbminute_candles = bbminute_candles.loc[(full_minute_candles['date']<bdate)]

                # print("BBMinute Candles:",bbminute_candles)
                peaks,bottoms = gather_range(minute_candles)
                # print("Peaks:",peaks)
                # print("Bottoms:",bottoms)

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
                    if maxp is not None:
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
                    if minute_candles.iloc[0]['volume']<bminute_candles['volume'].mean()*0.5:
                        prop_data, tickers_data, all_props = set_params(ticker,'Volume Lower Than Average',prop_data,tickers_data,all_props)
                if hammer_pattern(minute_candles.iloc[0]):
                    prop_data, tickers_data, all_props = set_params(ticker,'First Hammer',prop_data,tickers_data,all_props)
                if reverse_hammer_pattern(minute_candles.iloc[0]):
                    prop_data, tickers_data, all_props = set_params(ticker,'First Reverse Hammer',prop_data,tickers_data,all_props)
                if len(minute_candles)>1:
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
                if len(candles)>100:
                    levels[ticker] = find_levels(candles)
                else:
                    levels[ticker] = []
                if manualstocks:
                    print("Prop:",tickers_data[ticker])
                tickers_data = append_hash_set(tickers_data,ticker,'------------')
                with open('raw_data.csv', 'a') as f:
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
                    fieldnames = ['ticker','date','day','diff','diff_level','performance','profitable','winner']
                    row = {'ticker':ticker,'date':ldate,'day':datetime.strptime(ldate,'%Y-%m-%d').strftime('%A'),'diff':curdiff,'diff_level':dlvl,'performance':tcat,'profitable':profitable,'winner':0}
                    for pp in prop_list:
                        fieldnames.append(pp)
                        if pp in tickers_data[ticker]:
                            row[pp] = 1
                        else:
                            row[pp] = 0
                    writer = csv.DictWriter(f,fieldnames=fieldnames,extrasaction='ignore')
                    writer.writerow(row)

    print("End date:",end_date)
    test_props = []
    if ldate:
        with open('day_data.csv', 'a') as f:
            fieldnames = ['date','day']
            row = {'date':ldate,'day':datetime.strptime(ldate,'%Y-%m-%d').strftime('%A')}
            for pp in prop_list:
                fieldnames.append(pp)
                if pp in prop_data:
                    profitable = []
                    for cp in prop_data[pp]:
                        curdiff = max_price[cp][0] - first_price[cp][0]
                        if curdiff > 1:
                            profitable.append(cp)
                    row[pp] = profitable
                else:
                    row[pp] = 0
            writer = csv.DictWriter(f,fieldnames=fieldnames,extrasaction='ignore')
            writer.writerow(row)
        with open('day_count.csv', 'a') as f:
            fieldnames = ['date','day']
            row = {'date':ldate,'day':datetime.strptime(ldate,'%Y-%m-%d').strftime('%A')}
            for pp in prop_list:
                fieldnames.append(pp)
                if pp in prop_data:
                    profitable = []
                    for cp in prop_data[pp]:
                        curdiff = max_price[cp][0] - first_price[cp][0]
                        if curdiff > 1:
                            profitable.append(cp)
                    row[pp] = len(profitable)
                else:
                    row[pp] = 0
            writer = csv.DictWriter(f,fieldnames=fieldnames,extrasaction='ignore')
            writer.writerow(row)
    if len(test_props)>0: # and all(value in prop_data for value in test_props):
        firstprop = 0
        while not test_props[firstprop] in prop_data:
            firstprop += 1
        common_tckr = set(prop_data[test_props[firstprop]])
        for test in test_props:
            if test!=test_props[firstprop] and test in prop_data:
                common_tckr = common_tckr.intersection(prop_data[test])
    else:
        common_tckr = tickers
    negate_test_props = []
    toremove = []
    negated_props = {}
    if len(negate_test_props)>0: # and all(value in prop_data for value in test_props):
        for ctckr in common_tckr:
            if ctckr in tickers_data:
                negated_props[ctckr] = []
                for test in negate_test_props:
                    if test in tickers_data[ctckr]:
                        if end_of_trading:
                            negated_props = append_hash_set(negated_props,ctckr,test)
                        else:
                            negated_props[ctckr] = negate_test_props
                        toremove.append(ctckr)
            else:
                negated_props[ctckr] = negate_test_props
    else:
        for ctckr in common_tckr:
            negated_props[ctckr] = negate_test_props
    if not end_of_trading and len(test_props)>0:
        for tr in toremove:
            if tr in common_tckr:
                common_tckr.remove(tr)

    tckr_diff = {}
    with_price = []
    for ctckr in common_tckr:
        if ctckr in tickers_data:
            tckr_diff[ctckr] = max_price[ctckr][0] - first_price[ctckr][0]
            with_price.append({'date':latest_date[ctckr],'ticker':ctckr,'open':first_price[ctckr][0],'price':latest_price[ctckr][0],'max':max_price[ctckr][0],'diff':tckr_diff[ctckr],'prop':"\n".join(tickers_data[ctckr]), 'negate':"\n".join(negated_props[ctckr]),'levels':"\n".join([ str(lvl['level']) + ' --- ' + str(lvl['count']) for lvl in levels[ctckr] ])})
                    
    # with_price = [ {'date':latest_date[tckr],'ticker':tckr,'open':first_price[tckr][0],'price':latest_price[tckr][0],'max':max_price[tckr][0],'diff':tckr_diff[tckr],'prop':"\n".join(tickers_data[tckr]), 'negate':"\n".join(negated_props[tckr]),'levels':"\n".join([ str(lvl['level']) + ' --- ' + str(lvl['count']) for lvl in levels[tckr] ])} for tckr in common_tckr ]

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
    return with_price,end_of_trading,prop_count,fail_prop_count,price_levels,diff_levels,outstanding,top_prop

starttest = datetime.now()
data = []
with open('raw_data.csv', 'w') as f:
    fieldnames = ['ticker','date','day','diff','diff_level','performance','profitable','winner']
    for pp in prop_list:
        fieldnames.append(pp)
    writer = csv.DictWriter(f,fieldnames=fieldnames,extrasaction='ignore')
    writer.writeheader()
with open('day_data.csv', 'w') as f:
    fieldnames = ['date','day']
    for pp in prop_list:
        fieldnames.append(pp)
    writer = csv.DictWriter(f,fieldnames=fieldnames,extrasaction='ignore')
    writer.writeheader()
with open('day_count.csv', 'w') as f:
    fieldnames = ['date','day']
    for pp in prop_list:
        fieldnames.append(pp)
    writer = csv.DictWriter(f,fieldnames=fieldnames,extrasaction='ignore')
    writer.writeheader()
for day in range(60):
# for day in range(5):
    instockdate = starttest - timedelta(days=day)
    print("Processing ",instockdate)
    tmpkey = str(instockdate.date())
    print("Tmpkey :",tmpkey)
# result=sorted(findgap(),key=lambda x:x['diff'])
    result,endtrading,prop_count,fail_prop_count,price_levels,diff_levels,outstanding,top_prop = findgap()
    if len(result)>1:
        if endtrading:
            result=sorted(result,key=lambda x:x['diff'])
        else:
            result=sorted(result,key=lambda x:x['price'])

        data.append({
            'date':str(instockdate.date()),
            'top_ticker':result[-1]['ticker'],
            'top_price':result[-1]['price'],
            'top_diff':result[-1]['diff'],
            'outstanding':outstanding,
            'top_prop':top_prop,
            'prop':prop_count,
            'fail_prop':fail_prop_count,
            # 'price_levels':price_levels,
            'diff_levels':diff_levels
        })
    # print(tabulate(result,headers="keys",tablefmt="grid"))
with open(outfile, "w") as write_file:
    json.dump(data, write_file)
endtest = datetime.now()
print("Start:",starttest)
print("End:",endtest)
print("Time:",endtest-starttest)
alldata = pd.read_csv('raw_data.csv')
winners = alldata.groupby('date')['diff'].max().reset_index()
# for win in winners:
#     print("Win:",win)
#     alldata.loc[alldata['diff']==win,'winner'] = 1
# winners = winners.reindex(alldata['date'])
# winners = alldata['diff'].max()
# alldata_with_winners = alldata.merge(winners.to_frame('daywinner'), on='date')
# alldata['gotwin'] = (alldata['diff']==winners)
winners.to_csv('winners.csv',index=False)
