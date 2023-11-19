#!/usr/bin/env python

import pandas as pd 
import os
import sys
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
    kmeans = KMeans(n_clusters=kint,n_init=10).fit(highlevels.reshape(-1,1))
    highclusters = kmeans.predict(highlevels.reshape(-1,1))

    resistancelevels = {}

    for cidx in range(datalen):
        curcluster = highclusters[cidx]
        if curcluster not in resistancelevels:
            resistancelevels[curcluster] = 1
        else:
            resistancelevels[curcluster] += 1

    donecluster = []
    finalreslevels = {}
    dresponse = ""
    for cidx in range(datalen):
        candle = candles.iloc[cidx]
        curcluster = highclusters[cidx]
        if resistancelevels[curcluster] > 2:
            if curcluster not in donecluster:
                donecluster.append(curcluster)
                finalreslevels[curcluster] = {'level':candle['high'],'count':1}
            else:
                finalreslevels[curcluster] = {'level':(finalreslevels[curcluster]['level'] + candle['high'])/2,'count':finalreslevels[curcluster]['count']+1}

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
    kmeans = KMeans(n_clusters=kint,n_init=10).fit(lowlevels.reshape(-1,1))
    lowclusters = kmeans.predict(lowlevels.reshape(-1,1))

    supportlevels = {}

    for cidx in range(datalen):
        curcluster = lowclusters[cidx]
        if curcluster not in supportlevels:
            supportlevels[curcluster] = 1
        else:
            supportlevels[curcluster] += 1

    donecluster = []
    finalsuplevels = {}
    dresponse = ""
    for cidx in range(datalen):
        candle = candles.iloc[cidx]
        curcluster = lowclusters[cidx]
        if supportlevels[curcluster] > 2:
            if curcluster not in donecluster:
                donecluster.append(curcluster)
                finalsuplevels[curcluster] = {'level':candle['low'],'count':1}
            else:
                finalsuplevels[curcluster] = {'level':(finalsuplevels[curcluster]['level'] + candle['low'])/2,'count':finalsuplevels[curcluster]['count']+1}

    response += "\n\nSupport levels:"
    sortedsuplevels = []
    for lvl,clstr in sorted(finalsuplevels.items(),key=lambda x: x[1]['level']):
        sortedsuplevels.append(clstr)
        response += "\n" + str(clstr['level']) + " : " + str(clstr['count'])
    
    response += "\n\n" + dresponse
    return sortedreslevels, sortedsuplevels

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

def min_bottom(bottoms):
    curbottom = bottoms[0]
    for candle in bottoms:
        if candle['low'] < curbottom['low']:
            curbottom = candle
    return curbottom

def max_peak(peaks):
    curpeak = peaks[0]
    for candle in peaks:
        if candle['high'] > curpeak['high']:
            curpeak = candle
    return curpeak

inputfile = 'stocks.csv'
outfile = 'shorts.csv'
instockdate = None
openrangelimit = 1
purchaselimit = 300
completelist = False
trackunit = None
perctarget = 10
opts, args = getopt.getopt(sys.argv[1:],"i:o:d:r:p:c:u:x:",["input=","out=","date=","range=","purchaselimit=","complete=","unit=","perctarget="])
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

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
stocks = pd.read_csv(os.path.join(script_dir,inputfile),header=0)

def body_length(candle):
    return body_top(candle) - body_bottom(candle)

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
    return body_length(candle) < candle['range']/2 and body_bottom(candle) > candle['low'] + candle['range']/2

def reverse_hammer_pattern(candle):
    return body_length(candle) < candle['range']/2 and body_top(candle) < candle['low'] + candle['range']/1.5

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

def backtest():
    end_date = datetime.now()
    print("Got end date:",end_date)
    days = 100 
    start_date = end_date - timedelta(days=days)
    highrange = {}
    profitable = {}
    fullrange = {}
    stats = {}

    for i in range(len(stocks.index)):
    # for i in range(10):
        if isinstance(stocks.iloc[i]['Ticker'], str):
            ticker = stocks.iloc[i]['Ticker'].upper()
            dticker = yq.Ticker(ticker)
            candles = dticker.history(start=start_date,end=end_date,interval='1d')
            print("Processing ",ticker, " got ",len(candles))
            candles = candles.loc[(candles['volume']>0)]
        else:
            continue

        if len(candles.index):
            candles = candles.reset_index(level=[0,1])
            candles['range'] = candles['high'] - candles['low']
            for i in range(len(candles)):
                curcandle = candles.iloc[i]
                curkey = str(curcandle['date'])
                minute_end_date = datetime.strptime(curkey + ' 23:59:59', '%Y-%m-%d %H:%M:%S')
                minute_start_date = minute_end_date - timedelta(days=5)
                # print("Minute start:",minute_start_date, " Minute end:",minute_end_date)
                full_minute_candles = dticker.history(start=minute_start_date,end=minute_end_date,interval='15m')
                full_minute_candles['range'] = full_minute_candles['high'] - full_minute_candles['low']
                peaks = []
                bottoms = []
                minuteprofittickers = []
                avgtest = []
                passminute = 0
                gotprofit = 0 
                minuteprofit = 0
                if len(full_minute_candles)>0:
                    full_minute_candles = full_minute_candles.reset_index(level=[0,1])
                    # print("full minute candles:",full_minute_candles)
                    minutelastcandle = full_minute_candles.iloc[-2]
                    ldate = str(datetime.date(minutelastcandle['date'])-timedelta(days=1))
                    # print("ldate:",ldate)
                    minute_candles = full_minute_candles.loc[(full_minute_candles['date']>ldate)]
                    # print("minute candles:",minute_candles)

                    datediff = 1
                    bdate = str(datetime.date(minutelastcandle['date'])-timedelta(days=datediff))
                    bminute_candles = full_minute_candles.loc[(full_minute_candles['date']>bdate)]
                    bminute_candles = bminute_candles.loc[(full_minute_candles['date']<ldate)]
                    while len(bminute_candles)==0 and datediff<=5:
                        datediff += 1
                        bdate = str(datetime.date(minutelastcandle['date'])-timedelta(days=datediff))
                        bminute_candles = full_minute_candles.loc[(full_minute_candles['date']>bdate)]
                        bminute_candles = bminute_candles.loc[(full_minute_candles['date']<ldate)]

                    datediff += 1
                    bbdate = str(datetime.date(minutelastcandle['date'])-timedelta(days=datediff))
                    bbminute_candles = full_minute_candles.loc[(full_minute_candles['date']>bbdate)]
                    bbminute_candles = bbminute_candles.loc[(full_minute_candles['date']<bdate)]
                    while len(bbminute_candles)==0 and datediff<=5:
                        datediff += 1
                        bbdate = str(datetime.date(minutelastcandle['date'])-timedelta(days=datediff))
                        bbminute_candles = full_minute_candles.loc[(full_minute_candles['date']>bbdate)]
                        bbminute_candles = bbminute_candles.loc[(full_minute_candles['date']<bdate)]
                    peaks,bottoms = gather_range(minute_candles)
                    # print("Peaks:",peaks)
                    if minute_test(peaks,bottoms):
                        if curkey in highrange:
                            if curcandle['range']>highrange[curkey]['range']:
                                    highrange[curkey] = {'range':curcandle['range'],'ticker':ticker}
                                    profitable[curkey] = {'range':maxp['high'] - minute_candles.iloc[2]['high'],'ticker':ticker}
                        else:
                            highrange[curkey] = {'range':curcandle['range'],'ticker':ticker}
                            profitable[curkey] = {'range':maxp['high'] - minute_candles.iloc[2]['high'],'ticker':ticker}
                    if len(peaks):
                        maxp = max_peak(peaks)
                        profitamt = 0
                        if minute_test(peaks,bottoms) and green_candle(minute_candles.iloc[0]):
                            profitamt = maxp['high'] - minute_candles.iloc[2]['high']
                        if profitamt>0:
                            if curkey in profitable:
                                if profitamt > profitable[curkey]['range']:
                                    profitable[curkey] = {'range':profitamt,'ticker':ticker}
                            else:
                                profitable[curkey] = {'range':profitamt,'ticker':ticker}
                    if curkey in fullrange:
                        if curcandle['range']>fullrange[curkey]['range']:
                            fullrange[curkey] = {'range':curcandle['range'],'ticker':ticker}
                    else:
                        fullrange[curkey] = {'range':curcandle['range'],'ticker':ticker}
                    if curkey in stats:
                        passminute = stats[curkey]['passminute']
                        gotprofit = stats[curkey]['gotprofit']
                        minuteprofit = stats[curkey]['minuteprofit']
                        minuteprofittickers = stats[curkey]['minuteprofittickers']
                        avgtest = stats[curkey]['avgtest']
                    mtest = minute_test(peaks,bottoms)
                    gotprofit = profit_test(peaks,bottoms)
                    minute_profit = minute_profit_test(peaks,bottoms)
                    pattern = pattern_test(minute_candles)
                    prevtest = prev_avg_test(minute_candles,bminute_candles,2.5)
                    if mtest:
                        passminute += 1
                    if gotprofit:
                        gotprofit += 1
                    if minute_profit:
                        minuteprofit += 1
                        minuteprofittickers.append(ticker)
                    if mtest and pattern and prevtest:
                        avgtest.append(ticker)
                stats[curkey] = {'passminute':passminute,'gotprofit':gotprofit,'minuteprofit':minuteprofit,'minuteprofittickers':minuteprofittickers,'avgtest':avgtest}

    print("High Openers:")
    return highrange,fullrange,stats,profitable

starttest = datetime.now()

if 'cal' not in st.session_state:
    print("No session. Setting it")
    caldata = []
    highrange,fullrange,stats,profitable = backtest()
    for kdate,kval in profitable.items():
        prepend = '$'
        caldata.append({'title':prepend + kval['ticker'] + ' ' + str(kval['range']),'start':kdate,'end':kdate})

    for kdate,kval in highrange.items():
        prepend = '^'
        if kdate in fullrange and fullrange[kdate]['ticker']==kval['ticker']:
            prepend += '['
        caldata.append({'title':prepend + kval['ticker'] + ' ' + str(kval['range']),'start':kdate,'end':kdate})

    for kdate,kval in fullrange.items():
        if kdate in highrange and highrange[kdate]['ticker']==kval['ticker']:
            pass
        else:
            caldata.append({'title':'[' + kval['ticker'] + ' ' + str(kval['range']),'start':kdate,'end':kdate})
    for kdate,kval in stats.items():
        caldata.append({'title':'Minute:' + str(kval['passminute']) + '; Profit:' + str(kval['gotprofit']) + '; MinuteProfit:' + str(kval['minuteprofit']),'start':kdate,'end':kdate})
        caldata.append({'title':'MinuteProfit Tickers:' + str(','.join(kval['minuteprofittickers'])),'start':kdate,'end':kdate})
        caldata.append({'title':'Test Tickers:' + str(','.join(kval['avgtest'])),'start':kdate,'end':kdate})
    st.session_state['cal'] = caldata
else:
    print("No session need to set")


calendar_options = {
    "initialView":"listWeek"
}
calendar = calendar(events=st.session_state['cal'],options=calendar_options)
st.write(calendar)
endtest = datetime.now()
print("Start:",starttest)
print("End:",endtest)
print("Time:",endtest-starttest)
