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

def find_levels(candles):
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
    if c_pos>0 and c_pos<len(candles)-dlen:
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
    else:
        return False

def is_bottom(candles,c_pos,dlen=1):
    if c_pos>0 and c_pos<len(candles)-dlen:
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

outputfile = open(outfile,"w")
outdata = 'Ticker\tOpen\t\t\tClose\t\t\tPercent\t\t\tFirst\n'
outputfile.writelines(outdata)
outputfile.close()

def analyzedate(stockdateparam = None):
    end_date = datetime.now()
    if stockdateparam:
        if isinstance(stockdateparam,datetime):
            end_date = stockdateparam
        else:
            end_date = datetime.strptime(stockdateparam + ' 23:59:59', '%Y-%m-%d %H:%M:%S')
    print("Got end date:",end_date)
    days = 4
    start_date = end_date - timedelta(days=days)
    day_start_date = end_date - timedelta(days=days*10)
    hour_start_date = end_date - timedelta(days=days*4)

    filtered = []
    highrangetickers = []
    dayrangetickers = []
    hourrangetickers = []
    hourdetails = []
    openrangetickers = []
    highopeners = []

    for i in range(len(stocks.index)):
    # for i in range(100):
        if isinstance(stocks.iloc[i]['Ticker'], str):
            ticker = stocks.iloc[i]['Ticker'].upper()
            print("Processing ",ticker)
            dticker = yq.Ticker(ticker)
            candles = dticker.history(start=start_date,end=end_date,interval='15m')
            # day_candles = dticker.history(start=day_start_date,end=end_date,interval='1d')
            # hour_candles = dticker.history(start=hour_start_date,end=end_date,interval='1h')
            candles = candles.loc[(candles['volume']>0)]
        else:
            continue

        if len(candles.index):
            print("Processing ticker ",ticker)
            candles = candles.reset_index(level=[0,1])
            # day_candles = day_candles.reset_index(level=[0,1])
            # hour_candles = hour_candles.reset_index(level=[0,1])
            lastcandle = candles.iloc[-1]
            ldate = str(lastcandle['date'].date())
            bdate = str(datetime.date(lastcandle['date'])-timedelta(days=1))
            adate = str(datetime.date(lastcandle['date'])+timedelta(days=1))
            cdcandle = candles.loc[(candles['date']>ldate)]
            prevcdcandle = candles.loc[(candles['date']>bdate)]
            if cdcandle.iloc[-1]['date'].day != end_date.day:
                print("Got no data on last day")
                continue
            if trackunit:
                peaks,bottoms = gather_range_unit(trackunit,cdcandle)
                # hour_peaks,hour_bottoms = gather_range_unit(trackunit,hour_candles)
                # day_peaks,day_bottoms = gather_range_unit(trackunit,day_candles)
            else:
                peaks,bottoms = gather_range(cdcandle)
                # hour_peaks,hour_bottoms = gather_range_body(hour_candles)
                # day_peaks,day_bottoms = gather_range_body(day_candles)

            startat930 = cdcandle.iloc[0]['date'].hour==9 and cdcandle.iloc[0]['date'].minute==30
            secondat945 = False
            if len(cdcandle)>2:
                secondat945 = cdcandle.iloc[1]['date'].hour==9 and cdcandle.iloc[1]['date'].minute==45

            if not (startat930 and secondat945):
                print("Late first or second ticks ",startat930,",",secondat945)
                continue

            print("Got data on 9.30 and 9.45")

            if green_candle(cdcandle.iloc[0]):
                oprange = cdcandle.iloc[0]['close'] - cdcandle.iloc[0]['open']
                rangeperc = (oprange / cdcandle.iloc[0]['open']) * 100
                goodsec = False
                secabove = False
                volincrease = False
                volincrease2 = False
                dayvol = False
                if len(cdcandle)>2:
                    secrange = cdcandle.iloc[1]['close'] - cdcandle.iloc[1]['open']
                    goodsec = secrange > 0 or abs(secrange)<oprange/3
                    secabove = cdcandle.iloc[1]['open'] > cdcandle.iloc[0]['open']
                    volincrease = green_candle(cdcandle.iloc[1]) and cdcandle.iloc[1]['volume'] > (cdcandle.iloc[0]['volume']/2)
                    volincrease2 = red_candle(cdcandle.iloc[1]) and cdcandle.iloc[1]['volume'] < (cdcandle.iloc[0]['volume']/3)
                    dayvol = cdcandle.iloc[0]['volume'] > prevcdcandle.iloc[-1]['volume'] * 2
                if rangeperc > (perctarget/2) and goodsec and secabove and (volincrease or volincrease2) and dayvol:
                    outputfile = open(outfile,"a")
                    outdatatbl = {'Ticker':ticker,'Open':cdcandle.iloc[0]['open'],'Close':cdcandle.iloc[0]['close'],'Percent':rangeperc,'First':cdcandle.iloc[0]['date']}
                    outdata = ticker + '\t' + str(cdcandle.iloc[0]['open']) + '\t' + str(cdcandle.iloc[0]['close']) + '\t' + str(rangeperc) + '\t' + str(cdcandle.iloc[0]['date']) + '\n'
                    highopeners.append(outdatatbl)
                    outputfile.writelines(outdata)
                    outputfile.close()

    print("High Openers:")
    print(tabulate(highopeners,headers="keys"))
    return highopeners


testdates = ['2023-10-27','2023-10-26','2023-10-25','2023-10-24','2023-10-23','2023-10-22','2023-10-20','2023-10-19','2023-10-18','2023-10-17','2023-10-16','2023-10-13','2023-10-12','2023-10-11','2023-10-10','2023-10-09']

if len(testdates):
    starttest = datetime.now()
    totals = []
    for tdate in testdates:
        totals += analyzedate(tdate)
    print("Totals:")
    print(tabulate(totals,headers="keys"))
    endtest = datetime.now()
    print("Start:",starttest)
    print("End:",endtest)
    print("Time:",endtest-starttest)
else:
    if instockdate:
        print("Got instockdate:",instockdate)
        analyzedate(instockdate)
    else:
        analyzedate(datetime.now())
