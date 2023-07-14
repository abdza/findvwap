#!/usr/bin/env python

import pandas as pd 
import os
import math
from datetime import datetime,timedelta
import yahooquery as yq
from ta.volume import VolumeWeightedAveragePrice
import time
import numpy as np
from numerize import numerize
from sklearn.cluster import KMeans

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

def clean_bull(first,second):
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

def clean_bear(first,second):
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
    bounce = 0
    pullback = 0
    for i in range(-1,(len(candles.index) + 1)*-1,-1):
        finalcandle = candles.iloc[i]
        if finalcandle['volume']>0:
            if bounce==0:
                if green_candle(finalcandle):
                    bounce += 1
                else:
                    break
            elif bounce>=1:
                if green_candle(finalcandle):
                    nextcandle = candles.iloc[i+1]
                    if clean_bull(finalcandle,nextcandle)>2:
                        bounce += 1
                    else:
                        break
                elif pullback==0:
                    if red_candle(finalcandle):
                        pullback += bounce
                    else:
                        break
                else:
                    if i-1<len(candles.index)*-1:
                        i=(len(candles.index)*-1) + 1
                    prevcandle = candles.iloc[i-1]
                    if clean_bear(prevcandle,finalcandle)>1:
                        pullback += bounce
                    else:
                        break
    return pullback

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

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
stocks = pd.read_csv(os.path.join(script_dir,'stocks.csv'),header=0)

end_date = datetime.now()
days = 200
start_date = end_date - timedelta(days=days)

for i in range(int(len(stocks.index))-1):
    if isinstance(stocks.iloc[i]['Ticker'], str):
        ticker = stocks.iloc[i]['Ticker'].upper()
        dticker = yq.Ticker(ticker)
        candles = dticker.history(start=start_date,end=end_date,interval='1d')
    else:
        continue

    if len(candles.index):
        candles = candles.reset_index(level=[0,1])
            
        pullback = find_bounce(candles)
        if pullback>2:
            print("Ticker ===",ticker,"=== just bounced:",pullback)
            endloop = (pullback + 1) * -1
            curcandle = candles.iloc[-1]

            resistance,support = find_levels(candles)
            print("Resistance:")
            for res in resistance:
                print("Level:",res['level']," Count:",res['count'])
            print("Support:")
            for res in support:
                print("Level:",res['level']," Count:",res['count'])
            print("\n\n")
