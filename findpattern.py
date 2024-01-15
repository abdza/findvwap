#!/usr/bin/env python

import pandas as pd 
import os
import sys
import getopt
from datetime import datetime,timedelta
from pandas.plotting import table
import yahooquery as yq
import numpy as np
import math
from tabulate import tabulate
from props import *

def double_bottom(peaks,bottoms):
    lastpeak = peaks[-1]
    lastbottom = bottoms[-1]
    secondlastbottom = bottoms[-2]
    secondlastpeak = peaks[-2]
    score = 0
    if secondlastbottom['date'] < secondlastpeak['date'] < lastbottom['date'] < lastpeak['date']:
        score += 1
        if lastpeak['close'] > secondlastpeak['high']:
            score += 1
        if abs(secondlastbottom['low'] - lastbottom['low']) < 0.1:
            score += 1
    return score,str(lastpeak['close'])

def higher_high(peaks,bottoms):
    lastpeak = peaks[-1]
    lastbottom = bottoms[-1]
    secondlastbottom = bottoms[-2]
    secondlastpeak = peaks[-2]
    thirdlastbottom = bottoms[-3]
    thirdlastpeak = peaks[-3]
    score = 0
    if thirdlastbottom['date'] < thirdlastpeak['date'] < secondlastbottom['date'] < secondlastpeak['date'] < lastbottom['date'] < lastpeak['date']:
        score += 1
        if thirdlastbottom['low'] < secondlastbottom['low'] and thirdlastpeak['high'] < secondlastpeak['high']:
            score += 1
        if secondlastbottom['low'] < lastbottom['low'] and secondlastpeak['high'] < lastpeak['high']:
            score += 1
    return score,str(lastpeak['close'])

def supernova(candles):
    candles['range'] = abs(candles['close'] - candles['open'])
    score = 0
    ranges = []
    for i in range(len(candles)):
        curcandle = candles.iloc[-i]
        if curcandle['range'] > candles['range'].mean() * 5:
            score += 1
            ranges.append(str(curcandle['range']))
    return score,','.join(ranges)

def volumesupernova(candles):
    score = 0
    ranges = []
    for i in range(3):
        curcandle = candles.iloc[-i]
        if curcandle['volume'] > candles['volume'].mean() * 10:
            score += 1
            ranges.append(str(curcandle['close']))
    return score,','.join(ranges)


def findpattern(stocks,end_date):
    days = 30
    start_date = end_date - timedelta(days=days)
    possible_double = []
    possible_up = []
    possible_nova = []
    possible_volumenova = []
    for i in range(len(stocks.index)):
        try:
            ticker = stocks.iloc[i]['Ticker'].upper()
            dticker = yq.Ticker(ticker)
            candles = dticker.history(start=start_date,end=end_date,interval='1d')
            candles = candles.reset_index(level=[0,1])
            peaks,bottoms = gather_range(candles)

            score = double_bottom(peaks,bottoms)
            if score>2:
                possible_double.append({'ticker':ticker,'score':score})
            possible_double = sorted(possible_double,key=lambda x:x['score'],reverse=True)

            score = higher_high(peaks,bottoms)
            if score>2:
                possible_up.append({'ticker':ticker,'score':score})
            possible_up = sorted(possible_up,key=lambda x:x['score'],reverse=True)

            score,ranges = supernova(candles)
            if score>0:
                possible_nova.append({'ticker':ticker,'score':score,'ranges':ranges})
            possible_nova = sorted(possible_nova,key=lambda x:x['score'],reverse=True)

            score = volumesupernova(candles)
            if score>0:
                possible_volumenova.append({'ticker':ticker,'score':score})
            possible_volumenova = sorted(possible_volumenova,key=lambda x:x['score'],reverse=True)
        except Exception as exp:
            print("Error downloading candles:",exp)


    print("Possible double bottom:",tabulate(possible_double,headers="keys"))
    print("Possible up:",tabulate(possible_up,headers="keys"))
    print("Possible Nova:",tabulate(possible_nova,headers="keys"))
    print("Possible Volume Nova:",tabulate(possible_volumenova,headers="keys"))
    fullresult = pd.DataFrame()
    result = pd.DataFrame.from_dict(possible_double)
    result['type'] = 'double'
    fullresult = pd.concat([fullresult,result])
    result = pd.DataFrame.from_dict(possible_up)
    result['type'] = 'up'
    fullresult = pd.concat([fullresult,result])
    result = pd.DataFrame.from_dict(possible_nova)
    result['type'] = 'nova'
    fullresult = pd.concat([fullresult,result])
    result = pd.DataFrame.from_dict(possible_volumenova)
    result['type'] = 'volumenova'
    fullresult = pd.concat([fullresult,result])
    fullresult.to_csv(os.path.join(script_dir,'pattern.csv'),index=False)

end_date = None
stockdate = None
manualstocks = None
inputfile = 'filtered.csv'
opts, args = getopt.getopt(sys.argv[1:],"i:d:s:",["input=","date=","stock="])
for opt, arg in opts:
    if opt in ("-i", "--input"):
        inputfile = arg
    if opt in ("-d", "--date"):
        stockdate = datetime.strptime(arg + ' 23:59:59', '%Y-%m-%d %H:%M:%S')
    if opt in ("-s", "--stocks"):
        manualstocks = arg.split(',')

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
if stockdate:
    end_date = stockdate
else:
    end_date = datetime.now()
if manualstocks:
    stocks = pd.DataFrame({'Ticker':manualstocks})
else:
    stocks = pd.read_csv(os.path.join(script_dir,inputfile),header=0)


findpattern(stocks,end_date)
