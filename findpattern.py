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

def double_bottom(candles,peaks,bottoms):
    lastpeak = peaks[-1]
    lastbottom = bottoms[-1]
    secondlastbottom = bottoms[-2]
    secondlastpeak = peaks[-2]
    score = 0
    if secondlastbottom['date'] < secondlastpeak['date'] < lastbottom['date'] < lastpeak['date']:
        print("Sequence is correct")
        score += 1
        if lastpeak['close'] > secondlastpeak['high']:
            print("Last peak confirmed double bottom")
            score += 1
            if abs(secondlastbottom['low'] - lastbottom['low']) < 0.1:
                print("Bottom is close enough")
                score += 1
    return score

def higher_high(candles,peaks,bottoms):
    lastpeak = peaks[-1]
    lastbottom = bottoms[-1]
    secondlastbottom = bottoms[-2]
    secondlastpeak = peaks[-2]
    thirdlastbottom = bottoms[-3]
    thirdlastpeak = peaks[-3]
    score = 0
    if thirdlastbottom['date'] < thirdlastpeak['date'] < secondlastbottom['date'] < secondlastpeak['date'] < lastbottom['date'] < lastpeak['date']:
        print("Sequence is correct")
        score += 1
        if thirdlastbottom['low'] < secondlastbottom['low'] and thirdlastpeak['high'] < secondlastpeak['high']:
            print("Higher low")
            score += 1
        if secondlastbottom['low'] < lastbottom['low'] and secondlastpeak['high'] < lastpeak['high']:
            print("Confirm Higher low")
            score += 1
    return score

def findpattern(stocks,end_date):
    days = 30
    start_date = end_date - timedelta(days=days)
    possible_double = []
    possible_up = []
    for i in range(len(stocks.index)):
        try:
            ticker = stocks.iloc[i]['Ticker'].upper()
            dticker = yq.Ticker(ticker)
            candles = dticker.history(start=start_date,end=end_date,interval='1d')
            candles = candles.reset_index(level=[0,1])
            peaks,bottoms = gather_range(candles)
            print("Testing:",ticker)
            print("Peaks")
            print(tabulate(peaks,headers="keys"))
            print("Bottoms")
            print(tabulate(bottoms,headers="keys"))

            score = double_bottom(candles,peaks,bottoms)
            if score>0:
                possible_double.append({'ticker':ticker,'score':score})
            possible_double = sorted(possible_double,key=lambda x:x['score'],reverse=True)

            score = higher_high(candles,peaks,bottoms)
            if score>0:
                possible_up.append({'ticker':ticker,'score':score})
            possible_up = sorted(possible_up,key=lambda x:x['score'],reverse=True)
        except Exception as exp:
            print("Error downloading candles:",exp)


    print("Possible double bottom:",tabulate(possible_double,headers="keys"))
    print("Possible up:",tabulate(possible_up,headers="keys"))

end_date = None
stockdate = None
manualstocks = None
inputfile = 'stocks.csv'
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
