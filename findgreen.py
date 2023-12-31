#!/usr/bin/env python

import pandas as pd 
import os
import sys
import getopt
import math
from datetime import datetime,timedelta
import yahooquery as yq
from ta.volume import VolumeWeightedAveragePrice

high_limit = 4  # how many times the high volume has to be from the average 

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

def candle_bull(first,second):
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

def candle_bear(first,second):
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

def tribull(candles):
    limit = 3 
    totalbull = 0
    limit = min(3,len(candles.index)-1)
    for i in range(0,limit):
        curcandle = candles.iloc[i]
        if green_candle(curcandle):
            totalbull += clean_bull(candles.iloc[i+1],curcandle)
    if totalbull>10:
        return True
    else:
        return False

def tripattern(candles):
    candles['size'] = abs(candles['close'] - candles['open'])
    avg_size = candles['size'].mean()
    curcandle = candles.iloc[0]
    found = False
    if green_candle(curcandle) and curcandle['size'] > avg_size * 2:
        i = 1 
        gotrest = False
        while i<5 and not found:
            thiscandle = candles.iloc[i]
            if green_candle(thiscandle) and thiscandle['high'] < curcandle['open'] and thiscandle['size'] > avg_size * 2:
                found = True
            if not found and not gotrest and thiscandle['size'] < avg_size / 2 and thiscandle['high'] < curcandle['open']:
                gotrest = True
            i += 1
    return found

def bullrun(candles):
    runscore = 0 
    bearscore = 0
    limit = 0
    for i in range(0,len(candles.index)-1):
        if candle_bull(candles.iloc[i+1],candles.iloc[i])>limit:
            runscore += 1
            if limit<2:
                limit += 1
        else:
            bearscore += 1
            if limit>0:
                limit -= 1
    return runscore,bearscore

def find_bounce(candles):
    bounce = 0
    pullback = 0
    for i in range(0,len(candles.index)-1):
        finalcandle = candles.iloc[i]
        if finalcandle['volume']>0:
            if bounce==0:
                if green_candle(finalcandle):
                    bounce += 1
                else:
                    break
            elif bounce>=1:
                if green_candle(finalcandle):
                    prevcandle = candles.iloc[i+1]
                    if clean_bull(prevcandle,finalcandle)>2:
                        bounce += 1
                    else:
                        break
                elif pullback==0:
                    if red_candle(finalcandle):
                        pullback += bounce
                    else:
                        break
                else:
                    prevcandle = candles.iloc[i+1]
                    if clean_bear(prevcandle,finalcandle)>1:
                        pullback += bounce
                    else:
                        break
    return pullback

def count_above_vwap(candles):
    above = 0
    for i in range(0,len(candles.index)-1):
        finalcandle = candles.iloc[i]
        prevfinalcandle = candles.iloc[i+1]
        if finalcandle['low'] > finalcandle['vwap'] and prevfinalcandle['low']<finalcandle['low'] and prevfinalcandle['low'] > prevfinalcandle['vwap']:
            above += 1
        else:
            break
    return above

def green_high(candles):
    gothigh = None
    gotdiff = None
    highest = 5 
    if len(candles.index) + 1 < highest:
        highest = len(candles.index) - 1
    for i in range(0,highest):
        curcandle = candles.iloc[i]
        if curcandle['volume'] > 0:
            prevcandle = candles.iloc[i+1]
            if prevcandle['volume']:
                diff = curcandle['volume']/prevcandle['volume']
                if diff > high_limit and green_candle(curcandle) and curcandle['high']>prevcandle['high']:
                    gothigh = i
                    gotdiff = diff
    return gothigh, gotdiff

def red_high(candles):
    gothigh = None
    gotdiff = None
    highest = 5 
    if len(candles.index) + 1 < highest:
        highest = len(candles.index) - 1
    for i in range(0,highest):
        curcandle = candles.iloc[i]
        if curcandle['volume'] > 0:
            prevcandle = candles.iloc[i+1]
            if prevcandle['volume']:
                diff = curcandle['volume']/prevcandle['volume']
                if diff > high_limit and red_candle(curcandle) and curcandle['high']<prevcandle['high']:
                    gothigh = i
                    gotdiff = diff
    return gothigh, gotdiff

inputfile = 'shorts.csv'
opts, args = getopt.getopt(sys.argv[1:],"i:",["input=",])
for opt, arg in opts:
    if opt in ("-i", "--input"):
        inputfile = arg

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
stocks = pd.read_csv(os.path.join(script_dir,inputfile),header=0)

end_date = datetime.now()
days = 2
start_date = end_date - timedelta(days=days)

start_time = '21:30:00'
if end_date.time().strftime('%H:%M:%S') >= start_time:
    trade_start = datetime.strptime(end_date.date().strftime('%d:%m:%Y') + ' ' + start_time,'%d:%m:%Y %H:%M:%S')
else:
    trade_start = datetime.strptime((end_date - timedelta(days=1)).date().strftime('%d:%m:%Y') + ' ' + start_time,'%d:%m:%Y %H:%M:%S')
print("Trade start:",trade_start)
print("End Date:",end_date)
day_candles = 79
candle_count = day_candles 
if end_date > trade_start:
    time_diff = end_date - trade_start
    minutes_diff = math.floor(time_diff.seconds/60)
    candle_count = math.floor(minutes_diff/5)
    print("Minutes in:",minutes_diff)
else:
    time_diff = trade_start - end_date
    # candle_count = 72
if candle_count>day_candles:
    candle_count = day_candles
if candle_count<3:
    candle_count = 3
print("Candle count:",candle_count)
half_candles = math.floor(candle_count*0.25)
print("Half candles:",half_candles)

price_alert_limit = 0.3

for i in range(int(len(stocks.index))-1):
# for i in range(1,2):
    gotinput = False
    if isinstance(stocks.iloc[i]['Ticker'], str):
        ticker = stocks.iloc[i]['Ticker'].upper()
        dticker = yq.Ticker(ticker)
        candles = dticker.history(start=start_date,end=end_date,interval='5m')
    else:
        continue

    if len(candles.index):

        candles = candles.reset_index(level=[0,1])
        candles['vwap'] = VolumeWeightedAveragePrice(high=candles['high'],low=candles['low'],close=candles['close'],volume=candles['volume'],window=candle_count).volume_weighted_average_price()

        candles = candles.iloc[::-1]

        daycandle = candles.iloc[0:candle_count]
        maxhigh = daycandle['high'].max()
        minlow = daycandle['low'].min()
        highindex = daycandle[['high']].idxmax()['high']
        lowindex = daycandle[['low']].idxmin()['low']
        daydiff = maxhigh - minlow


        if daydiff>price_alert_limit and lowindex<highindex:
            gotinput = True
            print("Ticker ",ticker, " Grow already has day diff more than ",price_alert_limit,"  : ",daydiff, " High:",highindex," Low:",lowindex)
            if lowindex>day_candles:
                lowindex = day_candles - 1
            if highindex>day_candles:
                highindex = day_candles - 1
            print("High candle:",daycandle.iloc[highindex])
            print("Low candle:",daycandle.iloc[lowindex])
            print("First candle:",daycandle.iloc[0])
            print("Last candle:",daycandle.iloc[-1])

        if daydiff>price_alert_limit and lowindex>highindex:
            gotinput = True
            print("Ticker ",ticker, " Shrunk already has day diff more than ",price_alert_limit,"  : ",daydiff, " High:",highindex," Low:",lowindex)
            if lowindex>day_candles:
                lowindex = day_candles - 1
            if highindex>day_candles:
                highindex = day_candles - 1
            print("High candle:",daycandle.iloc[highindex])
            print("Low candle:",daycandle.iloc[lowindex])
            print("First candle:",daycandle.iloc[0])
            print("Last candle:",daycandle.iloc[-1])

        # runscore,bullscore = bullrun(daycandle)
        # if bullscore<=0:
        #     bullscore = 1
        # runratio = runscore/bullscore
        # if runratio > 1.3:
        #     gotinput = True
        #     print("Ticker ",ticker," on bullrun ",runscore, ":",bullscore, " Ratio:",(runratio))
        #
        # if tribull(candles):
        #     gotinput = True
        #     print("Ticker ",ticker," got a clean tribull")
        #
        # if tripattern(candles):
        #     gotinput = True
        #     print("Ticker ",ticker," just got a tripattern")
        #
        # above = count_above_vwap(candles)
        # if above>1:
        #     gotinput = True
        #     print("Ticker ",ticker," above vwap:",above)
        #
        # pullback = find_bounce(candles)
        # if pullback>2:
        #     gotinput = True
        #     print("Ticker ",ticker," just bounced:",pullback)
        #
        # green_h, green_diff = green_high(candles)
        # if green_h:
        #     gotinput = True
        #     print("Ticker ",ticker," got green high ",green_diff)
        #
        # red_h, red_diff = red_high(candles)
        # if red_h:
        #     gotinput = True
        #     print("Ticker ",ticker," got red high ",red_diff)

        if gotinput:
            print("\n")
