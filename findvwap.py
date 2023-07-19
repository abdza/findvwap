#!/usr/bin/env python

import pandas as pd 
import os
import math
from datetime import datetime,timedelta
import yahooquery as yq
from ta.volume import VolumeWeightedAveragePrice
import time


high_limit = 4

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
    for i in range(0,len(candles.index)):
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
    for i in range(0,len(candles.index)):
        finalcandle = candles.iloc[i]
        prevfinalcandle = candles.iloc[i+1]
        if finalcandle['volume']>0:
            if finalcandle['low'] > finalcandle['vwap'] and prevfinalcandle['low']<finalcandle['low']:
                above += 1
            else:
                break
    return above

def green_high(candles):
    gothigh = None
    gotdiff = None
    for i in range(0,3):
        curcandle = candles.iloc[i]
        if curcandle['volume'] > 0:
            prevcandle = candles.iloc[i+1]
            diff = curcandle['volume']/prevcandle['volume']
            if diff > high_limit and green_candle(curcandle) and curcandle['high']>prevcandle['high']:
                gothigh = i
                gotdiff = diff
    return gothigh, gotdiff

def red_high(candles):
    gothigh = None
    gotdiff = None
    for i in range(0,3):
        curcandle = candles.iloc[i]
        if curcandle['volume'] > 0:
            prevcandle = candles.iloc[i+1]
            diff = curcandle['volume']/prevcandle['volume']
            if diff > high_limit and red_candle(curcandle) and curcandle['high']<prevcandle['high']:
                gothigh = i
                gotdiff = diff

    return gothigh, gotdiff

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
stocks = pd.read_csv(os.path.join(script_dir,'stocks.csv'),header=0)

end_date = datetime.now()
days = 2
start_date = end_date - timedelta(days=days)

start_time = '21:30:00'
trade_start = datetime.strptime(end_date.date().strftime('%d:%m:%Y') + ' ' + start_time,'%d:%m:%Y %H:%M:%S')
print("Trade start:",trade_start)
print("End Date:",end_date)
if end_date > trade_start:
    time_diff = end_date - trade_start
    minutes_diff = math.floor(time_diff.seconds/60)
    candle_count = math.floor(minutes_diff/5)
    print("Minutes in:",minutes_diff)
else:
    time_diff = trade_start - end_date
    candle_count = 72
if candle_count>72:
    candle_count = 72
if candle_count<3:
    candle_count = 3
print("Candle count:",candle_count)

# dticker = yq.Ticker('SANA')
# candles = dticker.history(start=start_date,end=end_date,interval='5m')
# candles = candles.iloc[::-1]
# print('SANA',candles)

for i in range(int(len(stocks.index))-1):
# for i in range(1,2):
    if isinstance(stocks.iloc[i]['Ticker'], str):
        ticker = stocks.iloc[i]['Ticker'].upper()
        dticker = yq.Ticker(ticker)
        candles = dticker.history(start=start_date,end=end_date,interval='5m')
        candles = candles.iloc[::-1]
    else:
        continue

    if len(candles.index):
        candles = candles.reset_index(level=[0,1])
        candles['vwap'] = VolumeWeightedAveragePrice(high=candles['high'],low=candles['low'],close=candles['close'],volume=candles['volume'],window=candle_count).volume_weighted_average_price()
        # for i in range(-1,-10,-1):
        #     curcandle = candles.iloc[i]
        #     print(ticker,curcandle[1],curcandle['open'],curcandle['high'],curcandle['low'],curcandle['close'],curcandle['volume'],curcandle['vwap'])

        above = count_above_vwap(candles)
        if above==2:
            print("Ticker ",ticker," above vwap:",above)
            endloop = above + 1
            curcandle = candles.iloc[0]
            for i in range(0,endloop):
                curcandle = candles.iloc[i]
                if curcandle['volume']>0:
                    print(ticker,curcandle[1],curcandle['open'],curcandle['high'],curcandle['low'],curcandle['close'],curcandle['volume'],curcandle['vwap'])

        green_h, green_diff = green_high(candles)
        if green_h:
            print("Ticker ",ticker," got green high ",green_diff)
            for i in range(green_h,green_h+2):
                curcandle = candles.iloc[i]
                if curcandle['volume']>0:
                    print(ticker,curcandle[1],curcandle['open'],curcandle['high'],curcandle['low'],curcandle['close'],curcandle['volume'],curcandle['vwap'])

        red_h, red_diff = red_high(candles)
        if red_h:
            print("Ticker ",ticker," got red high ",red_diff)
            for i in range(red_h,red_h+2):
                curcandle = candles.iloc[i]
                if curcandle['volume']>0:
                    print(ticker,curcandle[1],curcandle['open'],curcandle['high'],curcandle['low'],curcandle['close'],curcandle['volume'],curcandle['vwap'])
            
        pullback = find_bounce(candles)
        if pullback>2:
            print("Ticker ",ticker," just bounced:",pullback)
            endloop = pullback + 1
            curcandle = candles.iloc[0]
            if endloop >= len(candles.index):
                endloop = len(candles.index) - 1
            for i in range(0,endloop):
                curcandle = candles.iloc[i]
                if curcandle['volume']>0:
                    print(ticker,curcandle[1],curcandle['open'],curcandle['high'],curcandle['low'],curcandle['close'],curcandle['volume'],curcandle['vwap'])
