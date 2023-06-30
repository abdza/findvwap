#!/usr/bin/env python

import pandas as pd 
import os
import math
from datetime import datetime,timedelta
import yahooquery as yq
from ta.volume import VolumeWeightedAveragePrice
import time

def count_above_vwap(candles):
    above = 0
    for i in range(-1,(len(candles.index) + 1)*-1,-1):
        finalcandle = candles.iloc[i]
        prevfinalcandle = candles.iloc[i-1]
        if finalcandle['volume']>0:
            if finalcandle['low'] > finalcandle['vwap'] and prevfinalcandle['low']<finalcandle['low']:
                above += 1
            else:
                break
    return above

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
print("Candle count:",candle_count)
if candle_count>72:
    candle_count = 72

for i in range(int(len(stocks.index))-1):
    if isinstance(stocks.iloc[i]['Ticker'], str):
        ticker = stocks.iloc[i]['Ticker'].upper()
        dticker = yq.Ticker(ticker)
        candles = dticker.history(start=start_date,end=end_date,interval='5m')
    else:
        continue

    if len(candles.index):
        candles['vwap'] = VolumeWeightedAveragePrice(high=candles['high'],low=candles['low'],close=candles['close'],volume=candles['volume'],window=candle_count).volume_weighted_average_price()
        candles = candles.reset_index(level=[0,1])
        # for i in range(-1,-10,-1):
        #     curcandle = candles.iloc[i]
        #     print(ticker,curcandle[1],curcandle['open'],curcandle['high'],curcandle['low'],curcandle['close'],curcandle['volume'],curcandle['vwap'])

        above = count_above_vwap(candles)
        if above==2:
            print("Ticker ",ticker," above vwap:",above)
            endloop = (above + 1) * -1
            curcandle = candles.iloc[-1]
            if curcandle['volume']==0:
                endloop -= 1

            for i in range(-1,endloop,-1):
                curcandle = candles.iloc[i]
                if curcandle['volume']>0:
                    print(ticker,curcandle[1],curcandle['open'],curcandle['high'],curcandle['low'],curcandle['close'],curcandle['volume'],curcandle['vwap'])
                else:
                    i -= 1
