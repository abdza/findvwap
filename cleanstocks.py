#!/usr/bin/env python
import pandas as pd
from datetime import datetime,timedelta
import yahooquery as yq
from tabulate import tabulate

stocks = pd.read_csv('stocks.csv')
end_date = datetime.now()
start_date = end_date - timedelta(days=10)
max_candles = 0
for i in range(len(stocks)):
    cstock = stocks.iloc[i]
    dticker = yq.Ticker(cstock['Ticker'])
    candles = dticker.history(start=start_date,end=end_date,interval='15m')
    if len(candles)>max_candles:
        max_candles=len(candles)
    candles['body_length'] = candles['close'] - candles['open']
    if len(candles)>0:
        print("Zero length candles for ",cstock['Ticker']," : ",len(candles[candles['body_length']==0])," from ",len(candles)," in %",round(len(candles[candles['body_length']==0])/len(candles),2))
    else:
        print("Got zero candles")
    stocks.loc[stocks['Ticker']==cstock['Ticker'],'zero_count'] = len(candles[candles['body_length']==0])
    stocks.loc[stocks['Ticker']==cstock['Ticker'],'candles_count'] = len(candles)
print("Need to drop because of zero candles:",stocks.loc[stocks['zero_count']>10]['Ticker'])
print("Need to drop because of less candles:",stocks.loc[stocks['candles_count']<max_candles]['Ticker'])

zero_index = stocks[stocks['zero_count']>10].index
stocks = stocks.drop(zero_index)
candles_index = stocks[stocks['candles_count']<max_candles-10].index
stocks = stocks.drop(candles_index)
stocks.to_csv('stocks.csv',index=False)
