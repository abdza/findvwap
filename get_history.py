#!/usr/bin/env python

from logging import lastResort
import pandas as pd 
import os
import json
import sys
import csv
import getopt
from datetime import datetime,timedelta
from pandas.core.indexing import convert_missing_indexer
import yahooquery as yq
import numpy as np
import math
from tabulate import tabulate
from sklearn.cluster import KMeans
from ta.trend import EMAIndicator
import streamlit as st
from streamlit_calendar import calendar
from props import *
from sklearn.preprocessing import MinMaxScaler

inputfile = 'filtered.csv'
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
    full_data = []
    ticker_marks = {}
    latest_price = {}
    latest_date = {}
    first_price = {}
    max_price = {}
    levels = {}
    end_of_trading = False
    ldate = None

    for i in range(len(stocks.index)):
    # for i in range(5):
        if isinstance(stocks.iloc[i]['Ticker'], str):
            ticker = stocks.iloc[i]['Ticker'].upper()
            dticker = yq.Ticker(ticker)
            candles = dticker.history(start=start_date,end=end_date,interval='1d')
            candles = candles.loc[(candles['volume']>0)]
            print("Processing ",ticker, " got ",len(candles))
        else:
            continue

        if len(candles)<3:
            continue
        else:
            candles = candles.reset_index(level=[0,1])
            candles['range'] = candles['high'] - candles['low']
            candles['body_length'] = candles['close'] - candles['open']
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
            minute_start_date = minute_end_date - timedelta(days=10)
            full_minute_candles = dticker.history(start=minute_start_date,end=minute_end_date,interval='15m')
            full_minute_candles['range'] = full_minute_candles['high'] - full_minute_candles['low']
            full_minute_candles['body_length'] = full_minute_candles['close'] - full_minute_candles['open']
            if len(full_minute_candles)>1:
                tickers.append(ticker)
                full_minute_candles = full_minute_candles.reset_index(level=[0,1])
                minutelastcandle = full_minute_candles.iloc[-2]
                ldate = str(minutelastcandle['date'].date())
                fdate = str(datetime.date(minutelastcandle['date'])+timedelta(days=1))
                minute_candles = full_minute_candles.loc[(full_minute_candles['date']>ldate)]
                minute_candles = minute_candles.loc[(minute_candles['date']<fdate)]
                latest_date[ticker] = minute_candles.iloc[-1]['date']

                if manualstocks:
                    print("Minute Candles:")
                    print(tabulate(minute_candles,headers='keys'))

                datediff = 1
                bdate = str(minute_candles.iloc[0]['date'].date()-timedelta(days=datediff))
                print("Bdate:",bdate," Ldate:",ldate)
                bminute_candles = full_minute_candles.loc[(full_minute_candles['date']>bdate)]
                bminute_candles = bminute_candles.loc[(bminute_candles['date']<ldate)]
                while len(bminute_candles)==0 and datediff<=10:
                    datediff += 1
                    bdate = str(datetime.date(minutelastcandle['date'])-timedelta(days=datediff))
                    bminute_candles = full_minute_candles.loc[(full_minute_candles['date']>bdate)]
                    bminute_candles = bminute_candles.loc[(bminute_candles['date']<ldate)]

                if len(bminute_candles):
                    datediff = 1
                    bbdate = str(bminute_candles.iloc[0]['date'].date()-timedelta(days=datediff))
                    twodaystart = bbdate
                    # twodayend = str(bminute_candles.iloc[0]['date'].date())
                    twodayend = bbdate + ' 23:00:00'
                    print("2 day start:",twodaystart," 2 day end:",twodayend)
                    bbminute_candles = full_minute_candles.loc[full_minute_candles['date']>twodaystart]
                    bbminute_candles = bbminute_candles.loc[bbminute_candles['date']<twodayend]
                    print("Len of bbminute:",len(bbminute_candles))
                    while len(bbminute_candles)==0 and datediff<=10:
                        datediff += 1
                        bbdate = str(bminute_candles.iloc[0]['date']-timedelta(days=datediff))
                        bbminute_candles = full_minute_candles.loc[full_minute_candles['date']>twodaystart]
                        bbminute_candles = bbminute_candles.loc[bbminute_candles['date']<twodayend]
                else:
                    bbminute_candles = pd.DataFrame()
                day_candles = candles[:-1]
                hourdate = str(day_candles.iloc[-1]['date']-timedelta(days=10))

                print("Hour start:",hourdate)
                hour_candles = dticker.history(period='7d',start=hourdate,interval='1h')
                hour_candles['range'] = hour_candles['high'] - hour_candles['low']
                hour_candles['body_length'] = hour_candles['close'] - hour_candles['open']
                if len(hour_candles.index.shape)>1:
                    hour_candles = hour_candles.reset_index(level=[0,1])
                else:
                    hour_candles = hour_candles.reset_index()
                hour_candles = hour_candles.loc[(hour_candles['date']<ldate)]

                if manualstocks:
                    print("First candle:",minute_candles.iloc[0]['date'].hour,':',minute_candles.iloc[0]['date'].minute)
                    print("First size:",minute_candles.iloc[0]['body_length'],':',(minute_candles.iloc[0]['body_length']==0))
                    print("Previous Minute Candles:")
                    print(tabulate(bminute_candles,headers='keys'))
                    print("Previous NaN Check:",bminute_candles['open'].isnull().sum())

                latest_price = append_hash_set(latest_price,ticker,minute_candles.iloc[-1]['close'])

                prop_data, tickers_data, all_props, summary = analyze_minute(ticker,minute_candles,bminute_candles,bbminute_candles,hour_candles,day_candles)

                fieldnames = ['ticker','date','day','range','diff','diff_level','performance','profitable','gap','price']
                row = {'ticker':ticker,'date':ldate,'day':datetime.strptime(ldate,'%Y-%m-%d').strftime('%A'),'range':curcandle['range'],'diff':summary['diff'],'diff_level':summary['diff_level'],'performance':summary['category'],'profitable':summary['profitable'],'gap':summary['gap'],'price':summary['final_price']}
                fieldnames.append('Minute Start')
                row['Minute Start'] = minute_candles.iloc[0]['date']
                fieldnames.append('Minute End')
                row['Minute End'] = minute_candles.iloc[-1]['date']
                fieldnames.append('Yesterday Start')
                fieldnames.append('Yesterday End')
                if len(bminute_candles):
                    row['Yesterday Start'] = bminute_candles.iloc[0]['date']
                    row['Yesterday End'] = bminute_candles.iloc[-1]['date']
                else:
                    row['Yesterday Start'] = None
                    row['Yesterday End'] = None
                fieldnames.append('2 Days Start')
                fieldnames.append('2 Days End')
                if len(bbminute_candles):
                    row['2 Days Start'] = bbminute_candles.iloc[0]['date']
                    row['2 Days End'] = bbminute_candles.iloc[-1]['date']
                else:
                    row['2 Days Start'] = None
                    row['2 Days End'] = None
                fieldnames.append('Hourly Start')
                fieldnames.append('Hourly End')
                if len(hour_candles):
                    row['Hourly Start'] = hour_candles.iloc[0]['date']
                    row['Hourly End'] = hour_candles.iloc[-1]['date']
                else:
                    row['Hourly Start'] = None
                    row['Hourly End'] = None
                fieldnames.append('Daily Start')
                fieldnames.append('Daily End')
                if len(day_candles):
                    row['Daily Start'] = day_candles.iloc[0]['date']
                    row['Daily End'] = day_candles.iloc[-1]['date']
                else:
                    row['Daily Start'] = None
                    row['Daily End'] = None
                for pp in prop_list:
                    fieldnames.append(pp)
                    if pp in tickers_data[ticker]:
                        row[pp] = 1
                    else:
                        row[pp] = 0
                full_data.append(row)

    print("End date:",end_date)
    return full_data

starttest = datetime.now()
alldata = pd.DataFrame()
for day in range(90):
# for day in range(10):
# for day in range(2):
    instockdate = starttest - timedelta(days=day)
    result = findgap()
    result = pd.DataFrame.from_dict(result)
    alldata = pd.concat([alldata,result])

alldata.to_csv(os.path.join(script_dir,'raw_data.csv'),index=False)
dates = alldata['date'].unique()
print("Dates:",dates)
dateperc = pd.DataFrame()
for cdate in dates:
    daytrade = alldata[alldata['date']==cdate]
    percdict = {}
    percdict['date'] = cdate
    for prop in prop_list:
        dayprop = daytrade[daytrade[prop]==1]
        propperc = round(len(dayprop)/len(daytrade),4)
        percdict['Perc ' + prop] = propperc
        percdict['Total ' + prop] = len(dayprop)
    percdf = pd.DataFrame.from_dict(percdict,orient='index').T
    dateperc = pd.concat([dateperc,percdf])
result_perc = alldata.set_index('date').join(dateperc.set_index('date'))
result_perc.to_csv(os.path.join(script_dir,'raw_data_perc.csv'))

endtest = datetime.now()
print("Start:",starttest)
print("End:",endtest)
print("Time:",endtest-starttest)
