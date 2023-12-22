#!/usr/bin/env python

from logging import lastResort
import pandas as pd 
import os
import sys
import csv
import getopt
from datetime import datetime,timedelta
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
from tensorflow.keras.models import load_model
import tensorflow as tf
import autokeras as ak
from props import *
from sklearn.preprocessing import MinMaxScaler

inputfile = 'stocks.csv'
outfile = 'shorts.csv'
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
    tickers_data = {}
    full_data = []
    prop_data = {}
    ticker_marks = {}
    latest_price = {}
    latest_date = {}
    first_price = {}
    max_price = {}
    maxmovement = {}
    levels = {}
    all_props = []
    end_of_trading = False

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

        if len(candles):
            candles = candles.reset_index(level=[0,1])
            candles['range'] = candles['high'] - candles['low']
            candles['body_length'] = candles['close'] - candles['open']
            curcandle = candles.iloc[-1]
            if isinstance(curcandle['date'],datetime):
                curkey = str(curcandle['date'].date())
            else:
                curkey = str(curcandle['date'])
            minute_end_date = datetime.strptime(curkey + ' 23:59:59', '%Y-%m-%d %H:%M:%S')
            minute_start_date = minute_end_date - timedelta(days=5)
            full_minute_candles = dticker.history(start=minute_start_date,end=minute_end_date,interval='15m')
            full_minute_candles['range'] = full_minute_candles['high'] - full_minute_candles['low']
            full_minute_candles['body_length'] = full_minute_candles['close'] - full_minute_candles['open']
            peaks = []
            bottoms = []
            if len(full_minute_candles)>1:
                tickers.append(ticker)
                full_minute_candles = full_minute_candles.reset_index(level=[0,1])
                minutelastcandle = full_minute_candles.iloc[-2]
                ldate = str(minutelastcandle['date'].date())
                fdate = str(datetime.date(minutelastcandle['date'])+timedelta(days=1))
                minute_candles = full_minute_candles.loc[(full_minute_candles['date']>ldate)]
                minute_candles = minute_candles.loc[(minute_candles['date']<fdate)]
                nowdate = str(datetime.now().date())
                if curkey!=nowdate:
                    end_of_trading = True

                if manualstocks:
                    print("Minute Candles:")
                    print(tabulate(minute_candles,headers='keys'))
                    print("Minute Size:",len(minute_candles))
                    print("NaN Check:",minute_candles['open'].isnull().sum())

                datediff = 1
                bdate = str(datetime.date(minutelastcandle['date'])-timedelta(days=datediff))
                bminute_candles = full_minute_candles.loc[(full_minute_candles['date']>bdate)]
                bminute_candles = bminute_candles.loc[(bminute_candles['date']<ldate)]
                while len(bminute_candles)==0 and datediff<=5:
                    datediff += 1
                    bdate = str(datetime.date(minutelastcandle['date'])-timedelta(days=datediff))
                    bminute_candles = full_minute_candles.loc[(full_minute_candles['date']>bdate)]
                    bminute_candles = bminute_candles.loc[(bminute_candles['date']<ldate)]

                datediff += 1
                bbdate = str(datetime.date(minutelastcandle['date'])-timedelta(days=datediff))
                bbminute_candles = full_minute_candles.loc[(full_minute_candles['date']>bbdate)]
                bbminute_candles = bbminute_candles.loc[(bbminute_candles['date']<bdate)]
                while len(bbminute_candles)==0 and datediff<=5:
                    datediff += 1
                    bbdate = str(datetime.date(minutelastcandle['date'])-timedelta(days=datediff))
                    bbminute_candles = full_minute_candles.loc[(full_minute_candles['date']>bbdate)]
                    bbminute_candles = bbminute_candles.loc[(full_minute_candles['date']<bdate)]

                prop_data, tickers_data, all_props, summary = analyze_minute(ticker,minute_candles,bminute_candles,bbminute_candles)

                if len(candles)>100:
                    levels[ticker] = find_levels(candles)
                else:
                    levels[ticker] = []
                latest_date[ticker] = minute_candles.iloc[-1]['date']
                if manualstocks:
                    print("Prop:",tickers_data[ticker])
                tickers_data = append_hash_set(tickers_data,ticker,'------------')
                maxmovement[ticker] = minute_candles['high'].max() - minute_candles['low'].min()
                if manualstocks:
                    print("Max Price:",summary['max_price']," First Price:",summary['first_price'])
                    print("Diff:",summary['diff']," Profitable:",summary['profitable'])
                fieldnames = ['ticker','date','day','diff','diff_level','performance','profitable','marks','yavg','yyavg','1range','1body','gap']
                row = {'ticker':ticker,'date':ldate,'day':datetime.strptime(ldate,'%Y-%m-%d').strftime('%A'),'diff':summary['diff'],'diff_level':summary['diff_level'],'performance':summary['category'],'profitable':summary['profitable'],'gap':summary['gap'],'price':summary['final_price']}
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
result = findgap()
result = pd.DataFrame.from_dict(result)

result.to_csv(os.path.join(script_dir,'results.csv'),index=False)
dates = result['date'].unique()
print("Dates:",dates)
dateperc = pd.DataFrame()
for cdate in dates:
    daytrade = result[result['date']==cdate]
    percdict = {}
    percdict['date'] = cdate
    for prop in prop_list:
        dayprop = daytrade[daytrade[prop]==1]
        propperc = round(len(dayprop)/len(daytrade),2)
        percdict['Perc ' + prop] = propperc
    percdf = pd.DataFrame.from_dict(percdict,orient='index').T
    dateperc = pd.concat([dateperc,percdf])
result_perc = result.set_index('date').join(dateperc.set_index('date'))
result_perc.reset_index(inplace=True)
result_perc.to_csv(os.path.join(script_dir,'results_perc.csv'),index=False)

result_perc = calc_marks(result_perc)
result_perc.to_csv(os.path.join(script_dir,'results_marks.csv'),index=False)


profitable_model = load_model(os.path.join(script_dir,"model_profitable"), custom_objects=ak.CUSTOM_OBJECTS)
profitablecsv = result_perc.copy()

print("Prepop Columns:",profitablecsv.columns)
topop = ['ticker','date','day','diff','diff_level','performance','profitable']
for tp in topop:
    profitablecsv.pop(tp)
for tp in ignore_prop:
    profitablecsv.pop(tp)
todrop = ['prev_marks','opening_marks','late_marks','marks','gap','price']
for tp in todrop:
    profitablecsv.pop(tp)
print("Columns:",profitablecsv.columns)
profitablefloat = np.asarray(profitablecsv).astype(np.float32)
result_perc['predicted_profitable'] = profitable_model.predict(profitablefloat)

fieldnames = ['date','ticker','diff_level','performance','profitable','predicted_profitable','prev_marks','opening_marks','late_marks','marks','gap']
minuscolumns = list(set(result_perc.columns.to_list()) - set(fieldnames))
finalcolumns = fieldnames + sorted(minuscolumns)

result_perc = result_perc[finalcolumns]

result_perc.sort_values(by=['marks'],ascending=False,inplace=True)
result_perc.to_csv(os.path.join(script_dir,'results_profitability.csv'),index=False)
todisp = result_perc[['ticker','date','profitable','predicted_profitable','diff_level','performance']]
print(tabulate(todisp[:10],headers="keys",tablefmt="grid"))
endtest = datetime.now()
print("Start:",starttest)
print("End:",endtest)
print("Time:",endtest-starttest)
