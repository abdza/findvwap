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
                row = {'ticker':ticker,'date':ldate,'day':datetime.strptime(ldate,'%Y-%m-%d').strftime('%A'),'diff':summary['diff'],'diff_level':summary['diff_level'],'performance':summary['category'],'profitable':summary['profitable'],'gap':summary['gap']}
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
# result = sorted(result,key=lambda x:x['marks'])
result = pd.DataFrame.from_dict(result)

# compulsory_prop = ['First Green','Gap Up']
#
# for cp in compulsory_prop:
#     result = result[result[cp]==1]

result.to_csv(os.path.join(script_dir,'results.csv'),index=False)
# alldata = pd.read_csv('gapup_raw_data.csv')
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
result_perc.to_csv(os.path.join(script_dir,'results_perc.csv'),index=False)

# result_perc = pd.read_csv('raw_data_perc.csv')

global_marks = pd.read_csv(os.path.join(script_dir,'analyze_global.csv'))
global_fail = pd.read_csv(os.path.join(script_dir,'analyze_global_fail.csv'))

result_perc.reset_index(inplace=True)
# result_perc.set_index(['date','ticker'])

result_perc['prev_marks'] = 1
result_perc['neg_prev_marks'] = 1
for prop in prev_prop_list:
    cgmark = global_marks[global_marks['Prop']==prop]
    if len(cgmark):
        if cgmark.iloc[0]['Marks']>0:
            # result_perc.loc[result_perc[prop]==1,'prev_marks'] *= cgmark.iloc[0]['Marks']
            result_perc.loc[result_perc[prop]==1,'prev_marks'] += result_perc[prop] * cgmark.iloc[0]['Marks']
        # result_perc.loc[result_perc['Perc ' + prop]>cgmark.iloc[0]['Median'],'prev_marks'] *= (1 + cgmark.iloc[0]['Median'])
    cgmark = global_fail[global_fail['Prop']==prop]
    if len(cgmark):
        result_perc.loc[result_perc[prop]==1,'neg_prev_marks'] *= cgmark.iloc[0]['Marks'] * 0.3
result_perc['final_prev_marks'] = result_perc['prev_marks'] - result_perc['neg_prev_marks']

result_perc['opening_marks'] = 1
result_perc['neg_opening_marks'] = 1
for prop in opening_prop_list:
    cgmark = global_marks[global_marks['Prop']==prop]
    if len(cgmark):
        if cgmark.iloc[0]['Marks']>0:
            # result_perc.loc[result_perc[prop]==1,'opening_marks'] *= cgmark.iloc[0]['Marks']
            result_perc.loc[result_perc[prop]==1,'opening_marks'] += result_perc[prop] * cgmark.iloc[0]['Marks']
        # result_perc.loc[result_perc['Perc ' + prop]>cgmark.iloc[0]['Median'],'opening_marks'] *= (1 + cgmark.iloc[0]['Median'])
    cgmark = global_fail[global_fail['Prop']==prop]
    if len(cgmark):
        result_perc.loc[result_perc[prop]==1,'neg_opening_marks'] *= cgmark.iloc[0]['Marks'] * 0.3
result_perc['final_opening_marks'] = result_perc['opening_marks'] - result_perc['neg_opening_marks']

result_perc['late_marks'] = 1
result_perc['neg_late_marks'] = 1
for prop in late_prop_list:
    cgmark = global_marks[global_marks['Prop']==prop]
    if len(cgmark):
        if cgmark.iloc[0]['Marks']>0:
            # result_perc.loc[result_perc[prop]==1,'late_marks'] *= cgmark.iloc[0]['Marks']
            result_perc.loc[result_perc[prop]==1,'late_marks'] += result_perc[prop] * cgmark.iloc[0]['Marks']
        # result_perc.loc[result_perc['Perc ' + prop]>cgmark.iloc[0]['Median'],'late_marks'] *= (1 + cgmark.iloc[0]['Median'])
    cgmark = global_fail[global_fail['Prop']==prop]
    if len(cgmark):
        result_perc.loc[result_perc[prop]==1,'neg_late_marks'] *= cgmark.iloc[0]['Marks'] * 0.3
result_perc['final_late_marks'] = result_perc['late_marks'] - result_perc['neg_late_marks']
        

    # print(tabulate(corr,headers='keys'))
#     dayprop = daytrade[daytrade[prop]==1]
#     print(corr.columns)

scaler = MinMaxScaler()
result_perc[['prev_marks','neg_prev_marks','final_prev_marks','opening_marks','neg_opening_marks','final_opening_marks','late_marks','neg_late_marks','final_late_marks']] = scaler.fit_transform(result_perc[['prev_marks','neg_prev_marks','final_prev_marks','opening_marks','neg_opening_marks','final_opening_marks','late_marks','neg_late_marks','final_late_marks']])
result_perc['early_marks'] = result_perc['final_prev_marks'] + result_perc['final_opening_marks']
result_perc['marks'] = result_perc['final_prev_marks'] + result_perc['final_opening_marks'] + result_perc['final_late_marks']
result_perc[['early_marks','marks']] = scaler.fit_transform(result_perc[['early_marks','marks']])


prop_corr = pd.read_csv(os.path.join(script_dir,'analyze_global_corr.csv'))
print("prop:",prop_corr.columns)
result_perc['corr_marks'] = 1
for prop in prop_list:
    print("Corr for ",prop)
    filtered = result_perc[result_perc['Perc ' + prop]>0.5]
    filtered = filtered[filtered[prop]==1]
    if prop not in ignore_prop:
        cgmark = global_marks[global_marks['Prop']==prop]
        for tp in cgmark['All CP']:
            # print("Tp:",tp)
            intp = eval(tp)
            for iii in intp:
                print("In tp:",iii)
                # if iii['CorrRatio']>0.3:
                filtered = filtered[filtered[iii['Prop']]==1]
                result_perc.loc[filtered.index,'corr_marks'] += result_perc['corr_marks'] * iii['CorrRatio']
                print("We've got ratio")
        # corr = prop_corr[prop_corr[prop]>0.5]
        # filtered = result_perc[result_perc['Perc ' + prop]>0.5]
        # filtered = filtered[filtered[prop]==1]
        # for cp in corr['Prop']:
        #     filtered = filtered[filtered[cp]==1]
        #     print("P:",cp," Filter size:",len(filtered))
        # if len(filtered)>0:
        #     result_perc.loc[filtered.index,'corr_marks'] += 1

result_perc.loc[result_perc['corr_marks']==1,'corr_marks'] = 0

result_perc[['corr_marks']] = scaler.fit_transform(result_perc[['corr_marks']])

result_perc['final_marks'] = result_perc['marks'] + result_perc['corr_marks']
result_perc[['final_marks']] = scaler.fit_transform(result_perc[['final_marks']])

fieldnames = ['date','ticker','diff_level','performance','profitable','prev_marks','neg_prev_marks','final_prev_marks','opening_marks','neg_opening_marks','final_opening_marks','late_marks','neg_late_marks','final_late_marks','early_marks','marks','corr_marks','final_marks','yavg','yyavg','1range','1body','gap']
minuscolumns = list(set(result_perc.columns.to_list()) - set(fieldnames))
finalcolumns = fieldnames + sorted(minuscolumns)

result_perc = result_perc[finalcolumns]

result_perc.to_csv(os.path.join(script_dir,'results_marks.csv'),index=False)


# alldata.to_csv(os.path.join(script_dir,'gapup_raw_data.csv'))
# loaded_model = load_model("model_autokeras", custom_objects=ak.CUSTOM_OBJECTS)
# diff_model = load_model(os.path.join(script_dir,"model_diff_level"), custom_objects=ak.CUSTOM_OBJECTS)
# [print('Fd:',i,i.shape, i.dtype) for i in loaded_model.inputs]

# tocsv = pd.read_csv(os.path.join(script_dir,'gapup_raw_data.csv'))
# highcount = pd.read_csv(os.path.join(script_dir,'highcount.csv'))
# finalmarks = []
# for i in range(len(tocsv)):
#     curdat = tocsv.iloc[i]
#     totalplus = 1
#     hpp = []
#     for hc in range(len(highcount)):
#         hcdata = highcount.iloc[hc]
#         totalhc = len(tocsv[tocsv[hcdata['prop']]==1])
#         totalperc = totalhc/len(tocsv)
#         if totalperc > 0.5:
#             hpp.append(hcdata['prop'])
#             if curdat[hcdata['prop']]==1:
#                 totalplus += hcdata['perc']
#     if totalplus>1:
#         finalmarks.append(curdat['marks'] * (totalplus * 300))
#         if manualstocks:
#             print("Got high prop:",",".join(hpp)," will add marks:",totalplus * 300," to final marks:",finalmarks[-1]," from:",curdat['marks'])
#     else:
#         finalmarks.append(curdat['marks'])
#
# tocsv.loc[:,'final_marks'] = finalmarks
# tocsv.to_csv('gapup_raw_data.csv',index=False)

profitable_model = load_model(os.path.join(script_dir,"model_profitable"), custom_objects=ak.CUSTOM_OBJECTS)
profitablecsv = result_perc.copy()
# diffcsv = tocsv.copy()

topop = ['ticker','date','day','diff','diff_level','performance','profitable']
for tp in topop:
    profitablecsv.pop(tp)
for tp in ignore_prop:
    profitablecsv.pop(tp)
# topop = ['yavg','yyavg','1range','1body','gap','marks']
# for tp in topop:
#     profitablecsv.pop(tp)
print("Columns:",profitablecsv.columns)
profitablefloat = np.asarray(profitablecsv).astype(np.float32)
result_perc['predicted_profitable'] = profitable_model.predict(profitablefloat)

fieldnames = ['date','ticker','diff_level','performance','profitable','final_marks','predicted_profitable','prev_marks','neg_prev_marks','final_prev_marks','opening_marks','neg_opening_marks','final_opening_marks','late_marks','neg_late_marks','final_late_marks','early_marks','marks','corr_marks','gap']
minuscolumns = list(set(result_perc.columns.to_list()) - set(fieldnames))
finalcolumns = fieldnames + sorted(minuscolumns)

result_perc = result_perc[finalcolumns]

# topop = ['ticker','date','day','Big Reverse','Bottom After Noon','Bottom Before Noon','Bottom Lunch','Peak After Noon','Peak Before Noon','Peak Lunch','diff','profitable','performance','diff_level']
# for tp in topop:
#     diffcsv.pop(tp)
# topop = ['ticker','date','day','diff','diff_level','profitable','performance']
# for tp in topop:
#     diffcsv.pop(tp)
# for tp in ignore_prop:
#     diffcsv.pop(tp)
# difffloat = np.asarray(diffcsv).astype(np.float32)
# tocsv['predicted_diff'] = diff_model.predict(difffloat)

# tocsv.sort_values(by=['predicted_profitable','predicted_diff'],ascending=False,inplace=True)
result_perc.sort_values(by=['predicted_profitable'],ascending=False,inplace=True)
result_perc.to_csv(os.path.join(script_dir,'results_profitability.csv'),index=False)
todisp = result_perc[['ticker','date','profitable','predicted_profitable','final_marks','diff_level','performance']]
print(tabulate(todisp[:10],headers="keys",tablefmt="grid"))
# toresult = tocsv.iloc[:10][['ticker','date','profitable','predicted_profitable','diff','diff_level','performance','marks']]
# toresult.to_csv(os.path.join(script_dir,'results_predicted.csv'),index=False)
# print("End trading:",endtrading)
endtest = datetime.now()
print("Start:",starttest)
print("End:",endtest)
print("Time:",endtest-starttest)
