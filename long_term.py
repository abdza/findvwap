#!/usr/bin/env python

import numpy as np
import pandas as pd
import tensorflow as tf
from tabulate import tabulate
from datetime import datetime,timedelta

import autokeras as ak
from sklearn.preprocessing import MinMaxScaler
from props import *

def add_prop(daytotal,dayprofitable,prop,toappend):
    prop_total = daytotal[daytotal[prop]==1]
    prop_profitable = dayprofitable[dayprofitable[prop]==1]
    toappend[prop + ' Total'] = str(len(prop_total)) + ' - ' + str(round(len(prop_total)/len(daytotal),2))
    toappend[prop + ' Profit'] = str(len(prop_profitable)) + ' - ' + str(round(len(prop_profitable)/len(dayprofitable),2)) 
    return toappend

def add_prop_data(daytotal,dayprofitable,prop,toappend):
    toappend[prop] = {}
    prop_total = daytotal[daytotal[prop]==1]
    prop_profitable = dayprofitable[dayprofitable[prop]==1]
    toappend[prop]['Total'] = len(prop_total)
    toappend[prop]['Total Perc'] = round(len(prop_total)/len(daytotal),2)
    toappend[prop]['Profit'] = len(prop_profitable)
    toappend[prop]['Profit Perc'] = round(len(prop_profitable)/len(dayprofitable),2) 
    return toappend

starttest = datetime.now()
raw_data = pd.read_csv('raw_data_perc.csv')
# raw_data = pd.read_csv('gapup_raw_data.csv')
raw_data = raw_data[raw_data['Sluggish Ticker']==0]
profitable = raw_data[raw_data['profitable']==1]
print("Total:",len(raw_data)," Profitable:",len(profitable))
dates = profitable['date'].unique()
date_format = '%Y-%m-%d'
toprint = []
highprops = {}
hightotal = {}
totalhighcount = {}
totalprofitcount = {}
for date in dates:
    date_obj = datetime.strptime(date, date_format)
    dayprofitable = profitable[profitable['date']==date]
    daytotal = raw_data[raw_data['date']==date]
    greatprofit = dayprofitable[dayprofitable['performance']=='Great']
    goodprofit = dayprofitable[dayprofitable['performance']=='Good']
    fairprofit = dayprofitable[dayprofitable['performance']=='Fair']
    # toappend = {'Date':date,'Day':date_obj.isoweekday(),'Total':len(daytotal),'Profitable':len(dayprofitable),'Great':len(greatprofit),'Good':len(goodprofit),'Fair':len(fairprofit)}
    toappend = {'Date':date,'Profitable':len(dayprofitable)}

    # if toappend['Great']>0:
    #     toappend['Great'] = str(toappend['Great']) + '\n' + '\n'.join(greatprofit['ticker'].tolist())
    # toappend['Good'] = str(toappend['Good']) + '\n' + '\n'.join(goodprofit['ticker'].tolist()[:10])
    # toappend['Fair'] = str(toappend['Fair']) + '\n' + '\n'.join(fairprofit['ticker'].tolist()[:10])
    # add_prop(daytotal,dayprofitable,'First Green',toappend)
    # add_prop(daytotal,dayprofitable,'Gap Up',toappend)
    prop_data = {}
    highboth = []
    curtop = []
    for prop in prop_list:
        if prop not in ignore_prop:
            add_prop_data(daytotal,dayprofitable,prop,prop_data)
            if prop_data[prop]['Total Perc']>0.5:
                curtop.append(prop)
                if prop_data[prop]['Profit Perc']>0.5:
                    highboth.append(prop)
    # print(tabulate(prop_data,headers="keys",tablefmt="simple_grid"))
    sorted_prop_data=sorted(prop_data.items(),key=lambda x:x[1]['Total Perc'],reverse=True)
    sorted_prop_profit=sorted(prop_data.items(),key=lambda x:x[1]['Profit Perc'],reverse=True)
    # print(sorted_prop_data[:5])

    topprop = ""
    for kp,kv in sorted_prop_data[:25]:
        topprop += kp + ' - ' + str(kv['Total Perc']) + '\n'
    topprofit = ""
    for kp,kv in sorted_prop_profit[:25]:
        topprofit += kp + ' - ' + str(kv['Profit Perc']) + '\n'

    toappend['Top Prop'] = topprop
    toappend['Profit Prop'] = topprofit
    toappend['High Both'] = '\n'.join(highboth)
    for hb in highboth:
        if hb in highprops:
            highprops[hb] += 1
        else:
            highprops[hb] = 1
    for ct in curtop:
        if ct in hightotal:
            hightotal[ct] += 1
        else:
            hightotal[ct] = 1
        totalhighcount[ct] = len(raw_data[raw_data[ct]==1])
        totalprofitable = raw_data[raw_data['profitable']==1]
        totalprofitcount[ct] = len(totalprofitable[totalprofitable[ct]==1])

    toprint.append(toappend)


    # print("Date:",date," Day:",date_obj.isoweekday()," Profitable:",len(dayprofitable), " Great:",len(greatprofit)," Good:",len(goodprofit)," Fair:",len(fairprofit))
print(tabulate(toprint,headers="keys",tablefmt="simple_grid"))
highcount = []
for hp,hv in highprops.items():
    highcount.append({'prop':hp,'count':hv,'profit_perc':round(highprops[hp]/hightotal[hp],2),'perc':round(totalprofitcount[hp]/totalhighcount[hp],5),'total_count':totalhighcount[hp],'profit_count':totalprofitcount[hp]})
highcount = sorted(highcount,key=lambda x:x['perc'],reverse=True)
print("Trades:",len(raw_data))
print(tabulate(highcount,headers="keys",tablefmt="simple_grid"))
highcountdf = pd.DataFrame.from_dict(highcount)
highcountdf.to_csv('highcount.csv',index=False)
endtest = datetime.now()
print("Start:",starttest)
print("End:",endtest)
print("Time:",endtest-starttest)
