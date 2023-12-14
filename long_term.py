#!/usr/bin/env python

import numpy as np
import pandas as pd
import tensorflow as tf
from tabulate import tabulate
from datetime import datetime,timedelta

import autokeras as ak
from sklearn.preprocessing import MinMaxScaler

prop_list = [
'Big Reverse',
'Bottom After Noon',
'Bottom Before Noon',
'Bottom Lunch',
'Consecutive Early Green',
'Consecutive Early Red',
'Consecutive FVG',
'Consecutive Green',
'Consecutive Late Green',
'Consecutive Late Red',
'Consecutive Negative FVG',
'Consecutive Negative Volume Gap',
'Consecutive Red',
'Consecutive Volume Gap',
'Continue Higher High',
'Continue Higher Low',
'Continue Lower High',
'Continue Lower Low',
'First Green',
'First Hammer',
'First Red',
'First Reverse Hammer',
'FVG First',
'FVG Second',
'Gap Down Above Average',
'Gap Down Above 2 Day Average',
'Gap Down Below Prev Min',
'Gap Down',
'Gap Up Above Average',
'Gap Up Above 2 Day Average',
'Gap Up Above Prev Max',
'Gap Up',
'Higher High',
'Higher Low',
'Lower High',
'Lower Low',
'Negative FVG First',
'Negative FVG Second',
'Negative Volume Gap First',
'Negative Volume Gap Second',
'Open Higher Than 2 Prev Max',
'Open Higher Than Prev Max Plus Average',
'Open Higher Than Prev Max',
'Open Lower Than 2 Prev Max',
'Open Lower Than Prev Min Minus Average',
'Open Lower Than Prev Min',
'Peak After Noon',
'Peak Before Noon',
'Peak Lunch',
'Range Above 2 Day Average',
'Range Above Average',
'Range Lower 2 Day Average',
'Range Lower Average',
'Range More Than Gap Down',
'Range More Than Gap Up',
'Second Green',
'Second Hammer',
'Second Long',
'Second Red',
'Second Reverse Hammer',
'Second Short',
'Third Green',
'Third Hammer',
'Third Long',
'Third Red',
'Third Reverse Hammer',
'Third Short',
'Two Small Reverse',
'Volume Gap First',
'Volume Gap Second',
'Volume Higher Than Average',
'Volume Lower Than Average',
'Volume Open Higher',
'Volume Open Lower',
'Volume Open Excedingly High',
'Volume Open Excedingly Low',
'Volume Open After High',
'Volume Open After Low',
'Early Top Level',
'Late Top Level',
'Top Level',
'Second Range Shorter',
'Second Range Longer',
'Third Range Longer',
'Third Range Shorter',
'Consecutive Shorter Range',
'Consecutive Longer Range',
'Second Range Very Shorter',
'Second Range Very Longer',
'Third Range Very Longer',
'Third Range Very Shorter',
'Consecutive Very Shorter Range',
'Consecutive Very Longer Range',
'Second Volume Lower',
'Second Volume Higher',
'Third Volume Higher',
'Third Volume Lower',
'Consecutive Lower Volume',
'Consecutive Higher Volume',
'Limp Second Diff',
'Limp Third Diff',
'Consecutive Limp Diff',
'Tiny Range',
'Second Tiny Range',
'Third Tiny Range',
'Consecutive Early Tiny Range',
'Consecutive Late Tiny Range',
'Consecutive Tiny Range',
'Huge Range',
'Second Huge Range',
'Third Huge Range',
'Consecutive Early Huge Range',
'Consecutive Late Huge Range',
'Consecutive Huge Range',
'Huge Negative Range',
'Second Huge Negative Range',
'Third Huge Negative Range',
'Consecutive Early Huge Negative Range',
'Consecutive Late Huge Negative Range',
'Consecutive Huge Negative Range',
'Max After Min',
'Min After Max',
'Yesterday End In Red',
'Yesterday End Volume Above Average',
'Volume Above 5 Time Average',
'Volume Above 10 Time Average',
'Volume Above 5 Time Before Average',
'Volume Above 10 Time Before Average',
'Volume Consecutive Above 5 Time Average',
'Volume Consecutive Above 10 Time Average',
'New IPO',
'Fairly New IPO',
'Sluggish Ticker',
'Continue Sluggish Ticker',
'Late Start',
'Yesterday Status Great',
'Yesterday Status Good',
'Yesterday Status Fair',
'Yesterday Status Fail',
'Yesterday Profitable',
'Yesterday Loss',
'Yesterday Absolute Loss',
'2 Days Ago Status Great',
'2 Days Ago Status Good',
'2 Days Ago Status Fair',
'2 Days Ago Status Fail',
'2 Days Ago Profitable',
'2 Days Ago Loss',
'2 Days Ago Absolute Loss',
    ]

ignore_prop = [
'Big Reverse',
'Two Small Reverse',
'Bottom After Noon',
'Bottom Before Noon',
'Bottom Lunch',
'Peak After Noon',
'Peak Before Noon',
'Peak Lunch',
'Min After Max',
'Max After Min',
]

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
raw_data = pd.read_csv('raw_data_20231212.csv')
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
