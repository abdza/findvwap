#!/usr/bin/env python
import pandas as pd 
import os
import sys
import csv
from datetime import datetime,timedelta
import numpy as np
import math
from statistics import mean
from tabulate import tabulate
from sklearn.preprocessing import MinMaxScaler
from props import *
import getopt

global_marks = pd.read_csv(os.path.join(script_dir,'analyze_global.csv'))

datas = pd.read_csv('results_profitability.csv')
profitables = datas[datas['profitable']==1]
propcount = []
for prop in prop_list:
    cgmark = global_marks[global_marks['Prop']==prop]
    if len(cgmark):
        coor = cgmark.iloc[0]['Corr']
    else:
        coor = None
    if len(datas[datas[prop]==1]):
        pp = len(profitables[profitables[prop]==1])/len(datas[datas[prop]==1])
    else:
        pp = 0

    curprop = {'prop':prop,'count':len(profitables[profitables[prop]==1]),'total count':len(datas[datas[prop]==1]),'pp':pp,'perc':profitables['Perc ' + prop].max(),'coor':coor}
    propcount.append(curprop)
propcount.sort(key=lambda x:x['count'],reverse=True)
print(tabulate(propcount,headers='keys'))


datas = pd.read_csv('raw_data_perc_marks.csv')
datas.sort_values(by=['diff_level'],ascending=False,inplace=True)
losses = datas[datas['profitable']==0]
profitables = datas[datas['profitable']==1]

dates = datas['date'].unique()
daydatas = []
for cdate in dates:
    daydata = {}
    daytrades = datas[datas['date']==cdate]
    daylosses = losses[losses['date']==cdate]
    dayprofits = profitables[profitables['date']==cdate]
    daydata['date'] = cdate
    daydata['profitable'] = len(dayprofits)
    daydata['losses'] = len(daylosses)
    daydata['maxprofit'] = dayprofits['diff_level'].max()
    topprop = {}
    topprop2 = {}
    blacklist = [
        'Limp Third Diff',
        'Limp Second Diff',
        'Big Reverse',
        'Range Smaller Than Daily Average',
        'Yesterday Range Smaller Than Daily Average',
        'Consecutive Limp Diff',
        '2 Days Ago Range Smaller Than Daily Average',
        'Two Small Reverse',
        'Volume Open Lower',
        'Open Lower Than 2 Prev Max',
        'Second Range Shorter',
        '2 Days Ago Status Fair',
        'Hour General Lower High',
        'Daily End In Green',
        'Second Volume Lower',
        'Daily Last Bottom Before Last Peak',
        'Higher Low',
        'Yesterday Range Larger Than Hourly Average',
        '2 Days Ago Positive Morning Range',
        'Hour Red More Than Green',
        'Bottom Lunch',
        'Bottom Before Noon',
        'Bottom After Noon',
    ]
    for cp in prop_list:
        curlabel = 'Perc ' + cp
        if cp not in blacklist:
            curmax = daytrades[curlabel].max()
            if len(topprop)==0:
                topprop = {'prop':cp,'perc':curmax}
            elif curmax>topprop['perc']:
                topprop = {'prop':cp,'perc':curmax}
            if cp!=topprop['prop']:
                if len(topprop2)==0:
                    topprop2 = {'prop':cp,'perc':curmax}
                elif curmax>topprop2['perc']:
                    topprop2 = {'prop':cp,'perc':curmax}
    daydata['topprop'] = topprop['prop'] + ':' + str(topprop['perc'])
    daydata['topprop2'] = topprop2['prop'] + ':' + str(topprop2['perc'])

    daydatas.append(daydata)

    # Conclusion: Beware of days with high amount of Yesterday Loss, Sluggish Ticker, Lower High


print(tabulate(daydatas,headers="keys"))
print("Amount Of Days:",len(dates))

# totest = datas.iloc[:20].T
# totest.to_csv('tmp.csv')

# print(tabulate(totest,headers="keys"))

# opts, args = getopt.getopt(sys.argv[1:],"p:h:",["prop=","highest="])
# print("Args:",args)
# print("Opts:",opts)
# if len(opts)>0:
#     for opt, arg in opts:
#         if opt in ("-h", "--highest"):
#             maxheight = int(arg)
#             todisp = datas.sort_values('diff_level',ascending=False)
#             # todisp = todisp[todisp['First Green']==1]
#             todisp = todisp[:maxheight]
#             pcount = []
#             for p in prop_list:
#                 if p not in ignore_prop:
#                     pcount.append({'prop':p,'count':todisp[p].sum()})
#             todisp = todisp.transpose()
#             print(tabulate(todisp,headers="keys"))
#             pcdf = pd.DataFrame(pcount)
#             pcdf.sort_values('count',inplace=True,ascending=False)
#             pcdf.to_csv("analyze_highest_prop.csv",index=False)
#             print(pcdf)
#             todisp.to_csv("analyze_highest.csv")
#
#
# if len(args)>0:
#     iprop = args[0].split(",")
#     print("Iprop:",iprop)
#     pfpdata = datas
#     pdat = {}
#     done = []
#     for ip in iprop:
#         ip = ip.strip()
#         pfpdata = pfpdata[pfpdata[ip]==1]
#         profit = pfpdata[pfpdata['profitable']==1]
#         if len(pfpdata):
#             ratio = round(len(profit) / len(pfpdata),2)
#         else:
#             ratio = 0
#         fullcount = datas[datas[ip]==1]
#         fullratio = round(len(fullcount)/len(datas),2)
#         nextprop = {}
#         done.append(ip)
#         for p in prop_list:
#             if p not in ignore_prop and p not in done:
#                 nextdat = pfpdata[pfpdata[p]==1]
#                 nextprof = nextdat[nextdat['profitable']==1]
#                 nextprop[p]=len(nextprof)
#         nextprop = sorted(nextprop.items(),key=lambda x:x[1],reverse=True)
#         pdat[ip] = {'Profit':len(profit),'Full':len(pfpdata),'Ratio':ratio,'Full Count':len(fullcount),'Full Ratio':fullratio,'Next Prop':'\n'.join([ k + ':' + str(v) for k,v in nextprop[:5]])}
#     df = pd.DataFrame().from_dict(pdat)
#     print(tabulate(df.transpose(),headers="keys"))
# else:
#     if len(opts)==0:
#         pdat = {}
#         fpdata = datas[datas['First Green']==1]
#         for ip in prev_prop_list:
#             ip = ip.strip()
#             pfpdata = fpdata[fpdata[ip]==1]
#             profit = pfpdata[pfpdata['profitable']==1]
#             if len(pfpdata):
#                 ratio = round(len(profit) / len(pfpdata),2)
#             else:
#                 ratio = 0
#             fullcount = datas[datas[ip]==1]
#             fullratio = round(len(fullcount)/len(datas),2)
#             pdat[ip] = {'Profit':len(profit),'Full':len(pfpdata),'Ratio':ratio,'Full Count':len(fullcount),'Full Ratio':fullratio}
#         df = pd.DataFrame().from_dict(pdat)
#         print(tabulate(df.transpose(),headers="keys"))
#
#         pdat = {}
#         fpdata = datas[datas['First Green']==1]
#         for ip in opening_prop_list:
#             ip = ip.strip()
#             pfpdata = fpdata[fpdata[ip]==1]
#             profit = pfpdata[pfpdata['profitable']==1]
#             if len(pfpdata):
#                 ratio = round(len(profit) / len(pfpdata),2)
#             else:
#                 ratio = 0
#             fullcount = datas[datas[ip]==1]
#             fullratio = round(len(fullcount)/len(datas),2)
#             pdat[ip] = {'Profit':len(profit),'Full':len(pfpdata),'Ratio':ratio,'Full Count':len(fullcount),'Full Ratio':fullratio}
#         df = pd.DataFrame().from_dict(pdat)
#         print(tabulate(df.transpose(),headers="keys"))
#
#         pdat = {}
#         fpdata = datas[datas['First Green']==1]
#         for ip in late_prop_list:
#             ip = ip.strip()
#             pfpdata = fpdata[fpdata[ip]==1]
#             profit = pfpdata[pfpdata['profitable']==1]
#             if len(pfpdata):
#                 ratio = round(len(profit) / len(pfpdata),2)
#             else:
#                 ratio = 0
#             fullcount = datas[datas[ip]==1]
#             fullratio = round(len(fullcount)/len(datas),2)
#             pdat[ip] = {'Profit':len(profit),'Full':len(pfpdata),'Ratio':ratio,'Full Count':len(fullcount),'Full Ratio':fullratio}
#         df = pd.DataFrame().from_dict(pdat)
#         print(tabulate(df.transpose(),headers="keys"))

