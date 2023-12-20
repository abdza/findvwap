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

datas = pd.read_csv('raw_data_perc.csv')

maxheight = 20
todisp = datas.sort_values('diff_level',ascending=False)
todisp = todisp[todisp['profitable']==1]
pcount = []
for p in prop_list:
    if p not in ignore_prop:
        pcount.append({'prop':p,'count':todisp[p].sum()})
pcdf = pd.DataFrame(pcount)
pcdf.sort_values('count',inplace=True,ascending=False)
pcdf.to_csv("analyze_highest_prop.csv",index=False)

tominus = datas.sort_values('diff_level')
tominus = tominus[:maxheight]
mpcount = []
for p in prop_list:
    if p not in ignore_prop:
        mpcount.append({'prop':p,'count':tominus[p].sum()})
mpcdf = pd.DataFrame(mpcount)
mpcdf.sort_values('count',inplace=True,ascending=False)
mpcdf.to_csv("analyze_lowest_prop.csv",index=False)

global_prop = []
global_fail = {}
global_negate = {}

dates = datas['date'].unique()
print("Dates:",dates)

fieldnames = ['diff_level']
fieldnames += prop_list
for prop in prop_list:
    fieldnames.append('Perc ' + prop)
profitable = datas[datas['profitable']==1]
propsprofitable = profitable[fieldnames]
corrprofit = propsprofitable.corr()

corrprofit.index.names = ['Prop']
corrprofit.to_csv("analyze_global_corr.csv")

propsprofitable = datas[['profitable'] + fieldnames]
# propsprofitable = propsprofitable[propsprofitable['First Green']==1]
for prop in prop_list:
    propsprofitable['Profit ' + prop] = 0
    appear = propsprofitable[propsprofitable[prop]==1]
    appearprofit = appear[appear['profitable']==1]
    if len(appearprofit):
        propsprofitable.loc[appearprofit.index,'Profit ' + prop] = 1

group_by_diff = propsprofitable.groupby('diff_level').sum()
group_by_diff.reset_index(inplace=True)
group_by_diff.to_csv("analyze_global_group_numbers.csv",index=False)
group_by_diff = group_by_diff.corr()
group_by_diff.index.names = ['Prop']
group_by_diff.to_csv("analyze_global_group_corr.csv")

for pl in prop_list:
    proprows = datas[datas[pl]==1]
    if len(proprows)>0 and pl not in ignore_prop:
        profitable = proprows[proprows['profitable']==1]
        great = proprows[proprows['performance']=='Great']
        good = proprows[proprows['performance']=='Good']
        failed = proprows[proprows['profitable']==0]
        greatratio = round(len(great)/len(proprows),2)
        goodratio = round(len(good)/len(proprows),2)
        # print("For prop:",pl," Profitable rows:",len(profitable)," Total rows:",len(proprows))
        print("Great ratio:",greatratio," Good ratio:",goodratio)
        currow = {}
        currow['Prop'] = pl
        currow['Category'] = ''
        if pl in prev_prop_list:
            currow['Category'] += 'Prev'
        if pl in opening_prop_list:
            currow['Category'] += 'Opening'
        if pl in late_prop_list:
            currow['Category'] += 'Late'
        currow['Ratio'] = len(profitable)/len(proprows)
        currow['Full Ratio'] = len(proprows)/len(datas)
        currow['Marks'] = round(currow['Ratio']*currow['Full Ratio'],4)
        currow['Failed Marks'] = round(len(failed)/len(datas),4)
        currow['Multiplier'] = 0
        currow['GreatRatio'] = greatratio
        currow['GoodRatio'] = goodratio
        currow['Len Profitable'] = len(profitable)
        currow['Len Property'] = len(proprows)
        currow['Median'] = round(datas['Perc ' + pl].median(),4)
        currow['Mean'] = round(datas['Perc ' + pl].mean(),4)
        currow['Min'] = round(datas['Perc ' + pl].min(),4)
        currow['Max'] = round(datas['Perc ' + pl].max(),4)
        allcp = []
        corr = corrprofit[corrprofit[pl]>0.3]
        filtered = datas[datas['Perc ' + pl]>0.5]
        filtered = filtered[filtered[pl]==1]
        for cp in corr.index:
            if cp!=pl:
                cpfiltered = filtered[filtered[cp]==1]
                print("P:",cp," Filter size:",len(cpfiltered))
                if len(cpfiltered)>0:
                    profitfilter = cpfiltered[cpfiltered['profitable']==1]
                    cprow = {}
                    cprow['Prop'] = cp
                    cprow['CorrProfit'] = len(profitfilter)
                    lossfilter = cpfiltered[cpfiltered['profitable']==0]
                    cprow['CorrLoss'] = len(lossfilter)
                    if cprow['CorrLoss']>0:
                        cprow['CorrRatio'] = cprow['CorrProfit']/cprow['CorrLoss']
                    else:
                        cprow['CorrRatio'] = cprow['CorrProfit']
                    allcp.append(cprow)
        currow['All CP'] = allcp
        cpcount = pcdf[pcdf['prop']==pl]
        if len(cpcount)>0 and cpcount.iloc[0]['count']>10:
            print("Gapping up prop ",pl)
            currow['Marks'] += currow['Marks'] * (cpcount.iloc[0]['count'] * 10)
            currow['Multiplier'] = cpcount.iloc[0]['count']
        else:
            print("Gapping down prop ",pl)
            mcpcount = mpcdf[mpcdf['prop']==pl]
            if len(mcpcount)>0 and mcpcount.iloc[0]['count']>0:
                global_fail[pl] = currow['Failed Marks'] * (mcpcount.iloc[0]['count'] * 10)
                # currow['Marks'] -= currow['Failed Marks'] * (cpcount.iloc[0]['count'] * 10)
                # currow['Multiplier'] = cpcount.iloc[0]['count'] * -1
        # if pl in punish_prop:
        #     if pl in global_fail:
        #         global_fail[pl] *= 100
        #     else:
        #         global_fail[pl] = round(len(failed) / len(proprows),2) * 100

        if len(profitable)>0:
            failedratio = round(len(failed)/len(profitable),2)
            if failedratio>50:
                print("Putting prop into negate:",pl," Ratio:",failedratio)
                global_negate[pl] = failedratio
        global_prop.append(currow)

outdata = pd.DataFrame.from_dict(global_prop)
outdata.to_csv('analyze_global.csv',index=False)
# outdata = pd.DataFrame(list(global_negate.items()), columns=['Prop','Marks'])
# outdata.to_csv('analyze_global_negate.csv',index=False)
outdata = pd.DataFrame(list(global_fail.items()), columns=['Prop','Marks'])
outdata.to_csv('analyze_global_fail.csv',index=False)


# for prop in prop_list:
#     sigprop = corrprofit[corrprofit[prop]>0.5]
#     print("For prop ",prop)
#     print(sigprop.index)
    # print(tabulate(sigprop,headers='keys'))
