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

global_prop = []
global_fail = {}
global_negate = {}

dates = datas['date'].unique()
print("Dates:",dates)

profitable = datas[datas['profitable']==1]
propsprofitable = profitable[prop_list]
corrprofit = propsprofitable.corr()

corrprofit.index.names = ['Prop']
corrprofit.to_csv("analyze_global_corr.csv")

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
        currow['Marks'] = round(len(profitable) / len(proprows),4) + ((greatratio) + (goodratio))
        currow['GreatRatio'] = greatratio
        currow['GoodRatio'] = goodratio
        currow['Len Profitable'] = len(profitable)
        currow['Len Property'] = len(proprows)
        currow['Ratio'] = round(len(profitable)/len(proprows),4)
        currow['Median'] = round(datas['Perc ' + pl].median(),4)
        currow['Mean'] = round(datas['Perc ' + pl].mean(),4)
        currow['Min'] = round(datas['Perc ' + pl].min(),4)
        currow['Max'] = round(datas['Perc ' + pl].max(),4)
        allcp = []
        if pl not in ignore_prop:
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
                        cprow['CorrRatio'] = cprow['CorrProfit']/cprow['CorrLoss']
                        allcp.append(cprow)
        currow['All CP'] = allcp
        if pl not in reward_prop:
            fail_percent = round(len(failed) / len(proprows),2)
            # if fail_percent>0.98:
            global_fail[pl] = fail_percent
            #     global_fail[pl] *= 100
        else:
            currow['Marks'] *= 5
        # if pl in punish_prop:
        #     if pl in global_fail:
        #         global_fail[pl] *= 100
        #     else:
        #         global_fail[pl] = round(len(failed) / len(proprows),2) * 100

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


for prop in prop_list:
    sigprop = corrprofit[corrprofit[prop]>0.5]
    print("For prop ",prop)
    print(sigprop.index)
    # print(tabulate(sigprop,headers='keys'))
