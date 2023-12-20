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

datas = pd.read_csv('raw_data_perc.csv')
opts, args = getopt.getopt(sys.argv[1:],"p:h:",["prop=","highest="])
# print("Args:",args)
# print("Opts:",opts)
if len(opts)>0:
    for opt, arg in opts:
        if opt in ("-h", "--highest"):
            maxheight = int(arg)
            todisp = datas.sort_values('diff_level',ascending=False)
            # todisp = todisp[todisp['First Green']==1]
            todisp = todisp[:maxheight]
            pcount = []
            for p in prop_list:
                if p not in ignore_prop:
                    pcount.append({'prop':p,'count':todisp[p].sum()})
            todisp = todisp.transpose()
            print(tabulate(todisp,headers="keys"))
            pcdf = pd.DataFrame(pcount)
            pcdf.sort_values('count',inplace=True,ascending=False)
            pcdf.to_csv("analyze_highest_prop.csv",index=False)
            print(pcdf)
            todisp.to_csv("analyze_highest.csv")


if len(args)>0:
    iprop = args[0].split(",")
    print("Iprop:",iprop)
    pfpdata = datas
    pdat = {}
    done = []
    for ip in iprop:
        ip = ip.strip()
        pfpdata = pfpdata[pfpdata[ip]==1]
        profit = pfpdata[pfpdata['profitable']==1]
        if len(pfpdata):
            ratio = round(len(profit) / len(pfpdata),2)
        else:
            ratio = 0
        fullcount = datas[datas[ip]==1]
        fullratio = round(len(fullcount)/len(datas),2)
        nextprop = {}
        done.append(ip)
        for p in prop_list:
            if p not in ignore_prop and p not in done:
                nextdat = pfpdata[pfpdata[p]==1]
                nextprof = nextdat[nextdat['profitable']==1]
                nextprop[p]=len(nextprof)
        nextprop = sorted(nextprop.items(),key=lambda x:x[1],reverse=True)
        pdat[ip] = {'Profit':len(profit),'Full':len(pfpdata),'Ratio':ratio,'Full Count':len(fullcount),'Full Ratio':fullratio,'Next Prop':'\n'.join([ k + ':' + str(v) for k,v in nextprop[:5]])}
    df = pd.DataFrame().from_dict(pdat)
    print(tabulate(df.transpose(),headers="keys"))
else:
    if len(opts)==0:
        pdat = {}
        fpdata = datas[datas['First Green']==1]
        for ip in prev_prop_list:
            ip = ip.strip()
            pfpdata = fpdata[fpdata[ip]==1]
            profit = pfpdata[pfpdata['profitable']==1]
            if len(pfpdata):
                ratio = round(len(profit) / len(pfpdata),2)
            else:
                ratio = 0
            fullcount = datas[datas[ip]==1]
            fullratio = round(len(fullcount)/len(datas),2)
            pdat[ip] = {'Profit':len(profit),'Full':len(pfpdata),'Ratio':ratio,'Full Count':len(fullcount),'Full Ratio':fullratio}
        df = pd.DataFrame().from_dict(pdat)
        print(tabulate(df.transpose(),headers="keys"))

        pdat = {}
        fpdata = datas[datas['First Green']==1]
        for ip in opening_prop_list:
            ip = ip.strip()
            pfpdata = fpdata[fpdata[ip]==1]
            profit = pfpdata[pfpdata['profitable']==1]
            if len(pfpdata):
                ratio = round(len(profit) / len(pfpdata),2)
            else:
                ratio = 0
            fullcount = datas[datas[ip]==1]
            fullratio = round(len(fullcount)/len(datas),2)
            pdat[ip] = {'Profit':len(profit),'Full':len(pfpdata),'Ratio':ratio,'Full Count':len(fullcount),'Full Ratio':fullratio}
        df = pd.DataFrame().from_dict(pdat)
        print(tabulate(df.transpose(),headers="keys"))

        pdat = {}
        fpdata = datas[datas['First Green']==1]
        for ip in late_prop_list:
            ip = ip.strip()
            pfpdata = fpdata[fpdata[ip]==1]
            profit = pfpdata[pfpdata['profitable']==1]
            if len(pfpdata):
                ratio = round(len(profit) / len(pfpdata),2)
            else:
                ratio = 0
            fullcount = datas[datas[ip]==1]
            fullratio = round(len(fullcount)/len(datas),2)
            pdat[ip] = {'Profit':len(profit),'Full':len(pfpdata),'Ratio':ratio,'Full Count':len(fullcount),'Full Ratio':fullratio}
        df = pd.DataFrame().from_dict(pdat)
        print(tabulate(df.transpose(),headers="keys"))

