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
            todisp = todisp[todisp['First Green']==1]
            todisp = todisp[:maxheight]
            todisp = todisp.transpose()
            print(tabulate(todisp,headers="keys"))
            todisp.to_csv("analyze_highest.csv")


if len(args)>0:
    iprop = args[0].split(",")
    print("Iprop:",iprop)
    pfpdata = datas
    pdat = {}
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
        pdat[ip] = {'Profit':len(profit),'Full':len(pfpdata),'Ratio':ratio,'Full Count':len(fullcount),'Full Ratio':fullratio}
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

