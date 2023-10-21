#!/usr/bin/env python

import pandas as pd 
import os
import sys
import getopt
from datetime import datetime,timedelta
import yahooquery as yq
import numpy as np
import math
from tabulate import tabulate
from numerize import numerize
from sklearn.cluster import KMeans
from ta.trend import EMAIndicator

def green_candle(candle):
    if candle['open']<candle['close']:
        return True
    else:
        return False

def red_candle(candle):
    if candle['open']>candle['close']:
        return True
    else:
        return False

def clean_bull(first,second):   # second is higher than first
    score = 0
    if first['low']<second['low']:
        score += 1
        if first['high']<second['high']:
            score += 1
            if first['open']<second['open']:
                score += 1
                if first['close']<second['close']:
                    score += 1
    return score

def clean_bear(first,second):   # second is lower than first
    score = 0
    if first['low']>second['low']:
        score += 1
        if first['high']>second['high']:
            score += 1
            if first['open']>second['open']:
                score += 1
                if first['close']>second['close']:
                    score += 1
    return score

def bull(first,second):   # second is higher than first
    score = 0
    if first['low']<second['low']:
        score += 1
    if first['high']<second['high']:
        score += 1
    if first['open']<second['open']:
        score += 1
    if first['close']<second['close']:
        score += 1
    return score

def bear(first,second):   # second is lower than first
    score = 0
    if first['low']>second['low']:
        score += 1
    if first['high']>second['high']:
        score += 1
    if first['open']>second['open']:
        score += 1
    if first['close']>second['close']:
        score += 1
    return score

def find_bounce(candles):
    reversed_candles = candles.iloc[::-1]
    bounce = 0
    pullback = 0
    for i in range(len(reversed_candles.index)-2):
        curcandle = reversed_candles.iloc[i]
        prevcandle = reversed_candles.iloc[i+1]
        if bounce==0:     # currently in pullback mode
            if green_candle(curcandle) and bull(prevcandle,curcandle)>2:
                pullback += 1
            else:     # curcandle is red, so now in bounce mode
                bounce += 1
        else:
            if red_candle(curcandle) and bear(prevcandle,curcandle)>2:
                bounce += 1
            else:
                break
    return pullback,bounce

def find_levels(candles):
    response = "Levels:"
    min = candles['low'].min()
    max = candles['high'].max()
    p_range = candles['high'] - candles['low']
    range_avg = p_range.mean()
    vol_avg = candles['volume'].mean()
    response += "\nStart: " + str(start_date)
    response += "\nEnd: " + str(end_date)
    response += "\nMin: " + str(min)
    response += "\nMax: " + str(max)
    response += "\nRange Avg: " + str(numerize.numerize(range_avg))
    response += "\nVol Avg: " + str(numerize.numerize(vol_avg))

    datarange = max - min
    if datarange < 50:
        kint = int(datarange / 0.5)
    else:
        kint = int(datarange % 20)

    datalen = len(candles)

    highlevels = np.array(candles['high'])
    kmeans = KMeans(n_clusters=kint,n_init=10).fit(highlevels.reshape(-1,1))
    highclusters = kmeans.predict(highlevels.reshape(-1,1))

    resistancelevels = {}

    for cidx in range(datalen):
        curcluster = highclusters[cidx]
        if curcluster not in resistancelevels:
            resistancelevels[curcluster] = 1
        else:
            resistancelevels[curcluster] += 1

    donecluster = []
    finalreslevels = {}
    dresponse = ""
    for cidx in range(datalen):
        candle = candles.iloc[cidx]
        curcluster = highclusters[cidx]
        if resistancelevels[curcluster] > 2:
            if curcluster not in donecluster:
                donecluster.append(curcluster)
                finalreslevels[curcluster] = {'level':candle['high'],'count':1}
            else:
                finalreslevels[curcluster] = {'level':(finalreslevels[curcluster]['level'] + candle['high'])/2,'count':finalreslevels[curcluster]['count']+1}

    response += "\n\nResistance levels:"
    sortedreslevels = []
    for lvl,clstr in sorted(finalreslevels.items(),key=lambda x: x[1]['level']):
        sortedreslevels.append(clstr)
        response += "\n" + str(clstr['level']) + " : " + str(clstr['count'])

    if datarange < 50:
        kint = int(datarange / 0.5)
    else:
        kint = int(datarange % 20)
    lowlevels = np.array(candles['low'])
    kmeans = KMeans(n_clusters=kint,n_init=10).fit(lowlevels.reshape(-1,1))
    lowclusters = kmeans.predict(lowlevels.reshape(-1,1))

    supportlevels = {}

    for cidx in range(datalen):
        curcluster = lowclusters[cidx]
        if curcluster not in supportlevels:
            supportlevels[curcluster] = 1
        else:
            supportlevels[curcluster] += 1

    donecluster = []
    finalsuplevels = {}
    dresponse = ""
    for cidx in range(datalen):
        candle = candles.iloc[cidx]
        curcluster = lowclusters[cidx]
        if supportlevels[curcluster] > 2:
            if curcluster not in donecluster:
                donecluster.append(curcluster)
                finalsuplevels[curcluster] = {'level':candle['low'],'count':1}
            else:
                finalsuplevels[curcluster] = {'level':(finalsuplevels[curcluster]['level'] + candle['low'])/2,'count':finalsuplevels[curcluster]['count']+1}

    response += "\n\nSupport levels:"
    sortedsuplevels = []
    for lvl,clstr in sorted(finalsuplevels.items(),key=lambda x: x[1]['level']):
        sortedsuplevels.append(clstr)
        response += "\n" + str(clstr['level']) + " : " + str(clstr['count'])
    
    response += "\n\n" + dresponse
    return sortedreslevels, sortedsuplevels

def body_top(candle):
    if candle['open']>candle['close']:
        return candle['open']
    else:
        return candle['close']

def body_bottom(candle):
    if candle['open']<candle['close']:
        return candle['open']
    else:
        return candle['close']

def is_peak_body(candles,c_pos,dlen=1):
    if c_pos>0 and c_pos<len(candles)-dlen:
        before = False
        cloop = dlen
        while cloop>0 and not before:
            before = body_top(candles.iloc[c_pos])>body_top(candles.iloc[c_pos-cloop])
            cloop -= 1
        after = False
        cloop = dlen
        while cloop>0 and not after:
            after = body_top(candles.iloc[c_pos])>body_top(candles.iloc[c_pos+cloop])
            cloop -= 1
        return before and after
    else:
        return False

def is_bottom_body(candles,c_pos,dlen=1):
    if c_pos>0 and c_pos<len(candles)-dlen:
        before = False
        cloop = dlen
        while cloop>0 and not before:
            before = body_bottom(candles.iloc[c_pos])<body_bottom(candles.iloc[c_pos-cloop])
            cloop -= 1
        after = False
        cloop = dlen
        while cloop>0 and not after:
            after = body_bottom(candles.iloc[c_pos])<body_bottom(candles.iloc[c_pos+cloop])
            cloop -= 1
        return before and after
    else:
        return False

def gather_range_body(candles):
    peaks = []
    bottoms = []
    peakindex = []
    bottomindex = []
    for i in range(len(candles)):
        if is_peak_body(candles,i):
            peaks.append(candles.iloc[i])
        if is_bottom_body(candles,i):
            bottoms.append(candles.iloc[i])
    if len(peaks)==0:
        for i in range(len(candles)):
            if is_peak_body(candles,i,2) and i-1 not in peakindex:
                peaks.append(candles.iloc[i])
                peakindex.append(i)
    if len(bottoms)==0:
        for i in range(len(candles)):
            if is_bottom_body(candles,i,2) and i-1 not in bottomindex:
                bottoms.append(candles.iloc[i])
                bottomindex.append(i)

    return peaks,bottoms

def is_peak_unit(unit,candles,c_pos,dlen=1):
    if c_pos>0 and c_pos<len(candles)-dlen:
        before = False
        cloop = dlen
        while cloop>0 and not before:
            before = candles.iloc[c_pos][unit]>candles.iloc[c_pos-cloop][unit]
            cloop -= 1
        after = False
        cloop = dlen
        while cloop>0 and not after:
            after = candles.iloc[c_pos][unit]>candles.iloc[c_pos+cloop][unit]
            cloop -= 1
        return before and after
    else:
        return False

def is_bottom_unit(unit,candles,c_pos,dlen=1):
    if c_pos>0 and c_pos<len(candles)-dlen:
        before = False
        cloop = dlen
        while cloop>0 and not before:
            before = candles.iloc[c_pos][unit]<candles.iloc[c_pos-cloop][unit]
            cloop -= 1
        after = False
        cloop = dlen
        while cloop>0 and not after:
            after = candles.iloc[c_pos][unit]<candles.iloc[c_pos+cloop][unit]
            cloop -= 1
        return before and after
    else:
        return False

def gather_range_unit(unit,candles):
    peaks = []
    bottoms = []
    for i in range(len(candles)):
        if is_peak_unit(unit,candles,i):
            peaks.append(candles.iloc[i])
        if is_bottom_unit(unit,candles,i):
            bottoms.append(candles.iloc[i])
    if len(peaks)==0:
        for i in range(len(candles)):
            if is_peak_unit(unit,candles,i,2):
                peaks.append(candles.iloc[i])
    if len(bottoms)==0:
        for i in range(len(candles)):
            if is_bottom_unit(unit,candles,i,2):
                bottoms.append(candles.iloc[i])

    return peaks,bottoms

def is_peak(candles,c_pos,dlen=1):
    if c_pos>0 and c_pos<len(candles)-dlen:
        before = False
        cloop = dlen
        while cloop>0 and not before:
            before = candles.iloc[c_pos]['high']>candles.iloc[c_pos-cloop]['high']
            cloop -= 1
        after = False
        cloop = dlen
        while cloop>0 and not after:
            after = candles.iloc[c_pos]['high']>candles.iloc[c_pos+cloop]['high']
            cloop -= 1
        return before and after
    else:
        return False

def is_bottom(candles,c_pos,dlen=1):
    if c_pos>0 and c_pos<len(candles)-dlen:
        before = False
        cloop = dlen
        while cloop>0 and not before:
            before = candles.iloc[c_pos]['low']<candles.iloc[c_pos-cloop]['low']
            cloop -= 1
        after = False
        cloop = dlen
        while cloop>0 and not after:
            after = candles.iloc[c_pos]['low']<candles.iloc[c_pos+cloop]['low']
            cloop -= 1
        return before and after
    else:
        return False

def gather_range(candles):
    peaks = []
    bottoms = []
    for i in range(len(candles)):
        if is_peak(candles,i):
            peaks.append(candles.iloc[i])
        if is_bottom(candles,i):
            bottoms.append(candles.iloc[i])
    if len(peaks)==0:
        for i in range(len(candles)):
            if is_peak(candles,i,2):
                peaks.append(candles.iloc[i])
    if len(bottoms)==0:
        for i in range(len(candles)):
            if is_bottom(candles,i,2):
                bottoms.append(candles.iloc[i])

    return peaks,bottoms

def min_bottom(bottoms):
    curbottom = bottoms[0]
    for candle in bottoms:
        if candle['low'] < curbottom['low']:
            curbottom = candle
    return curbottom

def max_peak(peaks):
    curpeak = peaks[0]
    for candle in peaks:
        if candle['high'] > curpeak['high']:
            curpeak = candle
    return curpeak

inputfile = 'stocks.csv'
outfile = 'shorts.csv'
stockdate = None
openrangelimit = 1
purchaselimit = 300
completelist = False
trackunit = None
opts, args = getopt.getopt(sys.argv[1:],"i:o:d:r:p:c:u:",["input=","out=","date=","range=","purchaselimit=","complete=","unit="])
for opt, arg in opts:
    if opt in ("-i", "--input"):
        inputfile = arg
    if opt in ("-o", "--out"):
        outfile = arg
    if opt in ("-d", "--date"):
        stockdate = datetime.strptime(arg + ' 23:59:59', '%Y-%m-%d %H:%M:%S')
    if opt in ("-r", "--range"):
        openrangelimit = float(arg)
    if opt in ("-p", "--purchaselimit"):
        purchaselimit = float(arg)
    if opt in ("-c", "--complete"):
        completelist = True
    if opt in ("-u", "--unit"):
        if arg in ['close','high','low','open']:
            trackunit = arg

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
stocks = pd.read_csv(os.path.join(script_dir,inputfile),header=0)

if stockdate:
    end_date = stockdate
else:
    end_date = datetime.now()
days = 4
start_date = end_date - timedelta(days=days)
day_start_date = end_date - timedelta(days=days*10)
hour_start_date = end_date - timedelta(days=days*4)

filtered = []
highrangetickers = []
dayrangetickers = []
hourrangetickers = []
hourdetails = []

for i in range(len(stocks.index)):
    if isinstance(stocks.iloc[i]['Ticker'], str):
        ticker = stocks.iloc[i]['Ticker'].upper()
        dticker = yq.Ticker(ticker)
        candles = dticker.history(start=start_date,end=end_date,interval='15m')
        day_candles = dticker.history(start=day_start_date,end=end_date,interval='1d')
        hour_candles = dticker.history(start=hour_start_date,end=end_date,interval='1h')
        candles = candles.loc[(candles['volume']>0)]
    else:
        continue

    if len(candles.index):
        print("Processing ticker ",ticker)
        candles = candles.reset_index(level=[0,1])
        day_candles = day_candles.reset_index(level=[0,1])
        hour_candles = hour_candles.reset_index(level=[0,1])
        lastcandle = candles.iloc[-1]
        ldate = str(lastcandle['date'].date())
        bdate = str(datetime.date(lastcandle['date'])-timedelta(days=1))
        adate = str(datetime.date(lastcandle['date'])+timedelta(days=1))
        # cdcandle = candles.loc[(candles['date']>ldate) & (candles['date']<adate)]
        cdcandle = candles.loc[(candles['date']>ldate)]
        peaks,bottoms = gather_range(cdcandle)
        # day_peaks,day_bottoms = gather_range(day_candles)
        if trackunit:
            day_peaks,day_bottoms = gather_range_unit(trackunit,day_candles)
        else:
            day_peaks,day_bottoms = gather_range_body(day_candles)
        if len(day_bottoms)>2 and len(day_peaks)>2:
            last_low_higher = day_bottoms[-1]['low']>day_bottoms[-2]['low']
            second_last_low_higher = day_bottoms[-2]['low']>day_bottoms[-3]['low']
            last_high_higher = day_peaks[-1]['high']>day_peaks[-2]['high']
            second_last_high_higher = day_peaks[-2]['high']>day_peaks[-3]['high']
            if last_low_higher and second_last_low_higher and last_high_higher and second_last_high_higher:
                bottom_last = False
                if day_bottoms[-1]['date']>day_peaks[-1]['date']:
                    bottom_last = True
                closing_range = body_top(day_peaks[-1]) - body_bottom(day_bottoms[-1])
                second_closing_range = body_top(day_peaks[-2]) - body_bottom(day_bottoms[-2])
                third_closing_range = body_top(day_peaks[-3]) - body_bottom(day_bottoms[-3])
                avg_range = (closing_range+second_closing_range+third_closing_range)/3
                est_closing = body_bottom(day_bottoms[-1]) + closing_range
                est_avg = body_bottom(day_bottoms[-1]) + avg_range
                last_top = body_top(day_bottoms[-1])
                last_bottom = body_bottom(day_bottoms[-1])
                last_high = day_bottoms[-1]['high']
                last_low = day_bottoms[-1]['low']
                last_height = last_high - last_low
                bullish_tick = last_top - last_bottom < 0.1 and last_high - last_top > (last_height / 2)
                if not bullish_tick:
                    if completelist or bottom_last:
                        dayrangetickers.append({'Ticker':ticker,'Bottom Last':bottom_last,'Closing Range':closing_range,'Avg Range':avg_range,'Est Closing': est_closing,'Est Avg':est_avg,'Bottom 1':day_bottoms[-1]['date'],'Bottom 2':day_bottoms[-2]['date'],'Bottom 3':day_bottoms[-3]['date'],'Peak 1':day_peaks[-1]['date'],'Peak 2':day_peaks[-2]['date'],'Peak 3':day_peaks[-3]['date']})

        if trackunit:
            hour_peaks,hour_bottoms = gather_range_unit(trackunit,hour_candles)
        else:
            hour_peaks,hour_bottoms = gather_range_body(hour_candles)
        if len(hour_bottoms)>2 and len(hour_peaks)>2:
            last_low_higher = hour_bottoms[-1]['low']>hour_bottoms[-2]['low']
            second_last_low_higher = hour_bottoms[-2]['low']>hour_bottoms[-3]['low']
            last_high_higher = hour_peaks[-1]['high']>hour_peaks[-2]['high']
            second_last_high_higher = hour_peaks[-2]['high']>hour_peaks[-3]['high']
            if trackunit:
                last_low_higher = hour_bottoms[-1][trackunit]>hour_bottoms[-2][trackunit]
                second_last_low_higher = hour_bottoms[-2][trackunit]>hour_bottoms[-3][trackunit]
                last_high_higher = hour_peaks[-1][trackunit]>hour_peaks[-2][trackunit]
                second_last_high_higher = hour_peaks[-2][trackunit]>hour_peaks[-3][trackunit]
            else:
                last_low_higher = hour_bottoms[-1]['low']>hour_bottoms[-2]['low']
                second_last_low_higher = hour_bottoms[-2]['low']>hour_bottoms[-3]['low']
                last_high_higher = hour_peaks[-1]['high']>hour_peaks[-2]['high']
                second_last_high_higher = hour_peaks[-2]['high']>hour_peaks[-3]['high']
            valid = last_low_higher and second_last_low_higher and last_high_higher and second_last_high_higher
            if valid:
                bottom_last = False
                if hour_bottoms[-1]['date']>hour_peaks[-1]['date']:
                    bottom_last = True
                closing_range = body_top(hour_peaks[-1]) - body_bottom(hour_bottoms[-1])
                second_closing_range = body_top(hour_peaks[-2]) - body_bottom(hour_bottoms[-2])
                third_closing_range = body_top(hour_peaks[-3]) - body_bottom(hour_bottoms[-3])
                avg_range = (closing_range+second_closing_range+third_closing_range)/3
                est_closing = body_bottom(hour_bottoms[-1]) + closing_range
                est_avg = body_bottom(hour_bottoms[-1]) + avg_range
                last_top = body_top(hour_bottoms[-1])
                last_bottom = body_bottom(hour_bottoms[-1])
                last_high = hour_bottoms[-1]['high']
                last_low = hour_bottoms[-1]['low']
                last_height = last_high - last_low
                bullish_tick = last_top - last_bottom < 0.1 and last_high - last_top > (last_height / 2)
                if not bullish_tick:
                    if completelist or bottom_last:
                        hourrangetickers.append({'Ticker':ticker,'Bottom Last':bottom_last,'Closing Range':closing_range,'Avg Range':avg_range,'Est Closing': est_closing,'Est Avg':est_avg})
                        hourdetails.append({'Ticker':ticker,'Bottom 1':hour_bottoms[-1]['date'],'Bottom 2':hour_bottoms[-2]['date'],'Bottom 3':hour_bottoms[-3]['date'],'Peak 1':hour_peaks[-1]['date'],'Peak 2':hour_peaks[-2]['date'],'Peak 3':hour_peaks[-3]['date']})
                        if trackunit:
                            hourdetails.append({'Ticker':ticker,'Bottom 1':hour_bottoms[-1][trackunit],'Bottom 2':hour_bottoms[-2][trackunit],'Bottom 3':hour_bottoms[-3][trackunit],'Peak 1':hour_peaks[-1][trackunit],'Peak 2':hour_peaks[-2][trackunit],'Peak 3':hour_peaks[-3][trackunit]})
                        else:
                            hourdetails.append({'Ticker':ticker,'Bottom 1':body_bottom(hour_bottoms[-1]),'Bottom 2':body_bottom(hour_bottoms[-2]),'Bottom 3':body_bottom(hour_bottoms[-3]),'Peak 1':body_top(hour_peaks[-1]),'Peak 2':body_top(hour_peaks[-2]),'Peak 3':body_top(hour_peaks[-3])})
                    # hourrangetickers.append({'Ticker':ticker,'Bottom Last':bottom_last,'Closing Range':closing_range,'Avg Range':avg_range,'Est Closing': est_closing,'Est Avg':est_avg,'Bottom 1':hour_bottoms[-1]['date'],'Bottom 2':hour_bottoms[-2]['date'],'Bottom 3':hour_bottoms[-3]['date'],'Peak 1':hour_peaks[-1]['date'],'Peak 2':hour_peaks[-2]['date'],'Peak 3':hour_peaks[-3]['date']})

        maxpeak = minbottom = None
        if len(peaks):
            maxpeak = max_peak(peaks)
        if len(bottoms):
            # minbottom = min_bottom(bottoms)
            minbottom = bottoms[0]
        if minbottom is not None:
            openingrange = cdcandle.iloc[0]['high']-bottoms[0]['low']
            before1045 = minbottom['date'].hour<11 # or (minbottom['date'].hour==10 and minbottom['date'].minute<45)
            bottombelowstart = cdcandle.iloc[0]['high']>minbottom['high']
            bottombelowend = cdcandle.iloc[-1]['high']>minbottom['high']
            if openingrange>openrangelimit:
                print("Opening range is ============================ ",openingrange)
                if before1045 and bottombelowstart: # and bottombelowend:
                    price = cdcandle.iloc[-1]['close']
                    lastcandle = cdcandle.iloc[-1]
                    if lastcandle['date'].hour>=15 and lastcandle['date'].minute>=30:   #it is past 3.30 so consider the day closed
                        lossrange = bottoms[0]['high'] - bottoms[0]['low']
                        profitrange = cdcandle.iloc[0]['open'] - bottoms[0]['high']
                    else:
                        lossrange = price - bottoms[0]['low']
                        profitrange = cdcandle.iloc[0]['open'] - price
                    maxunit = math.floor(purchaselimit/price)
                    lossamount = lossrange * maxunit
                    profitamount = profitrange * maxunit
                    ratio = profitamount/lossamount
                    highrangetickers.append({'Ticker':ticker,'Price':price,'Loss Range':lossrange,'Profit Range':profitrange,'Unit':maxunit,'Loss Amount':lossamount,'Profit Amount':profitamount,'Ratio':ratio,'Bottom':str(bottoms[0]['date'].hour) + ':' + str(bottoms[0]['date'].minute)})
            if maxpeak is not None:
                if maxpeak['date']>minbottom['date']:
                    print("Max peak happen after min bottom")
                    print("Range:",maxpeak['high']-minbottom['low']," Min:",minbottom['date'], " Max:",maxpeak['date'], " Bottom count:",len(bottoms), " Peak count:",len(peaks))
                    if peaks[0]['date'] < bottoms[0]['date']:
                        print("First peak happen before first bottom")
                        print("Min:",bottoms[0]['date'], " Max:",peaks[0]['date'])

print("High range tickers:")
print(tabulate(highrangetickers,headers="keys"))
print("Hour range tickers:")
print(tabulate(hourrangetickers,headers="keys"))
print("Hour details:")
print(tabulate(hourdetails,headers="keys"))
print("Day range tickers:")
print(tabulate(dayrangetickers,headers="keys"))

        # candles = candles.reset_index(level=[0,1])
        # volume_multiplier = 0.5
        #
        # candles['ema10'] = EMAIndicator(close=candles['close'],window=10,fillna=True).ema_indicator()
        # candles['ema20'] = EMAIndicator(close=candles['close'],window=20,fillna=True).ema_indicator()
        # candles['tr'] = candles['high'] - candles['low']
        #
        # latestcandle = candles.iloc[-1]
        # prevcandle = candles.iloc[-2]
        # emadifflatest = latestcandle['ema10'] - latestcandle['ema20']
        # emadiffprev = prevcandle['ema10'] - latestcandle['ema20']
        # atr = candles['tr'].mean()
        #
        # if latestcandle['ema10'] > latestcandle['ema20'] and emadifflatest > emadiffprev and latestcandle['tr'] > atr and prevcandle['tr'] > atr: # and latestcandle['volume'] > candles['volume'].mean() * volume_multiplier:
        #     filtered.append(ticker)
        #     print("Ticker ",ticker," got higher ema10")
            
        # pullback,bounce = find_bounce(candles)
        # if pullback>0:
        #     filtered.append(ticker)
        #     print("Ticker ===",ticker,"=== just bounced:",bounce," pullback:",pullback)

# outdata = pd.DataFrame()
# outdata['Ticker'] = filtered
# outdata.to_csv(outfile,header=True,index=False)
