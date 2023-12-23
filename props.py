import numpy as np
import pandas as pd 
import os
from numerize import numerize
from sklearn.cluster import KMeans

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

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
'Yesterday Negative Morning Range',
'2 Days Ago Negative Morning Range',
'Consecutive Negative Morning Range',
'Yesterday Positive Morning Range',
'2 Days Ago Positive Morning Range',
'Consecutive Positive Morning Range',
'Yesterday Negative Afternoon Range',
'2 Days Ago Negative Afternoon Range',
'Consecutive Negative Afternoon Range',
'Yesterday Positive Afternoon Range',
'2 Days Ago Positive Afternoon Range',
'Consecutive Positive Afternoon Range',
'Yesterday Morning Range Larger',
'2 Days Ago Morning Range Larger',
'Consecutive Morning Range Larger',
'Yesterday Afternoon Range Larger',
'2 Days Ago Afternoon Range Larger',
'Consecutive Afternoon Range Larger',
    ]


late_prop_list = [
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
'FVG First',
'FVG Second',
'Higher High',
'Higher Low',
'Lower High',
'Lower Low',
'Negative FVG First',
'Negative FVG Second',
'Negative Volume Gap First',
'Negative Volume Gap Second',
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
'Volume Gap First',
'Volume Gap Second',
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
'Second Tiny Range',
'Third Tiny Range',
'Consecutive Early Tiny Range',
'Consecutive Late Tiny Range',
'Consecutive Tiny Range',
'Second Huge Range',
'Third Huge Range',
'Consecutive Early Huge Range',
'Consecutive Late Huge Range',
'Consecutive Huge Range',
'Second Huge Negative Range',
'Third Huge Negative Range',
'Consecutive Early Huge Negative Range',
'Consecutive Late Huge Negative Range',
'Consecutive Huge Negative Range',
'Late Start',
    ]

opening_prop_list = [
'First Green',
'First Hammer',
'First Red',
'First Reverse Hammer',
'Gap Down Above Average',
'Gap Down Above 2 Day Average',
'Gap Down Below Prev Min',
'Gap Down',
'Gap Up Above Average',
'Gap Up Above 2 Day Average',
'Gap Up Above Prev Max',
'Gap Up',
'Open Higher Than 2 Prev Max',
'Open Higher Than Prev Max Plus Average',
'Open Higher Than Prev Max',
'Open Lower Than 2 Prev Max',
'Open Lower Than Prev Min Minus Average',
'Open Lower Than Prev Min',
'Range Above 2 Day Average',
'Range Above Average',
'Range Lower 2 Day Average',
'Range Lower Average',
'Range More Than Gap Down',
'Range More Than Gap Up',
'Volume Higher Than Average',
'Volume Lower Than Average',
'Volume Open Higher',
'Volume Open Lower',
'Volume Open Excedingly High',
'Volume Open Excedingly Low',
'Volume Open After High',
'Volume Open After Low',
'Tiny Range',
'Huge Range',
'Huge Negative Range',
'Volume Above 5 Time Average',
'Volume Above 10 Time Average',
'Volume Above 5 Time Before Average',
'Volume Above 10 Time Before Average',
'Volume Consecutive Above 5 Time Average',
'Volume Consecutive Above 10 Time Average',
    ]

prev_prop_list = [
'Yesterday End In Red',
'Yesterday End Volume Above Average',
'New IPO',
'Fairly New IPO',
'Sluggish Ticker',
'Continue Sluggish Ticker',
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
'Yesterday Negative Morning Range',
'2 Days Ago Negative Morning Range',
'Consecutive Negative Morning Range',
'Yesterday Positive Morning Range',
'2 Days Ago Positive Morning Range',
'Consecutive Positive Morning Range',
'Yesterday Negative Afternoon Range',
'2 Days Ago Negative Afternoon Range',
'Consecutive Negative Afternoon Range',
'Yesterday Positive Afternoon Range',
'2 Days Ago Positive Afternoon Range',
'Consecutive Positive Afternoon Range',
'Yesterday Morning Range Larger',
'2 Days Ago Morning Range Larger',
'Consecutive Morning Range Larger',
'Yesterday Afternoon Range Larger',
'2 Days Ago Afternoon Range Larger',
'Consecutive Afternoon Range Larger',
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

summary_prop_list = [
'Big Reverse',
'Bottom After Noon',
'Bottom Before Noon',
'Bottom Lunch',
'Peak After Noon',
'Peak Before Noon',
'Peak Lunch',
'Two Small Reverse',
'Max After Min',
'Min After Max',
    ]

punish_prop = [
'Sluggish Ticker',
'Continue Sluggish Ticker',
'Limp Second Diff',
'Limp Third Diff',
'Consecutive Limp Diff',
'Late Start',
'First Red',
]

reward_prop = [
'Huge Range',
'Higher Low',
'Range Above Average',
'Second Short',
'Volume Higher Than Average',
'Volume Open Higher',
'Third Range Shorter',
'Second Volume Lower',
'Third Volume Lower',
'Gap Up',
'Lower High',
'Open Higher Than 2 Prev Max',
'Range More Than Gap Up',
'Second Red',
'Third Red',
'Third Short',
'Second Range Shorter',
'Consecutive Lower Volume',
'Limp Second Diff',
'Limp Third Diff',
'Third Huge Negative Range',
'Yesterday Status Good',
'Yesterday Profitable',
'2 Days Ago Status Fair',
]

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

def range_avg(candles):
    p_range = candles['high'] - candles['low']
    return p_range.mean()

def find_levels(candles):
    start_date = candles.iloc[0]['date']
    end_date = candles.iloc[-1]['date']
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
    resistancelevels = {}
    donecluster = []
    finalreslevels = {}
    dresponse = ""
    try:
        kmeans = KMeans(n_clusters=kint,n_init=10).fit(highlevels.reshape(-1,1))
        highclusters = kmeans.predict(highlevels.reshape(-1,1))
        for cidx in range(datalen):
            curcluster = highclusters[cidx]
            if curcluster not in resistancelevels:
                resistancelevels[curcluster] = 1
            else:
                resistancelevels[curcluster] += 1
        for cidx in range(datalen):
            candle = candles.iloc[cidx]
            curcluster = highclusters[cidx]
            if resistancelevels[curcluster] > 2:
                if curcluster not in donecluster:
                    donecluster.append(curcluster)
                    finalreslevels[curcluster] = {'level':candle['high'],'count':1}
                else:
                    finalreslevels[curcluster] = {'level':(finalreslevels[curcluster]['level'] + candle['high'])/2,'count':finalreslevels[curcluster]['count']+1}
    except:
        print("Got error for levels")

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
    supportlevels = {}
    donecluster = []
    finalsuplevels = {}
    dresponse = ""
    try:
        kmeans = KMeans(n_clusters=kint,n_init=10).fit(lowlevels.reshape(-1,1))
        lowclusters = kmeans.predict(lowlevels.reshape(-1,1))
        for cidx in range(datalen):
            curcluster = lowclusters[cidx]
            if curcluster not in supportlevels:
                supportlevels[curcluster] = 1
            else:
                supportlevels[curcluster] += 1
        for cidx in range(datalen):
            candle = candles.iloc[cidx]
            curcluster = lowclusters[cidx]
            if supportlevels[curcluster] > 2:
                if curcluster not in donecluster:
                    donecluster.append(curcluster)
                    finalsuplevels[curcluster] = {'level':candle['low'],'count':1}
                else:
                    finalsuplevels[curcluster] = {'level':(finalsuplevels[curcluster]['level'] + candle['low'])/2,'count':finalsuplevels[curcluster]['count']+1}
    except:
        print("Got error for levels")

    response += "\n\nSupport levels:"
    sortedsuplevels = []
    for lvl,clstr in sorted(finalsuplevels.items(),key=lambda x: x[1]['level']):
        sortedsuplevels.append(clstr)
        response += "\n" + str(clstr['level']) + " : " + str(clstr['count'])
    
    response += "\n\n" + dresponse
    # return sortedreslevels, sortedsuplevels
    return sortedsuplevels

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

def body_length(candle):
    return body_top(candle) - body_bottom(candle)

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
    if c_pos==0 and len(candles)>1:
        after = candles.iloc[c_pos]['high']>candles.iloc[c_pos+1]['high']
        return after
    elif c_pos>0 and c_pos<len(candles)-dlen:
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
    elif c_pos==len(candles)-dlen:
        before = candles.iloc[c_pos]['high']>candles.iloc[c_pos-1]['high']
        return before
    else:
        return False

def is_bottom(candles,c_pos,dlen=1):
    if c_pos==0 and len(candles)>1:
        after = candles.iloc[c_pos]['low']<candles.iloc[c_pos+1]['low']
        return after
    elif c_pos>0 and c_pos<len(candles)-dlen:
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
    elif c_pos==len(candles)-dlen:
        before = candles.iloc[c_pos]['low']<candles.iloc[c_pos-1]['low']
        return before
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

def min_bottom(bottoms,exclude=None):
    curbottom = None
    for candle in bottoms:
        goon = True
        if exclude is not None:
            for intest in exclude:
                if intest['date']==candle['date']:
                    goon = False
        if curbottom is None and goon:
            curbottom = candle
        else:
            if goon and candle['low'] < curbottom['low']:
                curbottom = candle
    return curbottom

def max_peak(peaks,exclude=None):
    curpeak = None
    for candle in peaks:
        goon = True
        if exclude is not None:
            for intest in exclude:
                if intest['date']==candle['date']:
                    goon = False
        if curpeak is None and goon:
            curpeak = candle
        else:
            if goon and candle['high'] > curpeak['high']:
                curpeak = candle
    return curpeak

def minute_test(peaks,bottoms):
    if len(bottoms)>0 and len(peaks)>0:
        if bottoms[0]['date']<peaks[0]['date']:
            return True
    return False

minprofit = 0.5

def profit_test(peaks,bottoms):
    if len(bottoms)>0 and len(peaks)>0:
        maxp = max_peak(peaks)
        minb = min_bottom(bottoms)
        if minb['date']<maxp['date'] and maxp['high'] - minb['low']>minprofit:
            return maxp['high'] - minb['low']
    return False

def minute_profit_test(peaks,bottoms):
    if len(bottoms)>0 and len(peaks)>0:
        maxp = max_peak(peaks)
        minb = min_bottom(bottoms)
        if bottoms[0]['date']<peaks[0]['date'] and minb['date']<maxp['date'] and maxp['high'] - minb['low']>minprofit:
            return maxp['high'] - minb['low']
    return False

def hammer_pattern(candle):
    max_range = candle['range']/2.5
    return body_length(candle) < max_range and body_bottom(candle) > candle['low'] + max_range

def reverse_hammer_pattern(candle):
    max_range = candle['range']/2.5
    return body_length(candle) < max_range and body_top(candle) < candle['low'] + max_range

def pattern_test(today):
    minrange = 0.15
    candle1 = today.iloc[0]
    if red_candle(candle1):
        return False
    either_green = True
    if len(today)>2:
        either_green = green_candle(today.iloc[1]) or green_candle(today.iloc[2])
    first_hammer = hammer_pattern(candle1)
    if not (either_green or first_hammer):
        return False
    if len(today)>1:
        if reverse_hammer_pattern(today.iloc[1]):
            return False
        if today.iloc[1]['range'] < body_length(today.iloc[0])/3:
            return False
    if len(today)>2:
        if candle1['range']<minrange and today.iloc[1]['range']<minrange and today.iloc[2]['range']<minrange:
            return False
        for i in range(2):
            if body_length(today.iloc[i+1]) > body_length(today.iloc[i]) * 0.7 and red_candle(today.iloc[i+1]):
                return False
        if len(today)>3:
            if body_top(today.iloc[3]) < body_bottom(today.iloc[2]):
                return False
            for i in range(3):
                if today.iloc[i+1]['high']<today.iloc[i]['low']:
                    return False
                if today.iloc[i+1]['range']>today.iloc[i]['range']*0.75 and red_candle(today.iloc[i+1]):
                    return False
            if len(today)>4:
                for i in range(4):
                    if red_candle(today.iloc[i]) and red_candle(today.iloc[i+1]):
                        return False
    return True

def prev_avg_test(today,yesterday,target_multiple=4):
    y_avg = range_avg(yesterday)
    if len(today)>2:
        for i in range(2):
            if today.iloc[i]['range']>y_avg*target_multiple:
                return True
        early_range = today.iloc[1]['high'] - today.iloc[0]['low']
        if early_range>y_avg*target_multiple:
            return True
        late_range = today.iloc[2]['high'] - today.iloc[0]['low']
        if late_range>y_avg*target_multiple:
            return True
    return False

def append_hash_set(hashdata,key,value):
    if not key in hashdata:
        hashdata[key] = []
        hashdata[key].append(value)
    else:
        if not value in hashdata[key]:
            hashdata[key].append(value)
    return hashdata

def set_params(ticker,proptext,prop_data,tickers_data,all_props):
    if not proptext in prop_list:
        raise Exception("Proptext '" + proptext + "' does not exists in prop_list")
    prop_data = append_hash_set(prop_data,proptext,ticker)
    tickers_data = append_hash_set(tickers_data,ticker,proptext)
    all_props.append(proptext)
    return prop_data, tickers_data, all_props

def analyze_minute(ticker,minute_candles,bminute_candles,bbminute_candles):
    prop_data = {}
    tickers_data = {}
    all_props = []
    maxprice = minute_candles.iloc[0]['high']
    if len(minute_candles)>1:
        first_price = body_top(minute_candles.iloc[1])
    else:
        first_price = minute_candles.iloc[0]['open']
    peaks,bottoms = gather_range(minute_candles)
    start_late = minute_candles.iloc[0]['date'].hour!=9 and minute_candles.iloc[0]['date'].minute!=30
    zero_size = minute_candles.iloc[0]['body_length']==0
    if start_late or zero_size:
        prop_data, tickers_data, all_props = set_params(ticker,'Late Start',prop_data,tickers_data,all_props)
    if green_candle(minute_candles.iloc[0]):
        prop_data, tickers_data, all_props = set_params(ticker,'First Green',prop_data,tickers_data,all_props)
    if len(minute_candles)>1 and green_candle(minute_candles.iloc[1]):
        prop_data, tickers_data, all_props = set_params(ticker,'Second Green',prop_data,tickers_data,all_props)
        if 'First Green' in tickers_data[ticker]:
            prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Early Green',prop_data,tickers_data,all_props)
    if len(minute_candles)>2 and green_candle(minute_candles.iloc[2]):
        prop_data, tickers_data, all_props = set_params(ticker,'Third Green',prop_data,tickers_data,all_props)
        if 'Consecutive Early Green' in tickers_data[ticker]:
            prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Green',prop_data,tickers_data,all_props)
        elif 'Second Green' in tickers_data[ticker]:
            prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Late Green',prop_data,tickers_data,all_props)
    if red_candle(minute_candles.iloc[0]):
        prop_data, tickers_data, all_props = set_params(ticker,'First Red',prop_data,tickers_data,all_props)
    if len(minute_candles)>1 and red_candle(minute_candles.iloc[1]):
        prop_data, tickers_data, all_props = set_params(ticker,'Second Red',prop_data,tickers_data,all_props)
        if 'First Red' in tickers_data[ticker]:
            prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Early Red',prop_data,tickers_data,all_props)
    if len(minute_candles)>2 and red_candle(minute_candles.iloc[2]):
        prop_data, tickers_data, all_props = set_params(ticker,'Third Red',prop_data,tickers_data,all_props)
        if 'Consecutive Early Red' in tickers_data[ticker]:
            prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Red',prop_data,tickers_data,all_props)
        elif 'Second Red' in tickers_data[ticker]:
            prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Late Red',prop_data,tickers_data,all_props)
    if len(bottoms)>0:
        minb = min_bottom(bottoms,[minute_candles.iloc[0]])
        if minb is not None:
            if str(minb['date'].time())<'12:00:00':
                prop_data, tickers_data, all_props = set_params(ticker,'Bottom Before Noon',prop_data,tickers_data,all_props)
            elif str(minb['date'].time())>'13:00:00':
                prop_data, tickers_data, all_props = set_params(ticker,'Bottom After Noon',prop_data,tickers_data,all_props)
            else:
                prop_data, tickers_data, all_props = set_params(ticker,'Bottom Lunch',prop_data,tickers_data,all_props)
    if len(peaks)>0:
        maxp = max_peak(peaks,[minute_candles.iloc[0]])
        minp = min_bottom(bottoms,[minute_candles.iloc[0]])
        if maxp is not None:
            if minp is not None:
                if maxp['date']>minp['date']:
                    prop_data, tickers_data, all_props = set_params(ticker,'Max After Min',prop_data,tickers_data,all_props)
                else:
                    prop_data, tickers_data, all_props = set_params(ticker,'Min After Max',prop_data,tickers_data,all_props)
            max_price = maxp['high']
            if str(maxp['date'].time())<'12:00:00':
                prop_data, tickers_data, all_props = set_params(ticker,'Peak Before Noon',prop_data,tickers_data,all_props)
            elif str(maxp['date'].time())>'13:00:00':
                prop_data, tickers_data, all_props = set_params(ticker,'Peak After Noon',prop_data,tickers_data,all_props)
            else:
                prop_data, tickers_data, all_props = set_params(ticker,'Peak Lunch',prop_data,tickers_data,all_props)
        else:
            max_price = minute_candles.iloc[-1]['high']
        if len(minute_candles)>2:
            for i in range(len(minute_candles)):
                if red_candle(minute_candles.iloc[i]) and green_candle(minute_candles.iloc[i-1]) and minute_candles.iloc[i]['range'] > minute_candles.iloc[i-1]['range']*0.6:
                    prop_data, tickers_data, all_props = set_params(ticker,'Big Reverse',prop_data,tickers_data,all_props)
                if red_candle(minute_candles.iloc[i]) and red_candle(minute_candles.iloc[i-1]) and green_candle(minute_candles.iloc[i-2]) and minute_candles.iloc[i]['range'] + minute_candles.iloc[i-1]['range'] > minute_candles.iloc[i-2]['range']*0.6:
                    prop_data, tickers_data, all_props = set_params(ticker,'Two Small Reverse',prop_data,tickers_data,all_props)

    else:
        max_price = minute_candles.iloc[-1]['high']

    if len(minute_candles)>1:
        if minute_candles.iloc[0]['low']<minute_candles.iloc[1]['low']:
            prop_data, tickers_data, all_props = set_params(ticker,'Higher Low',prop_data,tickers_data,all_props)
            if len(minute_candles)>2:
                if minute_candles.iloc[1]['low']<minute_candles.iloc[2]['low']:
                    prop_data, tickers_data, all_props = set_params(ticker,'Continue Higher Low',prop_data,tickers_data,all_props)
        if minute_candles.iloc[0]['low']>minute_candles.iloc[1]['low']:
            prop_data, tickers_data, all_props = set_params(ticker,'Lower Low',prop_data,tickers_data,all_props)
            if len(minute_candles)>2:
                if minute_candles.iloc[1]['low']>minute_candles.iloc[2]['low']:
                    prop_data, tickers_data, all_props = set_params(ticker,'Continue Lower Low',prop_data,tickers_data,all_props)
        if minute_candles.iloc[0]['high']<minute_candles.iloc[1]['high']:
            prop_data, tickers_data, all_props = set_params(ticker,'Higher High',prop_data,tickers_data,all_props)
            if len(minute_candles)>2:
                if minute_candles.iloc[1]['high']<minute_candles.iloc[2]['high']:
                    prop_data, tickers_data, all_props = set_params(ticker,'Continue Higher High',prop_data,tickers_data,all_props)
        if minute_candles.iloc[0]['high']>minute_candles.iloc[1]['high']:
            prop_data, tickers_data, all_props = set_params(ticker,'Lower High',prop_data,tickers_data,all_props)
            if len(minute_candles)>2:
                if minute_candles.iloc[1]['high']>minute_candles.iloc[2]['high']:
                    prop_data, tickers_data, all_props = set_params(ticker,'Continue Lower High',prop_data,tickers_data,all_props)
        if minute_candles.iloc[0]['body_length']*0.4<minute_candles.iloc[1]['body_length']:
            prop_data, tickers_data, all_props = set_params(ticker,'Second Long',prop_data,tickers_data,all_props)
        if minute_candles.iloc[0]['body_length']*0.2>minute_candles.iloc[1]['body_length']:
            prop_data, tickers_data, all_props = set_params(ticker,'Second Short',prop_data,tickers_data,all_props)
        if minute_candles.iloc[1]['range']<0.05:
            prop_data, tickers_data, all_props = set_params(ticker,'Second Tiny Range',prop_data,tickers_data,all_props)
            if 'Tiny Range' in tickers_data[ticker]:
                prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Early Tiny Range',prop_data,tickers_data,all_props)
        if minute_candles.iloc[1]['range']>0.3 and green_candle(minute_candles.iloc[1]):
            prop_data, tickers_data, all_props = set_params(ticker,'Second Huge Range',prop_data,tickers_data,all_props)
            if 'Huge Range' in tickers_data[ticker]:
                prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Early Huge Range',prop_data,tickers_data,all_props)
        if minute_candles.iloc[1]['range']>0.3 and red_candle(minute_candles.iloc[1]):
            prop_data, tickers_data, all_props = set_params(ticker,'Second Huge Negative Range',prop_data,tickers_data,all_props)
            if 'Huge Negative Range' in tickers_data[ticker]:
                prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Early Huge Negative Range',prop_data,tickers_data,all_props)
        if minute_candles.iloc[0]['range']<minute_candles.iloc[1]['range']:
            prop_data, tickers_data, all_props = set_params(ticker,'Second Range Longer',prop_data,tickers_data,all_props)
            if minute_candles.iloc[0]['range']*2<minute_candles.iloc[1]['range']:
                prop_data, tickers_data, all_props = set_params(ticker,'Second Range Very Longer',prop_data,tickers_data,all_props)
        else:
            prop_data, tickers_data, all_props = set_params(ticker,'Second Range Shorter',prop_data,tickers_data,all_props)
            if minute_candles.iloc[0]['range']>minute_candles.iloc[1]['range']*2:
                prop_data, tickers_data, all_props = set_params(ticker,'Second Range Very Shorter',prop_data,tickers_data,all_props)
        if minute_candles.iloc[0]['volume']>minute_candles.iloc[1]['volume']:
                prop_data, tickers_data, all_props = set_params(ticker,'Second Volume Lower',prop_data,tickers_data,all_props)
        else:
                prop_data, tickers_data, all_props = set_params(ticker,'Second Volume Higher',prop_data,tickers_data,all_props)
        if minute_candles.iloc[1]['high']-minute_candles.iloc[0]['high']<0.20:
                prop_data, tickers_data, all_props = set_params(ticker,'Limp Second Diff',prop_data,tickers_data,all_props)
        if hammer_pattern(minute_candles.iloc[1]):
            prop_data, tickers_data, all_props = set_params(ticker,'Second Hammer',prop_data,tickers_data,all_props)
        if reverse_hammer_pattern(minute_candles.iloc[1]):
            prop_data, tickers_data, all_props = set_params(ticker,'Second Reverse Hammer',prop_data,tickers_data,all_props)
        if body_bottom(minute_candles.iloc[1])-body_top(minute_candles.iloc[0])>0.1:
            prop_data, tickers_data, all_props = set_params(ticker,'FVG First',prop_data,tickers_data,all_props)
        if minute_candles.iloc[1]['low']-minute_candles.iloc[0]['high']>0.1:
            prop_data, tickers_data, all_props = set_params(ticker,'Volume Gap First',prop_data,tickers_data,all_props)
        if abs(minute_candles.iloc[1]['high'] - minute_candles.iloc[0]['high']) < 0.05:
            prop_data, tickers_data, all_props = set_params(ticker,'Early Top Level',prop_data,tickers_data,all_props)
        if body_bottom(minute_candles.iloc[0])-body_top(minute_candles.iloc[1])>0.1:
            prop_data, tickers_data, all_props = set_params(ticker,'Negative FVG First',prop_data,tickers_data,all_props)
        if minute_candles.iloc[0]['low']-minute_candles.iloc[1]['high']>0.1:
            prop_data, tickers_data, all_props = set_params(ticker,'Negative Volume Gap First',prop_data,tickers_data,all_props)

    if len(minute_candles)>2:
        if minute_candles.iloc[1]['body_length']*0.4<minute_candles.iloc[2]['body_length']:
            prop_data, tickers_data, all_props = set_params(ticker,'Third Long',prop_data,tickers_data,all_props)
        if minute_candles.iloc[1]['body_length']*0.2>minute_candles.iloc[2]['body_length']:
            prop_data, tickers_data, all_props = set_params(ticker,'Third Short',prop_data,tickers_data,all_props)
        if minute_candles.iloc[2]['range']<0.05:
            prop_data, tickers_data, all_props = set_params(ticker,'Third Tiny Range',prop_data,tickers_data,all_props)
            if 'Second Tiny Range' in tickers_data[ticker]:
                prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Late Tiny Range',prop_data,tickers_data,all_props)
            if 'Consecutive Early Tiny Range' in tickers_data[ticker] and 'Consecutive Late Tiny Range' in tickers_data[ticker]:
                prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Tiny Range',prop_data,tickers_data,all_props)
        if minute_candles.iloc[2]['range']>0.3 and green_candle(minute_candles.iloc[2]):
            prop_data, tickers_data, all_props = set_params(ticker,'Third Huge Range',prop_data,tickers_data,all_props)
            if 'Second Huge Range' in tickers_data[ticker]:
                prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Late Huge Range',prop_data,tickers_data,all_props)
            if 'Consecutive Early Huge Range' in tickers_data[ticker] and 'Consecutive Late Huge Range' in tickers_data[ticker]:
                prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Huge Range',prop_data,tickers_data,all_props)
        if minute_candles.iloc[2]['range']>0.3 and red_candle(minute_candles.iloc[2]):
            prop_data, tickers_data, all_props = set_params(ticker,'Third Huge Negative Range',prop_data,tickers_data,all_props)
            if 'Second Huge Negative Range' in tickers_data[ticker]:
                prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Late Huge Negative Range',prop_data,tickers_data,all_props)
            if 'Consecutive Early Huge Negative Range' in tickers_data[ticker] and 'Consecutive Late Huge Negative Range' in tickers_data[ticker]:
                prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Huge Negative Range',prop_data,tickers_data,all_props)
        if minute_candles.iloc[1]['range']<minute_candles.iloc[2]['range']:
            prop_data, tickers_data, all_props = set_params(ticker,'Third Range Longer',prop_data,tickers_data,all_props)
            if minute_candles.iloc[1]['range']*2<minute_candles.iloc[2]['range']:
                prop_data, tickers_data, all_props = set_params(ticker,'Third Range Very Longer',prop_data,tickers_data,all_props)
        else:
            prop_data, tickers_data, all_props = set_params(ticker,'Third Range Shorter',prop_data,tickers_data,all_props)
            if minute_candles.iloc[1]['range']>minute_candles.iloc[2]['range']*2:
                prop_data, tickers_data, all_props = set_params(ticker,'Third Range Very Shorter',prop_data,tickers_data,all_props)
        if minute_candles.iloc[1]['volume']>minute_candles.iloc[2]['volume']:
                prop_data, tickers_data, all_props = set_params(ticker,'Third Volume Lower',prop_data,tickers_data,all_props)
                if 'Second Volume Lower' in tickers_data[ticker]:
                    prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Lower Volume',prop_data,tickers_data,all_props)
        else:
                prop_data, tickers_data, all_props = set_params(ticker,'Third Volume Higher',prop_data,tickers_data,all_props)
                if 'Second Volume Higher' in tickers_data[ticker]:
                    prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Higher Volume',prop_data,tickers_data,all_props)
        if minute_candles.iloc[2]['high']-minute_candles.iloc[1]['high']<0.20:
                prop_data, tickers_data, all_props = set_params(ticker,'Limp Third Diff',prop_data,tickers_data,all_props)
                if 'Limp Second Diff' in tickers_data[ticker]:
                    prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Limp Diff',prop_data,tickers_data,all_props)
        if hammer_pattern(minute_candles.iloc[2]):
            prop_data, tickers_data, all_props = set_params(ticker,'Third Hammer',prop_data,tickers_data,all_props)
        if reverse_hammer_pattern(minute_candles.iloc[2]):
            prop_data, tickers_data, all_props = set_params(ticker,'Third Reverse Hammer',prop_data,tickers_data,all_props)
        if abs(minute_candles.iloc[2]['high'] - minute_candles.iloc[1]['high']) < 0.05:
            prop_data, tickers_data, all_props = set_params(ticker,'Late Top Level',prop_data,tickers_data,all_props)
            if 'Early Top Level' in tickers_data[ticker]:
                prop_data, tickers_data, all_props = set_params(ticker,'Top Level',prop_data,tickers_data,all_props)
        if body_bottom(minute_candles.iloc[2])-body_top(minute_candles.iloc[1])>0.1:
            prop_data, tickers_data, all_props = set_params(ticker,'FVG Second',prop_data,tickers_data,all_props)
            if 'FVG First' in tickers_data[ticker]:
                prop_data, tickers_data, all_props = set_params(ticker,'Consecutive FVG',prop_data,tickers_data,all_props)
        if minute_candles.iloc[2]['low']- minute_candles.iloc[1]['high']>0.1:
            prop_data, tickers_data, all_props = set_params(ticker,'Volume Gap Second',prop_data,tickers_data,all_props)
            if 'Volume Gap First' in tickers_data[ticker]:
                prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Volume Gap',prop_data,tickers_data,all_props)
        if body_bottom(minute_candles.iloc[1])-body_top(minute_candles.iloc[2])>0.1:
            prop_data, tickers_data, all_props = set_params(ticker,'Negative FVG Second',prop_data,tickers_data,all_props)
            if 'Negative FVG First' in tickers_data[ticker]:
                prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Negative FVG',prop_data,tickers_data,all_props)
        if minute_candles.iloc[1]['low']- minute_candles.iloc[2]['high']>0.1:
            prop_data, tickers_data, all_props = set_params(ticker,'Negative Volume Gap Second',prop_data,tickers_data,all_props)
            if 'Negative Volume Gap First' in tickers_data[ticker]:
                prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Negative Volume Gap',prop_data,tickers_data,all_props)
        if 'Second Range Shorter' in tickers_data[ticker] and 'Third Range Shorter' in tickers_data[ticker]:
            prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Shorter Range',prop_data,tickers_data,all_props)
        if 'Second Range Longer' in tickers_data[ticker] and 'Third Range Longer' in tickers_data[ticker]:
            prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Longer Range',prop_data,tickers_data,all_props)
        if 'Second Range Very Shorter' in tickers_data[ticker] and 'Third Range Very Shorter' in tickers_data[ticker]:
            prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Very Shorter Range',prop_data,tickers_data,all_props)
        if 'Second Range Very Longer' in tickers_data[ticker] and 'Third Range Very Longer' in tickers_data[ticker]:
            prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Very Longer Range',prop_data,tickers_data,all_props)

    if hammer_pattern(minute_candles.iloc[0]):
        prop_data, tickers_data, all_props = set_params(ticker,'First Hammer',prop_data,tickers_data,all_props)
    if reverse_hammer_pattern(minute_candles.iloc[0]):
        prop_data, tickers_data, all_props = set_params(ticker,'First Reverse Hammer',prop_data,tickers_data,all_props)
    if minute_candles.iloc[0]['range']<0.05:
        prop_data, tickers_data, all_props = set_params(ticker,'Tiny Range',prop_data,tickers_data,all_props)
    if minute_candles.iloc[0]['range']>0.3 and green_candle(minute_candles.iloc[0]):
        prop_data, tickers_data, all_props = set_params(ticker,'Huge Range',prop_data,tickers_data,all_props)
    if minute_candles.iloc[0]['range']>0.3 and red_candle(minute_candles.iloc[0]):
        prop_data, tickers_data, all_props = set_params(ticker,'Huge Negative Range',prop_data,tickers_data,all_props)

    if len(bminute_candles)>0 and len(bminute_candles)<20:
        prop_data, tickers_data, all_props = set_params(ticker,'Sluggish Ticker',prop_data,tickers_data,all_props)
        if len(bbminute_candles)>0 and len(bbminute_candles)<20:
            prop_data, tickers_data, all_props = set_params(ticker,'Continue Sluggish Ticker',prop_data,tickers_data,all_props)

    if len(bminute_candles)==0 and len(bbminute_candles)==0:
        prop_data, tickers_data, all_props = set_params(ticker,'New IPO',prop_data,tickers_data,all_props)
    if len(bbminute_candles)==0:
        prop_data, tickers_data, all_props = set_params(ticker,'Fairly New IPO',prop_data,tickers_data,all_props)

    if len(bminute_candles)>0 and len(bbminute_candles)>0:
        y_avg = bminute_candles['range'].mean()
        yy_avg = bbminute_candles['range'].mean()
        avg_multiple = 3

        if minute_candles.iloc[0]['range'] > y_avg*avg_multiple:
            prop_data, tickers_data, all_props = set_params(ticker,'Range Above Average',prop_data,tickers_data,all_props)
            if minute_candles.iloc[0]['range'] > yy_avg*avg_multiple:
                prop_data, tickers_data, all_props = set_params(ticker,'Range Above 2 Day Average',prop_data,tickers_data,all_props)
        if minute_candles.iloc[0]['range'] < y_avg*avg_multiple:
            prop_data, tickers_data, all_props = set_params(ticker,'Range Lower Average',prop_data,tickers_data,all_props)
            if minute_candles.iloc[0]['range'] < yy_avg*avg_multiple:
                prop_data, tickers_data, all_props = set_params(ticker,'Range Lower 2 Day Average',prop_data,tickers_data,all_props)
        if minute_candles.iloc[0]['low']>bbminute_candles['high'].max():
            prop_data, tickers_data, all_props = set_params(ticker,'Open Higher Than 2 Prev Max',prop_data,tickers_data,all_props)
        if minute_candles.iloc[0]['low']<bbminute_candles['high'].max():
            prop_data, tickers_data, all_props = set_params(ticker,'Open Lower Than 2 Prev Max',prop_data,tickers_data,all_props)
        if red_candle(bminute_candles.iloc[-1]):
            prop_data, tickers_data, all_props = set_params(ticker,'Yesterday End In Red',prop_data,tickers_data,all_props)
        if bminute_candles.iloc[-1]['range'] > y_avg:
            prop_data, tickers_data, all_props = set_params(ticker,'Yesterday End Volume Above Average',prop_data,tickers_data,all_props)
        if body_top(minute_candles.iloc[0]) < body_bottom(bminute_candles.iloc[-1]):
            prop_data, tickers_data, all_props = set_params(ticker,'Gap Down',prop_data,tickers_data,all_props)
            if minute_candles.iloc[0]['range'] > body_bottom(bminute_candles.iloc[-1]) - body_top(minute_candles.iloc[0]):
                prop_data, tickers_data, all_props = set_params(ticker,'Range More Than Gap Down',prop_data,tickers_data,all_props)
            if body_bottom(minute_candles.iloc[0]) < bminute_candles['low'].min():
                prop_data, tickers_data, all_props = set_params(ticker,'Gap Down Below Prev Min',prop_data,tickers_data,all_props)
        if body_top(minute_candles.iloc[0]) + y_avg < body_bottom(bminute_candles.iloc[-1]):
            prop_data, tickers_data, all_props = set_params(ticker,'Gap Down Above Average',prop_data,tickers_data,all_props)
        if body_top(minute_candles.iloc[0]) + yy_avg < body_bottom(bminute_candles.iloc[-1]):
            prop_data, tickers_data, all_props = set_params(ticker,'Gap Down Above 2 Day Average',prop_data,tickers_data,all_props)
        if body_bottom(minute_candles.iloc[0]) > body_top(bminute_candles.iloc[-1]):
            prop_data, tickers_data, all_props = set_params(ticker,'Gap Up',prop_data,tickers_data,all_props)
            if minute_candles.iloc[0]['range'] > body_bottom(minute_candles.iloc[0]) - body_top(bminute_candles.iloc[-1]):
                prop_data, tickers_data, all_props = set_params(ticker,'Range More Than Gap Up',prop_data,tickers_data,all_props)
            if body_bottom(minute_candles.iloc[0]) > bminute_candles['high'].max():
                prop_data, tickers_data, all_props = set_params(ticker,'Gap Up Above Prev Max',prop_data,tickers_data,all_props)
        if body_bottom(minute_candles.iloc[0]) > body_top(bminute_candles.iloc[-1]) + y_avg:
            prop_data, tickers_data, all_props = set_params(ticker,'Gap Up Above Average',prop_data,tickers_data,all_props)
        if body_bottom(minute_candles.iloc[0]) > body_top(bminute_candles.iloc[-1]) + yy_avg:
            prop_data, tickers_data, all_props = set_params(ticker,'Gap Up Above 2 Day Average',prop_data,tickers_data,all_props)
        if minute_candles.iloc[0]['open']>bminute_candles['high'].max():
            prop_data, tickers_data, all_props = set_params(ticker,'Open Higher Than Prev Max',prop_data,tickers_data,all_props)
        if minute_candles.iloc[0]['open']>bminute_candles['high'].max() + y_avg:
            prop_data, tickers_data, all_props = set_params(ticker,'Open Higher Than Prev Max Plus Average',prop_data,tickers_data,all_props)
        if minute_candles.iloc[0]['open']<bminute_candles['low'].min() :
            prop_data, tickers_data, all_props = set_params(ticker,'Open Lower Than Prev Min',prop_data,tickers_data,all_props)
        if minute_candles.iloc[0]['open']<bminute_candles['low'].min() - y_avg :
            prop_data, tickers_data, all_props = set_params(ticker,'Open Lower Than Prev Min Minus Average',prop_data,tickers_data,all_props)
        if minute_candles.iloc[0]['volume']>bminute_candles.iloc[-1]['volume']:
            prop_data, tickers_data, all_props = set_params(ticker,'Volume Open Higher',prop_data,tickers_data,all_props)
        if minute_candles.iloc[0]['volume']<bminute_candles.iloc[-1]['volume']:
            prop_data, tickers_data, all_props = set_params(ticker,'Volume Open Lower',prop_data,tickers_data,all_props)
        if minute_candles.iloc[0]['volume']>bminute_candles['volume'].mean()*1.5:
            prop_data, tickers_data, all_props = set_params(ticker,'Volume Higher Than Average',prop_data,tickers_data,all_props)
            if minute_candles.iloc[0]['volume']>bminute_candles['volume'].mean()*5:
                prop_data, tickers_data, all_props = set_params(ticker,'Volume Above 5 Time Average',prop_data,tickers_data,all_props)
            if minute_candles.iloc[0]['volume']>bbminute_candles['volume'].mean()*5:
                prop_data, tickers_data, all_props = set_params(ticker,'Volume Above 5 Time Before Average',prop_data,tickers_data,all_props)
                if 'Volume Above 5 Time Average' in tickers_data[ticker]:
                    prop_data, tickers_data, all_props = set_params(ticker,'Volume Consecutive Above 5 Time Average',prop_data,tickers_data,all_props)
            if minute_candles.iloc[0]['volume']>bminute_candles['volume'].mean()*10:
                prop_data, tickers_data, all_props = set_params(ticker,'Volume Above 10 Time Average',prop_data,tickers_data,all_props)
            if minute_candles.iloc[0]['volume']>bbminute_candles['volume'].mean()*10:
                prop_data, tickers_data, all_props = set_params(ticker,'Volume Above 10 Time Before Average',prop_data,tickers_data,all_props)
                if 'Volume Above 10 Time Average' in tickers_data[ticker]:
                    prop_data, tickers_data, all_props = set_params(ticker,'Volume Consecutive Above 10 Time Average',prop_data,tickers_data,all_props)
        if minute_candles.iloc[0]['volume']<bminute_candles['volume'].mean()*0.5:
            prop_data, tickers_data, all_props = set_params(ticker,'Volume Lower Than Average',prop_data,tickers_data,all_props)
            if 'Volume Open Lower' in tickers_data[ticker]:
                prop_data, tickers_data, all_props = set_params(ticker,'Volume Open Excedingly Low',prop_data,tickers_data,all_props)
            else:
                prop_data, tickers_data, all_props = set_params(ticker,'Volume Open After Low',prop_data,tickers_data,all_props)

        bpeaks,bbottoms = gather_range(bminute_candles)
        if len(bpeaks)>0 and len(bbottoms)>0:
            morningpeak = None
            morningbottom = None
            afternoonpeak = None
            afternoonbottom = None
            for peak in bpeaks:
                if peak['date'].hour<13:
                    if morningpeak is not None and morningpeak['high'] < peak['high']:
                        morningpeak = peak
                    else:
                        morningpeak = peak
                else:
                    if afternoonpeak is not None and afternoonpeak['high'] < peak['high']:
                        afternoonpeak = peak
                    else:
                        afternoonpeak = peak
            for bottom in bbottoms:
                if bottom['date'].hour<13:
                    if morningbottom is not None and morningbottom['low'] > bottom['low']:
                        morningbottom = bottom
                    else:
                        morningbottom = bottom
                else:
                    if afternoonbottom is not None and afternoonbottom['low'] > bottom['low']:
                        afternoonbottom = bottom
                    else:
                        afternoonbottom = bottom
            morningrange = None
            afternoonrange = None
            if morningpeak is not None and morningbottom is not None:
                morningrange = morningpeak['high'] - morningbottom['low']
                if morningpeak['date']<morningbottom['date']:
                    prop_data, tickers_data, all_props = set_params(ticker,'Yesterday Negative Morning Range',prop_data,tickers_data,all_props)
                else:
                    prop_data, tickers_data, all_props = set_params(ticker,'Yesterday Positive Morning Range',prop_data,tickers_data,all_props)

            if afternoonpeak is not None and afternoonbottom is not None:
                afternoonrange = afternoonpeak['high'] - afternoonbottom['low']
                if afternoonpeak['date']<afternoonbottom['date']:
                    prop_data, tickers_data, all_props = set_params(ticker,'Yesterday Negative Afternoon Range',prop_data,tickers_data,all_props)
                else:
                    prop_data, tickers_data, all_props = set_params(ticker,'Yesterday Positive Afternoon Range',prop_data,tickers_data,all_props)
            if morningrange is not None and afternoonrange is not None:
                if morningrange>afternoonrange:
                    prop_data, tickers_data, all_props = set_params(ticker,'Yesterday Morning Range Larger',prop_data,tickers_data,all_props)
                elif afternoonrange>morningrange:
                    prop_data, tickers_data, all_props = set_params(ticker,'Yesterday Afternoon Range Larger',prop_data,tickers_data,all_props)


            maxp = max_peak(bpeaks,[bminute_candles.iloc[0]])
            minp = min_bottom(bbottoms,[bminute_candles.iloc[0]])
            if maxp is not None and minp is not None and maxp['high']>body_top(bminute_candles.iloc[1]) and maxp['date']>bminute_candles.iloc[1]['date']:
                ydiff = body_top(maxp) - body_top(bminute_candles.iloc[1])
                if ydiff > 5:
                    tcat = 'Great'
                elif ydiff > 1:
                    tcat = 'Good'
                elif ydiff > 0:
                    tcat = 'Fair'
                else:
                    tcat = 'Fail'
                prop_data, tickers_data, all_props = set_params(ticker,'Yesterday Status ' + tcat,prop_data,tickers_data,all_props)
                if ydiff > 0.7:
                    profitable = 1
                    prop_data, tickers_data, all_props = set_params(ticker,'Yesterday Profitable',prop_data,tickers_data,all_props)
                else:
                    profitable = 0
                    prop_data, tickers_data, all_props = set_params(ticker,'Yesterday Loss',prop_data,tickers_data,all_props)
            else:
                prop_data, tickers_data, all_props = set_params(ticker,'Yesterday Absolute Loss',prop_data,tickers_data,all_props)

        bpeaks,bbottoms = gather_range(bbminute_candles)
        if len(bpeaks)>0 and len(bbottoms)>0:
            morningpeak = None
            morningbottom = None
            afternoonpeak = None
            afternoonbottom = None
            for peak in bpeaks:
                if peak['date'].hour<13:
                    if morningpeak is not None and morningpeak['high'] < peak['high']:
                        morningpeak = peak
                    else:
                        morningpeak = peak
                else:
                    if afternoonpeak is not None and afternoonpeak['high'] < peak['high']:
                        afternoonpeak = peak
                    else:
                        afternoonpeak = peak
            for bottom in bbottoms:
                if bottom['date'].hour<13:
                    if morningbottom is not None and morningbottom['low'] > bottom['low']:
                        morningbottom = bottom
                    else:
                        morningbottom = bottom
                else:
                    if afternoonbottom is not None and afternoonbottom['low'] > bottom['low']:
                        afternoonbottom = bottom
                    else:
                        afternoonbottom = bottom
            morningrange = None
            afternoonrange = None
            if morningpeak is not None and morningbottom is not None:
                morningrange = morningpeak['high'] - morningbottom['low']
                if morningpeak['date']<morningbottom['date']:
                    prop_data, tickers_data, all_props = set_params(ticker,'2 Days Ago Negative Morning Range',prop_data,tickers_data,all_props)
                    if 'Yesterday Negative Morning Range' in tickers_data[ticker]:
                        prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Negative Morning Range',prop_data,tickers_data,all_props)
                else:
                    prop_data, tickers_data, all_props = set_params(ticker,'2 Days Ago Positive Morning Range',prop_data,tickers_data,all_props)
                    if 'Yesterday Positive Morning Range' in tickers_data[ticker]:
                        prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Positive Morning Range',prop_data,tickers_data,all_props)

            if afternoonpeak is not None and afternoonbottom is not None:
                afternoonrange = afternoonpeak['high'] - afternoonbottom['low']
                if afternoonpeak['date']<afternoonbottom['date']:
                    prop_data, tickers_data, all_props = set_params(ticker,'2 Days Ago Negative Afternoon Range',prop_data,tickers_data,all_props)
                    if 'Yesterday Negative Afternoon Range' in tickers_data[ticker]:
                        prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Negative Afternoon Range',prop_data,tickers_data,all_props)
                else:
                    prop_data, tickers_data, all_props = set_params(ticker,'2 Days Ago Positive Afternoon Range',prop_data,tickers_data,all_props)
                    if 'Yesterday Positive Afternoon Range' in tickers_data[ticker]:
                        prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Positive Afternoon Range',prop_data,tickers_data,all_props)
            if morningrange is not None and afternoonrange is not None:
                if morningrange>afternoonrange:
                    prop_data, tickers_data, all_props = set_params(ticker,'2 Days Ago Morning Range Larger',prop_data,tickers_data,all_props)
                    if 'Yesterday Morning Range Large' in tickers_data[ticker]:
                        prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Morning Range Larger',prop_data,tickers_data,all_props)
                elif afternoonrange>morningrange:
                    prop_data, tickers_data, all_props = set_params(ticker,'2 Days Ago Afternoon Range Larger',prop_data,tickers_data,all_props)
                    if 'Yesterday Afternoon Range Large' in tickers_data[ticker]:
                        prop_data, tickers_data, all_props = set_params(ticker,'Consecutive Afternoon Range Larger',prop_data,tickers_data,all_props)
            maxp = max_peak(bpeaks,[bbminute_candles.iloc[0]])
            minp = min_bottom(bbottoms,[bbminute_candles.iloc[0]])
            if maxp is not None and minp is not None and maxp['high']>body_top(bbminute_candles.iloc[1]) and maxp['date']>bbminute_candles.iloc[1]['date']:
                ydiff = body_top(maxp) - body_top(bbminute_candles.iloc[1])
                if ydiff > 5:
                    tcat = 'Great'
                elif ydiff > 1:
                    tcat = 'Good'
                elif ydiff > 0:
                    tcat = 'Fair'
                else:
                    tcat = 'Fail'
                prop_data, tickers_data, all_props = set_params(ticker,'2 Days Ago Status ' + tcat,prop_data,tickers_data,all_props)
                if ydiff > 0.7:
                    profitable = 1
                    prop_data, tickers_data, all_props = set_params(ticker,'2 Days Ago Profitable',prop_data,tickers_data,all_props)
                else:
                    profitable = 0
            else:
                prop_data, tickers_data, all_props = set_params(ticker,'2 Days Ago Absolute Loss',prop_data,tickers_data,all_props)

    curdiff = max_price - first_price
    if curdiff > 5:
        tcat = 'Great'
    elif curdiff > 1:
        tcat = 'Good'
    elif curdiff > 0:
        tcat = 'Fair'
    else:
        tcat = 'Fail'
    if curdiff > 0.7:
        profitable = 1
    else:
        profitable = 0
    dlvl = str(round(curdiff,1))
    try:
        gap = minute_candles.iloc[0]['open']-bminute_candles.iloc[-1]['close']
    except:
        gap = 0
    final_price = minute_candles.iloc[-1]['close']
    summary = {'max_price':max_price,'first_price':first_price,'category':tcat,'profitable':profitable,'diff':curdiff,'diff_level':dlvl,'gap':gap,'final_price':final_price}

    return prop_data, tickers_data, all_props, summary

def calc_marks(proparray):
    global_marks = pd.read_csv(os.path.join(script_dir,'analyze_global.csv'))
    proparray['prev_marks'] = 1.0
    print("Mean profitable:",global_marks['Profitable'].mean())
    for prop in prev_prop_list:
        cgmark = global_marks[global_marks['Prop']==prop]
        if len(cgmark):
            curmark = 0
            if cgmark.iloc[0]['Profitable'] > global_marks['Profitable'].mean():
                curmark += cgmark.iloc[0]['Profitable'] * 2
            curmark += cgmark.iloc[0]['Good'] * 4
            curmark += cgmark.iloc[0]['Great'] * 4
            curmark *= cgmark.iloc[0]['Corr']
            proparray.loc[proparray[prop]==1,'prev_marks'] += curmark
    proparray['opening_marks'] = 1.0
    for prop in opening_prop_list:
        cgmark = global_marks[global_marks['Prop']==prop]
        if len(cgmark):
            curmark = 0
            if cgmark.iloc[0]['Profitable'] > global_marks['Profitable'].mean():
                curmark += cgmark.iloc[0]['Profitable'] * 2
            curmark += cgmark.iloc[0]['Good'] * 4
            curmark += cgmark.iloc[0]['Great'] * 4
            curmark *= cgmark.iloc[0]['Corr']
            proparray.loc[proparray[prop]==1,'opening_marks'] += curmark
    proparray['late_marks'] = 1.0
    for prop in late_prop_list:
        cgmark = global_marks[global_marks['Prop']==prop]
        if len(cgmark):
            curmark = 0
            if cgmark.iloc[0]['Profitable'] > global_marks['Profitable'].mean():
                curmark += cgmark.iloc[0]['Profitable'] * 2
            curmark += cgmark.iloc[0]['Good'] * 4
            curmark += cgmark.iloc[0]['Great'] * 4
            curmark *= cgmark.iloc[0]['Corr']
            proparray.loc[proparray[prop]==1,'late_marks'] += curmark
 
    proparray['marks'] = proparray['prev_marks'] + proparray['opening_marks'] + proparray['late_marks']
    return proparray
