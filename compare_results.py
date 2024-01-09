#!/usr/bin/env python

import os
import pandas as pd 
from props import *
from tabulate import tabulate
import sys

if len(sys.argv)<3:
    print("Need two file names to compare")
    exit()
print("First file:",sys.argv[1]," Second file:",sys.argv[2])
first_df = pd.read_csv(sys.argv[1])
second_df = pd.read_csv(sys.argv[2])
first_result = calc_marks(first_df,verbose_part='prev_prop_list')
print("===================================================================")
second_result = calc_marks(second_df,verbose_part='prev_prop_list')
first_result = first_result.loc[first_result['ticker']=='AAOI']
second_result = second_result.loc[second_result['ticker']=='AAOI']
print("first:",first_result)
print("second:",second_result)
print("first:",first_result['prev_marks'].values)
print("second:",second_result['prev_marks'].values)
todisp = []
global_marks = pd.read_csv(os.path.join(script_dir,'analyze_global.csv'))
for prop in prev_prop_list:
    todisp.append({'Prop':prop,'First':first_df[prop].values,'Second':second_df[prop].values})
# print(tabulate(todisp,headers='keys'))


