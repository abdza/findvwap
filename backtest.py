#!/usr/bin/env python

from logging import lastResort
import pandas as pd 
import os
import json
import sys
import csv
import getopt
from datetime import datetime,timedelta
from pandas.core.indexing import convert_missing_indexer
import yahooquery as yq
import numpy as np
import math
from tabulate import tabulate
from numerize import numerize
from sklearn.cluster import KMeans
from ta.trend import EMAIndicator
import streamlit as st
from streamlit_calendar import calendar
from props import *
from sklearn.preprocessing import MinMaxScaler

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

starttest = datetime.now()

result_perc = pd.read_csv(os.path.join(script_dir,'raw_data_perc.csv'))

result_perc = calc_marks(result_perc)

fieldnames = ['date','ticker','diff_level','performance','profitable','marks','prev_marks','opening_marks','late_marks','gap']
minuscolumns = list(set(result_perc.columns.to_list()) - set(fieldnames))
finalcolumns = fieldnames + sorted(minuscolumns)

result_perc = result_perc[finalcolumns]

result_perc.to_csv(os.path.join(script_dir,'raw_data_perc_marks.csv'),index=False)
endtest = datetime.now()
print("Start:",starttest)
print("End:",endtest)
print("Time:",endtest-starttest)
